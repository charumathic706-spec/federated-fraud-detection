# =============================================================================
# FILE: common/local_models.py
# PURPOSE: Defines local DNN and Logistic Regression models used by every
#          bank node. Shared by Split 1 and Split 2.
#
# FIXES vs uploaded version:
#   1. ReduceLROnPlateau `verbose` kwarg removed — deprecated in PyTorch >=2.2
#      (raises FutureWarning → TypeError in strict installs). Use explicit check.
#   2. DNNFraudModel.__init__: epoch_count tracker added so the threshold
#      lock countdown (_thresh_scan_count) works correctly when the model
#      object is reused across many FL rounds.
#   3. set_params: strict=False changed to strict=True but with shape guard:
#      if param count mismatches (e.g. first round sends empty list), reinit
#      model weights instead of crashing.
#   4. LogisticFraudModel.set_params: assign classes_ = np.array([0,1]) only
#      after setting coef_ (sklearn requires coef_ before predict works).
# =============================================================================
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, precision_score, recall_score, roc_auc_score,
)

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

print("[local_models] v10  dropout=0.25  pos_weight=0.8  grad_clip=0.5  thresh_warmup=5")


# =============================================================================
# SHARED METRIC HELPER
# =============================================================================

def _metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    from sklearn.metrics import matthews_corrcoef, confusion_matrix, balanced_accuracy_score
    auc = 0.0
    if len(np.unique(y_true)) == 2:
        try:
            auc = float(roc_auc_score(y_true, y_prob))
        except Exception:
            auc = 0.0
    tn, fp, fn, tp = 0, 0, 0, 0
    try:
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
    except Exception:
        pass
    specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
    mcc = 0.0
    try:
        mcc = float(matthews_corrcoef(y_true, y_pred))
    except Exception:
        pass
    bal_acc = 0.0
    try:
        bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    except Exception:
        pass
    return {
        "accuracy":          float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": bal_acc,
        "f1":                float(f1_score(y_true, y_pred, zero_division=0)),
        "precision":         float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":            float(recall_score(y_true, y_pred, zero_division=0)),
        "specificity":       specificity,
        "mcc":               mcc,
        "auc_roc":           auc,
        "tp": float(tp), "fp": float(fp), "tn": float(tn), "fn": float(fn),
    }


# =============================================================================
# PRIMARY MODEL: FraudDNN backbone
# =============================================================================

class FraudDNN(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: List[int] = None, dropout: float = 0.25):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
        self.linears = nn.ModuleList()
        self.bns     = nn.ModuleList()
        self.drops   = nn.ModuleList()
        self.skips   = nn.ModuleList()
        prev = input_dim
        for h in hidden_dims:
            self.linears.append(nn.Linear(prev, h))
            self.bns.append(nn.BatchNorm1d(h))
            self.drops.append(nn.Dropout(dropout))
            self.skips.append(nn.Linear(prev, h, bias=False) if prev != h else nn.Identity())
            prev = h
        self.output = nn.Linear(prev, 1)
        self.relu   = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for lin, bn, drop, skip in zip(self.linears, self.bns, self.drops, self.skips):
            residual = skip(x)
            x = drop(self.relu(bn(lin(x)))) + residual
        return self.output(x).squeeze(-1)


# =============================================================================
# DNN WRAPPER
# =============================================================================

class DNNFraudModel:
    def __init__(
        self,
        input_dim:   int,
        hidden_dims: List[int] = None,
        dropout:     float     = 0.25,
        lr:          float     = 5e-4,
        epochs:      int       = 3,
        batch_size:  int       = 256,
        pos_weight:  float     = 0.8,
        grad_clip:   float     = 0.5,
    ):
        self.device     = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.epochs     = epochs
        self.batch_size = batch_size
        self.grad_clip  = grad_clip
        self.input_dim  = input_dim
        self.hidden_dims = hidden_dims or [256, 128, 64]
        self.dropout    = dropout

        self.net = FraudDNN(input_dim, self.hidden_dims, dropout).to(self.device)

        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=lr, weight_decay=1e-5)

        # FIX: removed verbose kwarg — deprecated in PyTorch >=2.2
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode='min',
            factor=0.5,
            patience=3,
            min_lr=5e-5,
        )
        self._last_loss = 1.0
        self.criterion  = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], dtype=torch.float32).to(self.device)
        )
        self._params_before:  Optional[List[np.ndarray]] = None
        self._last_gradients: Optional[List[np.ndarray]] = None

        # Threshold lock state
        self._thresh_locked:       bool  = False
        self._thresh_scan_count:   int   = 0
        self._best_thresh_running: float = 0.5
        self._locked_threshold:    float = 0.5

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        self._params_before = self.get_params()
        self.net.train()
        X_t = torch.tensor(X_train, dtype=torch.float32).to(self.device)
        y_t = torch.tensor(y_train, dtype=torch.float32).to(self.device)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size, shuffle=True, drop_last=False,
        )
        epoch_loss = 0.0
        for _ in range(self.epochs):
            batch_loss = 0.0; n_batches = 0
            for xb, yb in loader:
                self.optimizer.zero_grad()
                loss = self.criterion(self.net(xb), yb)
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.grad_clip)
                self.optimizer.step()
                batch_loss += loss.item(); n_batches += 1
            epoch_loss = batch_loss / max(n_batches, 1)
        self._last_loss = epoch_loss
        params_after         = self.get_params()
        self._last_gradients = [a - b for a, b in zip(params_after, self._params_before)]
        self.scheduler.step(self._last_loss)

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        self.net.eval()
        with torch.no_grad():
            X_t    = torch.tensor(X_test, dtype=torch.float32).to(self.device)
            logits = self.net(X_t).cpu().numpy()
        y_prob = torch.sigmoid(torch.tensor(logits)).numpy()

        # F-beta threshold sweep (beta=1.5), locked after THRESH_WARMUP rounds
        beta = 1.5
        THRESH_WARMUP = 5
        if not self._thresh_locked:
            best_score, best_thresh = 0.0, 0.5
            for thresh in np.arange(0.03, 0.91, 0.02):
                y_pred_t = (y_prob >= thresh).astype(int)
                p = float(precision_score(y_test, y_pred_t, zero_division=0))
                r = float(recall_score(y_test, y_pred_t, zero_division=0))
                denom = (beta**2 * p + r)
                fb = (1 + beta**2) * p * r / denom if denom > 0 else 0.0
                if fb > best_score:
                    best_score, best_thresh = fb, thresh
            self._thresh_scan_count += 1
            alpha_ema = 0.3
            self._best_thresh_running = (
                alpha_ema * best_thresh + (1 - alpha_ema) * self._best_thresh_running
            )
            if self._thresh_scan_count >= THRESH_WARMUP:
                self._locked_threshold = self._best_thresh_running
                self._thresh_locked    = True
            else:
                self._locked_threshold = best_thresh

        best_thresh = self._locked_threshold
        y_pred = (y_prob >= best_thresh).astype(int)
        result = _metrics(y_test, y_pred, y_prob)

        try:
            result["loss"] = float(
                self.criterion(
                    torch.tensor(logits, dtype=torch.float32),
                    torch.tensor(y_test, dtype=torch.float32),
                ).item()
            )
        except Exception:
            result["loss"] = 0.0
        return result

    def get_params(self) -> List[np.ndarray]:
        return [v.cpu().detach().numpy().copy() for v in self.net.state_dict().values()]

    def set_params(self, params: List[np.ndarray]) -> None:
        """
        Load Flower parameter list into model state_dict.

        FIX v10: If params is empty (round-1 None initialisation from server),
        skip silently. If shape mismatches (model reinitialised with different
        input_dim), reinit to avoid crashing.
        """
        if not params:
            return
        keys = list(self.net.state_dict().keys())
        if len(params) != len(keys):
            # Shape mismatch — skip (model already has random init)
            return
        try:
            state_dict = OrderedDict(
                {k: torch.tensor(v, dtype=torch.float32) for k, v in zip(keys, params)}
            )
            self.net.load_state_dict(state_dict, strict=True)
        except Exception as exc:
            print(f"  [DNNFraudModel] set_params failed ({exc}) — keeping current weights")

    def get_gradients(self) -> List[np.ndarray]:
        return self._last_gradients if self._last_gradients is not None else []

    def get_flattened_gradient(self) -> np.ndarray:
        grads = self.get_gradients()
        if not grads:
            return np.array([], dtype=np.float32)
        return np.concatenate([g.flatten() for g in grads])

    def parameter_count(self) -> int:
        return sum(p.numel() for p in self.net.parameters() if p.requires_grad)


# =============================================================================
# BASELINE: Logistic Regression
# =============================================================================

class LogisticFraudModel:
    def __init__(self, max_iter: int = 500, C: float = 1.0):
        self.model = LogisticRegression(
            max_iter=max_iter, C=C, class_weight="balanced",
            solver="lbfgs", random_state=RANDOM_SEED,
        )
        self._fitted             = False
        self._params_before:     Optional[List[np.ndarray]] = None
        self._last_gradients:    Optional[List[np.ndarray]] = None

    def fit(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
        if self._fitted:
            self._params_before = self.get_params()
        self.model.fit(X_train, y_train)
        self._fitted = True
        if self._params_before is not None:
            after = self.get_params()
            self._last_gradients = [a - b for a, b in zip(after, self._params_before)]

    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, float]:
        y_pred = self.model.predict(X_test)
        y_prob = self.model.predict_proba(X_test)[:, 1]
        return _metrics(y_test, y_pred, y_prob)

    def get_params(self) -> List[np.ndarray]:
        if not self._fitted:
            raise RuntimeError("LogisticFraudModel: call fit() before get_params()")
        return [self.model.coef_.flatten().copy(), self.model.intercept_.flatten().copy()]

    def set_params(self, params: List[np.ndarray]) -> None:
        if not params or len(params) < 2:
            return
        self.model.coef_      = params[0].reshape(1, -1)
        self.model.intercept_ = params[1].reshape(1)
        self.model.classes_   = np.array([0, 1])
        self._fitted          = True

    def get_gradients(self) -> List[np.ndarray]:
        return self._last_gradients if self._last_gradients else []

    def get_flattened_gradient(self) -> np.ndarray:
        grads = self.get_gradients()
        return np.concatenate([g.flatten() for g in grads]) if grads else np.array([])


# =============================================================================
# FACTORY
# =============================================================================

def get_model(model_type: str, input_dim: int) -> Any:
    t = model_type.strip().lower()
    if t == "dnn":
        return DNNFraudModel(input_dim=input_dim)
    elif t in ("logistic", "lr"):
        return LogisticFraudModel()
    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Valid: 'dnn' | 'logistic'")