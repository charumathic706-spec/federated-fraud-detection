# =============================================================================
# FILE: flower_client.py
# PURPOSE: Defines BankFederatedClient — the Flower client that represents a
#          single financial institution (bank node) in the federated network.
#
#          What it does each round:
#            1. Receives the current global model parameters from the server
#            2. Loads those parameters into its local DNN
#            3. Trains the DNN on its OWN private local transaction data
#            4. Returns the updated parameters + training metrics to the server
#            5. On evaluate() calls: runs inference on local held-out test set
#
#          PRIVACY GUARANTEE: Raw transaction data NEVER leaves the client.
#          Only model weight tensors (float32 arrays) are transmitted.
#
#          Also exports make_client_fn() — a factory closure required by
#          fl.simulation.start_simulation() in main.py.
# =============================================================================

from __future__ import annotations

import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns, EvaluateRes,
    FitIns, FitRes,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
# Context was added in flwr 1.5 — graceful fallback for older versions
try:
    from flwr.common import Context
    _HAVE_CONTEXT = True
except ImportError:
    Context = object   # dummy type annotation only
    _HAVE_CONTEXT = False
from typing import Dict, List

from local_models    import get_model
from data_partition  import apply_smote


# =============================================================================
# BANK FEDERATED CLIENT
# =============================================================================

class BankFederatedClient(fl.client.Client):
    """
    Flower client representing one bank node.

    Attributes:
        client_id  — integer identifier (0, 1, 2, ...)
        model      — local DNNFraudModel (or LogisticFraudModel)
        X_train / y_train — private local training data (never transmitted)
        X_test  / y_test  — private local test data (never transmitted)
    """

    def __init__(
        self,
        client_id:  int,
        X_train:    np.ndarray,
        y_train:    np.ndarray,
        X_test:     np.ndarray,
        y_test:     np.ndarray,
        model_type: str  = "dnn",
        use_smote:  bool = True,
    ):
        self.client_id  = client_id
        self.input_dim  = X_train.shape[1]

        # Optionally apply SMOTE to balance the local training set
        if use_smote:
            self.X_train, self.y_train = apply_smote(X_train, y_train)
        else:
            self.X_train, self.y_train = X_train.copy(), y_train.copy()

        self.X_test     = X_test.copy()
        self.y_test     = y_test.copy()
        self.n_train    = len(self.X_train)
        self.n_test     = len(self.X_test)

        # Instantiate local model via factory
        self.model = get_model(model_type, self.input_dim)

        print(
            f"  [Bank {client_id:02d}] Ready | model={model_type} | "
            f"train={self.n_train:,} | test={self.n_test:,} | "
            f"fraud_train={int(self.y_train.sum())} ({self.y_train.mean()*100:.1f}%)"
        )

    # -------------------------------------------------------------------------
    # FIT — called by Flower server each training round
    # -------------------------------------------------------------------------

    def fit(self, ins: FitIns) -> FitRes:
        """
        1. Deserialise global parameters and load into local model.
        2. Train locally on private data.
        3. Serialise updated parameters and return to server with metrics.
        """
        print(f"  [Bank {self.client_id:02d}] fit() round start")

        # Load global model weights into local model
        global_params = parameters_to_ndarrays(ins.parameters)
        if global_params:
            try:
                self.model.set_params(global_params)
            except Exception as exc:
                print(f"  [Bank {self.client_id:02d}] Note: set_params skipped ({exc})")

        # Local training
        self.model.fit(self.X_train, self.y_train)

        # Evaluate on training set for logging only
        train_metrics = self.model.evaluate(self.X_train, self.y_train)

        print(
            f"  [Bank {self.client_id:02d}] fit() done | "
            f"F1={train_metrics['f1']:.4f} | AUC={train_metrics['auc_roc']:.4f}"
        )

        return FitRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            parameters=ndarrays_to_parameters(self.model.get_params()),
            num_examples=self.n_train,
            metrics={
                "client_id":    float(self.client_id),
                "train_f1":     float(train_metrics["f1"]),
                "train_auc":    float(train_metrics["auc_roc"]),
                "train_recall": float(train_metrics["recall"]),
                "n_fraud":      float(int(self.y_train.sum())),
            },
        )

    # -------------------------------------------------------------------------
    # EVALUATE — called by Flower server to assess global model quality
    # -------------------------------------------------------------------------

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
        """
        1. Load received global parameters into local model.
        2. Run inference on local held-out test set.
        3. Return loss + metrics to server.
        """
        print(f"  [Bank {self.client_id:02d}] evaluate() start")

        global_params = parameters_to_ndarrays(ins.parameters)
        if global_params:
            try:
                self.model.set_params(global_params)
            except Exception as exc:
                print(f"  [Bank {self.client_id:02d}] Note: eval set_params skipped ({exc})")

        metrics = self.model.evaluate(self.X_test, self.y_test)

        print(
            f"  [Bank {self.client_id:02d}] evaluate() done | "
            f"F1={metrics['f1']:.4f} | AUC={metrics['auc_roc']:.4f} | "
            f"Recall={metrics['recall']:.4f}"
        )

        # loss = 1 - F1  (higher F1 = lower loss)
        loss = float(1.0 - metrics["f1"])

        return EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=loss,
            num_examples=self.n_test,
            metrics={
                "client_id": float(self.client_id),
                "f1":        float(metrics["f1"]),
                "auc_roc":   float(metrics["auc_roc"]),
                "precision": float(metrics["precision"]),
                "recall":    float(metrics["recall"]),
                "accuracy":  float(metrics["accuracy"]),
            },
        )

    def to_client(self) -> "BankFederatedClient":
        """
        Required by fl.client.start_client() when using the gRPC transport.
        Flower calls to_client() on the object passed to start_client() to
        get the raw Client instance. Since BankFederatedClient already IS a
        fl.client.Client, we just return self.
        """
        return self


# =============================================================================
# CLIENT FACTORY — passed to fl.simulation.start_simulation()
# =============================================================================

def make_client_fn(
    partitions: List[Dict],
    model_type: str  = "dnn",
    use_smote:  bool = True,
):
    """
    Returns a closure client_fn(context: Context) → BankFederatedClient.

    FIXED: Flower's simulation engine now requires the new Context-based
    signature:  def client_fn(context: Context) -> Client
    The old signature  def client_fn(cid: str) -> Client  is deprecated and
    causes a WARNING flood plus will break in future Flower versions.

    The client's integer ID is read from context.node_config["partition-id"]
    which Flower sets automatically during simulation.

    Args:
        partitions: Output of dirichlet_partition() from data_partition.py
        model_type: "dnn" (recommended) | "logistic"
        use_smote:  Apply SMOTE oversampling on local training data
    """
    def client_fn(context_or_cid) -> BankFederatedClient:
        # Support both old API (cid: str) and new API (context: Context)
        if _HAVE_CONTEXT and isinstance(context_or_cid, Context):
            cid = int(context_or_cid.node_config["partition-id"])
        else:
            cid = int(context_or_cid)   # old API: cid passed as str
        partition = partitions[cid]
        return BankFederatedClient(
            client_id  = cid,
            X_train    = partition["X_train"],
            y_train    = partition["y_train"],
            X_test     = partition["X_test"],
            y_test     = partition["y_test"],
            model_type = model_type,
            use_smote  = use_smote,
        )

    return client_fn