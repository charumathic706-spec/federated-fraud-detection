# =============================================================================
# FILE: common/flower_client.py
# PURPOSE: BankFederatedClient — Flower client for one bank node.
#          Shared by Split 1 and Split 2.
#
# FIXES vs uploaded flower_client.py:
#   1. Imports corrected to use common.local_models / common.data_partition
#   2. set_params: now guarded — empty params (round-1 None server init) skipped
#   3. to_client() returns self — correct for fl.client.start_client() gRPC usage
#   4. make_client_fn removed from this file (only used in old simulation API,
#      not in gRPC subprocess architecture)
# =============================================================================
from __future__ import annotations

import numpy as np
import flwr as fl
from flwr.common import (
    EvaluateIns, EvaluateRes, FitIns, FitRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from typing import Dict, List

from common.local_models   import get_model
from common.data_partition import apply_smote


class BankFederatedClient(fl.client.Client):
    """
    Flower client representing one bank node.

    Each round:
      1. Receives global model parameters from server
      2. Loads those parameters into local DNN
      3. Trains on private local transaction data
      4. Returns updated parameters + metrics to server
      5. On evaluate(): runs inference on local held-out test set
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
        self.client_id = client_id
        self.input_dim = X_train.shape[1]

        if use_smote:
            self.X_train, self.y_train = apply_smote(X_train, y_train)
        else:
            self.X_train, self.y_train = X_train.copy(), y_train.copy()

        self.X_test  = X_test.copy()
        self.y_test  = y_test.copy()
        self.n_train = len(self.X_train)
        self.n_test  = len(self.X_test)

        self.model = get_model(model_type, self.input_dim)

        print(
            f"  [Bank {client_id:02d}] Ready | model={model_type} | "
            f"train={self.n_train:,} | test={self.n_test:,} | "
            f"fraud_train={int(self.y_train.sum())} ({self.y_train.mean()*100:.1f}%)"
        )

    def fit(self, ins: FitIns) -> FitRes:
        print(f"  [Bank {self.client_id:02d}] fit() round start")

        global_params = parameters_to_ndarrays(ins.parameters)
        if global_params:
            try:
                self.model.set_params(global_params)
            except Exception as exc:
                print(f"  [Bank {self.client_id:02d}] Note: set_params skipped ({exc})")

        self.model.fit(self.X_train, self.y_train)
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

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:
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

        loss = float(1.0 - metrics["f1"])

        return EvaluateRes(
            status=fl.common.Status(code=fl.common.Code.OK, message="Success"),
            loss=loss,
            num_examples=self.n_test,
            metrics={
                "client_id":         float(self.client_id),
                "f1":                float(metrics.get("f1", 0)),
                "auc_roc":           float(metrics.get("auc_roc", 0)),
                "precision":         float(metrics.get("precision", 0)),
                "recall":            float(metrics.get("recall", 0)),
                "accuracy":          float(metrics.get("accuracy", 0)),
                "balanced_accuracy": float(metrics.get("balanced_accuracy", 0)),
                "mcc":               float(metrics.get("mcc", 0)),
                "specificity":       float(metrics.get("specificity", 0)),
                "tp":  float(metrics.get("tp", 0)),
                "fp":  float(metrics.get("fp", 0)),
                "tn":  float(metrics.get("tn", 0)),
                "fn":  float(metrics.get("fn", 0)),
            },
        )

    def to_client(self) -> "BankFederatedClient":
        """
        Required by fl.client.start_client() gRPC transport.
        BankFederatedClient IS a fl.client.Client — return self.
        """
        return self