# =============================================================================
# FILE: common/fedavg_strategy.py
# PURPOSE: Server-side FedAvg strategy for Split 1. Also re-used by Split 2.
#
# FIXES vs uploaded version:
#   1. Moved to common/ — imported by both split1/main.py and split2/main.py
#   2. No functional changes — logic was correct; only import path updated
# =============================================================================
from __future__ import annotations

import json
import os
from typing import Dict, List, Optional, Tuple, Union

import flwr as fl
import numpy as np
from flwr.common import EvaluateRes, FitRes, Metrics, Parameters, Scalar
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import FedAvg


def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    """
    Sample-count-weighted average over client metric dicts.
    AUC-ROC is weighted only over active (F1>0) clients to avoid dead-client drag.
    """
    total = sum(n for n, _ in metrics)
    if total == 0:
        return {}
    keys = [k for k in metrics[0][1].keys() if k != "client_id"]
    result: Dict[str, float] = {}
    for key in keys:
        if key == "auc_roc":
            active = [(n, m) for n, m in metrics if float(m.get("f1", 0)) > 0]
            if active:
                active_total = sum(n for n, _ in active)
                result[key]  = sum(n * float(m.get(key, 0.0)) for n, m in active) / active_total
            else:
                result[key]  = sum(n * float(m.get(key, 0.0)) for n, m in metrics) / total
        else:
            result[key] = sum(n * float(m.get(key, 0.0)) for n, m in metrics) / total
    return result


class InstrumentedFedAvg(FedAvg):
    """
    Standard FedAvg with full metric logging and recall-weighted aggregation.
    """

    def __init__(
        self,
        num_clients:           int,
        fraction_fit:          float = 1.0,
        fraction_evaluate:     float = 1.0,
        min_fit_clients:       int   = 2,
        min_evaluate_clients:  int   = 2,
        min_available_clients: int   = 2,
        log_dir:               str   = "logs",
    ):
        super().__init__(
            fraction_fit=fraction_fit,
            fraction_evaluate=fraction_evaluate,
            min_fit_clients=min_fit_clients,
            min_evaluate_clients=min_evaluate_clients,
            min_available_clients=min_available_clients,
            fit_metrics_aggregation_fn=weighted_average,
            evaluate_metrics_aggregation_fn=weighted_average,
        )
        self.num_clients = num_clients
        self.log_dir     = log_dir
        self.round_logs: List[Dict] = []
        self.best_f1     = 0.0
        self.best_round  = 0

        os.makedirs(log_dir, exist_ok=True)
        print(
            f"\n[Server] FedAvg ready | {num_clients} clients | "
            f"logs → {log_dir}/training_log.json\n"
        )

    def aggregate_fit(
        self,
        server_round: int,
        results:  List[Tuple[ClientProxy, FitRes]],
        failures: List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Recall-weighted FedAvg aggregation."""
        from flwr.common import parameters_to_ndarrays, ndarrays_to_parameters

        sep = "=" * 62
        print(f"\n{sep}")
        print(
            f"  Round {server_round:02d}  |  FIT (recall-weighted)  |  "
            f"{len(results)} clients  |  {len(failures)} failures"
        )
        print(sep)

        if not results:
            return None, {}

        client_data = []
        for _, fit_res in results:
            m      = fit_res.metrics
            cid    = int(m.get("client_id", -1))
            recall = float(m.get("train_recall", 0.0))
            params = parameters_to_ndarrays(fit_res.parameters)
            n      = fit_res.num_examples
            client_data.append({"cid": cid, "recall": recall,
                                 "params": params, "n": n, "metrics": m})
            print(
                f"  Bank {cid:02d} | samples={n:,} | recall={recall:.4f} | "
                f"train_F1={m.get('train_f1', 0):.4f} | "
                f"train_AUC={m.get('train_auc', 0):.4f} | "
                f"fraud_samples={int(m.get('n_fraud', 0))}"
            )

        recalls      = [cd["recall"] for cd in client_data]
        total_recall = sum(recalls)

        if total_recall > 1e-9:
            weights = [r / total_recall for r in recalls]
            print(f"\n  [Recall weights] " +
                  "  ".join(f"C{cd['cid']}={w:.3f}" for cd, w in zip(client_data, weights)))
        else:
            weights = [1.0 / len(client_data)] * len(client_data)
            print("  [Weights] fallback to uniform (no recall metric)")

        aggregated = [
            sum(w * arr for w, cd in zip(weights, client_data)
                for arr in [cd["params"][layer_idx]])
            for layer_idx in range(len(client_data[0]["params"]))
        ]
        agg_params = ndarrays_to_parameters(aggregated)

        agg_metrics: Dict[str, Scalar] = {}
        metric_keys = [k for k in client_data[0]["metrics"] if k != "client_id"]
        for key in metric_keys:
            agg_metrics[key] = sum(
                w * float(cd["metrics"].get(key, 0.0))
                for w, cd in zip(weights, client_data)
            )

        print(
            f"\n  [Aggregated train] "
            f"F1={agg_metrics.get('train_f1', 0):.4f}  "
            f"AUC={agg_metrics.get('train_auc', 0):.4f}  "
            f"Recall={agg_metrics.get('train_recall', 0):.4f}"
        )
        return agg_params, agg_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results:  List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        print(f"\n  Round {server_round:02d}  |  EVALUATE  |  {len(results)} clients")

        client_logs: List[Dict] = []
        for _, eval_res in results:
            m   = eval_res.metrics
            cid = int(m.get("client_id", -1))
            print(
                f"  Bank {cid:02d} | "
                f"F1={m.get('f1', 0):.4f} | AUC={m.get('auc_roc', 0):.4f} | "
                f"Recall={m.get('recall', 0):.4f} | Precision={m.get('precision', 0):.4f} | "
                f"Accuracy={m.get('accuracy', 0):.4f}"
            )
            client_logs.append({
                "client_id": cid,
                "f1":        float(m.get("f1", 0)),
                "auc_roc":   float(m.get("auc_roc", 0)),
                "recall":    float(m.get("recall", 0)),
                "precision": float(m.get("precision", 0)),
                "accuracy":  float(m.get("accuracy", 0)),
            })

        loss_agg, metrics_agg = super().aggregate_evaluate(server_round, results, failures)

        if metrics_agg:
            gf1  = float(metrics_agg.get("f1",        0))
            gauc = float(metrics_agg.get("auc_roc",   0))
            grec = float(metrics_agg.get("recall",    0))
            gpre = float(metrics_agg.get("precision", 0))
            gacc = float(metrics_agg.get("accuracy",  0))

            print(
                f"\n  [Global eval] "
                f"F1={gf1:.4f} | AUC={gauc:.4f} | "
                f"Recall={grec:.4f} | Precision={gpre:.4f} | Accuracy={gacc:.4f}"
            )

            if gf1 > self.best_f1:
                self.best_f1    = gf1
                self.best_round = server_round
                print(f"  ★  New best model! F1={gf1:.4f} at round {server_round}")

            self.round_logs.append({
                "round":            server_round,
                "global_f1":        gf1,
                "global_auc":       gauc,
                "global_recall":    grec,
                "global_precision": gpre,
                "global_accuracy":  gacc,
                "global_loss":      float(loss_agg) if loss_agg is not None else 0.0,
                "best_f1":          self.best_f1,
                "best_round":       self.best_round,
                "client_metrics":   client_logs,
            })
            self._save_log()

        return loss_agg, metrics_agg

    def _save_log(self) -> None:
        path = os.path.join(self.log_dir, "training_log.json")
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

    def print_summary(self) -> None:
        sep = "=" * 62
        print(f"\n{sep}")
        print(f"  FEDERATED TRAINING — SPLIT 1 COMPLETE")
        print(f"  Best Global F1   : {self.best_f1:.4f}  (round {self.best_round})")
        print(f"  Total rounds     : {len(self.round_logs)}")
        if self.round_logs:
            last = self.round_logs[-1]
            print(f"  Final round AUC  : {last['global_auc']:.4f}")
            print(f"  Final round Acc  : {last['global_accuracy']:.4f}")
        print(f"  Log              : {self.log_dir}/training_log.json")
        print(f"{sep}\n")


def get_fedavg_strategy(
    num_clients:  int,
    fraction_fit: float = 1.0,
    log_dir:      str   = "logs",
) -> InstrumentedFedAvg:
    n_selected = max(2, int(num_clients * fraction_fit))
    return InstrumentedFedAvg(
        num_clients=num_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=n_selected,
        min_evaluate_clients=n_selected,
        min_available_clients=num_clients,
        log_dir=log_dir,
    )