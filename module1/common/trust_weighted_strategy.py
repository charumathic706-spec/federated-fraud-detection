# =============================================================================
# FILE: common/trust_weighted_strategy.py
# PURPOSE: Split 2 Trust-Weighted Adaptive Aggregation Strategy.
#
# FIXES vs uploaded version:
#   1. configure_fit: FitIns(parameters, config) — when parameters=None on
#      round 1 (initialize_parameters returns None), Flower crashes clients.
#      FIX: wrap None as ndarrays_to_parameters([]) so clients receive a valid
#      empty Parameters object and initialise their own model weights.
#   2. configure_evaluate: same fix for None parameters.
#   3. TrustWeightedFedAvg now accepts min_fit_clients / min_evaluate_clients
#      kwargs so the caller can enforce quorum (previously hardcoded to 1).
#   4. Unicode symbols (Greek letters) replaced with ASCII in source comments
#      to avoid encoding issues on Windows terminals.
# =============================================================================
from __future__ import annotations

import os
import json
import hashlib
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from datetime import datetime, timezone

import flwr as fl
from flwr.common import (
    Parameters, Scalar, Metrics,
    FitRes, EvaluateRes,
    ndarrays_to_parameters, parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy
from flwr.server.strategy import Strategy

from common.trust_scoring  import TrustScorer, AggregationResult
from common.fedavg_strategy import weighted_average


# =============================================================================
# Trust-Weighted Aggregation Strategy
# =============================================================================

class TrustWeightedFedAvg(Strategy):
    """
    Custom Flower Strategy implementing trust-weighted adaptive aggregation.

    FedAvg:        w_i = n_i / sum(n_j)          (data size)
    TrustWeighted: w_i = tau_i / sum(tau_j)       (trust score)
    Malicious:     w_i -> 0 (near-zero weight)
    """

    def __init__(
        self,
        num_clients:           int,
        fraction_fit:          float = 1.0,
        fraction_evaluate:     float = 1.0,
        min_fit_clients:       int   = 2,
        min_evaluate_clients:  int   = 2,
        # Trust scorer hyperparameters
        lambda_cosine:         float = 0.3,
        lambda_distance:       float = 0.5,
        lambda_norm:           float = 0.2,
        gamma:                 float = 0.85,
        anomaly_threshold:     float = 0.45,
        malicious_weight:      float = 1e-6,
        min_trust_floor:       float = 0.85,
        # Logging
        log_dir:               str   = "logs_split2",
        attack_simulator               = None,
    ):
        self.num_clients         = num_clients
        self.fraction_fit        = fraction_fit
        self.fraction_evaluate   = fraction_evaluate
        self.min_fit_clients     = min_fit_clients
        self.min_evaluate_clients = min_evaluate_clients
        self.log_dir             = log_dir
        self.attacker            = attack_simulator

        self.trust_scorer = TrustScorer(
            num_clients=num_clients,
            lambda_cosine=lambda_cosine,
            lambda_distance=lambda_distance,
            lambda_norm=lambda_norm,
            gamma=gamma,
            anomaly_threshold=anomaly_threshold,
            malicious_weight=malicious_weight,
            min_trust_floor=min_trust_floor,
        )

        self.current_global_params: Optional[List[np.ndarray]] = None
        self.round_logs:  List[Dict] = []
        self.best_f1:     float = 0.0
        self.best_round:  int   = 0

        os.makedirs(log_dir, exist_ok=True)
        print(
            f"\n[Server] TrustWeightedFedAvg initialised | "
            f"{num_clients} clients | "
            f"anomaly_threshold={anomaly_threshold} | "
            f"gamma={gamma} | logs -> {log_dir}/\n"
        )

    # ── Flower Strategy interface ──────────────────────────────────────────────

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """
        Return empty (but valid) Parameters so clients initialise their own model.

        FIX v10: Returning Python None here is NOT the same as returning
        ndarrays_to_parameters([]). When None is later passed to FitIns in
        configure_fit(), Flower's gRPC serialiser raises:
            AttributeError: 'NoneType' has no attribute 'tensors'
        We return an empty Parameters object instead so each client's
        set_params([]) call just no-ops and uses its random init weights.
        """
        return ndarrays_to_parameters([])

    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients and send current global model for local training."""
        if self.attacker:
            self.attacker.set_round(server_round)

        # FIX: ensure parameters is never None (use empty if not set yet)
        if parameters is None:
            parameters = ndarrays_to_parameters([])

        sample_size = max(self.min_fit_clients, int(self.num_clients * self.fraction_fit))
        clients     = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_fit_clients
        )
        config = {"server_round": server_round}
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Select clients for federated evaluation."""
        # FIX: ensure parameters is never None
        if parameters is None:
            parameters = ndarrays_to_parameters([])

        sample_size = max(self.min_evaluate_clients, int(self.num_clients * self.fraction_evaluate))
        clients     = client_manager.sample(
            num_clients=sample_size, min_num_clients=self.min_evaluate_clients
        )
        config = {"server_round": server_round}
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def evaluate(self, server_round, parameters):
        """Server-side evaluation — not used (clients evaluate locally)."""
        return None

    # ── Core: Trust-Weighted Aggregation ──────────────────────────────────────

    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, FitRes]],
        failures:     List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n{'='*65}")
        print(f"  ROUND {server_round} -- TRUST-WEIGHTED AGGREGATION")
        print(f"  Clients: {len(results)} responded | {len(failures)} failures")
        print(f"{'='*65}")

        client_params:      Dict[int, List[np.ndarray]] = {}
        client_flat_params: Dict[int, np.ndarray]       = {}
        client_gradients:   Dict[int, np.ndarray]       = {}
        client_n_samples:   Dict[int, int]              = {}

        global_flat = (
            np.concatenate([p.flatten() for p in self.current_global_params])
            if self.current_global_params else None
        )

        for proxy, fit_res in results:
            m      = fit_res.metrics
            cid    = int(m.get("client_id", 0))
            params = parameters_to_ndarrays(fit_res.parameters)

            # Inject attack if applicable
            if self.attacker and self.attacker.is_malicious(cid):
                if self.current_global_params:
                    params = self.attacker.poison_params(cid, params, self.current_global_params)

            client_params[cid]    = params
            client_n_samples[cid] = fit_res.num_examples

            flat = np.concatenate([p.flatten() for p in params])
            client_flat_params[cid] = flat

            if global_flat is not None and len(global_flat) == len(flat):
                client_gradients[cid] = flat - global_flat
            else:
                client_gradients[cid] = flat  # first round: use params as gradient proxy

            print(
                f"  Client {cid:02d} | samples={fit_res.num_examples} | "
                f"F1={m.get('train_f1', 0):.4f} | AUC={m.get('train_auc', 0):.4f}"
            )

        # Run trust scoring
        global_flat_for_scoring = (
            global_flat if global_flat is not None
            else np.zeros_like(list(client_flat_params.values())[0])
        )

        trust_result = self.trust_scorer.score_round(
            round_num=server_round,
            client_gradients=client_gradients,
            client_params=client_flat_params,
            global_params=global_flat_for_scoring,
        )
        self.trust_scorer.print_round_report(trust_result)

        # Trust-weighted parameter aggregation
        aggregated_params = self._weighted_aggregate(client_params, trust_result.trust_weights)
        self.current_global_params = aggregated_params

        model_hash = self._hash_params(aggregated_params)
        print(f"\n  [ModelHash] {model_hash[:16]}...  (-> Split 3 blockchain log)")

        self._save_round_log(
            server_round, trust_result, model_hash,
            {cid: m for _, fit_res in results
             for cid, m in [(int(fit_res.metrics.get("client_id", 0)), fit_res.metrics)]}
        )

        aggregated_metrics = {
            "flagged_count": float(len(trust_result.flagged_clients)),
            "trusted_count": float(len(trust_result.trusted_clients)),
            "avg_trust":     float(np.mean([
                self.trust_scorer.trust_records[cid].trust_score
                for cid in client_params
            ])),
            "min_anomaly":   float(min(trust_result.anomaly_scores.values())),
            "max_anomaly":   float(max(trust_result.anomaly_scores.values())),
        }

        return ndarrays_to_parameters(aggregated_params), aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, EvaluateRes]],
        failures:     List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        if not results:
            return None, {}

        print(f"\n  ROUND {server_round} -- EVALUATE")
        for _, eval_res in results:
            m    = eval_res.metrics
            cid  = int(m.get("client_id", -1))
            trust = self.trust_scorer.trust_records.get(cid)
            flag  = "[ATTACK]" if (trust and trust.is_malicious) else "[OK]"
            print(
                f"  {flag} Client {cid:02d} | "
                f"F1={m.get('f1', 0):.4f} | AUC={m.get('auc_roc', 0):.4f} | "
                f"Recall={m.get('recall', 0):.4f}"
            )

        losses    = [eval_res.loss          for _, eval_res in results]
        n_samples = [eval_res.num_examples  for _, eval_res in results]
        metrics_list = [(n, eval_res.metrics) for n, (_, eval_res) in zip(n_samples, results)]

        weighted_loss  = sum(l * n for l, n in zip(losses, n_samples)) / max(sum(n_samples), 1)
        global_metrics = weighted_average(metrics_list)

        global_f1  = global_metrics.get("f1", 0)
        global_auc = global_metrics.get("auc_roc", 0)
        g_acc  = global_metrics.get("accuracy", 0)
        g_bal  = global_metrics.get("balanced_accuracy", 0)
        g_mcc  = global_metrics.get("mcc", 0)
        g_spec = global_metrics.get("specificity", 0)
        g_pre  = global_metrics.get("precision", 0)
        g_rec  = global_metrics.get("recall", 0)
        print(
            f"\n  [Global Eval R{server_round:02d}]"
            f"  F1={global_f1:.4f}  AUC={global_auc:.4f}"
            f"  Recall={g_rec:.4f}  Precision={g_pre:.4f}"
        )
        print(
            f"  [Metrics]     Accuracy={g_acc:.4f}  BalancedAcc={g_bal:.4f}"
            f"  MCC={g_mcc:.4f}  Specificity={g_spec:.4f}"
        )

        if global_f1 > self.best_f1:
            self.best_f1    = global_f1
            self.best_round = server_round
            print(f"  * New best model! F1={global_f1:.4f} at round {server_round}")

        # CRITICAL FIX: Match the log entry by server_round, not round_logs[-1].
        #
        # Root cause of metrics landing in wrong round:
        #   _save_round_log() (called from aggregate_fit) appends a new entry
        #   with global_f1=0.0 placeholder.  aggregate_evaluate() was then
        #   doing round_logs[-1].update(...) to fill in real metrics.
        #   BUT: in Flower's subprocess architecture, the server can call
        #   aggregate_evaluate(round N) AFTER aggregate_fit(round N+1) has
        #   already appended round N+1's entry — so round_logs[-1] is now
        #   round N+1, and round N's real metrics get written there instead.
        #   This caused exactly the "every 8th round has real metrics" pattern
        #   seen in the log: Flower's gRPC scheduling with 5 subprocess clients
        #   creates a consistent 8-round lag between fit and evaluate calls.
        #
        # Fix: search round_logs by "round" key for server_round.
        target = None
        for entry in self.round_logs:
            if entry.get("round") == server_round:
                target = entry
                break
        if target is None:
            # evaluate fired before fit wrote the log — create entry now
            target = {"round": server_round}
            self.round_logs.append(target)
            self.round_logs.sort(key=lambda x: x.get("round", 0))

        target.update({
            "global_f1":                global_f1,
            "global_auc":               global_auc,
            "global_recall":            global_metrics.get("recall", 0),
            "global_precision":         global_metrics.get("precision", 0),
            "global_accuracy":          global_metrics.get("accuracy", 0),
            "global_balanced_accuracy": global_metrics.get("balanced_accuracy", 0),
            "global_mcc":               global_metrics.get("mcc", 0),
            "global_specificity":       global_metrics.get("specificity", 0),
            "global_tp":                global_metrics.get("tp", 0),
            "global_fp":                global_metrics.get("fp", 0),
            "global_tn":                global_metrics.get("tn", 0),
            "global_fn":                global_metrics.get("fn", 0),
        })
        self._flush_logs()

        return float(weighted_loss), global_metrics

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _weighted_aggregate(
        self,
        client_params: Dict[int, List[np.ndarray]],
        trust_weights: Dict[int, float],
    ) -> List[np.ndarray]:
        first_params = next(iter(client_params.values()))
        aggregated   = [np.zeros_like(p, dtype=np.float64) for p in first_params]
        for cid, params in client_params.items():
            w = trust_weights.get(cid, 0.0)
            for i, p in enumerate(params):
                aggregated[i] += w * p.astype(np.float64)
        return [a.astype(np.float32) for a in aggregated]

    def _hash_params(self, params: List[np.ndarray]) -> str:
        hasher = hashlib.sha256()
        for p in params:
            hasher.update(p.tobytes())
        return hasher.hexdigest()

    def _save_round_log(
        self,
        round_num:          int,
        result:             AggregationResult,
        model_hash:         str,
        client_fit_metrics: Dict,
    ) -> None:
        trust_summary = self.trust_scorer.get_trust_summary()
        log = {
            "round":            round_num,
            "timestamp":        datetime.now(timezone.utc).isoformat(),
            "model_hash":       model_hash,
            "trusted_clients":  result.trusted_clients,
            "flagged_clients":  result.flagged_clients,
            "trust_weights":    {str(k): v for k, v in result.trust_weights.items()},
            "anomaly_scores":   {str(k): v for k, v in result.anomaly_scores.items()},
            "cos_similarities": {str(k): v for k, v in result.cos_similarities.items()},
            "euc_distances":    {str(k): v for k, v in result.euc_distances.items()},
            "trust_scores":     {str(cid): info["trust_score"] for cid, info in trust_summary.items()},
            "global_f1":        0.0,   # updated by aggregate_evaluate()
            "global_auc":       0.0,
        }
        self.round_logs.append(log)
        self._flush_logs()

    def _flush_logs(self) -> None:
        path = os.path.join(self.log_dir, "trust_training_log.json")
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

    def print_summary(self) -> None:
        trust_summary = self.trust_scorer.get_trust_summary()
        last = self.round_logs[-1] if self.round_logs else {}

        print(f"\n{'='*65}")
        print(f"  SPLIT 2 FEDERATED TRAINING COMPLETE")
        print(f"  Total Rounds:          {len(self.round_logs)}")
        print(f"  Best F1 (round):       {self.best_f1:.4f}  (Round {self.best_round})")
        print(f"\n  === FINAL ROUND METRICS ===")
        print(f"  F1 Score:              {last.get('global_f1', 0):.4f}")
        print(f"  AUC-ROC:               {last.get('global_auc', 0):.4f}")
        print(f"  Recall (Sensitivity):  {last.get('global_recall', 0):.4f}")
        print(f"  Precision:             {last.get('global_precision', 0):.4f}")
        print(f"  Accuracy:              {last.get('global_accuracy', 0):.4f}")
        print(f"  Balanced Accuracy:     {last.get('global_balanced_accuracy', 0):.4f}")
        print(f"  MCC:                   {last.get('global_mcc', 0):.4f}  (>0.5=good, >0.7=strong)")
        print(f"  Specificity:           {last.get('global_specificity', 0):.4f}")
        tp = int(last.get('global_tp', 0)); fp = int(last.get('global_fp', 0))
        tn = int(last.get('global_tn', 0)); fn = int(last.get('global_fn', 0))
        print(f"  TP / FP / TN / FN:     {tp} / {fp} / {tn} / {fn}")
        print(f"\n  === ATTACK DETECTION ===")
        flagged_total = sum(1 for info in trust_summary.values() if info["is_malicious"])
        print(f"  Clients flagged:       {flagged_total}")
        print(f"  Rounds flagged:        {sum(info['rounds_flagged'] for info in trust_summary.values())}")
        print(f"\n  === FINAL TRUST SCORES ===")
        for cid, info in sorted(trust_summary.items()):
            status = "[ATTACK] MALICIOUS" if info["is_malicious"] else "[OK]     trusted  "
            print(
                f"    Client {cid:02d} {status} | "
                f"tau={info['trust_score']:.4f} | "
                f"alpha={info['anomaly_score']:.4f} | "
                f"cos={info['cos_similarity']:+.4f} | "
                f"flagged={info['rounds_flagged']} rounds"
            )
        print(f"\n  Log: {self.log_dir}/trust_training_log.json")
        print(f"{'='*65}\n")


def get_trust_strategy(
    num_clients:       int,
    fraction_fit:      float = 1.0,
    log_dir:           str   = "logs_split2",
    attack_simulator           = None,
    malicious_clients: list    = None,
) -> TrustWeightedFedAvg:
    """Factory — build the trust-weighted strategy."""
    n_selected = max(2, int(num_clients * fraction_fit))
    return TrustWeightedFedAvg(
        num_clients=num_clients,
        fraction_fit=fraction_fit,
        fraction_evaluate=fraction_fit,
        min_fit_clients=n_selected,
        min_evaluate_clients=n_selected,
        log_dir=log_dir,
        attack_simulator=attack_simulator,
    )