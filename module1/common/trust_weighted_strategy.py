"""
trust_weighted_strategy.py
--------------------------
Split 2's core deliverable: Trust-Weighted Adaptive Aggregation Strategy.

Replaces the standard FedAvg from Split 1 with a custom Flower Strategy
that integrates the trust scoring engine (trust_scoring.py) directly into
the aggregation step.

Per-round pipeline:
  1. Receive model updates from all clients (via Flower FitRes)
  2. Extract gradient deltas and flattened parameters from each client
  3. Run TrustScorer.score_round() -> per-client anomaly scores + trust weights
  4. Aggregate: global_params = Σ w_i × θ_i  (trust-weighted, not uniform)
  5. Flag malicious clients (w_i ≈ 0) and log to blockchain-ready metadata
  6. Log round results -> JSON for Split 3 blockchain audit trail
  7. [MODULE 2 INTEGRATION] Call GovernanceEngine.process_round() live
     so blockchain governance runs automatically after every FL round.
"""

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

try:
    from common.trust_scoring import TrustScorer, AggregationResult
    from common.fedavg_strategy import weighted_average
except ImportError:
    from trust_scoring import TrustScorer, AggregationResult
    from fedavg_strategy import weighted_average


# =============================================================================
# TRUST-WEIGHTED FEDAVG STRATEGY
# =============================================================================

class TrustWeightedFedAvg(Strategy):
    """
    Trust-Weighted Federated Averaging with live Module 2 governance hook.

    MODULE 2 INTEGRATION:
        Pass a GovernanceEngine instance via governance_engine parameter.
        After every round, _save_round_log() calls
        governance_engine.process_round(log_entry) automatically.
        This means Module 1 and Module 2 run together in one execution.

        If governance_engine=None (default), behaviour is identical to
        the original — no governance, pure FL only.
    """

    def __init__(
        self,
        num_clients:        int,
        fraction_fit:       float = 1.0,
        fraction_evaluate:  float = 1.0,
        # Trust scorer hyperparameters
        lambda_cosine:      float = 0.5,
        lambda_distance:    float = 0.3,
        lambda_norm:        float = 0.2,
        gamma:              float = 0.85,
        anomaly_threshold:  float = 0.45,
        malicious_weight:   float = 1e-6,
        min_trust_floor:    float = 0.70,
        # Logging
        log_dir:            str   = "logs_split2",
        # Attack simulator (injected from main.py for simulation)
        attack_simulator          = None,
        # ── MODULE 2 INTEGRATION ──────────────────────────────────────────────
        # Pass a GovernanceEngine instance to enable live blockchain governance.
        # If None, Module 2 is skipped (original behaviour preserved).
        governance_engine         = None,
    ):
        self.num_clients       = num_clients
        self.fraction_fit      = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.log_dir           = log_dir
        self.attacker          = attack_simulator

        # ── MODULE 2: store governance engine ────────────────────────────────
        self.governance_engine = governance_engine
        if governance_engine is not None:
            print(f"[Server] Module 2 (Blockchain Governance) CONNECTED — "
                  f"live governance after every round.")
        else:
            print(f"[Server] Module 2 (Blockchain Governance) DISABLED — "
                  f"pass governance_engine= to enable.")

        # Trust scorer instance (maintains trust state across rounds)
        self.trust_scorer = TrustScorer(
            num_clients       = num_clients,
            lambda_cosine     = lambda_cosine,
            lambda_distance   = lambda_distance,
            lambda_norm       = lambda_norm,
            gamma             = gamma,
            anomaly_threshold = anomaly_threshold,
            malicious_weight  = malicious_weight,
            min_trust_floor   = min_trust_floor,
        )

        # State tracking
        self.current_global_params: Optional[List[np.ndarray]] = None
        self.round_logs:  List[Dict] = []
        self.best_f1:     float = 0.0
        self.best_round:  int   = 0

        os.makedirs(log_dir, exist_ok=True)
        print(
            f"\n[Server] TrustWeightedFedAvg initialised | "
            f"{num_clients} clients | "
            f"anomaly_threshold={anomaly_threshold} | "
            f"γ={gamma} | logs -> {log_dir}/\n"
        )

    # -- Flower Strategy interface ---------------------------------------------

    def initialize_parameters(self, client_manager):
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        config = {"server_round": server_round}
        if parameters is None:
            return []
        sample_size = max(1, int(self.fraction_fit * self.num_clients))
        clients = client_manager.sample(num_clients=sample_size,
                                        min_num_clients=sample_size)
        return [(c, fl.common.FitIns(parameters, config)) for c in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        if self.fraction_evaluate == 0.0:
            return []
        config = {"server_round": server_round}
        sample_size = max(1, int(self.fraction_evaluate * self.num_clients))
        clients = client_manager.sample(num_clients=sample_size,
                                        min_num_clients=sample_size)
        return [(c, fl.common.EvaluateIns(parameters, config)) for c in clients]

    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, FitRes]],
        failures:     List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Main aggregation step — trust-weighted average with live governance.
        """
        if not results:
            return None, {}

        print(f"\n{'='*65}")
        print(f"  ROUND {server_round} -- TRUST-WEIGHTED AGGREGATION")
        print(f"  Clients: {len(results)} responded | {len(failures)} failures")
        print(f"{'='*65}")

        # -- Extract parameters and metadata from each client -----------------
        client_params:      Dict[int, List[np.ndarray]] = {}
        client_flat_params: Dict[int, np.ndarray]       = {}
        client_gradients:   Dict[int, np.ndarray]       = {}
        client_n_samples:   Dict[int, int]              = {}

        global_flat = (
            np.concatenate([p.flatten() for p in self.current_global_params])
            if self.current_global_params else None
        )

        for proxy, fit_res in results:
            m   = fit_res.metrics
            cid = int(m.get("client_id", 0))
            params = parameters_to_ndarrays(fit_res.parameters)

            # -- Inject attack if applicable ----------------------------------
            if self.attacker and self.attacker.is_malicious(cid):
                if self.current_global_params:
                    params = self.attacker.poison_params(
                        cid, params, self.current_global_params
                    )

            client_params[cid]    = params
            client_n_samples[cid] = fit_res.num_examples

            flat = np.concatenate([p.flatten() for p in params])
            client_flat_params[cid] = flat

            if global_flat is not None and len(global_flat) == len(flat):
                client_gradients[cid] = flat - global_flat
            else:
                client_gradients[cid] = flat

            print(
                f"  Client {cid:02d} | samples={fit_res.num_examples} | "
                f"F1={m.get('train_f1', 0):.4f} | AUC={m.get('train_auc', 0):.4f}"
            )

        # -- Run trust scoring ------------------------------------------------
        global_flat_for_scoring = (
            global_flat if global_flat is not None
            else np.zeros_like(list(client_flat_params.values())[0])
        )

        trust_result = self.trust_scorer.score_round(
            round_num        = server_round,
            client_gradients = client_gradients,
            client_params    = client_flat_params,
            global_params    = global_flat_for_scoring,
        )
        self.trust_scorer.print_round_report(trust_result)

        # -- Trust-weighted parameter aggregation -----------------------------
        aggregated_params = self._weighted_aggregate(
            client_params, trust_result.trust_weights
        )
        self.current_global_params = aggregated_params

        # -- Compute model hash for blockchain audit trail --------------------
        model_hash = self._hash_params(aggregated_params)
        print(f"\n  [ModelHash] {model_hash[:16]}...  (→ Module 2 blockchain)")

        # -- Save round log + call Module 2 governance ------------------------
        client_fit_metrics = {
            int(fit_res.metrics.get("client_id", 0)): fit_res.metrics
            for _, fit_res in results
        }
        self._save_round_log(
            server_round, trust_result, model_hash, client_fit_metrics
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
        """Aggregate evaluation metrics from all clients."""
        if not results:
            return None, {}

        # -- Weighted average of all 13 metrics across clients ----------------
        metrics_list = [
            (res.num_examples, res.metrics) for _, res in results
        ]
        agg = weighted_average(metrics_list)

        global_f1  = float(agg.get("eval_f1",  agg.get("f1",  0.0)))
        global_auc = float(agg.get("eval_auc", agg.get("auc", 0.0)))

        if global_f1 > self.best_f1:
            self.best_f1    = global_f1
            self.best_round = server_round

        # -- Update last log entry with global evaluation metrics -------------
        if self.round_logs:
            last = self.round_logs[-1]
            last.update({
                "global_f1":               global_f1,
                "global_auc":              global_auc,
                "global_recall":           float(agg.get("eval_recall",    agg.get("recall",    0.0))),
                "global_precision":        float(agg.get("eval_precision", agg.get("precision", 0.0))),
                "global_accuracy":         float(agg.get("eval_accuracy",  agg.get("accuracy",  0.0))),
                "global_balanced_accuracy":float(agg.get("eval_balanced_accuracy", 0.0)),
                "global_mcc":              float(agg.get("eval_mcc",       agg.get("mcc",       0.0))),
                "global_specificity":      float(agg.get("eval_specificity",0.0)),
                "global_tp":               int(agg.get("eval_tp",  agg.get("tp",  0))),
                "global_fp":               int(agg.get("eval_fp",  agg.get("fp",  0))),
                "global_tn":               int(agg.get("eval_tn",  agg.get("tn",  0))),
                "global_fn":               int(agg.get("eval_fn",  agg.get("fn",  0))),
                "best_f1":                 self.best_f1,
                "best_round":              self.best_round,
            })
            self._flush_logs()

            # ── MODULE 2 INTEGRATION ─────────────────────────────────────────
            # Update governance engine with final F1/AUC after evaluate step.
            # We call process_round here (after F1 is known) rather than in
            # aggregate_fit (where F1 is still 0.0).
            if self.governance_engine is not None:
                try:
                    self.governance_engine.process_round(last)
                    print(f"  [Module 2] Governance committed for Round "
                          f"{last['round']} | "
                          f"F1={global_f1:.4f} | "
                          f"flagged={last.get('flagged_clients', [])}")
                except Exception as exc:
                    # Never let governance errors crash FL training
                    print(f"  [Module 2] WARNING: Governance error "
                          f"(training continues): {exc}")

        print(
            f"\n  [Round {server_round} Global] "
            f"F1={global_f1:.4f}  AUC={global_auc:.4f}"
        )

        return float(1.0 - global_f1), dict(agg)

    def evaluate(self, server_round, parameters):
        return None

    # -- Private helpers -------------------------------------------------------

    def _weighted_aggregate(
        self,
        client_params:  Dict[int, List[np.ndarray]],
        trust_weights:  Dict[int, float],
    ) -> List[np.ndarray]:
        """Compute trust-weighted average of model parameters."""
        if not client_params:
            return []
        num_layers = len(list(client_params.values())[0])
        aggregated = []
        for layer_idx in range(num_layers):
            layer_agg = sum(
                trust_weights.get(cid, 0.0) * params[layer_idx]
                for cid, params in client_params.items()
            )
            aggregated.append(layer_agg)
        return aggregated

    def _hash_params(self, params: List[np.ndarray]) -> str:
        """SHA-256 hash of model parameters — used by blockchain audit trail."""
        h = hashlib.sha256()
        for p in params:
            h.update(p.tobytes())
        return h.hexdigest()

    def _save_round_log(
        self,
        round_num:          int,
        result:             AggregationResult,
        model_hash:         str,
        client_fit_metrics: Dict,
    ) -> None:
        """
        Save round metadata structured for Module 2 blockchain consumption.
        Called from aggregate_fit() — F1/AUC are 0.0 at this point and
        will be updated by aggregate_evaluate() once clients report metrics.
        """
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
            "trust_scores": {
                str(cid): info["trust_score"]
                for cid, info in trust_summary.items()
            },
            "client_metrics": [
                {
                    "client_id": cid,
                    "f1":        float(m.get("train_f1",  0.0)),
                    "auc":       float(m.get("train_auc", 0.0)),
                    "recall":    float(m.get("train_recall", 0.0)),
                }
                for cid, m in client_fit_metrics.items()
            ],
            # Populated by aggregate_evaluate() once clients report:
            "global_f1":  0.0,
            "global_auc": 0.0,
        }
        self.round_logs.append(log)
        self._flush_logs()

    def _flush_logs(self) -> None:
        """Write the full log list to trust_training_log.json."""
        path = os.path.join(self.log_dir, "trust_training_log.json")
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

    def print_summary(self) -> None:
        """Final summary after all rounds."""
        trust_summary = self.trust_scorer.get_trust_summary()
        last = self.round_logs[-1] if self.round_logs else {}

        print(f"\n{'='*65}")
        print(f"  SPLIT 2 FEDERATED TRAINING COMPLETE")
        print(f"  Total Rounds:          {len(self.round_logs)}")
        print(f"  Best F1 (round):       {self.best_f1:.4f}  (Round {self.best_round})")
        print(f"\n  === FINAL ROUND METRICS ===")
        print(f"  F1 Score:              {last.get('global_f1', 0):.4f}")
        print(f"  AUC-ROC:               {last.get('global_auc', 0):.4f}")
        print(f"  Recall:                {last.get('global_recall', 0):.4f}")
        print(f"  Precision:             {last.get('global_precision', 0):.4f}")
        print(f"  Balanced Accuracy:     {last.get('global_balanced_accuracy', 0):.4f}")
        print(f"  MCC:                   {last.get('global_mcc', 0):.4f}")
        print(f"  Specificity:           {last.get('global_specificity', 0):.4f}")
        tp = int(last.get('global_tp', 0))
        fp = int(last.get('global_fp', 0))
        tn = int(last.get('global_tn', 0))
        fn = int(last.get('global_fn', 0))
        print(f"  TP / FP / TN / FN:     {tp} / {fp} / {tn} / {fn}")
        print(f"\n  === ATTACK DETECTION ===")
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


# =============================================================================
# FACTORY  — imported by split2/main.py as get_trust_strategy()
# =============================================================================

def get_trust_strategy(
    num_clients:        int,
    fraction_fit:       float = 1.0,
    log_dir:            str   = "logs_split2",
    attack_simulator          = None,
    # TrustScorer hyperparameters
    lambda_cosine:      float = 0.5,
    lambda_distance:    float = 0.3,
    lambda_norm:        float = 0.2,
    gamma:              float = 0.85,
    anomaly_threshold:  float = 0.45,
    malicious_weight:   float = 1e-6,
    min_trust_floor:    float = 0.70,
    # ── MODULE 2 INTEGRATION ─────────────────────────────────────────────────
    governance_engine         = None,
) -> TrustWeightedFedAvg:
    """
    Factory function — called by split2/main.py.

    MODULE 2 INTEGRATION:
        Pass governance_engine=GovernanceEngine(...) to enable live blockchain
        governance. The engine's process_round() will be called automatically
        after every FL round completes evaluation.

        Pass governance_engine=None (default) to run Module 1 only — original
        behaviour, no change.

    Called by split2/main.py:
        from common.governance_bridge import build_governance_engine
        engine   = build_governance_engine(log_dir=args.log_dir)
        strategy = get_trust_strategy(
            num_clients       = args.num_clients,
            fraction_fit      = args.fraction_fit,
            log_dir           = args.log_dir,
            attack_simulator  = attacker,
            governance_engine = engine,
        )
    """
    return TrustWeightedFedAvg(
        num_clients       = num_clients,
        fraction_fit      = fraction_fit,
        fraction_evaluate = fraction_fit,
        lambda_cosine     = lambda_cosine,
        lambda_distance   = lambda_distance,
        lambda_norm       = lambda_norm,
        gamma             = gamma,
        anomaly_threshold = anomaly_threshold,
        malicious_weight  = malicious_weight,
        min_trust_floor   = min_trust_floor,
        log_dir           = log_dir,
        attack_simulator  = attack_simulator,
        governance_engine = governance_engine,
    )
