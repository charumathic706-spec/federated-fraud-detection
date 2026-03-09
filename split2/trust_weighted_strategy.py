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
  4. Aggregate: global_params = ? w_i ? ?_i  (trust-weighted, not uniform)
  5. Flag malicious clients (w_i ? 0) and log to blockchain-ready metadata
  6. Log round results -> JSON for Split 3 blockchain audit trail
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

from trust_scoring import TrustScorer, AggregationResult
from fedavg_strategy import weighted_average   # reuse metric aggregation from Split 1


# ??????????????????????????????????????????????????????????????????????????
# Trust-Weighted Aggregation Strategy
# ??????????????????????????????????????????????????????????????????????????

class TrustWeightedFedAvg(Strategy):
    """
    Custom Flower Strategy implementing trust-weighted adaptive aggregation.

    Key difference from standard FedAvg:
      FedAvg:       w_i = n_i / ? n_j          (proportional to data size)
      TrustWeighted: w_i = ?_i / ? ?_j         (proportional to trust score)
      Malicious:     w_i = ? ? 0               (near-zero weight)

    The global model update formula:
      ?_global(t+1) = ?_i  w_i(t) ? ?_i(t)
    where w_i(t) is derived from the dynamic trust score ?_i(t).
    """

    def __init__(
        self,
        num_clients:        int,
        fraction_fit:       float = 1.0,
        fraction_evaluate:  float = 1.0,
        # Trust scorer hyperparameters
        lambda_cosine:      float = 0.3,   # full-vector cos diluted by hidden layers
        lambda_distance:    float = 0.5,   # euc distance is the most reliable signal
        lambda_norm:        float = 0.2,
        gamma:              float = 0.85,  # faster trust decay (was 0.9) -- C1 neutralised quicker
        anomaly_threshold:  float = 0.40,  # label-flip scores ~0.70, well above threshold
        malicious_weight:   float = 1e-6,
        min_trust_floor:    float = 0.75,  # innocent clients never drop below this
        # Logging
        log_dir:            str   = "logs_split2",
        # Attack simulator (injected from main.py for simulation)
        attack_simulator    = None,
    ):
        self.num_clients       = num_clients
        self.fraction_fit      = fraction_fit
        self.fraction_evaluate = fraction_evaluate
        self.log_dir           = log_dir
        self.attacker          = attack_simulator

        # Trust scorer instance (maintains trust state across rounds)
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
            f"?={gamma} | logs -> {log_dir}/\n"
        )

    # -- Flower Strategy interface ---------------------------------------------

    def initialize_parameters(self, client_manager) -> Optional[Parameters]:
        """No initial parameters -- clients initialise their own models."""
        return None

    def configure_fit(self, server_round, parameters, client_manager):
        """Select clients and send global model for local training."""
        if self.attacker:
            self.attacker.set_round(server_round)

        sample_size = max(1, int(self.num_clients * self.fraction_fit))
        clients     = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
        config      = {"server_round": server_round}
        return [(client, fl.common.FitIns(parameters, config)) for client in clients]

    def configure_evaluate(self, server_round, parameters, client_manager):
        """Select clients for federated evaluation."""
        sample_size = max(1, int(self.num_clients * self.fraction_evaluate))
        clients     = client_manager.sample(num_clients=sample_size, min_num_clients=sample_size)
        config      = {"server_round": server_round}
        return [(client, fl.common.EvaluateIns(parameters, config)) for client in clients]

    def evaluate(self, server_round, parameters):
        """Server-side evaluation -- not used (clients evaluate locally)."""
        return None

    # -- Core: Trust-Weighted Aggregation -------------------------------------

    def aggregate_fit(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, FitRes]],
        failures:     List[Union[Tuple[ClientProxy, FitRes], BaseException]],
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """
        Main aggregation step -- replaces FedAvg weighted average with
        trust-weighted aggregation using gradient validation.
        """
        if not results:
            return None, {}

        print(f"\n{'='*65}")
        print(f"  ROUND {server_round} -- TRUST-WEIGHTED AGGREGATION")
        print(f"  Clients: {len(results)} responded | {len(failures)} failures")
        print(f"{'='*65}")

        # -- Extract parameters and metadata from each client ------------------
        client_params:     Dict[int, List[np.ndarray]] = {}
        client_flat_params: Dict[int, np.ndarray]       = {}
        client_gradients:  Dict[int, np.ndarray]        = {}
        client_n_samples:  Dict[int, int]               = {}

        global_flat = (
            np.concatenate([p.flatten() for p in self.current_global_params])
            if self.current_global_params else None
        )

        for proxy, fit_res in results:
            m   = fit_res.metrics
            cid = int(m.get("client_id", 0))
            params = parameters_to_ndarrays(fit_res.parameters)

            # -- Inject attack if applicable -----------------------------------
            if self.attacker and self.attacker.is_malicious(cid):
                if self.current_global_params:
                    params = self.attacker.poison_params(
                        cid, params, self.current_global_params
                    )

            client_params[cid]     = params
            client_n_samples[cid]  = fit_res.num_examples

            flat = np.concatenate([p.flatten() for p in params])
            client_flat_params[cid] = flat

            # Gradient delta = client params - global params
            if global_flat is not None and len(global_flat) == len(flat):
                client_gradients[cid] = flat - global_flat
            else:
                client_gradients[cid] = flat   # first round: use params as gradient proxy

            print(
                f"  Client {cid:02d} | samples={fit_res.num_examples} | "
                f"F1={m.get('train_f1', 0):.4f} | AUC={m.get('train_auc', 0):.4f}"
            )

        # -- Run trust scoring -------------------------------------------------
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

        # -- Trust-weighted parameter aggregation ------------------------------
        aggregated_params = self._weighted_aggregate(
            client_params, trust_result.trust_weights
        )
        self.current_global_params = aggregated_params

        # -- Compute model hash for blockchain audit trail (Split 3) -----------
        model_hash = self._hash_params(aggregated_params)
        print(f"\n  [ModelHash] {model_hash[:16]}...  (-> Split 3 blockchain log)")

        # -- Save round metadata -----------------------------------------------
        self._save_round_log(
            server_round, trust_result, model_hash,
            {cid: m for _, fit_res in results
             for cid, m in [(int(fit_res.metrics.get("client_id", 0)), fit_res.metrics)]}
        )

        aggregated_metrics = {
            "flagged_count":   float(len(trust_result.flagged_clients)),
            "trusted_count":   float(len(trust_result.trusted_clients)),
            "avg_trust":       float(np.mean([
                                   self.trust_scorer.trust_records[cid].trust_score
                                   for cid in client_params
                               ])),
            "min_anomaly":     float(min(trust_result.anomaly_scores.values())),
            "max_anomaly":     float(max(trust_result.anomaly_scores.values())),
        }

        return ndarrays_to_parameters(aggregated_params), aggregated_metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results:      List[Tuple[ClientProxy, EvaluateRes]],
        failures:     List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation results from clients."""
        if not results:
            return None, {}

        print(f"\n  ROUND {server_round} -- EVALUATE")
        for _, eval_res in results:
            m   = eval_res.metrics
            cid = int(m.get("client_id", -1))
            trust = self.trust_scorer.trust_records.get(cid)
            flag  = "[ATTACK]" if (trust and trust.is_malicious) else "[OK]"
            print(
                f"  {flag} Client {cid:02d} | "
                f"F1={m.get('f1', 0):.4f} | AUC={m.get('auc_roc', 0):.4f} | "
                f"Recall={m.get('recall', 0):.4f}"
            )

        losses       = [eval_res.loss      for _, eval_res in results]
        n_samples    = [eval_res.num_examples for _, eval_res in results]
        metrics_list = [(n, eval_res.metrics) for n, (_, eval_res) in zip(n_samples, results)]

        weighted_loss    = sum(l * n for l, n in zip(losses, n_samples)) / max(sum(n_samples), 1)
        global_metrics   = weighted_average(metrics_list)

        global_f1  = global_metrics.get("f1", 0)
        global_auc = global_metrics.get("auc_roc", 0)
        print(
            f"\n  [Global Eval] F1={global_f1:.4f} | AUC={global_auc:.4f} | "
            f"Recall={global_metrics.get('recall', 0):.4f}"
        )

        if global_f1 > self.best_f1:
            self.best_f1    = global_f1
            self.best_round = server_round
            print(f"  * New best model! F1={global_f1:.4f} at round {server_round}")

        # Update last round log with eval metrics
        if self.round_logs:
            self.round_logs[-1].update({
                "global_f1":        global_f1,
                "global_auc":       global_auc,
                "global_recall":    global_metrics.get("recall", 0),
                "global_precision": global_metrics.get("precision", 0),
            })
            self._flush_logs()

        return float(weighted_loss), global_metrics

    # -- Parameter aggregation -------------------------------------------------

    def _weighted_aggregate(
        self,
        client_params:  Dict[int, List[np.ndarray]],
        trust_weights:  Dict[int, float],
    ) -> List[np.ndarray]:
        """
        Perform trust-weighted parameter aggregation.
        ?_global = ?_i  w_i ? ?_i
        """
        # Get shape reference from first client
        first_params = next(iter(client_params.values()))
        aggregated   = [np.zeros_like(p, dtype=np.float64) for p in first_params]

        for cid, params in client_params.items():
            w = trust_weights.get(cid, 0.0)
            for i, p in enumerate(params):
                aggregated[i] += w * p.astype(np.float64)

        return [a.astype(np.float32) for a in aggregated]

    # -- Hashing (for Split 3 blockchain) -------------------------------------

    def _hash_params(self, params: List[np.ndarray]) -> str:
        """Compute SHA-256 hash of model parameters for blockchain logging."""
        hasher = hashlib.sha256()
        for p in params:
            hasher.update(p.tobytes())
        return hasher.hexdigest()

    # -- Logging ---------------------------------------------------------------

    def _save_round_log(
        self,
        round_num:    int,
        result:       AggregationResult,
        model_hash:   str,
        client_fit_metrics: Dict,
    ) -> None:
        """Save round metadata -- structured for Split 3 blockchain consumption."""
        trust_summary = self.trust_scorer.get_trust_summary()
        log = {
            "round":           round_num,
            "timestamp":       datetime.now(timezone.utc).isoformat(),
            "model_hash":      model_hash,
            "trusted_clients": result.trusted_clients,
            "flagged_clients": result.flagged_clients,
            "trust_weights":   {str(k): v for k, v in result.trust_weights.items()},
            "anomaly_scores":  {str(k): v for k, v in result.anomaly_scores.items()},
            "cos_similarities": {str(k): v for k, v in result.cos_similarities.items()},
            "euc_distances":   {str(k): v for k, v in result.euc_distances.items()},
            "trust_scores":    {
                str(cid): info["trust_score"]
                for cid, info in trust_summary.items()
            },
            # Placeholder -- updated after evaluate_aggregate()
            "global_f1":       0.0,
            "global_auc":      0.0,
        }
        self.round_logs.append(log)
        self._flush_logs()

    def _flush_logs(self) -> None:
        path = os.path.join(self.log_dir, "trust_training_log.json")
        with open(path, "w") as f:
            json.dump(self.round_logs, f, indent=2)

    def print_summary(self) -> None:
        """Final summary after all rounds."""
        trust_summary = self.trust_scorer.get_trust_summary()
        print(f"\n{'='*65}")
        print(f"  SPLIT 2 FEDERATED TRAINING COMPLETE")
        print(f"  Best Global F1:    {self.best_f1:.4f} (Round {self.best_round})")
        print(f"  Total Rounds:      {len(self.round_logs)}")
        print(f"\n  Final Trust Scores:")
        for cid, info in sorted(trust_summary.items()):
            status = "[ATTACK] MALICIOUS" if info["is_malicious"] else "[OK] trusted"
            print(
                f"    Client {cid:02d} {status} | "
                f"?={info['trust_score']:.4f} | "
                f"?={info['anomaly_score']:.4f} | "
                f"cos={info['cos_similarity']:+.4f} | "
                f"flagged_rounds={info['rounds_flagged']}"
            )
        print(f"\n  Log: {self.log_dir}/trust_training_log.json")
        print(f"{'='*65}\n")