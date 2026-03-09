"""
trust_scoring.py
----------------
Core mathematical trust scoring engine for Split 2.

For each client's gradient update, computes:
  1. Cosine Similarity   -- direction alignment with global gradient
  2. Euclidean Distance  -- magnitude deviation from global model
  3. Norm Ratio          -- detects scaling/amplification attacks
  4. Anomaly Score (?_i) -- composite score combining all three metrics
  5. Trust Weight (w_i)  -- derived from anomaly score via trust update rule

Mathematical formulas (from project proposal):
  cos_sim(g_i, g_global) = (g_i ? g_global) / (||g_i|| ||g_global||)
  euc_dist(?_i, ?_global) = ||?_i - ?_global||?
  anomaly_score ?_i = ??(1 - cos_sim) + ???norm(euc_dist) + ???norm_ratio_penalty
  trust_update:  ?_i(t+1) = ? ? ?_i(t) + (1-?) ? (1 - ?_i)
  trust_weight:  w_i = ?_i / ? ?_j   (softmax-normalised)

Attack detection thresholds:
  - Label-flipping: cos_sim < 0 (gradient points in opposite direction)
  - Gradient scaling: norm_ratio >> 1 (amplified update magnitude)
  - Combined: anomaly_score > threshold -> near-zero weight assigned
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ??????????????????????????????????????????????????????????????????????????
# Data structures
# ??????????????????????????????????????????????????????????????????????????

@dataclass
class ClientTrustRecord:
    """Tracks trust state for a single bank node across all rounds."""
    client_id:      int
    trust_score:    float = 1.0      # ?_i -- starts at full trust
    anomaly_score:  float = 0.0      # ?_i -- 0=benign, 1=malicious
    cos_similarity: float = 1.0      # direction alignment
    euc_distance:   float = 0.0      # magnitude deviation
    norm_ratio:     float = 1.0      # scaling factor
    is_malicious:   bool  = False    # flagged by threshold
    rounds_flagged: int   = 0        # consecutive rounds flagged
    history: List[Dict]  = field(default_factory=list)

    def log_round(self, round_num: int) -> None:
        self.history.append({
            "round":        round_num,
            "trust_score":  self.trust_score,
            "anomaly_score": self.anomaly_score,
            "cos_similarity": self.cos_similarity,
            "euc_distance": self.euc_distance,
            "is_malicious": self.is_malicious,
        })


@dataclass
class AggregationResult:
    """Result of one round of trust-weighted aggregation."""
    round_num:          int
    trusted_clients:    List[int]
    flagged_clients:    List[int]
    trust_weights:      Dict[int, float]
    anomaly_scores:     Dict[int, float]
    cos_similarities:   Dict[int, float]
    euc_distances:      Dict[int, float]
    global_f1:          float = 0.0
    global_auc:         float = 0.0


# ??????????????????????????????????????????????????????????????????????????
# Trust Scorer
# ??????????????????????????????????????????????????????????????????????????

class TrustScorer:
    """
    Computes per-client trust scores and weights for each aggregation round.

    The trust scoring pipeline per round:
      1. Flatten all client gradients to 1D vectors
      2. Compute a pseudo-global gradient (median of all clients, robust to outliers)
      3. For each client:
           a. cosine_similarity(g_i, g_global)
           b. euclidean_distance(?_i, ?_global)
           c. norm_ratio = ||g_i|| / (||g_global|| + ?)
           d. anomaly_score ?_i  (weighted combination)
           e. trust_update rule  ?_i(t+1) = ???_i(t) + (1-?)?(1 - ?_i)
      4. Normalise trust scores -> aggregation weights
      5. Flag malicious clients (?_i > threshold -> w_i ? 0)
    """

    def __init__(
        self,
        num_clients:        int,
        # Anomaly score weights (?? + ?? + ?? = 1.0)
        lambda_cosine:      float = 0.3,   # ?? -- cosine penalty (reduced: full-vector cos is diluted by hidden layers)
        lambda_distance:    float = 0.5,   # ?? -- euclidean distance penalty (increased: most reliable signal)
        lambda_norm:        float = 0.2,   # ?? -- norm ratio penalty
        # Trust update decay (temporal smoothing)
        gamma:              float = 0.9,   # ? -- trust decay factor
        # Detection thresholds
        anomaly_threshold:  float = 0.42,  # ?_i > this -> malicious flag (label-flip scores ~0.70)
        malicious_weight:   float = 1e-6,  # w_i assigned to flagged clients
        # Norm clipping
        max_norm_ratio:     float = 3.0,   # norm_ratio above this is penalised
        # Trust floor: prevents innocent data-rich clients (high euc_dist) from
        # being unfairly eroded below this value over many rounds
        min_trust_floor:    float = 0.75,  # tau_i >= this for all non-flagged clients
    ):
        self.num_clients       = num_clients
        self.lambda_cosine     = lambda_cosine
        self.lambda_distance   = lambda_distance
        self.lambda_norm       = lambda_norm
        self.gamma             = gamma
        self.anomaly_threshold = anomaly_threshold
        self.malicious_weight  = malicious_weight
        self.max_norm_ratio    = max_norm_ratio
        self.min_trust_floor   = min_trust_floor

        # Initialise trust records for all clients
        self.trust_records: Dict[int, ClientTrustRecord] = {
            cid: ClientTrustRecord(client_id=cid)
            for cid in range(num_clients)
        }

        self.round_results: List[AggregationResult] = []

    # -- Core scoring ---------------------------------------------------------

    def cosine_similarity(self, g_i: np.ndarray, g_global: np.ndarray) -> float:
        """
        Measures gradient direction alignment.
        cos_sim = 1.0  -> perfectly aligned (benign)
        cos_sim = 0.0  -> orthogonal
        cos_sim < 0.0  -> opposing direction (strong label-flip signal)
        """
        norm_i = np.linalg.norm(g_i)
        norm_g = np.linalg.norm(g_global)
        if norm_i < 1e-10 or norm_g < 1e-10:
            return 0.0
        return float(np.dot(g_i, g_global) / (norm_i * norm_g))

    def euclidean_distance(self, theta_i: np.ndarray, theta_global: np.ndarray) -> float:
        """
        Measures parameter space deviation from global model.
        Large distance -> potential gradient manipulation attack.
        """
        return float(np.linalg.norm(theta_i - theta_global))

    def norm_ratio(self, g_i: np.ndarray, g_global: np.ndarray) -> float:
        """
        Detects gradient scaling/amplification attacks.
        ratio >> 1 -> client is amplifying its update (scaling attack)
        """
        norm_i = np.linalg.norm(g_i)
        norm_g = np.linalg.norm(g_global)
        if norm_g < 1e-10:
            return 1.0
        return float(norm_i / (norm_g + 1e-10))

    def anomaly_score(
        self,
        cos_sim:    float,
        euc_dist:   float,
        norm_ratio: float,
        euc_max:    float = 1.0,
    ) -> float:
        """
        Composite anomaly score ?_i ? [0, 1].

        ?_i = ???(1 - cos_sim)/2 + ???min(euc_dist/euc_max, 1) + ???norm_penalty

        Higher score -> more suspicious update.

        Args:
            cos_sim:    Cosine similarity (range [-1, 1])
            euc_dist:   Euclidean distance (raw)
            norm_ratio: Gradient norm ratio
            euc_max:    Normalisation factor for euclidean distance
        """
        # Cosine penalty: maps [-1,1] -> [0,1]
        cos_penalty  = (1.0 - cos_sim) / 2.0

        # Distance penalty: normalised to [0, 1]
        dist_penalty = min(euc_dist / (euc_max + 1e-10), 1.0)

        # Norm ratio penalty: penalise when ratio >> 1 (amplification)
        norm_penalty = min(max(norm_ratio - 1.0, 0.0) / (self.max_norm_ratio - 1.0 + 1e-10), 1.0)

        alpha = (
            self.lambda_cosine   * cos_penalty  +
            self.lambda_distance * dist_penalty  +
            self.lambda_norm     * norm_penalty
        )
        return float(np.clip(alpha, 0.0, 1.0))

    def trust_update(self, client_id: int, alpha_i: float) -> float:
        """
        Trust update rule: ?_i(t+1) = ? ? ?_i(t) + (1-?) ? (1 - ?_i)

        - Rewards consistently benign clients (high ? -> high weight)
        - Punishes suspicious clients (? decays toward 0)
        - ? (decay factor) controls how quickly trust changes
        """
        record = self.trust_records[client_id]
        tau_prev = record.trust_score
        tau_new  = self.gamma * tau_prev + (1.0 - self.gamma) * (1.0 - alpha_i)
        return float(np.clip(tau_new, 0.0, 1.0))

    def apply_trust_floor(self, client_id: int) -> None:
        """Enforce min_trust_floor for non-flagged clients only.
        Prevents innocent data-rich clients from being unfairly penalised
        for having larger model updates (higher euc_dist) than small clients.
        """
        record = self.trust_records[client_id]
        if not record.is_malicious:
            record.trust_score = max(record.trust_score, self.min_trust_floor)

    # -- Main scoring round ----------------------------------------------------

    def score_round(
        self,
        round_num:       int,
        client_gradients: Dict[int, np.ndarray],   # {cid: flat gradient vector}
        client_params:    Dict[int, np.ndarray],   # {cid: flat param vector}
        global_params:    np.ndarray,              # flat global param vector
    ) -> AggregationResult:
        """
        Score all clients for one federated round.

        Args:
            round_num:        Current federation round number
            client_gradients: Per-client flattened gradient deltas
            client_params:    Per-client flattened model parameters (post-training)
            global_params:    Current global model parameters (flattened)

        Returns:
            AggregationResult with trust weights and flagged clients
        """
        if not client_gradients:
            raise ValueError("No client gradients provided for scoring.")

        # -- Step 1: Gradient matrix + leave-one-out means for cosine/norm_ratio ----
        # FIX: Using LOO-mean means a malicious client is compared against the
        # consensus of its PEERS, so it cannot hide inside the global median/mean.
        grad_matrix = np.stack(list(client_gradients.values()))  # (n_clients, param_dim)
        client_ids  = list(client_gradients.keys())
        loo_means   = {}
        for idx, cid in enumerate(client_ids):
            others = np.delete(grad_matrix, idx, axis=0)
            loo_means[cid] = np.mean(others, axis=0)   # mean of all OTHER clients
        global_grad = np.mean(grad_matrix, axis=0)     # fallback reference

        # -- Step 2: Per-client leave-one-out MAX for euclidean normalisation ------
        # FIX: Using global max lets the malicious client (highest euc) self-normalise
        # to dist_penalty=1.0 while benign outliers also get high penalties.
        # LOO-max: each client is normalised by the max distance of all OTHER clients,
        # so the malicious client (euc=2335 vs loo_max=1202) correctly gets penalty=1.0
        # while benign clients (euc<=1202 vs loo_max=2335) get penalty<=0.52.
        euc_dist_map = {
            cid: self.euclidean_distance(client_params.get(cid, global_params), global_params)
            for cid in client_gradients
        }
        loo_euc_max = {}
        cid_list = list(euc_dist_map.keys())
        for cid in cid_list:
            others_euc = [v for k, v in euc_dist_map.items() if k != cid]
            loo_euc_max[cid] = max(others_euc) if others_euc else 1.0

        # -- Step 3: Score each client ---------------------------------------------
        anomaly_scores:   Dict[int, float] = {}
        cos_similarities: Dict[int, float] = {}
        euc_distances:    Dict[int, float] = {}
        norm_ratios:      Dict[int, float] = {}

        for cid in client_gradients:
            g_i      = client_gradients[cid]
            theta_i  = client_params.get(cid, global_params)
            ref_grad = loo_means[cid]        # compare against peers, not self-inclusive mean
            euc_norm = loo_euc_max[cid]      # normalise against max of ALL OTHER clients

            cs  = self.cosine_similarity(g_i, ref_grad)
            ed  = euc_dist_map[cid]
            nr  = self.norm_ratio(g_i, ref_grad)
            alpha_i = self.anomaly_score(cs, ed, nr, euc_norm)

            cos_similarities[cid] = cs
            euc_distances[cid]    = ed
            norm_ratios[cid]      = nr
            anomaly_scores[cid]   = alpha_i

            # Update trust record
            tau_new = self.trust_update(cid, alpha_i)
            record  = self.trust_records[cid]
            record.trust_score    = tau_new
            record.anomaly_score  = alpha_i
            record.cos_similarity = cs
            record.euc_distance   = ed
            record.norm_ratio     = nr
            record.is_malicious   = alpha_i > self.anomaly_threshold
            if record.is_malicious:
                record.rounds_flagged += 1
            else:
                record.rounds_flagged = 0
                # Apply floor: innocent clients cannot be eroded below min_trust_floor
                self.apply_trust_floor(cid)
            record.log_round(round_num)

        # -- Step 4: Compute normalised trust weights -----------------------------
        raw_weights: Dict[int, float] = {}
        for cid in client_gradients:
            record = self.trust_records[cid]
            if record.is_malicious:
                raw_weights[cid] = self.malicious_weight
            else:
                raw_weights[cid] = max(record.trust_score, self.malicious_weight)

        weight_sum = sum(raw_weights.values()) + 1e-10
        trust_weights = {cid: w / weight_sum for cid, w in raw_weights.items()}

        # -- Step 5: Identify trusted vs flagged ----------------------------------
        trusted  = [cid for cid in client_gradients if not self.trust_records[cid].is_malicious]
        flagged  = [cid for cid in client_gradients if self.trust_records[cid].is_malicious]

        result = AggregationResult(
            round_num=round_num,
            trusted_clients=trusted,
            flagged_clients=flagged,
            trust_weights=trust_weights,
            anomaly_scores=anomaly_scores,
            cos_similarities=cos_similarities,
            euc_distances=euc_distances,
        )
        self.round_results.append(result)
        return result

    # -- Reporting -------------------------------------------------------------

    def get_trust_summary(self) -> Dict[int, Dict]:
        """Return current trust state for all clients."""
        return {
            cid: {
                "trust_score":    r.trust_score,
                "anomaly_score":  r.anomaly_score,
                "cos_similarity": r.cos_similarity,
                "is_malicious":   r.is_malicious,
                "rounds_flagged": r.rounds_flagged,
            }
            for cid, r in self.trust_records.items()
        }

    def print_round_report(self, result: AggregationResult) -> None:
        """Print a formatted trust report for a round."""
        print(f"\n  +- TRUST SCORES -- Round {result.round_num} {'-'*30}")
        for cid in sorted(result.trust_weights.keys()):
            record = self.trust_records[cid]
            status = "[ATTACK] MALICIOUS" if record.is_malicious else "[OK] trusted  "
            print(
                f"  | Client {cid:02d} {status} | "
                f"?={record.trust_score:.4f} | "
                f"?={result.anomaly_scores[cid]:.4f} | "
                f"cos={result.cos_similarities[cid]:+.4f} | "
                f"w={result.trust_weights[cid]:.4f}"
            )
        flagged = result.flagged_clients
        print(f"  +- Trusted: {result.trusted_clients} | Flagged: {flagged if flagged else 'none'}")