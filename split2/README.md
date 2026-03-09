# Split 2: Trust-Weighted Aggregation Engine
### Blockchain-Based Dynamic Trust Modeling for Federated Fraud Detection

**Builds directly on Split 1.** Replaces the standard FedAvg aggregation with a
dynamic trust-weighted mechanism that detects and neutralises adversarial clients.

---

## What's New in Split 2

| Component | Split 1 | Split 2 |
|-----------|---------|---------|
| Local model | XGBoost (primary) | **DNN** (primary, gradient-based) |
| Aggregation | Standard FedAvg | **Trust-Weighted FedAvg** |
| Attack defense | None | **Cosine sim + Euclidean dist + Norm ratio** |
| Trust scores | None | **τ_i per client, updated each round** |
| Blockchain metadata | None | **Model hash + trust scores logged** |

---

## File Structure

```
split2_trust_aggregation/
│
├── main.py                      ← Entry point with attack simulation + comparison
├── trust_scoring.py             ← Core math: anomaly score, trust update rule
├── trust_weighted_strategy.py   ← Flower Strategy replacing FedAvg
├── attack_simulator.py          ← Label-flip + gradient scaling attack injection
│
└── (imports from split1_federated_core/)
        local_models.py          ← Updated with DNN as primary model
        data_partition.py
        flower_client.py
        fedavg_strategy.py       ← Used for comparison baseline
```

---

## Setup

```bash
# Install dependencies (same as Split 1, no additions needed)
pip install -r ../split1_federated_core/requirements.txt
```

---

## Running Simulations

### Clean run (no attacks)
```bash
python main.py --synthetic --num_clients 5 --rounds 10
```

### Label-flipping attack (clients 1 and 3 poisoned)
```bash
python main.py --synthetic \
  --attack label_flip \
  --malicious_clients 1 3 \
  --num_clients 5 \
  --rounds 10
```

### Gradient scaling attack
```bash
python main.py --synthetic \
  --attack gradient_scale \
  --malicious_clients 1 \
  --scale_factor 8.0 \
  --num_clients 5 \
  --rounds 10
```

### Full comparison: FedAvg vs Trust-Weighted (generates comparison plot)
```bash
python main.py --synthetic \
  --attack label_flip \
  --malicious_clients 1 2 \
  --compare \
  --rounds 10
```

### With real dataset
```bash
python main.py \
  --data_path ../data/creditcard.csv \
  --attack combined \
  --malicious_clients 1 3 \
  --model dnn \
  --rounds 15 \
  --compare
```

---

## CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--attack` | none | `label_flip` \| `gradient_scale` \| `combined` \| `none` |
| `--malicious_clients` | [] | Space-separated client IDs e.g. `1 3` |
| `--scale_factor` | 5.0 | Amplification factor for gradient scaling attack |
| `--anomaly_threshold` | 0.6 | α_i above this → client flagged malicious |
| `--gamma` | 0.9 | Trust decay factor (higher = slower trust change) |
| `--compare` | False | Also run FedAvg baseline for side-by-side comparison |
| `--model` | dnn | `dnn` \| `logistic` \| `xgboost` |

---

## Trust Scoring Mathematics

### Anomaly Score  α_i
```
α_i = λ₁·(1 - cos_sim)/2  +  λ₂·norm(euc_dist)  +  λ₃·norm_penalty

where:
  cos_sim  = dot(g_i, g_global) / (||g_i|| · ||g_global||)
  euc_dist = ||θ_i - θ_global||₂
  norm_ratio = ||g_i|| / ||g_global||
```

### Trust Update Rule  τ_i(t+1)
```
τ_i(t+1) = γ · τ_i(t) + (1 - γ) · (1 - α_i)
```

### Trust Weights  w_i
```
w_i = τ_i / Σ_j τ_j          (if α_i ≤ threshold)
w_i = ε ≈ 0                   (if α_i > threshold — malicious)
```

### Attack Detection Signals
| Attack Type | Detection Signal |
|-------------|-----------------|
| Label-flipping | cos_sim < 0 (gradient reversal) |
| Gradient scaling | norm_ratio >> 1 (amplified update) |
| Combined | Both α > threshold |

---

## Outputs

After a run, `logs_split2/` contains:
- `trust_training_log.json` — per-round trust scores, anomaly scores, model hashes
- `trust_training_curves.png` — F1, AUC, per-client τ and α plots
- `comparison_fedavg_vs_trust.png` — side-by-side F1 comparison (if `--compare`)

The `trust_training_log.json` is structured as the input for **Split 3's blockchain
audit trail** — each entry contains `model_hash`, `trust_scores`, and `flagged_clients`.

---

## Interface for Split 3

Split 3 (Blockchain Governance Layer) consumes the round logs produced here:

```python
# Each round log entry in trust_training_log.json:
{
  "round":            3,
  "timestamp":        "2025-01-01T12:00:00+00:00",
  "model_hash":       "sha256:abc123...",   ← stored on Hyperledger Fabric
  "trusted_clients":  [0, 2, 4],
  "flagged_clients":  [1, 3],              ← smart contract updates their trust
  "trust_scores":     {"0": 0.92, "1": 0.03, ...},
  "anomaly_scores":   {"0": 0.08, "1": 0.87, ...},
  "global_f1":        0.89
}
```