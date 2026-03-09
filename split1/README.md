# Module 1 — Federated Training Core
### Blockchain-Based Dynamic Trust Modeling for Federated Fraud Detection

> **Part of a 3-module system.**
> Module 1 builds the federated training foundation.
> Module 2 (Split 2) replaces FedAvg with trust-weighted aggregation and attack detection.
> Module 3 (Split 3) adds Hyperledger Fabric blockchain audit trail.

---

## Table of Contents
1. [What This Module Does](#1-what-this-module-does)
2. [Directory Structure](#2-directory-structure)
3. [Preprocessing Pipeline](#3-preprocessing-pipeline)
4. [Algorithm — Federated Averaging (FedAvg)](#4-algorithm--federated-averaging-fedavg)
5. [DNN Architecture](#5-dnn-architecture)
6. [Non-IID Data Partitioning — Dirichlet Distribution](#6-non-iid-data-partitioning--dirichlet-distribution)
7. [Class Imbalance Handling — SMOTE](#7-class-imbalance-handling--smote)
8. [File-by-File Explanation](#8-file-by-file-explanation)
9. [Tools and Libraries](#9-tools-and-libraries)
10. [Supported Datasets](#10-supported-datasets)
11. [Installation](#11-installation)
12. [Running the Simulation](#12-running-the-simulation)
13. [Output Files](#13-output-files)
14. [Key Design Decisions](#14-key-design-decisions)
15. [Module 2 Interface Contract](#15-module-2-interface-contract)

---

## 1. What This Module Does

This module simulates a **federated learning network** across multiple bank nodes (financial institutions). Each bank holds its own private transaction dataset and trains a local Deep Neural Network (DNN) for fraud detection. No raw transaction data is ever shared — only model weight arrays are transmitted between banks and the central server.

The full round-by-round flow is:

```
┌─────────────────────────────────────────────────────────────┐
│                        SERVER                               │
│  Maintains global model  │  Aggregates client updates       │
│  Selects clients per round│  Logs metrics to JSON           │
└──────────────┬──────────────────────────────┬───────────────┘
               │  Global model weights        │
               │  (float32 arrays)            │ Updated weights
               ▼                              │ + metrics
┌──────────────────────────────────────────────────────────────┐
│  Bank 00        Bank 01        Bank 02     ...  Bank N-1     │
│  ─────────      ─────────      ─────────        ─────────    │
│  Private        Private        Private          Private      │
│  Transactions   Transactions   Transactions     Transactions │
│  ↓              ↓              ↓                ↓            │
│  Local DNN      Local DNN      Local DNN        Local DNN    │
│  training       training       training         training     │
│                                                              │
│  Raw data NEVER leaves each bank node                       │
└──────────────────────────────────────────────────────────────┘
```

**Privacy guarantee:** Only weight tensors cross the network boundary, never raw financial records. This satisfies GDPR and PCI-DSS data residency requirements.

---

## 2. Directory Structure

```
split1_federated_core/
│
├── data_partition.py       ← Preprocessing pipeline + Dirichlet partitioning
├── local_models.py         ← DNN architecture + Logistic Regression baseline
├── flower_client.py        ← BankFederatedClient (one per bank node)
├── fedavg_strategy.py      ← FedAvg server strategy with JSON logging
├── main.py                 ← Entry point — wires everything together
├── split1_colab.ipynb      ← Google Colab notebook (ready to run)
└── requirements.txt        ← Pinned dependencies
```

---

## 3. Preprocessing Pipeline

**File:** `data_partition.py` → `load_dataset(csv_path)`

The entire preprocessing pipeline runs automatically when you call `load_dataset()`. Nothing needs to be done manually. The steps in order are:

### Step 1 — Load CSV
```python
df = pd.read_csv(csv_path, low_memory=False)
```
Loads the raw CSV. `low_memory=False` prevents mixed-type column warnings on large financial datasets.

### Step 2 — Remove Duplicate Rows
```python
df = df.drop_duplicates()
```
**Why:** Fraud datasets from real banking systems often contain duplicate log entries due to retry mechanisms or ETL pipeline errors. Duplicates inflate training accuracy artificially and must be removed before any other step.

### Step 3 — Auto-Detect Fraud Label Column
```python
label_candidates = ["Class", "isFraud", "is_fraud", "fraud", "label", "target"]
```
Searches for the binary fraud label column by name. Supports Kaggle Credit Card Fraud (`Class`), IEEE-CIS (`isFraud`), PaySim (`isFraud`), and generic datasets. Raises a clear error with column names listed if none found.

### Step 4 — Enforce Binary Labels
```python
y = np.clip(df[target_col].values.astype(float).astype(int), 0, 1)
```
**Why:** Some datasets have float labels (`1.0`), multi-class extensions, or stray values. This step forces all labels into `{0, 1}` cleanly regardless of original encoding.

### Step 5 — Drop Identifier, String, and Leakage Columns
```python
id_cols = {"transactionid", "accountid", "customerid", "id",
           "nameorig", "namedest", "step", "time", "unnamed: 0"}
```
**Why:** Columns like `TransactionID`, `NameOrig`, `NameDest` are unique identifiers — they have no predictive signal and would cause the DNN to overfit to individual transaction IDs. The `Time` column in the Credit Card dataset is a raw timestamp with no normalisation that adds noise without feature engineering. All `object` dtype columns (strings) are also dropped since the DNN requires numeric input.

### Step 6 — Fill Missing Values (Median Imputation)
```python
X_df = X_df.fillna(X_df.median(numeric_only=True))
```
**Why median, not mean:** Transaction amounts and account balances have heavily right-skewed distributions with extreme outliers. The median is robust to these outliers — it represents the "typical" value. The mean would be pulled upward by a single $1M transaction, producing a poor imputation value for typical $50 transactions.

### Step 7 — Drop High-NaN Columns
```python
threshold = 0.5 * len(X_df)
X_df = X_df.loc[:, X_df.isnull().sum() <= threshold]
```
**Why:** Columns that are more than 50% missing (common in IEEE-CIS identity features) cannot be reliably imputed — the imputed values would not represent real observations. These columns are dropped entirely rather than imputed.

### Step 8 — Outlier Capping (Winsorisation)
```python
lo = np.percentile(col, 1)   # 1st percentile
hi = np.percentile(col, 99)  # 99th percentile
df[col] = df[col].clip(lower=lo, upper=hi)
```
**Why:** Financial transaction data has extreme outliers — a single wire transfer of $1,000,000 alongside thousands of $10–$50 transactions. Without capping, these outliers cause exploding gradients in the DNN during certain mini-batches. Winsorisation (capping at percentile bounds) neutralises outliers without removing any rows, preserving all training samples.

**Why 1st/99th and not 5th/95th:** Fraud transactions often have legitimately large amounts. Capping at 5% would eliminate many real fraud signals. The 1st/99th percentile removes only the most extreme statistical noise while keeping the fraud amount distribution intact.

### Step 9 — StandardScaler Normalisation
```python
scaler = StandardScaler()
X = scaler.fit_transform(X_df.values).astype(np.float32)
```
**Why:** DNNs require input features to be on the same scale for gradient descent to converge. Without normalisation:
- A feature with range [0, 1,000,000] dominates gradients
- A feature with range [0, 1] gets almost no gradient signal
- The network learns to rely entirely on the large-scale features

StandardScaler transforms each feature to **zero mean and unit variance**: `x_scaled = (x - mean) / std`

### Step 9b — Final NaN/Inf Safety Check
```python
mask = np.isfinite(X).all(axis=1)
X = X[mask]
y = y[mask]
```
**Why:** StandardScaler divides by standard deviation. If a column has zero variance (all values identical, e.g., a constant feature), division produces `NaN`. This final check removes any such rows silently rather than crashing during DNN training.

### Complete pipeline summary

| Step | Operation | Library | Why |
|---|---|---|---|
| 1 | Load CSV | pandas | Raw data ingestion |
| 2 | Drop duplicates | pandas | Prevent artificial accuracy inflation |
| 3 | Detect label column | pandas | Multi-dataset compatibility |
| 4 | Binary label enforcement | numpy | Handle float/multi-class edge cases |
| 5 | Drop ID/string columns | pandas | Remove non-predictive noise |
| 6 | Median imputation | pandas | Robust to financial data skewness |
| 7 | Drop high-NaN columns | pandas | Remove unrecoverable missing features |
| 8 | Winsorisation [1%, 99%] | numpy | Prevent gradient explosion from outliers |
| 9 | StandardScaler | sklearn | Enable DNN gradient convergence |
| 9b | NaN/Inf removal | numpy | Safety against zero-variance columns |

---

## 4. Algorithm — Federated Averaging (FedAvg)

**File:** `fedavg_strategy.py`

FedAvg is the standard algorithm for federated learning, introduced by McMahan et al. (2017). It trains a shared global model across decentralised nodes without sharing raw data.

### The FedAvg Round (repeated R times)

```
Round t:
  1. Server broadcasts current global weights θ_global(t) to all K clients
  2. Each client k independently:
       a. Initialises local model with θ_global(t)
       b. Runs E epochs of SGD on local private data D_k
       c. Produces updated local weights θ_k(t+1)
       d. Sends θ_k(t+1) back to server
  3. Server aggregates:
       θ_global(t+1) = Σ_k [ (n_k / N) · θ_k(t+1) ]
       where n_k = number of local training samples
             N   = total samples across all clients
```

### The aggregation formula

```
θ_global = (n_0/N)·θ_0 + (n_1/N)·θ_1 + ... + (n_K/N)·θ_K
```

Clients with more data contribute more to the global model. This is implemented in `weighted_average()` in `fedavg_strategy.py` and is applied to all metric dicts as well as the parameter arrays.

### Why FedAvg is the baseline (not the final algorithm)

Standard FedAvg has a critical weakness: **it treats all client updates equally**. A malicious bank that submits poisoned gradients (label-flipping attack) gets the same weight as a honest bank. Module 2 replaces FedAvg with a trust-weighted variant that detects and suppresses malicious updates.

### Flower simulation mode

Instead of running a real network, Flower's `start_simulation()` runs all bank nodes as threads on a single machine. This is functionally identical to the distributed case for algorithmic purposes — each client receives parameters, trains locally, and returns results independently. This makes development and testing possible without multi-machine infrastructure.

---

## 5. DNN Architecture

**File:** `local_models.py` → `FraudDNN` + `DNNFraudModel`

### Network structure

```
Input (n_features)
    │
    ├─ Linear(n_features → 256)
    │  BatchNorm1d(256)
    │  ReLU
    │  Dropout(0.3)          ← + Skip connection Linear(n_features → 256)
    │
    ├─ Linear(256 → 128)
    │  BatchNorm1d(128)
    │  ReLU
    │  Dropout(0.3)          ← + Skip connection Linear(256 → 128)
    │
    ├─ Linear(128 → 64)
    │  BatchNorm1d(64)
    │  ReLU
    │  Dropout(0.3)          ← + Skip connection Linear(128 → 64)
    │
    └─ Linear(64 → 1)
       (raw logit — no sigmoid here, applied in loss)
```

**Output:** A single logit. Sigmoid is applied at inference time: `P(fraud) = sigmoid(logit)`. Classification threshold is 0.5.

### Why each component was chosen

**BatchNorm1d** — Each bank has a different data distribution (different fraud rate, different transaction types). BatchNorm normalises layer inputs during training, stabilising gradient flow even when the input distribution shifts between FL rounds. Without it, training becomes unstable as the global model arrives with weights calibrated to a different data distribution than the local one.

**Residual skip connections** — After multiple federated rounds, the global model accumulates small weight updates from many clients. Skip connections (ResNet-style) preserve gradient magnitude through the network during backpropagation, preventing gradients from vanishing to zero by round 5–10 and ensuring meaningful parameter updates continue throughout training.

**Dropout(0.3)** — Each bank has a small local dataset (a fraction of the global data). Without regularisation, the DNN memorises local training samples and performs poorly when the global model is evaluated across all banks. Dropout randomly disables 30% of neurons per forward pass, forcing the network to learn robust distributed representations.

**BCEWithLogitsLoss with pos_weight=10** — Fraud detection datasets have extreme class imbalance (typically 0.1%–3% fraud). If the loss treats fraud and legit samples equally, the DNN learns to predict "legit" for everything and achieves 99% accuracy while detecting zero fraud. Setting `pos_weight=10` means the loss penalises missed fraud 10× more than missed legitimate transactions.

**Adam optimiser with weight_decay=1e-5** — Adam adapts per-parameter learning rates using moment estimates, converging faster than SGD on fraud detection tasks. L2 weight decay (1e-5) adds mild regularisation without affecting convergence speed.

**Gradient clipping at norm 1.0** — In federated learning, clients sometimes receive a global model that is far from their local optimum. The first few gradient steps can be very large, causing weight explosion. Clipping all gradient tensors to a maximum norm of 1.0 prevents this without affecting training when gradients are already small.

**StepLR scheduler (step=3, gamma=0.5)** — Halves the learning rate every 3 epochs. This allows aggressive initial learning and fine-tuned convergence as local training progresses.

### Why DNN over XGBoost for this project

XGBoost was the original candidate but was replaced by DNN because:

| Property | DNN | XGBoost |
|---|---|---|
| Parameter space | Continuous float32 tensors | JSON tree structures |
| Gradient delta | Meaningful continuous vector | No equivalent |
| Cosine similarity | Clean and interpretable | Undefined |
| Euclidean distance | Clean L2 norm | Undefined |
| Flower transport | Direct state_dict serialisation | Byte-string hack |

Module 2's trust scoring requires computing cosine similarity and Euclidean distance between client gradient updates and the global gradient. These operations are only mathematically meaningful for continuous vector spaces — which DNN parameters provide naturally.

---

## 6. Non-IID Data Partitioning — Dirichlet Distribution

**File:** `data_partition.py` → `dirichlet_partition()`

### The problem with IID splits

In real federated learning, each bank sees completely different customers and transaction patterns. Bank A (retail banking) processes thousands of $10–$100 card payments. Bank B (corporate banking) processes hundreds of $100,000+ wire transfers. Their local fraud patterns are completely different. A simple random IID split (each bank gets a uniform random sample) does not model this reality.

### The Dirichlet solution

The Dirichlet distribution `Dir(α)` generates a probability vector over K classes. When used for data partitioning:

```
For each class c (legit=0, fraud=1):
  1. Sample proportions: p = Dir(α, α, ..., α)  ← K values summing to 1
  2. Assign p[k] × |class_c| samples to client k
```

The concentration parameter `α` controls heterogeneity:

```
α = 0.1  → Very heterogeneous
           Some banks get almost all the fraud, others get almost none
           Realistic for specialised financial institutions

α = 0.5  → Moderately heterogeneous (default, recommended)
           Each bank has noticeably different fraud rates
           Reflects typical multi-bank federated learning conditions

α = 5.0  → Near-IID
           All banks get roughly equal fraud rates
           Not realistic but useful for debugging
```

**Example with α=0.5, 5 banks:**
```
Bank 00: 2,100 samples | fraud rate: 1.8%
Bank 01: 1,850 samples | fraud rate: 5.2%
Bank 02: 2,400 samples | fraud rate: 0.4%
Bank 03: 1,600 samples | fraud rate: 7.1%
Bank 04: 2,050 samples | fraud rate: 2.5%
```

This is the realistic scenario the FL algorithm must handle — and why simple FedAvg without trust scoring struggles when one bank's model is corrupted.

---

## 7. Class Imbalance Handling — SMOTE

**File:** `data_partition.py` → `apply_smote()`

### The imbalance problem

Fraud detection datasets are severely imbalanced. The Kaggle Credit Card Fraud dataset has 0.172% fraud. After Dirichlet partitioning, some banks may receive even fewer fraud examples (e.g., 5 fraud out of 2,000 transactions). Training a DNN on 5 positive examples produces a model that predicts "legit" for everything.

### SMOTE — Synthetic Minority Over-sampling Technique

SMOTE generates **synthetic fraud samples** by interpolating between real fraud examples in feature space:

```
For each real fraud sample x_i:
  1. Find its k nearest neighbours among other fraud samples
  2. Pick a random neighbour x_j
  3. Generate synthetic sample: x_new = x_i + λ(x_j - x_i)
     where λ is uniform random in [0, 1]
```

The result is a new synthetic fraud transaction that lies on the line segment between two real fraud transactions — it is a plausible fraud pattern, not a random noise point.

**In this module:** SMOTE is applied *per bank node* after Dirichlet partitioning, upsampling local fraud to 30% of the local legit count (`sampling_strategy=0.3`).

**Example:**
```
Before SMOTE:  1,680 legit  |  20 fraud  (1.2% fraud rate)
After SMOTE:   1,680 legit  | 504 fraud  (23% fraud rate, 484 synthetic)
```

**Safety guards in the code:**
- Skips if fewer than 2 real fraud samples (SMOTE needs at least 1 neighbour)
- Reduces `k_neighbors` from 5 to `n_fraud - 1` for tiny fraud sets
- Falls back to original data silently if SMOTE fails for any edge case

---

## 8. File-by-File Explanation

### `data_partition.py`
**Role:** Data loading, preprocessing, and partitioning.

| Function | What it does |
|---|---|
| `load_dataset(csv_path)` | Runs the full 9-step preprocessing pipeline on your CSV. Returns cleaned `(X, y)` arrays. |
| `_winsorise(df, lower, upper)` | Internal helper. Clips each column at given percentile bounds. |
| `dirichlet_partition(X, y, ...)` | Splits data across N bank nodes using Dirichlet distribution. Returns list of per-client train/test dicts. |
| `apply_smote(X_train, y_train)` | Applies SMOTE oversampling to a single bank node's training data. |
| `make_synthetic_data()` | Generates a fake 10,000-sample fraud dataset for testing without Kaggle access. |

### `local_models.py`
**Role:** DNN architecture and model wrappers.

| Class / Function | What it does |
|---|---|
| `FraudDNN` | PyTorch `nn.Module`. Three hidden layers with BatchNorm, ReLU, Dropout, residual skips. |
| `DNNFraudModel` | Wrapper. Handles `fit()`, `evaluate()`, Flower param serialisation, and gradient delta storage for Module 2. |
| `LogisticFraudModel` | Sklearn Logistic Regression baseline with the same interface as `DNNFraudModel`. |
| `get_model(type, dim)` | Factory function. Returns `DNNFraudModel` for `"dnn"`, `LogisticFraudModel` for `"logistic"`. |

### `flower_client.py`
**Role:** Defines how each bank node behaves during a FL round.

| Class / Function | What it does |
|---|---|
| `BankFederatedClient` | Flower `Client` subclass. Receives global weights, trains locally, returns updated weights. |
| `BankFederatedClient.fit()` | Called by Flower server each round. Loads global params → trains DNN → returns updated params + training metrics. |
| `BankFederatedClient.evaluate()` | Called by Flower server for evaluation. Loads global params → runs inference on local test set → returns loss and metrics. |
| `make_client_fn(partitions, ...)` | Returns a `client_fn(cid) → BankFederatedClient` closure for `fl.simulation.start_simulation()`. |

### `fedavg_strategy.py`
**Role:** Server-side aggregation logic.

| Class / Function | What it does |
|---|---|
| `weighted_average(metrics)` | Computes sample-count-weighted average of metric dicts. Used for both fit and evaluate aggregation. Also re-used by Module 2. |
| `InstrumentedFedAvg` | Extends Flower's built-in `FedAvg`. Adds per-round console logging, JSON log writing, and best-F1 tracking. |
| `InstrumentedFedAvg.aggregate_fit()` | Logs per-client training metrics, then delegates to parent FedAvg for actual parameter averaging. |
| `InstrumentedFedAvg.aggregate_evaluate()` | Logs per-client eval metrics, writes round to `training_log.json`, tracks best F1. |
| `get_fedavg_strategy(...)` | Factory function. Constructs `InstrumentedFedAvg` with correct `min_fit_clients` settings. |

### `main.py`
**Role:** Entry point. Wires all components together and runs the simulation.

| Function | What it does |
|---|---|
| `main()` | Loads data → partitions → builds client factory → builds strategy → runs `fl.simulation.start_simulation()` → prints summary → plots curves. |
| `plot_training_curves(log_path, dir)` | Reads `training_log.json` and saves a 3-panel PNG with F1, AUC-ROC, and Recall curves per round. |
| `build_parser()` | Defines all CLI arguments. |

### `split1_colab.ipynb`
**Role:** Self-contained Google Colab notebook. Upload the 5 Python files, optionally upload your CSV, adjust the config cell, and run all cells in order. Includes IEEE-CIS merge helper.

---

## 9. Tools and Libraries

| Library | Version | Role in this module |
|---|---|---|
| **Flower (flwr)** | 1.7.0 | Federated learning framework. Provides `Client`, `Strategy`, and `start_simulation()`. Handles the communication protocol between server and clients. |
| **PyTorch** | 2.1.0 | DNN implementation. `nn.Module`, `BCEWithLogitsLoss`, `Adam`, `DataLoader`, `clip_grad_norm_`. |
| **scikit-learn** | 1.3.2 | `StandardScaler` for normalisation. `train_test_split` for per-client data splits. `F1Score`, `ROC-AUC`, `Precision`, `Recall` metrics. `LogisticRegression` baseline. |
| **imbalanced-learn** | 0.11.0 | `SMOTE` for synthetic oversampling of local fraud minority class. |
| **pandas** | 2.1.4 | CSV loading, duplicate removal, NaN handling, column dropping, `median()` imputation. |
| **numpy** | 1.26.2 | Array operations, Dirichlet sampling, Winsorisation, data type management. |
| **matplotlib** | 3.8.2 | Training curve plots (F1, AUC-ROC, Recall over rounds). |

---

## 10. Supported Datasets

| Dataset | Label Column | Notes |
|---|---|---|
| Kaggle Credit Card Fraud | `Class` | 284,807 transactions, 0.172% fraud, 30 PCA features |
| IEEE-CIS Fraud Detection | `isFraud` | Must merge `train_transaction.csv` + `train_identity.csv` first (see below) |
| PaySim Synthetic | `isFraud` | Mobile money simulation, 6.3M transactions |
| Generic fraud CSV | `is_fraud`, `fraud`, `label`, `target` | Any binary fraud CSV |

### IEEE-CIS dataset merge (run once before training)

```python
import pandas as pd
transaction = pd.read_csv("train_transaction.csv")
identity    = pd.read_csv("train_identity.csv")
merged      = transaction.merge(identity, on="TransactionID", how="left")
merged.to_csv("ieee_cis_merged.csv", index=False)
```

---

## 11. Installation

```bash
# Clone/download the project then:
cd split1_federated_core

pip install -r requirements.txt
```

**Python version:** 3.9 or higher recommended.

**GPU support (optional):** If CUDA is available, the DNN will automatically train on GPU. No configuration needed — `torch.device("cuda" if torch.cuda.is_available() else "cpu")` handles it.

---

## 12. Running the Simulation

### Quick test — synthetic data (no dataset needed)
```bash
python main.py --synthetic --num_clients 5 --rounds 5
```

### With Kaggle Credit Card Fraud dataset
```bash
python main.py --data_path ../data/creditcard.csv --num_clients 5 --rounds 10
```

### With IEEE-CIS merged dataset
```bash
python main.py --data_path ../data/ieee_cis_merged.csv --num_clients 8 --rounds 15
```

### Logistic Regression baseline (fast, no GPU needed)
```bash
python main.py --data_path ../data/creditcard.csv --model logistic --rounds 5
```

### Highly heterogeneous partition (α=0.1)
```bash
python main.py --data_path ../data/creditcard.csv --alpha 0.1 --num_clients 5 --rounds 10
```

### All CLI arguments

| Argument | Default | Description |
|---|---|---|
| `--data_path` | None | Path to your fraud CSV file |
| `--synthetic` | False | Use synthetic data (ignores `--data_path`) |
| `--num_clients` | 5 | Number of simulated bank nodes |
| `--rounds` | 10 | Number of federated training rounds |
| `--model` | `dnn` | Local model: `dnn` or `logistic` |
| `--alpha` | 0.5 | Dirichlet concentration (lower = more heterogeneous) |
| `--fraction_fit` | 1.0 | Fraction of clients selected per round |
| `--no_smote` | False | Disable SMOTE oversampling |
| `--log_dir` | `logs` | Output directory for JSON log and plots |

### Google Colab
Open `split1_colab.ipynb`. The notebook has 9 cells:

| Cell | Action |
|---|---|
| 1 | Install dependencies |
| 2 | Upload the 5 Python files |
| 3 | Upload your dataset CSV (or skip for synthetic) |
| 4 | Configure NUM_CLIENTS, NUM_ROUNDS, MODEL_TYPE, etc. |
| 5 | Run the full simulation |
| 6 | Plot training curves |
| 7 | View per-bank metrics table |
| 8 | Download results |
| 9 | IEEE-CIS merge helper (if needed) |

---

## 13. Output Files

After a run, the `logs/` directory (or your `--log_dir`) contains:

### `training_log.json`
Per-round log with global and per-client metrics. Structure:
```json
[
  {
    "round": 1,
    "global_f1": 0.7823,
    "global_auc": 0.9241,
    "global_recall": 0.8102,
    "global_precision": 0.7612,
    "best_f1": 0.7823,
    "best_round": 1,
    "client_metrics": [
      {"client_id": 0, "f1": 0.7901, "auc_roc": 0.9312, "recall": 0.82, "precision": 0.76},
      ...
    ]
  },
  ...
]
```

This JSON is also the **input consumed by Module 2** — the trust-weighted aggregation engine reads it as a baseline for comparison plots.

### `training_curves.png`
Three-panel plot showing global F1 Score, AUC-ROC, and Recall across all federation rounds.

---

## 14. Key Design Decisions

### Why Flower over TensorFlow Federated or PySyft?

| Framework | Issue |
|---|---|
| TensorFlow Federated | TensorFlow-only. Cannot use PyTorch DNN or custom aggregation logic without major rewrites. |
| PySyft | High setup complexity. Requires running separate workers. Not suitable for simulation. |
| FedML | Good framework but less mature custom Strategy API. |
| **Flower** | Framework-agnostic (PyTorch, sklearn, XGBoost all work). `Strategy` API lets Module 2 replace `aggregate_fit()` with one subclass. `start_simulation()` runs multi-client FL on a single machine. |

### Why the gradient delta design matters for Module 2

In `DNNFraudModel.fit()`:
```python
self._params_before = self.get_params()        # snapshot before training
# ... training happens ...
params_after = self.get_params()
self._last_gradients = [after - before         # parameter delta
                        for after, before in zip(params_after, self._params_before)]
```

This delta vector is the **gradient proxy** used by Module 2's trust scorer. When a malicious bank performs a label-flipping attack, its gradient delta points in the opposite direction to the global gradient — this produces a strongly negative cosine similarity, triggering the anomaly detection.

### Why pos_weight=10 and not class-ratio-based

The Credit Card Fraud dataset has ~0.172% fraud, giving a theoretical pos_weight of ~580. In practice, SMOTE has already partially rebalanced the local training data. Using a pos_weight proportional to the actual SMOTE-adjusted ratio would require computing it per client and per round. The fixed pos_weight=10 is a practical compromise that consistently biases the DNN toward detecting fraud without making the loss too aggressive on the already-oversampled data.

---

## 15. Module 2 Interface Contract

Module 2 (`split2_trust_aggregation/`) imports from this module using relative paths. The interface is:

```python
# Imports Module 2 uses from Module 1:
from data_partition  import load_dataset, dirichlet_partition, make_synthetic_data
from flower_client   import BankFederatedClient, make_client_fn
from fedavg_strategy import get_fedavg_strategy, weighted_average
from local_models    import get_model
```

The `FitRes.metrics` dict returned by `BankFederatedClient.fit()` contains the keys Module 2's trust scorer reads:

| Key | Type | Used by Module 2 for |
|---|---|---|
| `client_id` | float | Identifying which trust record to update |
| `train_f1` | float | Performance-based anomaly signal |
| `train_auc` | float | Performance-based anomaly signal |
| `train_recall` | float | Fraud detection capacity signal |
| `n_fraud` | float | Validating non-zero fraud exposure |

Additionally, `DNNFraudModel.get_gradients()` and `get_flattened_gradient()` are called by Module 2 to access the parameter delta for cosine similarity computation.

---

*Module 1 is intentionally the clean, attack-free baseline. All adversarial robustness is added in Module 2.*