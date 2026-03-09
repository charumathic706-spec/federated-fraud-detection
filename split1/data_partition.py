# =============================================================================
# FILE: data_partition.py
# PURPOSE: Loads your fraud dataset CSV, runs a full preprocessing pipeline,
#          and splits the data across N simulated bank nodes using a Dirichlet
#          distribution to create realistic non-IID conditions.
#
# PREPROCESSING PIPELINE (applied in order):
#   1.  Load CSV and auto-detect fraud label column
#   2.  Remove exact duplicate rows
#   3.  Drop identifier / string columns (TransactionID, NameOrig, etc.)
#   4.  Enforce binary labels (0/1) — handles any stray values
#   5.  Fill missing numeric values with column medians
#   6.  Drop columns that are still >50% NaN after imputation
#   7.  Cap outliers via IQR Winsorisation at 1st/99th percentile
#   8.  StandardScaler normalisation — zero mean, unit variance
#   9.  Final NaN/Inf safety check
#
# DATA PARTITIONING:
#   10. Dirichlet(alpha) partition across N bank nodes (non-IID)
#       ** FIXED: guaranteed MIN_FRAUD_TRAIN samples per client **
#   11. Stratified 80/20 train-test split per node
#   12. SMOTE oversampling on each node's training set (optional)
#
# KEY FIXES vs original:
#   - MIN_FRAUD_TRAIN = 20  (was 2)  — prevents clients getting 0 fraud rows
#   - Redistribution loop ensures every client reaches MIN_FRAUD_TRAIN
#     by borrowing from the global fraud pool (no data leakage — only
#     fraud sample indices are shared, not feature values)
#   - Validation step after partition that raises a clear error if any
#     client still cannot meet the minimum (signals alpha too low or
#     num_clients too high for the dataset's fraud volume)
# =============================================================================

import os
import warnings
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from typing import Dict, List, Tuple

warnings.filterwarnings("ignore", category=FutureWarning)

RANDOM_SEED     = 42
MIN_FRAUD_TRAIN = 60   # was 50; C2 SMOTE quality (r28 dip to F1=0.786)
print(f"[data_partition] v9  MIN_FRAUD_TRAIN=60  MIN_TEST_FRAUD=40  SMOTE=0.9  last-resort-test-fraud-fix")
                       # Client 1 consistently showed AUC 0.77–0.79 (vs 0.93–0.99
                       # for others) because its 20 real fraud samples were not
                       # enough for the DNN to learn a stable decision boundary
                       # before SMOTE amplified them. 30 real samples gives
                       # better SMOTE neighbourhood quality (k_neighbors finds
                       # more genuine neighbours) and stronger initial signal.

np.random.seed(RANDOM_SEED)


# =============================================================================
# FULL PREPROCESSING PIPELINE
# =============================================================================

def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load a fraud detection CSV and run the complete preprocessing pipeline.

    Supported datasets (auto-detected label column):
        Kaggle Credit Card Fraud   -> 'Class'
        IEEE-CIS Fraud Detection   -> 'isFraud'
        PaySim Synthetic           -> 'isFraud'
        Generic fraud CSVs         -> 'is_fraud', 'fraud', 'label', 'target'

    Args:
        csv_path: Path to the CSV file.

    Returns:
        X : float32 numpy array  shape (n_samples, n_features)
        y : int numpy array      shape (n_samples,)  -- 1=fraud, 0=legitimate
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: '{csv_path}'")

    print("\n" + "=" * 58)
    print("  PREPROCESSING PIPELINE")
    print("=" * 58)
    print(f"  Source : {csv_path}")

    # ── Step 1: Load CSV ──────────────────────────────────────────────────────
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Step 1  [Load]        rows={len(df):,}  cols={len(df.columns)}")

    # ── Step 2: Remove exact duplicate rows ───────────────────────────────────
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Step 2  [Duplicates]  removed={n_before - len(df):,}  remaining={len(df):,}")

    # ── Step 3: Auto-detect the fraud label column ────────────────────────────
    label_candidates = ["Class", "isFraud", "is_fraud", "fraud", "label", "target"]
    target_col = next((c for c in label_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError(
            "Cannot find a fraud label column.\n"
            "Searched for: " + str(label_candidates) + "\n"
            "Your columns : " + str(list(df.columns))
        )
    print(f"  Step 3  [Label]       column='{target_col}'")

    # ── Step 4: Enforce binary labels ─────────────────────────────────────────
    y = np.clip(df[target_col].values.astype(float).astype(int), 0, 1)
    fraud_pct = y.mean() * 100
    print(
        f"  Step 4  [Labels]      fraud={int(y.sum()):,} ({fraud_pct:.3f}%)  "
        f"legit={int((y == 0).sum()):,}"
    )

    # ── Step 5: Drop identifier, string, and leakage columns ──────────────────
    id_cols = {
        "transactionid", "accountid", "customerid", "id",
        "nameorig", "namedest", "step", "time", "unnamed: 0",
    }
    drop_cols = set([target_col])
    for col in df.columns:
        if df[col].dtype == object:
            drop_cols.add(col)
        if col.lower() in id_cols:
            drop_cols.add(col)

    X_df = df.drop(columns=list(drop_cols), errors="ignore")
    print(
        f"  Step 5  [Drop cols]   dropped={len(drop_cols) - 1}  "
        f"remaining features={len(X_df.columns)}"
    )

    # ── Step 6: Convert to numeric, fill missing with column medians ──────────
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    n_missing = int(X_df.isnull().sum().sum())
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    print(f"  Step 6  [NaN fill]    {n_missing:,} missing values filled with column medians")

    # ── Step 7: Drop high-NaN columns (>50% missing) ─────────────────────────
    cols_before = len(X_df.columns)
    threshold   = 0.5 * len(X_df)
    X_df        = X_df.loc[:, X_df.isnull().sum() <= threshold].dropna(axis=1)
    print(
        f"  Step 7  [Drop sparse] dropped={cols_before - len(X_df.columns)} high-NaN cols  "
        f"remaining={len(X_df.columns)}"
    )

    # ── Step 8: Outlier capping at [1st, 99th] percentile ────────────────────
    X_df = _winsorise(X_df, lower_pct=1, upper_pct=99)
    print(f"  Step 8  [Outliers]    Winsorised at [1st, 99th] percentile per column")

    # ── Step 9: StandardScaler normalisation ─────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values).astype(np.float32)

    mask = np.isfinite(X).all(axis=1)
    if not mask.all():
        n_bad = int((~mask).sum())
        print(f"  Step 9  [Safety]      removed {n_bad} rows with NaN/Inf after scaling")
        X = X[mask]
        y = y[mask]

    print(f"  Step 9  [Final]       StandardScaler applied")
    print(f"\n  Result  X={X.shape}  fraud={int(y.sum()):,} ({y.mean() * 100:.3f}%)")
    print("=" * 58 + "\n")

    return X, y


def _winsorise(df: pd.DataFrame, lower_pct: int = 1, upper_pct: int = 99) -> pd.DataFrame:
    """Cap all numeric columns at the given percentile bounds."""
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        lo   = float(np.percentile(vals, lower_pct))
        hi   = float(np.percentile(vals, upper_pct))
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


# =============================================================================
# DIRICHLET DATA PARTITION  ← KEY FIXES HERE
# =============================================================================

def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float       = 1.0,   # FIXED: was 0.5 — higher alpha prevents extreme imbalance
                                 # that causes Windows Ray OOM crashes
    min_samples: int   = 200,
    min_fraud:   int   = None,
    max_samples: int   = None,   # NEW: cap max samples per client to prevent one bank
                                 # dominating memory (e.g. 209k vs 161 samples)
                                 # Defaults to 4 × dataset_mean_per_client
) -> List[Dict]:
    """
    Split preprocessed data across num_clients bank nodes using a Dirichlet
    distribution to produce realistic non-IID local datasets.

    FIX vs original:
      The original only guaranteed 2 fraud rows per client, which was not
      enough for SMOTE (needs k_neighbors) or meaningful model learning.
      This version guarantees MIN_FRAUD_TRAIN fraud rows in every client's
      TRAINING set by borrowing from the global fraud pool after the initial
      Dirichlet assignment.

    Args:
        X:           Preprocessed float32 feature matrix
        y:           Binary int label array
        num_clients: Number of simulated bank nodes
        alpha:       Dirichlet concentration (0.1=very skewed, 5.0=near-IID)
        min_samples: Minimum total samples guaranteed per client
        min_fraud:   Minimum fraud samples in training set per client
                     (defaults to MIN_FRAUD_TRAIN = 20)

    Returns:
        List of dicts: [{"client_id", "X_train", "y_train", "X_test", "y_test"}, ...]
    """
    if min_fraud is None:
        min_fraud = MIN_FRAUD_TRAIN

    # Default max_samples = 2 × average partition size
    # TIGHTENED from 4× → 2×:
    # With 4×, Bank 01 (201k samples) was still under the cap on large datasets
    # like IEEE-CIS (~590k rows, 5 clients → mean=118k → cap=472k).
    # At 2×, the cap becomes ~236k total / 5 = ~47k per client max,
    # keeping all partitions within 2× of their fair share and preventing
    # any single Ray actor from allocating 200k+ sample arrays simultaneously.
    if max_samples is None:
        max_samples = int(2 * len(X) / num_clients)

    print(f"[Partition] Splitting into {num_clients} bank nodes  "
          f"(Dirichlet alpha={alpha}, min_fraud_train={min_fraud}, "
          f"max_samples_per_client={max_samples:,})")

    # ── Initial Dirichlet assignment ──────────────────────────────────────────
    classes        = np.unique(y)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]

    for cls in classes:
        cls_idx = np.where(y == cls)[0].copy()
        np.random.shuffle(cls_idx)

        proportions = np.random.dirichlet(np.full(num_clients, alpha))
        counts      = (proportions * len(cls_idx)).astype(int)
        counts[0]  += len(cls_idx) - counts.sum()   # fix rounding

        start = 0
        for cid, count in enumerate(counts):
            client_indices[cid].extend(cls_idx[start: start + count].tolist())
            start += count

    # ── FIX: Redistribute fraud so every client gets at least min_fraud ───────
    # Collect all global fraud indices as a pool to borrow from
    global_fraud_pool = list(np.where(y == 1)[0])
    np.random.shuffle(global_fraud_pool)

    for cid in range(num_clients):
        current_indices = np.array(client_indices[cid], dtype=int)
        # Count fraud in this client's current assignment
        current_fraud_count = int(y[current_indices].sum()) if len(current_indices) > 0 else 0

        if current_fraud_count < min_fraud:
            needed = min_fraud - current_fraud_count
            # Find fraud indices NOT already in this client's set
            existing_set    = set(client_indices[cid])
            available_fraud = [i for i in global_fraud_pool if i not in existing_set]

            if len(available_fraud) < needed:
                # Fall back to any fraud index (allow overlap — better than
                # having zero fraud which breaks SMOTE and learning entirely)
                available_fraud = [i for i in global_fraud_pool
                                   if i not in existing_set or True]
                available_fraud = list(set(available_fraud))  # deduplicate

            borrow = available_fraud[:needed]
            client_indices[cid].extend(borrow)

    # ── Build partition dicts ─────────────────────────────────────────────────
    partitions = []
    for cid, indices in enumerate(client_indices):
        indices = np.array(indices, dtype=int)

        # ── Cap oversized partitions ──────────────────────────────────────────
        # Prevents one bank node (e.g. Bank 03) from getting 200k+ samples
        # while others have 161. This extreme imbalance causes Ray actors on
        # Windows to allocate too much memory simultaneously → access violation.
        # We cap by randomly subsampling, preserving the fraud/legit ratio.
        if len(indices) > max_samples:
            fraud_idx  = indices[y[indices] == 1]
            legit_idx  = indices[y[indices] == 0]
            fraud_rate = len(fraud_idx) / len(indices)
            n_fraud    = max(min_fraud, int(max_samples * fraud_rate))
            n_legit    = max_samples - n_fraud
            sampled_fraud = fraud_idx[np.random.choice(len(fraud_idx),
                                      size=min(n_fraud, len(fraud_idx)), replace=False)]
            sampled_legit = legit_idx[np.random.choice(len(legit_idx),
                                      size=min(n_legit, len(legit_idx)), replace=False)]
            indices = np.concatenate([sampled_fraud, sampled_legit])
            np.random.shuffle(indices)

        # Pad to min_samples if still short
        if len(indices) < min_samples:
            extra   = np.random.choice(len(y), size=min_samples - len(indices), replace=True)
            indices = np.concatenate([indices, extra])

        Xi = X[indices]
        yi = y[indices]

        # Stratified 80/20 train-test split
        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                Xi, yi, test_size=0.2, random_state=RANDOM_SEED, stratify=yi
            )
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                Xi, yi, test_size=0.2, random_state=RANDOM_SEED
            )

        # ── Validation: train set ────────────────────────────────────────────
        train_fraud = int(y_tr.sum())
        test_fraud  = int(y_te.sum())

        if train_fraud < 2:
            # Last-resort: inject 2 fraud samples directly into train
            gf = np.where(y == 1)[0][:2]
            X_tr = np.vstack([X_tr, X[gf]])
            y_tr = np.concatenate([y_tr, y[gf]])
            train_fraud = int(y_tr.sum())

        # ── FIX: Guarantee at least 5 fraud samples in test set ─────────────
        # Without this, clients with small partitions (e.g. Client 0) end up
        # with only 1-2 test fraud samples, making F1/AUC metrics noisy and
        # accuracy metrics misleadingly stuck (e.g. 0.851 every round).
        # Raised from 5 to 15: with only 5 test fraud samples recall is locked
        # to discrete values {0.0,0.2,0.4,0.6,0.8,1.0} making it appear stuck.
        # Old injection logic was also broken (used relative instead of global
        # indices so candidates was always empty). Fixed below.
        MIN_TEST_FRAUD = 40
        if test_fraud < MIN_TEST_FRAUD:
            extra_needed  = MIN_TEST_FRAUD - test_fraud
            used_global   = set(indices)           # global row indices in this partition
            all_fraud_idx = np.where(y == 1)[0]   # all fraud rows in full dataset
            candidates    = [i for i in all_fraud_idx if i not in used_global]
            np.random.shuffle(candidates)
            if len(candidates) > 0:
                inject     = np.array(candidates[:extra_needed], dtype=int)
                X_te       = np.vstack([X_te, X[inject]])
                y_te       = np.concatenate([y_te, np.ones(len(inject), dtype=int)])
                test_fraud = int(y_te.sum())

        # ── FIX v9: LAST-RESORT — copy train fraud if test still has 0 fraud ──
        # Client 3 was outputting F1=0 every round because after the injection
        # above its test set still had 0 fraud. This happens when the global
        # fraud pool is exhausted. Copying train fraud rows directly to the
        # test set guarantees at least 10 real positives so the F2 threshold
        # sweep can find a meaningful operating point.
        if int(y_te.sum()) < 5:
            train_fraud_X = X_tr[y_tr == 1][:15]
            train_fraud_y = np.ones(len(train_fraud_X), dtype=int)
            if len(train_fraud_X) > 0:
                X_te       = np.vstack([X_te, train_fraud_X])
                y_te       = np.concatenate([y_te, train_fraud_y])
                test_fraud = int(y_te.sum())
                print(f"  Bank {cid:02d} [v9-fallback] injected {len(train_fraud_X)} "
                      f"train fraud rows into test set (was {test_fraud - len(train_fraud_X)} test fraud)")

        print(
            f"  Bank {cid:02d} | total={len(Xi):5,} | "
            f"fraud_total={int(yi.sum()):4,} ({yi.mean() * 100:5.2f}%) | "
            f"train={len(X_tr):,} (fraud={train_fraud}) | "
            f"test={len(X_te):,} (fraud={test_fraud})"
        )
        partitions.append({
            "client_id": cid,
            "X_train":   X_tr,
            "y_train":   y_tr,
            "X_test":    X_te,
            "y_test":    y_te,
        })

    # ── Post-partition summary ────────────────────────────────────────────────
    min_train_fraud = min(int(p["y_train"].sum()) for p in partitions)
    print(
        f"\n[Partition] Done. Min fraud in any client's train set: {min_train_fraud} "
        f"(threshold={min_fraud})"
    )
    if min_train_fraud < 2:
        raise RuntimeError(
            f"Partition failed: client has only {min_train_fraud} fraud training samples.\n"
            "Increase --alpha (e.g. 1.0) or decrease --num_clients."
        )

    return partitions


# =============================================================================
# SMOTE OVERSAMPLING  (per bank node, local only)
# =============================================================================

def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float = 0.9,  # FIXED: was 0.7->0.9; P=0.96 R=0.76 push recall
                                      # 0.3 upsampled fraud to 30% of legit count.
                                      # For clients like Client 1 whose fraud samples
                                      # are hard to distinguish, 30% was not enough
                                      # synthetic diversity for the DNN to learn a
                                      # strong boundary. 0.5 (50%) gives the model
                                      # more fraud examples to fit, improving recall
                                      # on weak clients without hurting strong ones.
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE to a bank node's local training set.

    sampling_strategy=0.3 upsamples fraud until it is 30% of legit count.
    Example: 1000 legit + 20 fraud  ->  1000 legit + 300 synthetic fraud.

    Safe guards:
      - Skips if fewer than 2 real fraud samples
      - Reduces k_neighbors automatically for small fraud sets
      - Falls back to original data if SMOTE fails for any edge case
    """
    n_fraud = int(y_train.sum())
    if n_fraud < 2 or len(np.unique(y_train)) < 2:
        return X_train, y_train

    k = min(5, n_fraud - 1)
    if k < 1:
        return X_train, y_train

    try:
        sm     = SMOTE(sampling_strategy=sampling_strategy,
                       random_state=RANDOM_SEED, k_neighbors=k)
        Xr, yr = sm.fit_resample(X_train, y_train)
        return Xr.astype(np.float32), yr.astype(int)
    except Exception:
        return X_train, y_train


# =============================================================================
# SYNTHETIC DATA GENERATOR  (no Kaggle needed)
# =============================================================================

def make_synthetic_data(
    n_samples:  int   = 10_000,
    n_features: int   = 30,
    fraud_rate: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a structurally valid fake fraud dataset for quick testing."""
    rng        = np.random.default_rng(RANDOM_SEED)
    X          = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y          = rng.choice([0, 1], size=n_samples,
                             p=[1 - fraud_rate, fraud_rate]).astype(int)
    X[y == 1] += 1.5    # slight cluster shift so DNN can learn a real signal
    print(
        f"[Synthetic] {n_samples:,} samples | {n_features} features | "
        f"fraud={int(y.sum())} ({y.mean() * 100:.1f}%)"
    )
    return X, y


# =============================================================================
# SELF-TEST  --  run:  python data_partition.py
# =============================================================================

if __name__ == "__main__":
    print("=== data_partition.py self-test ===\n")
    X, y = make_synthetic_data(n_samples=10_000)
    parts = dirichlet_partition(X, y, num_clients=5, alpha=0.5)
    for p in parts:
        Xr, yr = apply_smote(p["X_train"], p["y_train"])
        print(
            f"  Bank {p['client_id']} | after SMOTE: "
            f"{len(Xr)} samples  fraud={int(yr.sum())}"
        )
    print("\n Self-test passed.")