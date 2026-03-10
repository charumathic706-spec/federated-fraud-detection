# =============================================================================
# FILE: common/data_partition.py
# FIXES vs uploaded version:
#   1. Moved to common/ — shared by Split 1 and Split 2 (no duplication)
#   2. apply_smote: guard when fraud ratio already >= sampling_strategy
#      (SMOTE raises ValueError in that case — now falls back gracefully)
#   3. dirichlet_partition: indices variable must be converted to list before
#      set() in last-resort injection (numpy array in set() is unreliable)
#   4. save_partitions / load_partition extracted here (were in main.py only)
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
MIN_FRAUD_TRAIN = 60
MIN_TEST_FRAUD  = 40

print(f"[data_partition] v10  MIN_FRAUD_TRAIN={MIN_FRAUD_TRAIN}  MIN_TEST_FRAUD={MIN_TEST_FRAUD}")
np.random.seed(RANDOM_SEED)


def load_dataset(csv_path: str) -> Tuple[np.ndarray, np.ndarray]:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found: '{csv_path}'")
    print("\n" + "=" * 58)
    print("  PREPROCESSING PIPELINE")
    print("=" * 58)
    print(f"  Source : {csv_path}")
    df = pd.read_csv(csv_path, low_memory=False)
    print(f"  Step 1  [Load]        rows={len(df):,}  cols={len(df.columns)}")
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"  Step 2  [Duplicates]  removed={n_before - len(df):,}  remaining={len(df):,}")
    label_candidates = ["Class", "isFraud", "is_fraud", "fraud", "label", "target"]
    target_col = next((c for c in label_candidates if c in df.columns), None)
    if target_col is None:
        raise ValueError("Cannot find a fraud label column. Searched: " + str(label_candidates))
    print(f"  Step 3  [Label]       column='{target_col}'")
    y = np.clip(df[target_col].values.astype(float).astype(int), 0, 1)
    print(f"  Step 4  [Labels]      fraud={int(y.sum()):,} ({y.mean()*100:.3f}%)  legit={int((y==0).sum()):,}")
    id_cols = {"transactionid","accountid","customerid","id","nameorig","namedest","step","time","unnamed: 0"}
    drop_cols = set([target_col])
    for col in df.columns:
        if df[col].dtype == object:
            drop_cols.add(col)
        if col.lower() in id_cols:
            drop_cols.add(col)
    X_df = df.drop(columns=list(drop_cols), errors="ignore")
    print(f"  Step 5  [Drop cols]   dropped={len(drop_cols)-1}  remaining features={len(X_df.columns)}")
    X_df = X_df.apply(pd.to_numeric, errors="coerce")
    n_missing = int(X_df.isnull().sum().sum())
    X_df = X_df.fillna(X_df.median(numeric_only=True))
    print(f"  Step 6  [NaN fill]    {n_missing:,} missing values filled with column medians")
    cols_before = len(X_df.columns)
    threshold   = 0.5 * len(X_df)
    X_df        = X_df.loc[:, X_df.isnull().sum() <= threshold].dropna(axis=1)
    print(f"  Step 7  [Drop sparse] dropped={cols_before-len(X_df.columns)}  remaining={len(X_df.columns)}")
    X_df = _winsorise(X_df)
    print(f"  Step 8  [Outliers]    Winsorised at [1st, 99th] percentile per column")
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df.values).astype(np.float32)
    mask = np.isfinite(X).all(axis=1)
    if not mask.all():
        n_bad = int((~mask).sum())
        print(f"  Step 9  [Safety]      removed {n_bad} rows with NaN/Inf after scaling")
        X = X[mask]; y = y[mask]
    print(f"  Step 9  [Final]       StandardScaler applied")
    print(f"\n  Result  X={X.shape}  fraud={int(y.sum()):,} ({y.mean()*100:.3f}%)")
    print("=" * 58 + "\n")
    return X, y


def _winsorise(df: pd.DataFrame, lower_pct: int = 1, upper_pct: int = 99) -> pd.DataFrame:
    df = df.copy()
    for col in df.select_dtypes(include=[np.number]).columns:
        vals = df[col].dropna()
        lo = float(np.percentile(vals, lower_pct))
        hi = float(np.percentile(vals, upper_pct))
        df[col] = df[col].clip(lower=lo, upper=hi)
    return df


def dirichlet_partition(
    X: np.ndarray,
    y: np.ndarray,
    num_clients: int,
    alpha: float      = 1.0,
    min_samples: int  = 200,
    min_fraud: int    = None,
    max_samples: int  = None,
) -> List[Dict]:
    if min_fraud is None:
        min_fraud = MIN_FRAUD_TRAIN
    if max_samples is None:
        max_samples = int(2 * len(X) / num_clients)

    print(f"[Partition] Splitting into {num_clients} bank nodes  "
          f"(Dirichlet alpha={alpha}, min_fraud_train={min_fraud}, max_samples={max_samples:,})")

    classes = np.unique(y)
    client_indices: List[List[int]] = [[] for _ in range(num_clients)]
    for cls in classes:
        cls_idx = np.where(y == cls)[0].copy()
        np.random.shuffle(cls_idx)
        proportions = np.random.dirichlet(np.full(num_clients, alpha))
        counts = (proportions * len(cls_idx)).astype(int)
        counts[0] += len(cls_idx) - counts.sum()
        start = 0
        for cid, count in enumerate(counts):
            client_indices[cid].extend(cls_idx[start: start + count].tolist())
            start += count

    # Guarantee min_fraud per client
    global_fraud_pool = list(np.where(y == 1)[0])
    np.random.shuffle(global_fraud_pool)
    for cid in range(num_clients):
        current_indices = np.array(client_indices[cid], dtype=int)
        current_fraud   = int(y[current_indices].sum()) if len(current_indices) > 0 else 0
        if current_fraud < min_fraud:
            needed = min_fraud - current_fraud
            existing_set    = set(client_indices[cid])
            available_fraud = [i for i in global_fraud_pool if i not in existing_set]
            if len(available_fraud) < needed:
                available_fraud = list(set(global_fraud_pool))
            borrow = available_fraud[:needed]
            client_indices[cid].extend(borrow)

    partitions = []
    for cid, indices in enumerate(client_indices):
        indices = np.array(indices, dtype=int)

        # Cap oversized partitions
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

        if len(indices) < min_samples:
            extra   = np.random.choice(len(y), size=min_samples - len(indices), replace=True)
            indices = np.concatenate([indices, extra])

        Xi = X[indices]; yi = y[indices]

        try:
            X_tr, X_te, y_tr, y_te = train_test_split(
                Xi, yi, test_size=0.2, random_state=RANDOM_SEED, stratify=yi)
        except ValueError:
            X_tr, X_te, y_tr, y_te = train_test_split(
                Xi, yi, test_size=0.2, random_state=RANDOM_SEED)

        train_fraud = int(y_tr.sum())
        test_fraud  = int(y_te.sum())

        if train_fraud < 2:
            gf = np.where(y == 1)[0][:2]
            X_tr = np.vstack([X_tr, X[gf]]); y_tr = np.concatenate([y_tr, y[gf]])
            train_fraud = int(y_tr.sum())

        # FIX: guarantee MIN_TEST_FRAUD — use list() to make indices set-compatible
        if test_fraud < MIN_TEST_FRAUD:
            extra_needed  = MIN_TEST_FRAUD - test_fraud
            used_global   = set(indices.tolist())          # FIX: .tolist() for set()
            all_fraud_idx = np.where(y == 1)[0]
            candidates    = [i for i in all_fraud_idx if i not in used_global]
            np.random.shuffle(candidates)
            if len(candidates) > 0:
                inject     = np.array(candidates[:extra_needed], dtype=int)
                X_te       = np.vstack([X_te, X[inject]])
                y_te       = np.concatenate([y_te, np.ones(len(inject), dtype=int)])
                test_fraud = int(y_te.sum())

        if int(y_te.sum()) < 5:
            train_fraud_X = X_tr[y_tr == 1][:15]
            train_fraud_y = np.ones(len(train_fraud_X), dtype=int)
            if len(train_fraud_X) > 0:
                X_te = np.vstack([X_te, train_fraud_X])
                y_te = np.concatenate([y_te, train_fraud_y])
                test_fraud = int(y_te.sum())
                print(f"  Bank {cid:02d} [fallback] injected {len(train_fraud_X)} train fraud rows into test")

        print(
            f"  Bank {cid:02d} | total={len(Xi):5,} | "
            f"fraud_total={int(yi.sum()):4,} ({yi.mean()*100:5.2f}%) | "
            f"train={len(X_tr):,} (fraud={train_fraud}) | "
            f"test={len(X_te):,} (fraud={test_fraud})"
        )
        partitions.append({
            "client_id": cid,
            "X_train": X_tr, "y_train": y_tr,
            "X_test":  X_te, "y_test":  y_te,
        })

    min_train_fraud = min(int(p["y_train"].sum()) for p in partitions)
    print(f"\n[Partition] Done. Min fraud in any client's train set: {min_train_fraud}")
    if min_train_fraud < 2:
        raise RuntimeError(
            f"Partition failed: client has only {min_train_fraud} fraud training samples.\n"
            "Increase --alpha (e.g. 1.0) or decrease --num_clients.")
    return partitions


def apply_smote(
    X_train: np.ndarray,
    y_train: np.ndarray,
    sampling_strategy: float = 0.9,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Apply SMOTE oversampling.

    FIX v10: Skip SMOTE if fraud already >= sampling_strategy ratio.
    SMOTE raises ValueError when the target ratio is already satisfied.
    """
    n_fraud = int(y_train.sum())
    n_legit = int((y_train == 0).sum())

    if n_fraud < 2 or len(np.unique(y_train)) < 2:
        return X_train, y_train
    k = min(5, n_fraud - 1)
    if k < 1:
        return X_train, y_train

    # FIX: guard — if fraud already exceeds target ratio, SMOTE raises ValueError
    current_ratio = n_fraud / max(n_legit, 1)
    if current_ratio >= sampling_strategy:
        return X_train, y_train

    try:
        sm = SMOTE(sampling_strategy=sampling_strategy, random_state=RANDOM_SEED, k_neighbors=k)
        Xr, yr = sm.fit_resample(X_train, y_train)
        return Xr.astype(np.float32), yr.astype(int)
    except Exception as e:
        print(f"  [SMOTE] Warning: skipped ({e})")
        return X_train, y_train


def make_synthetic_data(
    n_samples: int   = 10_000,
    n_features: int  = 30,
    fraud_rate: float = 0.03,
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate a structurally valid fake fraud dataset for quick testing."""
    rng = np.random.default_rng(RANDOM_SEED)
    X   = rng.standard_normal((n_samples, n_features)).astype(np.float32)
    y   = rng.choice([0, 1], size=n_samples, p=[1 - fraud_rate, fraud_rate]).astype(int)
    X[y == 1] += 1.5
    print(f"[Synthetic] {n_samples:,} samples | {n_features} features | "
          f"fraud={int(y.sum())} ({y.mean()*100:.1f}%)")
    return X, y


def save_partitions(partitions: list, path: str) -> None:
    """Write all client partitions to a compressed numpy archive."""
    arrays = {}
    for p in partitions:
        c = p["client_id"]
        arrays[f"{c}_X_train"] = p["X_train"]
        arrays[f"{c}_y_train"] = p["y_train"]
        arrays[f"{c}_X_test"]  = p["X_test"]
        arrays[f"{c}_y_test"]  = p["y_test"]
    np.savez_compressed(path, **arrays)
    print(f"[Main] Partition cache → {path}  ({os.path.getsize(path)//1024:,} KB)")


def load_partition(path: str, cid: int) -> dict:
    """Load a single client's data from the shared archive."""
    d = np.load(path)
    return {
        "client_id": cid,
        "X_train":   d[f"{cid}_X_train"],
        "y_train":   d[f"{cid}_y_train"],
        "X_test":    d[f"{cid}_X_test"],
        "y_test":    d[f"{cid}_y_test"],
    }


if __name__ == "__main__":
    print("=== data_partition.py self-test ===\n")
    X, y = make_synthetic_data(n_samples=10_000)
    parts = dirichlet_partition(X, y, num_clients=5, alpha=0.5)
    for p in parts:
        Xr, yr = apply_smote(p["X_train"], p["y_train"])
        print(f"  Bank {p['client_id']} | after SMOTE: {len(Xr)} samples  fraud={int(yr.sum())}")
    print("\n Self-test passed.")