# =============================================================================
# FILE: split1/main.py
# PURPOSE: Split 1 — FedAvg Baseline (Flower gRPC, subprocess architecture)
#
# Architecture:
#   Orchestrator (main thread):
#     1. Loads/partitions data, writes .npz cache
#     2. Spawns N client subprocesses (each connects via gRPC)
#     3. Starts fl.server.start_server() in main thread (blocks until done)
#     4. Waits for clients to exit, plots results
#
#   Client subprocess (re-invoked with --_client_mode):
#     Loads its partition from cache, builds BankFederatedClient,
#     connects to server via fl.client.start_client("127.0.0.1:8080")
#
# Usage:
#   python -m split1.main --synthetic
#   python -m split1.main --data_path ../data/creditcard.csv
#   python -m split1.main --synthetic --num_clients 5 --rounds 10
# =============================================================================

import argparse
import json
import multiprocessing
import os
import subprocess
import sys
import threading
import time

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend (no display needed)
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

# Add module1/ root to sys.path so 'common' package resolves correctly
_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from common.data_partition  import (
    load_dataset, dirichlet_partition, make_synthetic_data,
    save_partitions, load_partition,
)
from common.fedavg_strategy import get_fedavg_strategy
from common.flower_client   import BankFederatedClient

SERVER_ADDRESS = "127.0.0.1:8080"


# =============================================================================
# PLOTTING
# =============================================================================

def plot_training_curves(log_path: str, save_dir: str) -> None:
    if not os.path.exists(log_path):
        print("[Plot] Log not found — skipping.")
        return
    with open(log_path) as f:
        logs = json.load(f)
    if not logs:
        return

    rounds  = [l["round"]         for l in logs]
    f1s     = [l["global_f1"]     for l in logs]
    aucs    = [l["global_auc"]    for l in logs]
    recalls = [l["global_recall"] for l in logs]

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("Split 1 — Federated Training Curves (FedAvg Baseline)",
                 fontsize=13, fontweight="bold")
    for ax, (vals, title, color) in zip(axes, [
        (f1s,     "Global F1 Score", "#2196F3"),
        (aucs,    "Global AUC-ROC",  "#4CAF50"),
        (recalls, "Global Recall",   "#FF9800"),
    ]):
        ax.plot(rounds, vals, marker="o", linewidth=2, color=color, markersize=6)
        ax.fill_between(rounds, vals, alpha=0.12, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Federation Round")
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))
    plt.tight_layout()
    out = os.path.join(save_dir, "split1_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {out}")


# =============================================================================
# CLIENT SUBPROCESS ENTRY POINT
# =============================================================================

def _run_as_client(cid: int, cache: str, model_type: str, use_smote: bool) -> None:
    """Runs inside a subprocess — load partition, connect to server via gRPC."""
    import flwr as fl

    partition = load_partition(cache, cid)

    client = BankFederatedClient(
        client_id  = cid,
        X_train    = partition["X_train"],
        y_train    = partition["y_train"],
        X_test     = partition["X_test"],
        y_test     = partition["y_test"],
        model_type = model_type,
        use_smote  = use_smote,
    )

    for attempt in range(30):
        try:
            fl.client.start_client(
                server_address = SERVER_ADDRESS,
                client         = client.to_client(),
            )
            break
        except Exception as exc:
            if attempt < 29:
                time.sleep(1.0)
            else:
                print(f"  [Bank {cid:02d}] Failed to connect after 30s: {exc}")
                sys.exit(1)


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Split 1 — FedAvg Baseline (Flower gRPC)")
    p.add_argument("--data_path",    type=str,   default=None)
    p.add_argument("--synthetic",    action="store_true")
    p.add_argument("--num_clients",  type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=25)
    p.add_argument("--model",        type=str,   default="dnn", choices=["dnn", "logistic"])
    p.add_argument("--alpha",        type=float, default=1.0)
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--no_smote",     action="store_true")
    p.add_argument("--max_samples",  type=int,   default=None)
    p.add_argument("--log_dir",      type=str,   default="logs_split1")
    # Hidden flags for subprocess re-invocation
    p.add_argument("--_client_mode", action="store_true",  help=argparse.SUPPRESS)
    p.add_argument("--_cid",   type=int, default=-1,       help=argparse.SUPPRESS)
    p.add_argument("--_cache", type=str, default="",       help=argparse.SUPPRESS)
    p.add_argument("--_model", type=str, default="dnn",    help=argparse.SUPPRESS)
    p.add_argument("--_smote", type=str, default="true",   help=argparse.SUPPRESS)
    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    # ── CLIENT MODE ───────────────────────────────────────────────────────────
    if args._client_mode:
        _run_as_client(
            cid        = args._cid,
            cache      = args._cache,
            model_type = args._model,
            use_smote  = (args._smote.lower() == "true"),
        )
        return

    # ── ORCHESTRATOR MODE ─────────────────────────────────────────────────────
    import flwr as fl

    os.makedirs(args.log_dir, exist_ok=True)

    sep = "=" * 62
    print(f"\n{sep}")
    print("  SPLIT 1 — FEDERATED TRAINING CORE (FedAvg Baseline)")
    print(f"  Model : {args.model.upper()} | Clients: {args.num_clients} | Rounds: {args.rounds}")
    print(f"  Engine: Flower gRPC | Server: {SERVER_ADDRESS}")
    print(f"{sep}\n")

    # Step 1: Load data
    if args.synthetic or args.data_path is None:
        if not args.synthetic:
            print("[Main] No --data_path given — using synthetic data.\n")
        X, y = make_synthetic_data()
    else:
        X, y = load_dataset(args.data_path)

    # Step 2: Partition
    partitions = dirichlet_partition(
        X, y,
        num_clients = args.num_clients,
        alpha       = args.alpha,
        max_samples = args.max_samples,
    )

    # Step 3: Write shared partition cache
    cache_path = os.path.join(args.log_dir, ".partition_cache.npz")
    save_partitions(partitions, cache_path)

    # Step 4: Build FedAvg strategy
    strategy = get_fedavg_strategy(
        num_clients  = args.num_clients,
        fraction_fit = args.fraction_fit,
        log_dir      = args.log_dir,
    )

    # Step 5: Spawn client subprocesses BEFORE starting server
    smote_flag   = "false" if args.no_smote else "true"
    client_procs = []

    print(f"[Main] Spawning {args.num_clients} client subprocesses...")
    for cid in range(args.num_clients):
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--_client_mode",
            "--_cid",   str(cid),
            "--_cache", os.path.abspath(cache_path),
            "--_model", args.model,
            "--_smote", smote_flag,
        ]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        client_procs.append((cid, proc))
        print(f"  Spawned Bank {cid:02d}  PID={proc.pid}")

    def _stream(cid, proc):
        for line in proc.stdout:
            sys.stdout.write(f"  [Bank {cid:02d}] {line}")
            sys.stdout.flush()

    stream_threads = []
    for cid, proc in client_procs:
        t = threading.Thread(target=_stream, args=(cid, proc), daemon=True)
        t.start()
        stream_threads.append(t)

    print("\n[Main] Clients waiting for server. Starting server now...\n")
    time.sleep(2.0)

    # Step 6: Start Flower gRPC server in MAIN THREAD (blocks until done)
    try:
        fl.server.start_server(
            server_address = SERVER_ADDRESS,
            config         = fl.server.ServerConfig(num_rounds=args.rounds),
            strategy       = strategy,
        )
    except Exception as exc:
        print(f"\n[Main] Server error: {exc}")
        for _, proc in client_procs:
            proc.terminate()
        raise

    print("\n[Main] All rounds complete. Waiting for clients to disconnect...")

    for t in stream_threads:
        t.join(timeout=60)

    all_ok = True
    for cid, proc in client_procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"  [Warning] Bank {cid:02d} did not exit cleanly — terminating.")
            proc.terminate()
            all_ok = False
        if proc.returncode not in (0, None):
            print(f"  [Warning] Bank {cid:02d} exit code: {proc.returncode}")

    # Cleanup
    try:
        os.remove(cache_path)
    except OSError:
        pass

    strategy.print_summary()

    log_path = os.path.join(args.log_dir, "training_log.json")
    plot_training_curves(log_path, args.log_dir)

    print(f"\n[Main] {'OK' if all_ok else 'WARN'} Split 1 complete.")
    print(f"         Log  -> {log_path}")
    print(f"         Plot -> {os.path.join(args.log_dir, 'split1_training_curves.png')}")
    print(f"\n  -> Next: run Split 2 (trust-weighted aggregation + attack simulation)")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()