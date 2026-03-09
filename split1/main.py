# =============================================================================
# FILE: main.py
# PURPOSE: Split 1 federated simulation using the modern Flower API.
#
# FLOWER VERSION COMPATIBILITY:
#   This file targets Flower >= 1.8 (new ServerApp / ClientApp API).
#   If you have Flower 1.7.x pinned (as in the Colab notebook), run:
#       pip install "flwr>=1.8" imbalanced-learn==0.11.0
#   OR use the Colab notebook which pins flwr==1.7.0 and uses
#   fl.simulation.start_simulation() (works on Linux/Colab, not Windows).
#
# WHY NOT fl.simulation.start_simulation() ON WINDOWS:
#   start_simulation() uses Ray's plasma shared-memory store. On Windows,
#   concurrent large numpy array allocations across Ray worker processes
#   cause "Windows fatal exception: access violation". This is a known
#   Ray/Windows incompatibility that cannot be fixed with resource limits.
#
# SOLUTION — Real Flower FL, no Ray, no shared memory:
#   Server runs in the MAIN THREAD (required on Windows — signal.signal()
#   only works in the main thread). Clients run as separate subprocesses,
#   each connecting to the server via gRPC over localhost:8080. This is
#   exactly how production FL works — just with localhost instead of real IPs.
#
#   Architecture:
#     main.py (main thread) → fl.server.start_server() ← blocks until done
#                                    ↕ gRPC TCP :8080
#     Subprocess 0 → fl.client.start_client("127.0.0.1:8080")  [Bank 00]
#     Subprocess 1 → fl.client.start_client("127.0.0.1:8080")  [Bank 01]
#     ...
#     Subprocess N → fl.client.start_client("127.0.0.1:8080")  [Bank 0N]
#
#   Clients are launched BEFORE the server starts (they will retry until
#   the server is available). The server starts last, in main thread.
#
# USAGE:
#   python main.py --data_path ../data/creditcard.csv
#   python main.py --data_path ../data/ieee_cis_merged.csv
#   python main.py --synthetic
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
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

from data_partition  import load_dataset, dirichlet_partition, make_synthetic_data
from fedavg_strategy import get_fedavg_strategy
from flower_client   import BankFederatedClient
from data_partition  import apply_smote

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
    out = os.path.join(save_dir, "training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {out}")


# =============================================================================
# PARTITION CACHE  (written by orchestrator, read by each client subprocess)
# Subprocesses cannot receive large numpy arrays via arguments.
# We write all partitions to a .npz file; each subprocess reads only its own.
# =============================================================================

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
    print(f"[Main] Partition cache → {path}  "
          f"({os.path.getsize(path) // 1024:,} KB)")


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


# =============================================================================
# CLIENT SUBPROCESS ENTRY POINT
# When main.py is re-invoked with --_client_mode, it becomes a Flower client.
# =============================================================================

def _run_as_client(cid: int, cache: str, model_type: str, use_smote: bool) -> None:
    """
    Load local data, build BankFederatedClient, connect to server via gRPC.
    This runs inside a subprocess — completely isolated memory space.
    """
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

    # Retry loop: server may not be ready immediately
    for attempt in range(30):
        try:
            fl.client.start_client(
                server_address = SERVER_ADDRESS,
                client         = client.to_client(),
            )
            break   # connected and finished — exit loop
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
    p = argparse.ArgumentParser(
        description="Split 1 — Federated Training (Flower gRPC, Windows-safe)"
    )
    p.add_argument("--data_path",    type=str,   default=None)
    p.add_argument("--synthetic",    action="store_true")
    p.add_argument("--num_clients",  type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=25)
    p.add_argument("--model",        type=str,   default="dnn",
                   choices=["dnn", "logistic"])
    p.add_argument("--alpha",        type=float, default=1.0)
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--no_smote",     action="store_true")
    p.add_argument("--max_samples",  type=int,   default=None)
    p.add_argument("--log_dir",      type=str,   default="logs")

    # Hidden flags used internally when re-invoking as a client subprocess
    p.add_argument("--_client_mode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_cid",   type=int, default=-1,  help=argparse.SUPPRESS)
    p.add_argument("--_cache", type=str, default="",  help=argparse.SUPPRESS)
    p.add_argument("--_model", type=str, default="dnn", help=argparse.SUPPRESS)
    p.add_argument("--_smote", type=str, default="true", help=argparse.SUPPRESS)
    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    # ── CLIENT MODE ───────────────────────────────────────────────────────────
    # Subprocess re-invocation: become a Flower client and connect to server
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
    print("  SPLIT 1 — FEDERATED TRAINING CORE")
    print("  Blockchain-Based Dynamic Trust Modeling")
    print(f"  Model : {args.model.upper()} | Clients: {args.num_clients} | Rounds: {args.rounds}")
    print(f"  Engine: Flower gRPC — server in main thread, clients as subprocesses")
    print(f"  Server: {SERVER_ADDRESS}")
    print(f"{sep}\n")

    # ── Step 1: Load data ─────────────────────────────────────────────────────
    if args.synthetic or args.data_path is None:
        if not args.synthetic:
            print("[Main] No --data_path given — using synthetic data.\n")
        X, y = make_synthetic_data()
    else:
        X, y = load_dataset(args.data_path)

    # ── Step 2: Partition ─────────────────────────────────────────────────────
    partitions = dirichlet_partition(
        X, y,
        num_clients = args.num_clients,
        alpha       = args.alpha,
        max_samples = args.max_samples,
    )

    # ── Step 3: Write shared partition cache ──────────────────────────────────
    cache_path = os.path.join(args.log_dir, ".partition_cache.npz")
    save_partitions(partitions, cache_path)

    # ── Step 4: Build FedAvg strategy ────────────────────────────────────────
    strategy = get_fedavg_strategy(
        num_clients  = args.num_clients,
        fraction_fit = args.fraction_fit,
        log_dir      = args.log_dir,
    )

    # ── Step 5: Spawn client subprocesses BEFORE starting server ─────────────
    # Clients start with a retry loop (up to 30s) so they wait for the server.
    # This must happen BEFORE fl.server.start_server() because start_server()
    # BLOCKS the main thread until all rounds complete.
    smote_flag    = "false" if args.no_smote else "true"
    client_procs  = []

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
        proc = subprocess.Popen(
            cmd,
            stdout = subprocess.PIPE,
            stderr = subprocess.STDOUT,
            text   = True,
        )
        client_procs.append((cid, proc))
        print(f"  Spawned Bank {cid:02d}  PID={proc.pid}")

    # Stream each client's stdout to console in background threads
    def _stream(cid, proc):
        for line in proc.stdout:
            sys.stdout.write(f"  [Bank {cid:02d}] {line}")
            sys.stdout.flush()

    stream_threads = []
    for cid, proc in client_procs:
        t = threading.Thread(target=_stream, args=(cid, proc), daemon=True)
        t.start()
        stream_threads.append(t)

    # Brief pause to let subprocesses start their retry loops
    print("\n[Main] Clients are waiting for server. Starting server now...\n")
    time.sleep(2.0)

    # ── Step 6: Start Flower gRPC server IN MAIN THREAD ──────────────────────
    # MUST be main thread on Windows — fl.server.start_server() registers
    # signal handlers via signal.signal() which Python only permits in the
    # main thread. Running it in a background thread raises:
    #   "ValueError: signal only works in main thread of the main interpreter"
    #
    # start_server() BLOCKS here until all num_rounds complete, then returns.
    try:
        fl.server.start_server(
            server_address = SERVER_ADDRESS,
            config         = fl.server.ServerConfig(num_rounds=args.rounds),
            strategy       = strategy,
        )
    except Exception as exc:
        print(f"\n[Main] Server error: {exc}")
        # Kill client subprocesses if server failed
        for _, proc in client_procs:
            proc.terminate()
        raise

    print("\n[Main] All rounds complete. Waiting for clients to disconnect...")

    # ── Step 7: Wait for all client subprocesses to exit ─────────────────────
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

    # ── Step 8: Cleanup, summary, plot ───────────────────────────────────────
    try:
        os.remove(cache_path)
    except OSError:
        pass

    strategy.print_summary()

    log_path = os.path.join(args.log_dir, "training_log.json")
    plot_training_curves(log_path, args.log_dir)

    print(f"\n[Main] {'✅' if all_ok else '⚠️ '} Split 1 complete.")
    print(f"         Log  → {log_path}")
    print(f"         Plot → {os.path.join(args.log_dir, 'training_curves.png')}")
    print(f"\n  → Next: run Split 2 (trust-weighted aggregation + attack simulation)")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for Windows subprocess spawning
    main()