# =============================================================================
# FILE: split2/main.py
# PURPOSE: Split 2 — Trust-Weighted Aggregation + Attack Simulation
#          (Flower gRPC, subprocess architecture — same pattern as Split 1)
#
# What Split 2 adds on top of Split 1:
#   - TrustWeightedFedAvg replaces InstrumentedFedAvg
#   - AttackSimulator injects label-flip / gradient-scale attacks on chosen clients
#   - Per-round trust scores, anomaly scores, flagged clients are logged to JSON
#   - Model SHA-256 hash logged each round (-> Split 3 blockchain audit trail)
#
# Attack types:
#   --attack none            -- clean baseline (trust scoring only, no attack)
#   --attack label_flip      -- client flips fraud/legit labels during training
#   --attack gradient_scale  -- client amplifies gradient update by scale_factor
#   --attack combined        -- both attacks simultaneously
#
# Usage:
#   python -m split2.main --synthetic
#   python -m split2.main --synthetic --attack label_flip --malicious 1
#   python -m split2.main --synthetic --attack gradient_scale --malicious 1 3
#   python -m split2.main --data_path ../data/creditcard.csv --attack label_flip
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
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

_MODULE_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _MODULE_ROOT not in sys.path:
    sys.path.insert(0, _MODULE_ROOT)

from common.data_partition        import (
    load_dataset, dirichlet_partition, make_synthetic_data,
    save_partitions, load_partition,
)
from common.trust_weighted_strategy import get_trust_strategy
from common.attack_simulator        import AttackSimulator
from common.flower_client           import BankFederatedClient

SERVER_ADDRESS = "127.0.0.1:8081"   # different port from Split 1


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

    rounds  = [l["round"]                    for l in logs]
    f1s     = [l.get("global_f1", 0)         for l in logs]
    aucs    = [l.get("global_auc", 0)        for l in logs]
    recalls = [l.get("global_recall", 0)     for l in logs]
    flagged = [len(l.get("flagged_clients", [])) for l in logs]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Split 2 — Trust-Weighted FL + Attack Detection", fontsize=13, fontweight="bold")

    for ax, (vals, title, color) in zip(axes[:3], [
        (f1s,     "Global F1",    "#2196F3"),
        (aucs,    "Global AUC",   "#4CAF50"),
        (recalls, "Global Recall","#FF9800"),
    ]):
        ax.plot(rounds, vals, marker="o", linewidth=2, color=color, markersize=6)
        ax.fill_between(rounds, vals, alpha=0.12, color=color)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("Round"); ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.05); ax.grid(True, alpha=0.3)
        ax.yaxis.set_major_formatter(mtick.FormatStrFormatter("%.2f"))

    axes[3].bar(rounds, flagged, color="#E53935", alpha=0.8)
    axes[3].set_title("Flagged Clients / Round", fontsize=11)
    axes[3].set_xlabel("Round"); axes[3].set_ylabel("Count")
    axes[3].set_ylim(0, max(flagged or [1]) + 1)
    axes[3].grid(True, alpha=0.3)

    plt.tight_layout()
    out = os.path.join(save_dir, "split2_training_curves.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved -> {out}")


# =============================================================================
# CLIENT SUBPROCESS ENTRY POINT
# =============================================================================

def _run_as_client(cid: int, cache: str, model_type: str, use_smote: bool,
                   server_address: str) -> None:
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
                server_address = server_address,
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
    p = argparse.ArgumentParser(description="Split 2 — Trust-Weighted FL + Attack Simulation")
    p.add_argument("--data_path",    type=str,   default=None)
    p.add_argument("--synthetic",    action="store_true")
    p.add_argument("--num_clients",  type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=25)
    p.add_argument("--model",        type=str,   default="dnn", choices=["dnn", "logistic"])
    p.add_argument("--alpha",        type=float, default=1.0)
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--no_smote",     action="store_true")
    p.add_argument("--max_samples",  type=int,   default=None)
    p.add_argument("--log_dir",      type=str,   default="logs_split2")
    p.add_argument("--port",         type=int,   default=8081,
                   help="gRPC port (default 8081 to avoid collision with Split 1)")
    # Attack configuration
    p.add_argument("--attack",       type=str,   default="none",
                   choices=["none", "label_flip", "gradient_scale", "combined"])
    p.add_argument("--malicious",    type=int,   nargs="+", default=[1],
                   help="Client IDs to designate as malicious (default: [1])")
    p.add_argument("--scale_factor", type=float, default=5.0,
                   help="Gradient amplification factor for gradient_scale attack")
    p.add_argument("--attack_start", type=int,   default=1,
                   help="Round to begin attacking (default: 1)")
    # Hidden subprocess flags
    p.add_argument("--_client_mode",   action="store_true",  help=argparse.SUPPRESS)
    p.add_argument("--_cid",     type=int, default=-1,       help=argparse.SUPPRESS)
    p.add_argument("--_cache",   type=str, default="",       help=argparse.SUPPRESS)
    p.add_argument("--_model",   type=str, default="dnn",    help=argparse.SUPPRESS)
    p.add_argument("--_smote",   type=str, default="true",   help=argparse.SUPPRESS)
    p.add_argument("--_server",  type=str, default="127.0.0.1:8081", help=argparse.SUPPRESS)
    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    # ── CLIENT MODE ───────────────────────────────────────────────────────────
    if args._client_mode:
        _run_as_client(
            cid            = args._cid,
            cache          = args._cache,
            model_type     = args._model,
            use_smote      = (args._smote.lower() == "true"),
            server_address = args._server,
        )
        return

    # ── ORCHESTRATOR MODE ─────────────────────────────────────────────────────
    import flwr as fl

    server_address = f"127.0.0.1:{args.port}"
    os.makedirs(args.log_dir, exist_ok=True)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  SPLIT 2 — TRUST-WEIGHTED FEDERATED LEARNING")
    print(f"  Model : {args.model.upper()} | Clients: {args.num_clients} | Rounds: {args.rounds}")
    print(f"  Attack: {args.attack.upper()} | Malicious: {args.malicious}")
    print(f"  Engine: Flower gRPC | Server: {server_address}")
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
    cache_path = os.path.join(args.log_dir, ".partition_cache_s2.npz")
    save_partitions(partitions, cache_path)

    # Step 4: Build attack simulator
    attacker = AttackSimulator(
        attack_type        = args.attack,
        malicious_clients  = args.malicious if args.attack != "none" else [],
        scale_factor       = args.scale_factor,
        attack_start_round = args.attack_start,
    )

    # Step 5: Build trust-weighted strategy
    strategy = get_trust_strategy(
        num_clients      = args.num_clients,
        fraction_fit     = args.fraction_fit,
        log_dir          = args.log_dir,
        attack_simulator = attacker,
    )

    # Step 6: Spawn client subprocesses
    smote_flag   = "false" if args.no_smote else "true"
    client_procs = []

    print(f"[Main] Spawning {args.num_clients} client subprocesses...")
    for cid in range(args.num_clients):
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--_client_mode",
            "--_cid",    str(cid),
            "--_cache",  os.path.abspath(cache_path),
            "--_model",  args.model,
            "--_smote",  smote_flag,
            "--_server", server_address,
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

    # Step 7: Start Flower gRPC server in MAIN THREAD
    try:
        fl.server.start_server(
            server_address = server_address,
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

    log_path = os.path.join(args.log_dir, "trust_training_log.json")
    plot_training_curves(log_path, args.log_dir)

    print(f"\n[Main] {'OK' if all_ok else 'WARN'} Split 2 complete.")
    print(f"         Log  -> {log_path}")
    print(f"         Plot -> {os.path.join(args.log_dir, 'split2_training_curves.png')}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()