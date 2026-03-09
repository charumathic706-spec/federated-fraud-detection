"""
main.py — Split 2: Trust-Weighted Aggregation Engine
------------------------------------------------------
Windows-safe architecture — NO Ray, NO fl.simulation.start_simulation().

Uses the exact same gRPC + subprocess design as Split 1:
  - Server runs in the MAIN THREAD via fl.server.start_server()
  - Each client runs as a separate OS subprocess
  - Data shared via a temporary .npz partition cache file
  - Clients connect via gRPC over 127.0.0.1:8080

Supports:
  - Trust-weighted aggregation (replaces FedAvg)
  - Label-flipping attack simulation
  - Gradient scaling attack simulation
  - Optional FedAvg baseline comparison run

Usage:
    # Clean run (no attacks), synthetic data
    python main.py --synthetic --num_clients 5 --rounds 10

    # Label-flip attack on client 1, real dataset
    python main.py --data_path ..\data\creditcard.csv --attack label_flip --malicious_clients 1 --rounds 25

    # Gradient scaling attack
    python main.py --synthetic --attack gradient_scale --malicious_clients 1 --scale_factor 8.0

    # Combined attack + compare vs FedAvg baseline
    python main.py --data_path ..\data\creditcard.csv --attack combined --malicious_clients 1 2 --compare --rounds 25
"""

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

# ── Add Split 1 directory to path so we can import its modules ────────────────
# Split 1 must be in the sibling folder named 'split1'
_HERE      = os.path.dirname(os.path.abspath(__file__))
_SPLIT1    = os.path.join(_HERE, "..", "split1")
_SPLIT2    = _HERE
sys.path.insert(0, _SPLIT1)
sys.path.insert(0, _SPLIT2)

from data_partition  import load_dataset, dirichlet_partition, make_synthetic_data, apply_smote
from flower_client   import BankFederatedClient
from fedavg_strategy import get_fedavg_strategy    # Split 1 baseline (for --compare)
from attack_simulator import AttackSimulator
from trust_weighted_strategy import TrustWeightedFedAvg

SERVER_ADDRESS = "127.0.0.1:8081"   # 8081 to avoid conflict with Split 1 (8080)


# =============================================================================
# PARTITION CACHE  (identical helpers from Split 1 main.py)
# =============================================================================

def save_partitions(partitions: list, path: str) -> None:
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
    d = np.load(path)
    return {
        "client_id": cid,
        "X_train":   d[f"{cid}_X_train"],
        "y_train":   d[f"{cid}_y_train"],
        "X_test":    d[f"{cid}_X_test"],
        "y_test":    d[f"{cid}_y_test"],
    }


# =============================================================================
# ATTACKED BANK CLIENT
# Wraps BankFederatedClient, injects label-flip BEFORE local training
# =============================================================================

class AttackedBankClient(BankFederatedClient):
    """
    BankFederatedClient with label-flip data poisoning.
    The attack is applied to training labels BEFORE SMOTE and training.
    Gradient-scaling attack is applied server-side in TrustWeightedFedAvg.
    """
    def __init__(self, client_id, X_train, y_train, X_test, y_test,
                 model_type, use_smote, attacker: AttackSimulator):
        X_p, y_p = attacker.poison_data(client_id, X_train, y_train)
        super().__init__(client_id, X_p, y_p, X_test, y_test,
                         model_type, use_smote)
        self._attacker = attacker


# =============================================================================
# CLIENT SUBPROCESS ENTRY POINT
# =============================================================================

def _run_as_client(
    cid:        int,
    cache:      str,
    model_type: str,
    use_smote:  bool,
    attack_type: str,
    malicious:  list,
    scale_factor: float,
    attack_start_round: int,
) -> None:
    """
    Runs inside a subprocess. Loads local data, builds the client,
    connects to the gRPC server, and runs until training completes.
    """
    import flwr as fl

    partition = load_partition(cache, cid)

    attacker = AttackSimulator(
        attack_type=attack_type,
        malicious_clients=malicious,
        scale_factor=scale_factor,
        attack_start_round=attack_start_round,
    )

    client = AttackedBankClient(
        client_id  = cid,
        X_train    = partition["X_train"],
        y_train    = partition["y_train"],
        X_test     = partition["X_test"],
        y_test     = partition["y_test"],
        model_type = model_type,
        use_smote  = use_smote,
        attacker   = attacker,
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
# SERVER RUN HELPER  (generic — used for both trust and baseline runs)
# =============================================================================

def _run_server_with_clients(
    strategy,
    partitions:  list,
    num_clients: int,
    num_rounds:  int,
    log_dir:     str,
    model_type:  str,
    use_smote:   bool,
    attack_type: str,
    malicious:   list,
    scale_factor: float,
    attack_start_round: int,
    run_label:   str = "run",
) -> None:
    """
    Core routine: write partition cache, spawn client subprocesses,
    start gRPC server in main thread, wait for completion.
    """
    cache_path = os.path.join(log_dir, f".partition_cache_{run_label}.npz")
    save_partitions(partitions, cache_path)

    import flwr as fl

    # ── Build subprocess command template ─────────────────────────────────────
    smote_flag = "false" if not use_smote else "true"
    mal_str    = ",".join(str(m) for m in malicious) if malicious else ""

    client_procs = []
    print(f"\n[Main] Spawning {num_clients} client subprocesses ({run_label})...")
    for cid in range(num_clients):
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--_client_mode",
            "--_cid",         str(cid),
            "--_cache",       os.path.abspath(cache_path),
            "--_model",       model_type,
            "--_smote",       smote_flag,
            "--_attack",      attack_type,
            "--_malicious",   mal_str,
            "--_scale",       str(scale_factor),
            "--_atk_start",   str(attack_start_round),
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client_procs.append((cid, proc))
        print(f"  Spawned Bank {cid:02d}  PID={proc.pid}")

    # Stream each subprocess's stdout to console via background threads
    def _stream(cid, proc):
        for line in proc.stdout:
            sys.stdout.write(f"  [Bank {cid:02d}] {line}")
            sys.stdout.flush()

    stream_threads = []
    for cid, proc in client_procs:
        t = threading.Thread(target=_stream, args=(cid, proc), daemon=True)
        t.start()
        stream_threads.append(t)

    print(f"\n[Main] Clients waiting. Starting gRPC server ({run_label})...\n")
    time.sleep(2.0)

    # ── Start server IN MAIN THREAD (required on Windows) ────────────────────
    try:
        fl.server.start_server(
            server_address = SERVER_ADDRESS,
            config         = fl.server.ServerConfig(num_rounds=num_rounds),
            strategy       = strategy,
        )
    except Exception as exc:
        print(f"\n[Main] Server error: {exc}")
        for _, proc in client_procs:
            proc.terminate()
        raise

    print(f"\n[Main] All rounds complete ({run_label}). Waiting for clients...")

    for t in stream_threads:
        t.join(timeout=60)

    for cid, proc in client_procs:
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            print(f"  [Warning] Bank {cid:02d} did not exit cleanly — terminating.")
            proc.terminate()
        if proc.returncode not in (0, None):
            print(f"  [Warning] Bank {cid:02d} exit code: {proc.returncode}")

    try:
        os.remove(cache_path)
    except OSError:
        pass


# =============================================================================
# PLOTTING
# =============================================================================

def plot_trust_results(log_path: str, save_dir: str,
                       attack_type: str, malicious: list) -> None:
    if not os.path.exists(log_path):
        print("[Plot] Trust log not found — skipping.")
        return

    with open(log_path) as f:
        logs = json.load(f)

    rounds = [l["round"]      for l in logs]
    f1s    = [l.get("global_f1",  0) for l in logs]
    aucs   = [l.get("global_auc", 0) for l in logs]
    client_ids = sorted({int(k) for l in logs for k in l.get("trust_scores", {})})

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle(
        f"Split 2: Trust-Weighted Aggregation\n"
        f"Attack: {attack_type} | Malicious clients: {malicious if malicious else 'none'}",
        fontsize=12, fontweight="bold"
    )
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(client_ids), 1)))

    # F1
    ax = axes[0, 0]
    ax.plot(rounds, f1s, "o-", color="#2196F3", lw=2, ms=5)
    ax.set_title("Global F1 Score"); ax.set_xlabel("Round"); ax.set_ylabel("F1")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # AUC
    ax = axes[0, 1]
    ax.plot(rounds, aucs, "s-", color="#4CAF50", lw=2, ms=5)
    ax.set_title("Global AUC-ROC"); ax.set_xlabel("Round"); ax.set_ylabel("AUC")
    ax.set_ylim(0, 1.05); ax.grid(True, alpha=0.3)

    # Trust scores per client
    ax = axes[1, 0]
    for cid, color in zip(client_ids, colors):
        vals  = [l["trust_scores"].get(str(cid), 1.0) for l in logs]
        style = "--" if cid in malicious else "-"
        label = f"Client {cid} ⚠" if cid in malicious else f"Client {cid}"
        ax.plot(rounds, vals, style, color=color, lw=2, ms=4, marker="o", label=label)
    ax.axhline(y=0.4, color="red", linestyle=":", alpha=0.7)
    ax.set_title("Per-Client Trust Score (τ)"); ax.set_xlabel("Round")
    ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    # Anomaly scores per client
    ax = axes[1, 1]
    for cid, color in zip(client_ids, colors):
        vals  = [l["anomaly_scores"].get(str(cid), 0.0) for l in logs]
        style = "--" if cid in malicious else "-"
        label = f"Client {cid} ⚠" if cid in malicious else f"Client {cid}"
        ax.plot(rounds, vals, style, color=color, lw=2, ms=4, marker="o", label=label)
    ax.axhline(y=0.6, color="red", linestyle=":", alpha=0.7)
    ax.set_title("Per-Client Anomaly Score (α)"); ax.set_xlabel("Round")
    ax.set_ylim(-0.05, 1.1); ax.legend(fontsize=7); ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "trust_training_curves.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Saved → {path}")


def plot_comparison(log1: str, log2: str, save_dir: str, attack_type: str) -> None:
    if not (os.path.exists(log1) and os.path.exists(log2)):
        return
    with open(log1) as f: bl = json.load(f)
    with open(log2) as f: tl = json.load(f)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot([l["round"] for l in bl],
            [l.get("global_f1", 0) for l in bl],
            "o--", color="#FF5252", lw=2, label="Standard FedAvg (vulnerable)")
    ax.plot([l["round"] for l in tl],
            [l.get("global_f1", 0) for l in tl],
            "s-", color="#2196F3", lw=2, label="Trust-Weighted FedAvg (defended)")
    ax.set_title(f"FedAvg vs Trust-Weighted — Attack: {attack_type}",
                 fontsize=12, fontweight="bold")
    ax.set_xlabel("Round"); ax.set_ylabel("Global F1")
    ax.set_ylim(0, 1.05); ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(save_dir, "comparison_fedavg_vs_trust.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[Plot] Comparison saved → {path}")


# =============================================================================
# ARGUMENT PARSER
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Split 2: Trust-Weighted Aggregation (Windows gRPC, no Ray)"
    )
    # Public flags
    p.add_argument("--data_path",         type=str,   default=None)
    p.add_argument("--synthetic",         action="store_true")
    p.add_argument("--num_clients",       type=int,   default=5)
    p.add_argument("--rounds",            type=int,   default=30)
    p.add_argument("--alpha",             type=float, default=1.0)
    p.add_argument("--model",             type=str,   default="dnn")
    p.add_argument("--attack",            type=str,   default="none",
                   choices=["none", "label_flip", "gradient_scale", "combined"])
    p.add_argument("--malicious_clients", type=int,   nargs="+", default=[])
    p.add_argument("--scale_factor",      type=float, default=5.0)
    p.add_argument("--anomaly_threshold", type=float, default=0.40)
    p.add_argument("--gamma",             type=float, default=0.85)
    p.add_argument("--min_trust_floor",    type=float, default=0.75,  help="Trust floor for innocent clients")
    p.add_argument("--compare",           action="store_true",
                   help="Also run FedAvg baseline for comparison")
    p.add_argument("--no_smote",          action="store_true")
    p.add_argument("--log_dir",           type=str,   default="logs_split2")

    # Hidden flags — used when re-invoking as a client subprocess
    p.add_argument("--_client_mode", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--_cid",      type=int,   default=-1,    help=argparse.SUPPRESS)
    p.add_argument("--_cache",    type=str,   default="",    help=argparse.SUPPRESS)
    p.add_argument("--_model",    type=str,   default="dnn", help=argparse.SUPPRESS)
    p.add_argument("--_smote",    type=str,   default="true",help=argparse.SUPPRESS)
    p.add_argument("--_attack",   type=str,   default="none",help=argparse.SUPPRESS)
    p.add_argument("--_malicious",type=str,   default="",    help=argparse.SUPPRESS)
    p.add_argument("--_scale",    type=float, default=5.0,   help=argparse.SUPPRESS)
    p.add_argument("--_atk_start",type=int,   default=1,     help=argparse.SUPPRESS)
    return p


# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    args = build_parser().parse_args()

    # ── CLIENT MODE ───────────────────────────────────────────────────────────
    # When a subprocess is spawned it calls this same file with --_client_mode
    if args._client_mode:
        malicious = [int(x) for x in args._malicious.split(",") if x.strip()]
        _run_as_client(
            cid        = args._cid,
            cache      = args._cache,
            model_type = args._model,
            use_smote  = (args._smote.lower() == "true"),
            attack_type = args._attack,
            malicious   = malicious,
            scale_factor = args._scale,
            attack_start_round = args._atk_start,
        )
        return

    # ── ORCHESTRATOR MODE ─────────────────────────────────────────────────────
    os.makedirs(args.log_dir, exist_ok=True)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  SPLIT 2: TRUST-WEIGHTED AGGREGATION ENGINE")
    print("  Blockchain-Based Dynamic Trust Modeling")
    print(f"  Model: {args.model.upper()} | Clients: {args.num_clients} | Rounds: {args.rounds}")
    print(f"  Attack: {args.attack} | Malicious: {args.malicious_clients}")
    print(f"  Engine: Flower gRPC — server in main thread, clients as subprocesses")
    print(f"{sep}\n")

    # ── Load data ─────────────────────────────────────────────────────────────
    if args.synthetic or not args.data_path:
        print("[Main] Using synthetic data.\n")
        X, y = make_synthetic_data()
    else:
        X, y = load_dataset(args.data_path)

    partitions = dirichlet_partition(
        X, y, num_clients=args.num_clients, alpha=args.alpha
    )

    use_smote = not args.no_smote

    # ── Optional: FedAvg baseline comparison ──────────────────────────────────
    if args.compare:
        print("\n--- Running STANDARD FedAvg baseline (for comparison) ---\n")
        baseline_log_dir = os.path.join(args.log_dir, "baseline_fedavg")
        os.makedirs(baseline_log_dir, exist_ok=True)

        baseline_strategy = get_fedavg_strategy(
            num_clients  = args.num_clients,
            log_dir      = baseline_log_dir,
        )
        _run_server_with_clients(
            strategy     = baseline_strategy,
            partitions   = partitions,
            num_clients  = args.num_clients,
            num_rounds   = args.rounds,
            log_dir      = baseline_log_dir,
            model_type   = args.model,
            use_smote    = use_smote,
            attack_type  = args.attack,          # same attacked clients for fair compare
            malicious    = args.malicious_clients,
            scale_factor = args.scale_factor,
            attack_start_round = 1,
            run_label    = "baseline",
        )
        baseline_strategy.print_summary()
        print("\n--- Baseline complete. Now running Trust-Weighted ---\n")

    # ── Trust-Weighted run ────────────────────────────────────────────────────
    trust_strategy = TrustWeightedFedAvg(
        num_clients       = args.num_clients,
        anomaly_threshold = args.anomaly_threshold,
        gamma             = args.gamma,
        min_trust_floor   = args.min_trust_floor,
        log_dir           = args.log_dir,
        attack_simulator  = AttackSimulator(  # server-side gradient scaling
            attack_type       = args.attack,
            malicious_clients = args.malicious_clients,
            scale_factor      = args.scale_factor,
        ),
    )

    _run_server_with_clients(
        strategy     = trust_strategy,
        partitions   = partitions,
        num_clients  = args.num_clients,
        num_rounds   = args.rounds,
        log_dir      = args.log_dir,
        model_type   = args.model,
        use_smote    = use_smote,
        attack_type  = args.attack,
        malicious    = args.malicious_clients,
        scale_factor = args.scale_factor,
        attack_start_round = 1,
        run_label    = "trust",
    )

    trust_strategy.print_summary()

    # ── Plot ──────────────────────────────────────────────────────────────────
    trust_log = os.path.join(args.log_dir, "trust_training_log.json")
    plot_trust_results(trust_log, args.log_dir, args.attack, args.malicious_clients)

    if args.compare:
        baseline_log = os.path.join(args.log_dir, "baseline_fedavg", "training_log.json")
        plot_comparison(baseline_log, trust_log, args.log_dir, args.attack)

    print(f"\n[Main] Split 2 complete.")
    print(f"  → Trust log : {trust_log}")
    print(f"  → Plots     : {args.log_dir}/")
    print(f"\n  Next: Split 3 — push model hashes + trust scores to Hyperledger Fabric.")


if __name__ == "__main__":
    multiprocessing.freeze_support()   # required for Windows subprocess spawning
    main()