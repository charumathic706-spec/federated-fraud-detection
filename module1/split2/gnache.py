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
#   --attack combined        -- BOTH attacks on the SAME malicious clients
#
# HOW TO RUN MULTIPLE ATTACK TYPES ON DIFFERENT CLIENTS:
#   argparse only keeps the LAST value for repeated flags, so:
#       --attack label_flip --malicious 1 --attack gradient_scale --malicious 3
#   is WRONG — argparse sees attack=gradient_scale, malicious=[3] only.
#
#   CORRECT approaches:
#     a) Combined attack on same clients:
#        python -m split2.main --attack combined --malicious 1 3
#        (both label_flip AND gradient_scale applied to clients 1 and 3)
#
#     b) Label-flip only:
#        python -m split2.main --attack label_flip --malicious 1
#
#     c) Gradient scale only on client 3:
#        python -m split2.main --attack gradient_scale --malicious 3 --scale_factor 10
#
#     d) Label-flip on client 1, gradient scale on client 3 (different attack per client):
#        python -m split2.main --attack label_flip --malicious 1 3 --gs_clients 3 --scale_factor 10
#        (use --gs_clients to specify which subset also gets gradient scaling)
#
# Usage examples:
#   python -m split2.main --synthetic
#   python -m split2.main --data_path ../data/creditcard.csv --attack label_flip --malicious 1
#   python -m split2.main --data_path ../data/creditcard.csv --attack combined --malicious 1 3 --scale_factor 10
#   python -m split2.main --data_path ../data/creditcard.csv --attack label_flip --malicious 1 --gs_clients 3 --scale_factor 10
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

from common.data_partition          import (
    load_dataset, dirichlet_partition, make_synthetic_data,
    save_partitions, load_partition,
)
from common.trust_weighted_strategy import get_trust_strategy
from common.attack_simulator        import AttackSimulator
from common.flower_client           import BankFederatedClient

# MODULE 2: import GovernanceEngine from split3
# Wrapped in try/except so Module 1 still works if split3 is missing
import sys as _sys, os as _os
_split3_candidates = [
    _os.path.join(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__))), "split3"),
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "..", "split3"),
]
for _p in _split3_candidates:
    if _os.path.isdir(_p) and _p not in _sys.path:
        _sys.path.insert(0, _p)
        break
try:
    from governance import GovernanceEngine, GovernanceConfig
    _GOVERNANCE_AVAILABLE = True
except ImportError:
    _GOVERNANCE_AVAILABLE = False

# ── SERVER ADDRESS ────────────────────────────────────────────────────────────
# LOCAL MODE  (single machine — original behaviour):
#   SERVER_ADDRESS = "127.0.0.1:8081"
#
# NETWORK MODE (multi-machine — server listens on all interfaces):
#   Set to "0.0.0.0:8081" so clients on other machines can connect.
#   On the client machine, point --_server to the server's LAN IP e.g. 192.168.1.5:8081
#
# Switch by commenting/uncommenting one of the two lines below:
# ── SERVER ADDRESS ────────────────────────────────────────────────────────────
# LOCAL MODE  (original — single machine, all clients on same PC):
#   Uncomment the line below and comment out NETWORK MODE line
#   Run command: python -m split2.main --data_path ../data/creditcard.csv --attack label_flip --malicious 1
#
# NETWORK MODE (multi-machine — server and clients on different PCs):
#   Uncomment the NETWORK MODE line and comment out LOCAL MODE line
#   On System 1 (server): python -m split2.main --data_path ../data/creditcard.csv --remote_clients 1 2
#   On System 2 (client): python run_client.py --cid 1 --server <System1_IP>:8081 --attack
#   Find System 1 IP by running: ipconfig  (look for IPv4 Address under WiFi)
#   Also open firewall on System 1: netsh advfirewall firewall add rule name="FL Server" dir=in action=allow protocol=TCP localport=8081

SERVER_ADDRESS = "127.0.0.1:8081"     # LOCAL  MODE — comment this out when using network mode
# SERVER_ADDRESS = "0.0.0.0:8081"     # NETWORK MODE — uncomment this when clients run on another machine


# =============================================================================
# MULTI-ATTACK WRAPPER
# =============================================================================

class MultiAttackSimulator:
    """
    Wraps two AttackSimulator instances to apply DIFFERENT attack types to
    DIFFERENT subsets of malicious clients in the same run.

    Example: label_flip on client 1, gradient_scale on client 3.

    If only one attack type is needed (--attack label_flip or combined),
    use the regular AttackSimulator directly — this wrapper is only activated
    when --gs_clients is specified alongside --attack label_flip/none.
    """

    def __init__(
        self,
        flip_clients:  list,
        scale_clients: list,
        scale_factor:  float = 5.0,
        attack_start_round: int = 1,
    ):
        all_malicious = list(set(flip_clients) | set(scale_clients))

        # Attacker A: label-flip on flip_clients
        self.flip_attacker = AttackSimulator(
            attack_type        = "label_flip" if flip_clients else "none",
            malicious_clients  = flip_clients,
            attack_start_round = attack_start_round,
        )

        # Attacker B: gradient-scale on scale_clients
        self.scale_attacker = AttackSimulator(
            attack_type        = "gradient_scale" if scale_clients else "none",
            malicious_clients  = scale_clients,
            scale_factor       = scale_factor,
            attack_start_round = attack_start_round,
        )

        print(
            f"[MultiAttack] label_flip on {flip_clients} | "
            f"gradient_scale on {scale_clients} (×{scale_factor})"
        )

    def set_round(self, round_num: int) -> None:
        self.flip_attacker.set_round(round_num)
        self.scale_attacker.set_round(round_num)

    def is_malicious(self, client_id: int) -> bool:
        return (self.flip_attacker.is_malicious(client_id) or
                self.scale_attacker.is_malicious(client_id))

    def poison_data(self, client_id, X, y):
        return self.flip_attacker.poison_data(client_id, X, y)

    def poison_params(self, client_id, params, global_params):
        return self.scale_attacker.poison_params(client_id, params, global_params)

    def get_attack_summary(self):
        return {
            "flip_clients":  self.flip_attacker.malicious_clients,
            "scale_clients": self.scale_attacker.malicious_clients,
            "scale_factor":  self.scale_attacker.scale_factor,
        }


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

    rounds  = [l["round"]                        for l in logs]
    f1s     = [l.get("global_f1", 0)             for l in logs]
    aucs    = [l.get("global_auc", 0)            for l in logs]
    recalls = [l.get("global_recall", 0)         for l in logs]
    flagged = [len(l.get("flagged_clients", [])) for l in logs]

    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    fig.suptitle("Split 2 — Trust-Weighted FL + Attack Detection", fontsize=13, fontweight="bold")

    for ax, (vals, title, color) in zip(axes[:3], [
        (f1s,     "Global F1",     "#2196F3"),
        (aucs,    "Global AUC",    "#4CAF50"),
        (recalls, "Global Recall", "#FF9800"),
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
                   server_address: str, is_label_flip: bool = False) -> None:
    """Runs inside a subprocess — load partition, connect to server via gRPC."""
    import flwr as fl

    partition = load_partition(cache, cid)

    client = BankFederatedClient(
        client_id     = cid,
        X_train       = partition["X_train"],
        y_train       = partition["y_train"],
        X_test        = partition["X_test"],
        y_test        = partition["y_test"],
        model_type    = model_type,
        use_smote     = use_smote,
        is_label_flip = is_label_flip,
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
    p = argparse.ArgumentParser(
        description="Split 2 — Trust-Weighted FL + Attack Simulation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
ATTACK EXAMPLES
  Single attack type, single malicious client:
    python -m split2.main --attack label_flip --malicious 1

  Single attack type, multiple malicious clients:
    python -m split2.main --attack gradient_scale --malicious 1 3 --scale_factor 8

  BOTH attacks on the SAME malicious clients (combined):
    python -m split2.main --attack combined --malicious 1 3 --scale_factor 10

  DIFFERENT attacks on DIFFERENT clients (label_flip on C1, gradient_scale on C3):
    python -m split2.main --attack label_flip --malicious 1 --gs_clients 3 --scale_factor 10

  NOTE: Running --attack twice is NOT supported by argparse (only last value kept).
        Use --attack combined OR --gs_clients for mixed attacks.
        """,
    )

    # Data
    p.add_argument("--data_path",    type=str,   default=None,
                   help="Path to CSV dataset (creditcard.csv or ieee_cis_merged.csv)")
    p.add_argument("--synthetic",    action="store_true",
                   help="Use built-in synthetic fraud data (no CSV needed)")
    p.add_argument("--max_samples",  type=int,   default=None,
                   help="Cap total dataset rows (speeds up testing)")

    # Federated learning
    p.add_argument("--num_clients",  type=int,   default=5)
    p.add_argument("--rounds",       type=int,   default=25)
    p.add_argument("--model",        type=str,   default="dnn", choices=["dnn", "logistic"])
    p.add_argument("--alpha",        type=float, default=1.0,
                   help="Dirichlet alpha for non-IID partitioning (lower=more non-IID)")
    p.add_argument("--fraction_fit", type=float, default=1.0)
    p.add_argument("--no_smote",     action="store_true")
    p.add_argument("--log_dir",      type=str,   default="logs_split2")
    p.add_argument("--port",         type=int,   default=8081)

    # Module 2 governance
    p.add_argument("--no_governance", action="store_true",
                   help="Disable Module 2 blockchain (Module 1 only mode)")

    # Attack configuration
    p.add_argument("--attack",       type=str,   default="none",
                   choices=["none", "label_flip", "gradient_scale", "combined"],
                   help=(
                       "Attack type to inject:\n"
                       "  none           = no attack (clean baseline)\n"
                       "  label_flip     = flip fraud labels during training\n"
                       "  gradient_scale = amplify gradient update by scale_factor\n"
                       "  combined       = BOTH label_flip AND gradient_scale on --malicious clients"
                   ))
    p.add_argument("--malicious",    type=int,   nargs="+", default=[1],
                   help="Client IDs to attack with --attack type (default: [1])")
    p.add_argument("--gs_clients",   type=int,   nargs="+", default=None,
                   help=(
                       "OPTIONAL: additional client IDs to apply gradient_scale on top of "
                       "--attack label_flip. Enables different attacks on different clients. "
                       "Example: --attack label_flip --malicious 1 --gs_clients 3 --scale_factor 10"
                   ))
    p.add_argument("--scale_factor", type=float, default=5.0,
                   help="Gradient amplification multiplier for gradient_scale attack (default: 5.0)")
    p.add_argument("--attack_start", type=int,   default=1,
                   help="Round number to begin attacking (default: 1 = from the start)")

    # Hidden subprocess flags (used when re-invoking self as a client process)
    p.add_argument("--_client_mode",  action="store_true",  help=argparse.SUPPRESS)
    p.add_argument("--_cid",    type=int, default=-1,       help=argparse.SUPPRESS)
    p.add_argument("--_cache",  type=str, default="",       help=argparse.SUPPRESS)
    p.add_argument("--_model",  type=str, default="dnn",    help=argparse.SUPPRESS)
    p.add_argument("--_smote",  type=str, default="true",   help=argparse.SUPPRESS)
    p.add_argument("--_server", type=str, default="127.0.0.1:8081", help=argparse.SUPPRESS)
    p.add_argument("--_label_flip", type=str, default="false",      help=argparse.SUPPRESS)

    # ── NETWORK MODE: remote clients argument ─────────────────────────────────
    # LOCAL MODE (original): not needed — all clients are spawned as local subprocesses
    #
    # NETWORK MODE: specify which client IDs will connect from another machine.
    # The server will skip spawning those clients locally and wait for them to connect.
    # Example: --remote_clients 1 2  → clients 1 and 2 connect from System 2
    #          clients 0, 3, 4 are still spawned as local subprocesses on System 1
    p.add_argument("--remote_clients", type=int, nargs="+", default=[],
                   help=(
                       "NETWORK MODE ONLY: client IDs that will connect from a remote machine. "
                       "These clients will NOT be spawned locally — they must be started manually "
                       "on the remote machine using run_client.py. "
                       "Example: --remote_clients 1 2"
                   ))
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
            is_label_flip  = (args._label_flip.lower() == "true"),
        )
        return

    # ── ORCHESTRATOR MODE ─────────────────────────────────────────────────────
    import flwr as fl

    # LOCAL MODE (original): server_address was hardcoded to 127.0.0.1
    # server_address = f"127.0.0.1:{args.port}"
    #
    # NETWORK MODE: server_address is derived from SERVER_ADDRESS constant at top of file.
    # Switching between modes only requires changing that one constant:
    #   SERVER_ADDRESS = "127.0.0.1:8081"  → local only (original behaviour, single machine)
    #   SERVER_ADDRESS = "0.0.0.0:8081"    → listens on all interfaces (multi-machine)
    _base = SERVER_ADDRESS.rsplit(":", 1)[0]   # "127.0.0.1" or "0.0.0.0"
    server_address = f"{_base}:{args.port}"
    os.makedirs(args.log_dir, exist_ok=True)

    sep = "=" * 65
    print(f"\n{sep}")
    print("  SPLIT 2 — TRUST-WEIGHTED FEDERATED LEARNING")
    print(f"  Model    : {args.model.upper()} | Clients: {args.num_clients} | Rounds: {args.rounds}")
    print(f"  Attack   : {args.attack.upper()} | Malicious: {args.malicious}")
    if args.gs_clients:
        print(f"  GS attack: gradient_scale also on clients {args.gs_clients} (×{args.scale_factor})")
    print(f"  Engine   : Flower gRPC | Server: {server_address}")
    print(f"{sep}\n")

    # ── Step 1: Build attack simulator ────────────────────────────────────────
    # Handle the case where the user wants DIFFERENT attacks on DIFFERENT clients
    # using --gs_clients (e.g. label_flip on C1, gradient_scale on C3)
    if args.gs_clients is not None and args.attack in ("label_flip", "none"):
        # Different attacks per client: use MultiAttackSimulator
        attacker = MultiAttackSimulator(
            flip_clients       = args.malicious if args.attack == "label_flip" else [],
            scale_clients      = args.gs_clients,
            scale_factor       = args.scale_factor,
            attack_start_round = args.attack_start,
        )
    else:
        # Standard: one attack type on all --malicious clients
        attacker = AttackSimulator(
            attack_type        = args.attack,
            malicious_clients  = args.malicious if args.attack != "none" else [],
            scale_factor       = args.scale_factor,
            attack_start_round = args.attack_start,
        )

    # ── Step 2: Load data ─────────────────────────────────────────────────────
    if args.synthetic or args.data_path is None:
        if not args.synthetic:
            print("[Main] No --data_path given — using synthetic data.\n")
        X, y = make_synthetic_data()
    else:
        # Resolve relative paths against the directory the user ran from (cwd),
        # not against __file__ — so ../data/creditcard.csv works from any location.
        data_path = os.path.abspath(args.data_path)
        X, y = load_dataset(data_path)

    # ── Step 3: Partition data across bank nodes ──────────────────────────────
    partitions = dirichlet_partition(
        X, y,
        num_clients = args.num_clients,
        alpha       = args.alpha,
        max_samples = args.max_samples,
    )

    # ── Step 4: Write shared partition cache for subprocess clients ───────────
    cache_path = os.path.join(args.log_dir, ".partition_cache_s2.npz")
    save_partitions(partitions, cache_path)

    # ── Step 5: Build GovernanceEngine (Module 2) ────────────────────────────
    _gov_engine = None
    if _GOVERNANCE_AVAILABLE and not getattr(args, "no_governance", False):
        _gov_cfg = GovernanceConfig(
            use_simulation          = False,  # real Ganache (set True to use sim)
            output_dir              = _os.path.join(args.log_dir, "governance_output"),
            anomaly_threshold       = 0.5,
            consecutive_flag_limit  = 3,
        )
        _gov_engine = GovernanceEngine(config=_gov_cfg)
    elif getattr(args, "no_governance", False):
        print("[Main] Module 2 disabled via --no_governance flag.")
    else:
        print("[Main] split3/ not found — running Module 1 only.")

    # ── Step 6: Build trust-weighted strategy with Module 2 hook ─────────────
    strategy = get_trust_strategy(
        num_clients       = args.num_clients,
        fraction_fit      = args.fraction_fit,
        log_dir           = args.log_dir,
        attack_simulator  = attacker,
        governance_engine = _gov_engine,
    )

    # ── Step 6: Spawn client subprocesses ─────────────────────────────────────
    smote_flag = "false" if args.no_smote else "true"

    # Determine which clients should flip labels locally inside their fit() call
    if args.gs_clients is not None and args.attack in ("label_flip", "none"):
        label_flip_set = set(args.malicious) if args.attack == "label_flip" else set()
    elif args.attack in ("label_flip", "combined"):
        label_flip_set = set(args.malicious)
    else:
        label_flip_set = set()

    # ── NETWORK MODE: identify which clients connect from a remote machine ────
    # LOCAL MODE (original): remote_cids is always empty — all clients spawned locally
    # NETWORK MODE: clients listed in --remote_clients are skipped here;
    #               they must be started manually on System 2 using run_client.py
    remote_cids = set(args.remote_clients)   # empty set in local mode (original behaviour)

    client_procs = []

    # LOCAL MODE (original spawn loop — no remote_clients check):
    # print(f"[Main] Spawning {args.num_clients} client subprocesses...")
    # if label_flip_set:
    #     print(f"  Label-flip clients (local): {sorted(label_flip_set)}")
    # for cid in range(args.num_clients):
    #     flip_flag = "true" if cid in label_flip_set else "false"
    #     cmd = [
    #         sys.executable, os.path.abspath(__file__),
    #         "--_client_mode",
    #         "--_cid",        str(cid),
    #         "--_cache",      os.path.abspath(cache_path),
    #         "--_model",      args.model,
    #         "--_smote",      smote_flag,
    #         "--_server",     server_address,
    #         "--_label_flip", flip_flag,
    #     ]
    #     proc = subprocess.Popen(
    #         cmd,
    #         stdout=subprocess.PIPE,
    #         stderr=subprocess.STDOUT,
    #         text=True,
    #     )
    #     client_procs.append((cid, proc))
    #     print(f"  Spawned Bank {cid:02d}  PID={proc.pid}  label_flip={flip_flag}")

    # NETWORK MODE compatible spawn loop:
    # - In LOCAL MODE: remote_cids is empty so all clients are spawned as before
    # - In NETWORK MODE: clients in remote_cids are skipped (they connect from System 2)
    print(f"[Main] Spawning client subprocesses...")
    if label_flip_set:
        print(f"  Label-flip clients (local): {sorted(label_flip_set)}")
    if remote_cids:
        print(f"  Remote clients (connect from System 2): {sorted(remote_cids)}")
        print(f"  On System 2, run for each remote client:")
        for cid in sorted(remote_cids):
            flip_flag = "true" if cid in label_flip_set else "false"
            attack_flag = "--attack" if flip_flag == "true" else ""
            print(f"    python run_client.py --cid {cid} --server <System1_IP>:{args.port} "
                  f"--data_path <path_to_csv> {attack_flag}")

    for cid in range(args.num_clients):
        # NETWORK MODE: skip clients that will connect from a remote machine
        if cid in remote_cids:
            print(f"  Skipping Bank {cid:02d} — will connect from remote machine (System 2)")
            continue

        # LOCAL MODE (original): spawn every client as a subprocess (same as before)
        flip_flag = "true" if cid in label_flip_set else "false"
        cmd = [
            sys.executable, os.path.abspath(__file__),
            "--_client_mode",
            "--_cid",        str(cid),
            "--_cache",      os.path.abspath(cache_path),
            "--_model",      args.model,
            "--_smote",      smote_flag,
            "--_server",     server_address,
            "--_label_flip", flip_flag,
        ]
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
        client_procs.append((cid, proc))
        print(f"  Spawned Bank {cid:02d}  PID={proc.pid}  label_flip={flip_flag}")

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

    # ── Step 7: Start Flower gRPC server in MAIN THREAD ───────────────────────
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

    # Cleanup partition cache
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
    print(f"\n  Next: python -m split3.main --trust_log {log_path}")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
