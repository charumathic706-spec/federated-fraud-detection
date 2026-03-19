# =============================================================================
# FILE: run_client.py
# PURPOSE: Standalone client runner for NETWORK MODE (multi-machine setup).
#
#          In LOCAL MODE you do NOT need this file at all.
#          split2/main.py spawns all clients automatically as subprocesses.
#
#          In NETWORK MODE this file runs on System 2 (client machine) and
#          connects to the Flower server running on System 1 (server machine).
#
# =============================================================================
#
# HOW TO SWITCH BETWEEN MODES
# ─────────────────────────────────────────────────────────────────────────────
#
# LOCAL MODE  (single machine — original behaviour, no changes needed):
#   Just run split2/main.py as normal. This file is not used at all.
#   Command:
#       python -m split2.main --data_path ../data/creditcard.csv \
#           --attack label_flip --malicious 1
#
# NETWORK MODE (two machines — System 1 = server, System 2 = client):
#   Step 1 — On System 1, open split2/main.py and change:
#       SERVER_ADDRESS = "127.0.0.1:8081"    <- comment this line out
#       # SERVER_ADDRESS = "0.0.0.0:8081"   <- uncomment this line
#
#   Step 2 — On System 1, find System 1's IP address:
#       Open Command Prompt -> type: ipconfig
#       Look for IPv4 Address under WiFi adapter (e.g. 192.168.1.5)
#
#   Step 3 — On System 1, open Windows Firewall port 8081:
#       netsh advfirewall firewall add rule name="FL Server" dir=in
#           action=allow protocol=TCP localport=8081
#
#   Step 4 — On System 1, start the server:
#       python -m split2.main --data_path ../data/creditcard.csv \
#           --attack none --remote_clients 1 2
#       (spawns clients 0,3,4 locally and waits for clients 1 and 2 from System 2)
#
#   Step 5 — On System 2, open a terminal for each remote client:
#       Benign client:
#           python run_client.py --cid 2 --server 192.168.1.5:8081 \
#               --data_path ../data/creditcard.csv
#       Malicious client (label-flip attack):
#           python run_client.py --cid 1 --server 192.168.1.5:8081 \
#               --data_path ../data/creditcard.csv --attack
#
# IMPORTANT: Both machines must have the same creditcard.csv dataset file.
#            Both must use the same --alpha and --num_clients values.
#
# =============================================================================

import argparse
import os
import sys
import time

# Path setup — add module1/ root so all common.* imports work
_ROOT = os.path.dirname(os.path.abspath(__file__))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from common.flower_client  import BankFederatedClient
from common.data_partition import load_dataset, dirichlet_partition, load_partition
import flwr as fl


def build_parser():
    p = argparse.ArgumentParser(
        description="Run a single federated bank client on System 2 (NETWORK MODE only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
EXAMPLES
  Benign client:
    python run_client.py --cid 2 --server 192.168.1.5:8081 --data_path ../data/creditcard.csv

  Malicious client (label-flip attack):
    python run_client.py --cid 1 --server 192.168.1.5:8081 --data_path ../data/creditcard.csv --attack

  Using shared cache from System 1:
    python run_client.py --cid 1 --server 192.168.1.5:8081 --cache logs_split2/.partition_cache_s2.npz
        """
    )
    p.add_argument("--cid",         type=int,   required=True,
                   help="Client ID — must match one of --remote_clients on System 1 (0-4)")
    p.add_argument("--server",      type=str,   required=True,
                   help="Server address  e.g.  192.168.1.5:8081")

    # Two ways to load data:
    # Way 1: Both machines have the CSV — re-run Dirichlet with same seed
    # Way 2: Copy the .npz cache from System 1 — guarantees identical partitions
    p.add_argument("--data_path",   type=str,   default=None,
                   help="Path to creditcard.csv on THIS machine (Way 1)")
    p.add_argument("--cache",       type=str,   default=None,
                   help="Path to .npz cache exported by System 1 (Way 2 — takes priority)")

    p.add_argument("--num_clients", type=int,   default=5,
                   help="Must match System 1 value (default: 5)")
    p.add_argument("--alpha",       type=float, default=1.0,
                   help="Dirichlet alpha — must match System 1 (default: 1.0)")
    p.add_argument("--model",       type=str,   default="dnn", choices=["dnn", "logistic"])
    p.add_argument("--no_smote",    action="store_true")
    p.add_argument("--attack",      action="store_true",
                   help="Enable label-flip attack on this client")
    p.add_argument("--retry",       type=int,   default=60,
                   help="Seconds to retry connecting (default: 60)")
    return p


def main():
    args = build_parser().parse_args()

    print(f"\n{'='*58}")
    print(f"  run_client.py  --  NETWORK MODE")
    print(f"  Bank ID    : {args.cid}")
    print(f"  Server     : {args.server}")
    print(f"  Attack     : {'LABEL-FLIP ENABLED' if args.attack else 'None (benign)'}")
    print(f"{'='*58}\n")

    # ── Step 1: Load partition ─────────────────────────────────────────────────
    # ORIGINAL (split2/main.py _run_as_client read from local .npz cache):
    # partition = load_partition(cache, cid)
    # X_train, y_train, X_test, y_test = partition values
    #
    # NETWORK MODE: two options below — cache takes priority over data_path

    if args.cache and os.path.exists(args.cache):
        # Way 2: shared cache copied from System 1 (most reliable)
        print(f"[Client {args.cid}] Loading partition from cache: {args.cache}")
        partition = load_partition(args.cache, args.cid)
        X_train   = partition["X_train"]
        y_train   = partition["y_train"]
        X_test    = partition["X_test"]
        y_test    = partition["y_test"]

    elif args.data_path:
        # Way 1: re-partition from raw CSV on this machine
        # random seed = 42 (fixed in data_partition.py) guarantees same split as System 1
        data_path = os.path.abspath(args.data_path)
        print(f"[Client {args.cid}] Loading dataset: {data_path}")
        print(f"[Client {args.cid}] Partitioning: alpha={args.alpha}, "
              f"num_clients={args.num_clients} ...")
        X, y = load_dataset(data_path)
        partitions = dirichlet_partition(X, y,
                                         num_clients=args.num_clients,
                                         alpha=args.alpha)
        p       = partitions[args.cid]
        X_train = p["X_train"]
        y_train = p["y_train"]
        X_test  = p["X_test"]
        y_test  = p["y_test"]

    else:
        print("[ERROR] Provide --data_path or --cache. See --help.")
        sys.exit(1)

    print(f"[Client {args.cid}] Partition ready: "
          f"train={len(X_train):,}  test={len(X_test):,}  "
          f"fraud_train={int(y_train.sum())}")

    # ── Step 2: Build client ───────────────────────────────────────────────────
    # ORIGINAL (split2/main.py _run_as_client):
    # client = BankFederatedClient(
    #     client_id  = cid,
    #     X_train    = partition["X_train"],  y_train = partition["y_train"],
    #     X_test     = partition["X_test"],   y_test  = partition["y_test"],
    #     model_type = model_type,
    #     use_smote  = use_smote,
    # )
    #
    # NETWORK MODE: identical call — is_label_flip comes from --attack flag
    # instead of the hidden --_label_flip subprocess arg
    client = BankFederatedClient(
        client_id     = args.cid,
        X_train       = X_train,
        y_train       = y_train,
        X_test        = X_test,
        y_test        = y_test,
        model_type    = args.model,
        use_smote     = not args.no_smote,
        is_label_flip = args.attack,
    )

    # ── Step 3: Connect to server ──────────────────────────────────────────────
    # ORIGINAL (split2/main.py _run_as_client — 30 retries):
    # for attempt in range(30):
    #     try:
    #         fl.client.start_client(server_address=server_address,
    #                                client=client.to_client())
    #         break
    #     except Exception as exc:
    #         if attempt < 29:
    #             time.sleep(1.0)
    #         else:
    #             print(f"Failed after 30s: {exc}")
    #             sys.exit(1)
    #
    # NETWORK MODE: same logic, retry raised to 60s for slower network startup
    print(f"\n[Client {args.cid}] Connecting to {args.server} "
          f"(will retry up to {args.retry}s) ...\n")

    for attempt in range(args.retry):
        try:
            fl.client.start_client(
                server_address = args.server,
                client         = client.to_client(),
            )
            print(f"\n[Client {args.cid}] All rounds complete. Disconnected cleanly.")
            break
        except Exception as exc:
            if attempt < args.retry - 1:
                print(f"  [Client {args.cid}] Attempt {attempt+1}/{args.retry} failed "
                      f"— retrying... ({exc})")
                time.sleep(1.0)
            else:
                print(f"\n[Client {args.cid}] Could not connect after {args.retry}s.")
                print(f"  Last error : {exc}")
                print(f"\n  Checklist:")
                print(f"    1. Is split2/main.py running on System 1?")
                print(f"    2. Is the IP correct?  ({args.server})")
                print(f"    3. Is port 8081 open on System 1 firewall?")
                print(f"       Run on System 1: netsh advfirewall firewall add rule "
                      f"name=\"FL Server\" dir=in action=allow protocol=TCP localport=8081")
                print(f"    4. Are both machines on the same WiFi/LAN?")
                sys.exit(1)


if __name__ == "__main__":
    main()
