"""
main.py
-------
Split 3 — Blockchain Governance Layer
CLI entry point.

Usage:
  # Run with simulated blockchain (default, Colab-safe):
  python main.py --trust_log ../split2/trust_training_log.json

  # Run with real Hyperledger Fabric:
  python main.py --trust_log ../split2/trust_training_log.json --fabric

  # Include tamper simulation (security demo):
  python main.py --trust_log ../split2/trust_training_log.json --tamper_round 5

  # Custom output directory:
  python main.py --trust_log ./logs/trust_training_log.json --output_dir ./output

  # Demo mode (generates synthetic trust log, no Split 2 required):
  python main.py --demo
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Dict, List

import numpy as np

from governance import GovernanceConfig, GovernanceEngine
from model_hasher import ModelHasher, verify_hash_chain_from_log


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Demo trust log generator
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def generate_demo_trust_log(
    num_rounds:      int = 10,
    num_clients:     int = 5,
    malicious_client: int = 2,
    output_path:     str = "./demo_trust_log.json",
) -> str:
    """
    Generate a synthetic trust_training_log.json for demo / testing.
    Simulates a realistic FL run with one misbehaving client.
    """
    import hashlib
    rng = np.random.default_rng(42)
    log = []

    for rnd in range(1, num_rounds + 1):
        # Simulate improving F1 over rounds
        base_f1 = min(0.65 + rnd * 0.025 + rng.normal(0, 0.005), 0.92)

        client_metrics = []
        trust_scores   = {}
        anomaly_scores = {}
        flagged        = []
        trusted        = []

        for cid in range(num_clients):
            if cid == malicious_client and rnd > 4:
                # Client starts misbehaving after round 4
                anomaly  = float(rng.uniform(0.6, 0.95))
                trust    = max(0.05, 1.0 - anomaly - rng.uniform(0, 0.1))
                flagged.append(cid)
            else:
                anomaly  = float(rng.uniform(0.0, 0.15))
                trust    = min(1.0, 0.85 + rng.uniform(0, 0.15))
                trusted.append(cid)

            trust_scores[str(cid)]   = round(trust, 4)
            anomaly_scores[str(cid)] = round(anomaly, 4)

        # Produce a deterministic fake model hash
        model_seed = f"round_{rnd}_f1_{base_f1:.4f}"
        model_hash = hashlib.sha256(model_seed.encode()).hexdigest()

        log.append({
            "round":            rnd,
            "timestamp":        time.time(),
            "model_hash":       model_hash,
            "trusted_clients":  trusted,
            "flagged_clients":  flagged,
            "trust_scores":     trust_scores,
            "anomaly_scores":   anomaly_scores,
            "cos_similarities": {str(c): round(float(rng.uniform(0.7, 1.0)), 4)
                                 for c in range(num_clients)},
            "euc_distances":    {str(c): round(float(rng.uniform(0.0, 0.5)), 4)
                                 for c in range(num_clients)},
            "global_f1":        round(base_f1, 6),
            "global_auc":       round(min(base_f1 + 0.10, 0.99), 6),
        })

    with open(output_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"[Demo] Synthetic trust log written: {output_path}")
    print(f"       {num_rounds} rounds, {num_clients} clients, "
          f"client {malicious_client} goes malicious after round 4")
    return output_path


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Main
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Split 3 — Blockchain Governance Layer for Federated Fraud Detection"
    )
    p.add_argument(
        "--trust_log", type=str,
        default=None,
        help="Path to trust_training_log.json from Split 2",
    )
    p.add_argument(
        "--output_dir", type=str,
        default="./governance_output",
        help="Directory for governance reports and hash chain",
    )
    p.add_argument(
        "--fabric", action="store_true",
        help="Use real Hyperledger Fabric instead of simulation",
    )
    p.add_argument(
        "--tamper_round", type=int, default=None,
        help="Inject and detect a simulated tamper at this round number",
    )
    p.add_argument(
        "--demo", action="store_true",
        help="Generate a synthetic trust log and run in demo mode",
    )
    p.add_argument(
        "--demo_rounds", type=int, default=10,
        help="Number of rounds for demo mode (default: 10)",
    )
    p.add_argument(
        "--anomaly_threshold", type=float, default=0.5,
        help="Anomaly score threshold for client flagging (default: 0.5)",
    )
    p.add_argument(
        "--quarantine_after", type=int, default=3,
        help="Auto-quarantine after N consecutive flagged rounds (default: 3)",
    )
    p.add_argument(
        "--verify_every", type=int, default=5,
        help="Re-verify full hash chain every N rounds (default: 5)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print("\n" + "="*65)
    print("  SPLIT 3 — BLOCKCHAIN GOVERNANCE LAYER")
    print("  Federated Fraud Detection Framework")
    print("="*65)

    # ── Demo mode ─────────────────────────────────────────────────────────────
    if args.demo:
        demo_log_path = "./demo_trust_log.json"
        trust_log_path = generate_demo_trust_log(
            num_rounds=args.demo_rounds,
            output_path=demo_log_path,
        )
    elif args.trust_log is None:
        print("\n[Error] --trust_log is required unless --demo is used.")
        print("        Run: python main.py --demo")
        sys.exit(1)
    else:
        trust_log_path = args.trust_log

    if not os.path.exists(trust_log_path):
        print(f"\n[Error] Trust log not found: {trust_log_path}")
        sys.exit(1)

    # ── Configure governance engine ───────────────────────────────────────────
    config = GovernanceConfig(
        anomaly_threshold=args.anomaly_threshold,
        consecutive_flag_limit=args.quarantine_after,
        verify_chain_every_n=args.verify_every,
        use_simulation=not args.fabric,
        output_dir=args.output_dir,
    )

    # ── Run governance ────────────────────────────────────────────────────────
    engine = GovernanceEngine(config)

    t0 = time.time()
    report = engine.process_trust_log(trust_log_path)
    elapsed = time.time() - t0

    # ── Tamper simulation (optional) ──────────────────────────────────────────
    if args.tamper_round is not None:
        tamper_result = engine.run_tamper_simulation(
            round_to_tamper=args.tamper_round
        )
        print(f"\n  Tamper simulation result: {tamper_result['detection_message']}")

    # ── Final output ──────────────────────────────────────────────────────────
    print(f"\n[Main] Processing time: {elapsed:.2f}s")
    print(f"\n[Main] ✅ Split 3 complete.")
    print(f"  → Governance report: "
          f"{os.path.join(args.output_dir, config.report_filename)}")
    print(f"  → Hash chain:        "
          f"{os.path.join(args.output_dir, config.hash_chain_filename)}")
    print(f"\n  Summary:")
    print(f"    Rounds processed:    {report.total_rounds}")
    print(f"    Hash chain intact:   {'✅ YES' if report.chain_intact else '🚨 BROKEN'}")
    print(f"    Tamper events:       {report.tamper_events}")
    print(f"    Quarantined clients: {report.quarantined_clients}")
    print(f"    Best Global F1:      {report.best_f1:.4f} @ Round {report.best_round}")


if __name__ == "__main__":
    main()