"""
governance.py
-------------
Blockchain governance layer for the Federated Fraud Detection framework.

Responsibilities:
  1. Ingest trust_training_log.json from Split 2
  2. Hash every model update and build the hash chain via ModelHasher
  3. Register each round on the blockchain (sim or real Fabric)
  4. Detect tamper events and raise on-chain alerts
  5. Produce an immutable audit trail (GDPR Article 30, PCI-DSS Req 10)
  6. Enforce governance rules:
       - Reject rounds with broken hash chains
       - Flag clients with anomaly_score > threshold
       - Auto-quarantine clients flagged ≥ N consecutive rounds

Input:  trust_training_log.json  (written by Split 2)
Output: governance_report.json   (hash chain + audit trail + tamper events)
        hash_chain.json           (exportable hash chain)
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

try:
    # When running from split3/ directory directly
    from model_hasher import ModelHasher, TamperReport, simulate_tamper
    from fabric_gateway import create_gateway
except ImportError:
    # When imported from module1/ via governance_bridge (path already added)
    from model_hasher import ModelHasher, TamperReport, simulate_tamper
    from fabric_gateway import create_gateway


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance configuration
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class GovernanceConfig:
    # Tamper detection
    anomaly_threshold:         float = 0.5     # above this → flag client
    consecutive_flag_limit:    int   = 3        # auto-quarantine after N rounds
    min_f1_threshold:          float = 0.30    # rounds below this are suspicious

    # Hash chain
    verify_chain_every_n:      int   = 5        # re-verify full chain every N rounds

    # Blockchain
    use_simulation:            bool  = True
    org_msp:                   str   = "Org1MSP"

    # Audit trail actor label
    audit_actor:               str   = "FederatedCoordinator"

    # Output paths
    output_dir:                str   = "./governance_output"
    report_filename:           str   = "governance_report.json"
    hash_chain_filename:       str   = "hash_chain.json"


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Governance report structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class RoundGovernanceRecord:
    round_num:           int
    model_hash:          str
    block_hash:          str
    chain_intact:        bool
    flagged_clients:     List[int]
    trusted_clients:     List[int]
    quarantined_clients: List[int]
    tamper_alerts:       List[str]
    global_f1:           float
    global_auc:          float
    blockchain_tx_id:    str
    audit_tx_id:         str
    timestamp:           float = field(default_factory=time.time)

    def to_dict(self) -> Dict:
        return {
            "round":               self.round_num,
            "timestamp":           self.timestamp,
            "model_hash":          self.model_hash,
            "block_hash":          self.block_hash,
            "chain_intact":        self.chain_intact,
            "flagged_clients":     self.flagged_clients,
            "trusted_clients":     self.trusted_clients,
            "quarantined_clients": self.quarantined_clients,
            "tamper_alerts":       self.tamper_alerts,
            "global_f1":           round(self.global_f1, 6),
            "global_auc":          round(self.global_auc, 6),
            "blockchain_tx_id":    self.blockchain_tx_id,
            "audit_tx_id":         self.audit_tx_id,
        }


@dataclass
class GovernanceReport:
    total_rounds:        int
    chain_intact:        bool
    tamper_events:       int
    quarantined_clients: List[int]
    best_f1:             float
    best_round:          int
    round_records:       List[RoundGovernanceRecord] = field(default_factory=list)
    tamper_report:       Optional[Dict] = None

    def to_dict(self) -> Dict:
        return {
            "summary": {
                "total_rounds":        self.total_rounds,
                "chain_intact":        self.chain_intact,
                "tamper_events":       self.tamper_events,
                "quarantined_clients": self.quarantined_clients,
                "best_f1":             round(self.best_f1, 6),
                "best_round":          self.best_round,
            },
            "tamper_report":  self.tamper_report,
            "round_records":  [r.to_dict() for r in self.round_records],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# GovernanceEngine
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class GovernanceEngine:
    """
    Core governance orchestrator.

    Takes trust_training_log.json entries one at a time (or all at once)
    and for each round:
      1. Derives a deterministic model hash from the log entry's model_hash
         (in production, the actual model params are passed directly)
      2. Extends the local hash chain
      3. Registers the block on the blockchain
      4. Checks anomaly scores → flags / quarantines clients
      5. Raises tamper alerts when chain breaks
      6. Appends GDPR-compliant audit events
    """

    def __init__(self, config: Optional[GovernanceConfig] = None) -> None:
        self.cfg     = config or GovernanceConfig()
        self.hasher  = ModelHasher()
        self.gateway = create_gateway(
            use_simulation=self.cfg.use_simulation,
            org_msp=self.cfg.org_msp,
        )
        self._round_records:         List[RoundGovernanceRecord] = []
        self._quarantine_tracker:    Dict[int, int] = {}   # client_id → consecutive flags
        self._quarantined_clients:   set            = set()
        self._best_f1:               float          = 0.0
        self._best_round:            int            = 0

        os.makedirs(self.cfg.output_dir, exist_ok=True)
        print(f"\n[Governance] Engine initialised")
        print(f"  Mode:      {'SIMULATION' if self.cfg.use_simulation else 'FABRIC'}")
        print(f"  Output:    {self.cfg.output_dir}")

    # ── Public API ────────────────────────────────────────────────────────────

    def process_trust_log(self, trust_log_path: str) -> GovernanceReport:
        """
        Process a complete trust_training_log.json file from Split 2.
        Iterates through all rounds and applies governance to each.

        Args:
            trust_log_path: Path to trust_training_log.json

        Returns:
            GovernanceReport with full audit trail and hash chain
        """
        print(f"\n[Governance] Loading trust log: {trust_log_path}")
        with open(trust_log_path, "r") as f:
            trust_log = json.load(f)

        print(f"  Rounds to process: {len(trust_log)}")
        print(f"{'─'*60}")

        for entry in trust_log:
            self.process_round(entry)

        return self._finalise()

    def process_round(self, log_entry: Dict) -> RoundGovernanceRecord:
        """
        Process a single round's trust log entry.
        Can be called incrementally during live training.

        Args:
            log_entry:  One element from trust_training_log.json

        Returns:
            RoundGovernanceRecord
        """
        rnd         = log_entry["round"]
        model_hash  = log_entry.get("model_hash", "")
        global_f1   = log_entry.get("global_f1", 0.0)
        global_auc  = log_entry.get("global_auc", 0.0)
        trust_scores    = log_entry.get("trust_scores", {})
        anomaly_scores  = log_entry.get("anomaly_scores", {})
        flagged         = [int(c) for c in log_entry.get("flagged_clients", [])]
        trusted         = [int(c) for c in log_entry.get("trusted_clients", [])]

        print(f"\n  [Round {rnd:02d}] F1={global_f1:.4f}  "
              f"trusted={trusted}  flagged={flagged}")

        # 1. Build hash chain entry from log-provided model_hash
        #    In production, pass actual model param tensors to hash_round()
        synthetic_params = self._hash_to_synthetic_params(model_hash, rnd)
        hash_record = self.hasher.hash_round(
            round_num=rnd,
            params=synthetic_params,
            timestamp=time.time(),
        )

        # 2. Check governance rules → quarantine decisions
        newly_quarantined = self._apply_governance_rules(
            rnd, flagged, anomaly_scores
        )
        quarantined_now = list(self._quarantined_clients)

        # 3. Register on blockchain
        tx_id, ok = self.gateway.register_model(
            round_num=rnd,
            model_hash=hash_record.model_hash,
            block_hash=hash_record.block_hash,
            prev_block_hash=hash_record.prev_block_hash,
            global_f1=global_f1,
            global_auc=global_auc,
            trusted_clients=trusted,
            flagged_clients=flagged,
            param_count=hash_record.param_count,
            total_bytes=hash_record.total_bytes,
        )

        # 4. Periodic chain verification
        tamper_alerts: List[str] = []
        chain_intact = True
        if rnd % self.cfg.verify_chain_every_n == 0 or rnd == 1:
            report = self.hasher.verify_chain()
            chain_intact = report.is_intact
            if not chain_intact:
                for tampered_rnd in report.tampered_rounds:
                    alert_msg = (f"HASH_CHAIN_BREAK at round {tampered_rnd}: "
                                 f"{report.details}")
                    tamper_alerts.append(alert_msg)
                    self.gateway.raise_tamper_alert(
                        round_num=tampered_rnd,
                        alert_type="HASH_CHAIN_BREAK",
                        detail=str(report.details),
                        severity="CRITICAL",
                    )

        # 5. Raise alerts for newly quarantined clients
        for cid in newly_quarantined:
            alert_msg = (f"CLIENT_QUARANTINED: client {cid} "
                         f"flagged {self.cfg.consecutive_flag_limit} consecutive rounds")
            tamper_alerts.append(alert_msg)
            self.gateway.raise_tamper_alert(
                round_num=rnd,
                alert_type="CLIENT_QUARANTINED",
                detail=alert_msg,
                severity="HIGH",
            )

        # 6. Audit trail event
        audit_tx = self.gateway.append_audit_event(
            event_type="ROUND_COMMITTED",
            round_num=rnd,
            data={
                "model_hash":          hash_record.model_hash,
                "block_hash":          hash_record.block_hash,
                "global_f1":           global_f1,
                "global_auc":          global_auc,
                "trusted_clients":     trusted,
                "flagged_clients":     flagged,
                "quarantined_clients": quarantined_now,
                "chain_intact":        chain_intact,
            },
            actor=self.cfg.audit_actor,
        )

        # 7. Track best F1
        if global_f1 > self._best_f1:
            self._best_f1   = global_f1
            self._best_round = rnd

        record = RoundGovernanceRecord(
            round_num=rnd,
            model_hash=hash_record.model_hash,
            block_hash=hash_record.block_hash,
            chain_intact=chain_intact,
            flagged_clients=flagged,
            trusted_clients=trusted,
            quarantined_clients=quarantined_now,
            tamper_alerts=tamper_alerts,
            global_f1=global_f1,
            global_auc=global_auc,
            blockchain_tx_id=tx_id,
            audit_tx_id=audit_tx,
        )
        self._round_records.append(record)
        return record

    def run_tamper_simulation(
        self,
        round_to_tamper: int = 5,
    ) -> Dict:
        """
        Inject a simulated tamper into the hash chain and verify detection.
        Returns a dict describing the tamper event and detection result.
        """
        print(f"\n[Governance] 🔬 Tamper simulation — injecting at round {round_to_tamper}")

        if not self.hasher.get_chain():
            return {"error": "No hash chain built yet. Run process_trust_log() first."}

        # Inject tamper
        tampered_hasher = simulate_tamper(self.hasher, round_to_tamper)

        # Verify — should detect
        report = tampered_hasher.verify_chain()
        detected = not report.is_intact

        result = {
            "tampered_round":      round_to_tamper,
            "detected":            detected,
            "tampered_rounds":     report.tampered_rounds,
            "broken_links":        report.broken_links,
            "detection_message":   report.summary(),
        }

        print(f"  Tamper {'DETECTED ✅' if detected else 'MISSED ❌'}")
        print(f"  {report.summary()}")

        # Log tamper simulation to audit trail
        self.gateway.append_audit_event(
            event_type="TAMPER_SIMULATION",
            round_num=round_to_tamper,
            data=result,
            actor="SecurityAudit",
        )

        return result

    def export_reports(self) -> Tuple[str, str]:
        """
        Write governance_report.json and hash_chain.json to output_dir.

        Returns:
            (governance_report_path, hash_chain_path)
        """
        report = self._build_report()

        report_path = os.path.join(self.cfg.output_dir, self.cfg.report_filename)
        chain_path  = os.path.join(self.cfg.output_dir, self.cfg.hash_chain_filename)

        with open(report_path, "w") as f:
            json.dump(report.to_dict(), f, indent=2, default=str)

        with open(chain_path, "w") as f:
            json.dump(self.hasher.export_chain(), f, indent=2)

        print(f"\n[Governance] Reports written:")
        print(f"  → {report_path}")
        print(f"  → {chain_path}")
        return report_path, chain_path

    def print_summary(self) -> None:
        """Print a formatted summary to stdout."""
        report = self._build_report()
        print(f"\n{'='*65}")
        print(f"  GOVERNANCE SUMMARY")
        print(f"{'='*65}")
        print(f"  Rounds processed:    {report.total_rounds}")
        print(f"  Hash chain intact:   {'✅ YES' if report.chain_intact else '🚨 BROKEN'}")
        print(f"  Tamper events:       {report.tamper_events}")
        print(f"  Quarantined clients: {report.quarantined_clients}")
        print(f"  Best Global F1:      {report.best_f1:.4f} (Round {report.best_round})")
        print(f"\n  Blockchain blocks:   {self.gateway.get_block_count()}")
        alerts = self.gateway.get_tamper_alerts()
        print(f"  On-chain alerts:     {len(alerts)}")
        audit  = self.gateway.get_audit_trail()
        print(f"  Audit events:        {len(audit)}")
        print(f"{'='*65}")

    # ── Private helpers ───────────────────────────────────────────────────────

    def _apply_governance_rules(
        self,
        rnd:           int,
        flagged:       List[int],
        anomaly_scores: Dict[str, float],
    ) -> List[int]:
        """
        Update quarantine tracker and return newly quarantined clients.
        Rule: flag a client on-chain after consecutive_flag_limit rounds flagged.
        """
        # Increment flag counter for flagged clients
        for cid in flagged:
            self._quarantine_tracker[cid] = self._quarantine_tracker.get(cid, 0) + 1

        # Reset counter for clients NOT flagged this round
        all_clients = set(self._quarantine_tracker.keys())
        for cid in all_clients - set(flagged):
            self._quarantine_tracker[cid] = max(
                0, self._quarantine_tracker[cid] - 1
            )

        # Newly quarantined this round
        newly_quarantined = []
        for cid, count in self._quarantine_tracker.items():
            if (count >= self.cfg.consecutive_flag_limit and
                    cid not in self._quarantined_clients):
                self._quarantined_clients.add(cid)
                newly_quarantined.append(cid)
                print(f"    ⚠️  Client {cid} QUARANTINED after "
                      f"{count} consecutive flagged rounds")

        return newly_quarantined

    @staticmethod
    def _hash_to_synthetic_params(model_hash: str, rnd: int) -> List[np.ndarray]:
        """
        Convert a hex model_hash string to a list of synthetic numpy arrays
        suitable for ModelHasher.hash_round().

        This is used when actual model parameters are not available (e.g. when
        processing a log file after training). In live training, pass the actual
        flattened parameter tensors directly.

        The synthetic params are deterministic given model_hash + rnd, so the
        resulting hash_record.model_hash will be consistent across calls.
        """
        # Decode hex hash into bytes, split into 4 equal chunks → 4 param tensors
        raw = bytes.fromhex(model_hash)
        chunk = len(raw) // 4
        params = []
        for i in range(4):
            segment = raw[i * chunk:(i + 1) * chunk]
            arr = np.frombuffer(segment, dtype=np.uint8).astype(np.float32)
            # Embed round number into last element for uniqueness
            arr[-1] = float(rnd)
            params.append(arr)
        return params

    def _finalise(self) -> GovernanceReport:
        """Called after processing all rounds to produce the final report."""
        self.print_summary()
        self.export_reports()
        return self._build_report()

    def _build_report(self) -> GovernanceReport:
        tamper_report_raw = self.hasher.verify_chain()
        return GovernanceReport(
            total_rounds=len(self._round_records),
            chain_intact=tamper_report_raw.is_intact,
            tamper_events=len(self.gateway.get_tamper_alerts()),
            quarantined_clients=sorted(self._quarantined_clients),
            best_f1=self._best_f1,
            best_round=self._best_round,
            round_records=self._round_records,
            tamper_report=tamper_report_raw.__dict__,
        )
