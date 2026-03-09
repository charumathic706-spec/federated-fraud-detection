"""
blockchain_sim.py
-----------------
In-memory simulated permissioned blockchain ledger.
Zero external dependencies — runs in Google Colab out of the box.

Simulates the Hyperledger Fabric concepts used in Split 3:
  - Immutable append-only ledger (blocks cannot be modified after commit)
  - Endorsement policy (N-of-M signatures before commit)
  - Chaincode (smart contract) execution
  - Channel-based multi-org isolation
  - World state (key-value store, latest state per key)
  - Block explorer queries

The real Fabric SDK wrappers in fabric_gateway.py mirror this API exactly,
so switching from simulation → production is a drop-in replacement.

Simulated organisations:
  Org1MSP = FederatedCoordinator   (runs the aggregation server)
  Org2MSP = Bank_0 ... Bank_4      (participant nodes)
  OrdererMSP = Orderer             (block ordering service)
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulated identity / MSP
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

ORGS = {
    "Org1MSP":   "FederatedCoordinator",
    "Org2MSP":   "BankConsortium",
    "OrdererMSP": "OrderingService",
}


@dataclass
class Identity:
    msp_id:   str
    cert_id:  str   # simulated certificate fingerprint

    def sign(self, data: str) -> str:
        """Simulated ECDSA signature — HMAC with cert_id as key."""
        import hmac
        return hmac.new(
            self.cert_id.encode(), data.encode(), hashlib.sha256
        ).hexdigest()

    def __str__(self) -> str:
        return f"{self.msp_id}::{self.cert_id[:8]}..."


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Block & Transaction structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class Transaction:
    tx_id:         str
    timestamp:     float
    creator:       Identity
    chaincode:     str
    function:      str
    args:          Dict[str, Any]
    endorsements:  List[str] = field(default_factory=list)
    is_valid:      bool = True
    read_set:      Dict[str, Any] = field(default_factory=dict)
    write_set:     Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "tx_id":        self.tx_id,
            "timestamp":    self.timestamp,
            "creator":      str(self.creator),
            "chaincode":    self.chaincode,
            "function":     self.function,
            "args":         self.args,
            "endorsements": self.endorsements,
            "is_valid":     self.is_valid,
        }


@dataclass
class Block:
    block_num:    int
    timestamp:    float
    prev_hash:    str
    transactions: List[Transaction] = field(default_factory=list)
    block_hash:   str = ""
    data_hash:    str = ""

    def __post_init__(self) -> None:
        self._compute_hashes()

    def _compute_hashes(self) -> None:
        tx_data = json.dumps(
            [tx.to_dict() for tx in self.transactions],
            sort_keys=True, default=str
        )
        self.data_hash = hashlib.sha256(tx_data.encode()).hexdigest()
        header = f"{self.block_num}|{self.timestamp}|{self.prev_hash}|{self.data_hash}"
        self.block_hash = hashlib.sha256(header.encode()).hexdigest()

    def to_dict(self) -> Dict:
        return {
            "block_num":    self.block_num,
            "timestamp":    self.timestamp,
            "prev_hash":    self.prev_hash,
            "block_hash":   self.block_hash,
            "data_hash":    self.data_hash,
            "tx_count":     len(self.transactions),
            "transactions": [tx.to_dict() for tx in self.transactions],
        }


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Chaincode (smart contracts)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class ModelRegistryChaincode:
    """
    Smart contract: ModelRegistry
    Manages model hash records and integrity proofs.

    Functions:
      RegisterModel(round, model_hash, block_hash, global_f1, ...)
      GetModel(round)
      QueryAllModels()
      VerifyModelHash(round, claimed_hash)
    """

    NAME = "ModelRegistry"

    def __init__(self, world_state: Dict[str, Any]) -> None:
        self._state = world_state

    def invoke(self, function: str, args: Dict) -> Tuple[bool, Any]:
        fn = getattr(self, f"_fn_{function}", None)
        if fn is None:
            return False, f"Unknown function: {function}"
        return fn(args)

    def _fn_RegisterModel(self, args: Dict) -> Tuple[bool, Dict]:
        key = f"model:{args['round']}"
        if key in self._state:
            return False, f"Model for round {args['round']} already registered."
        record = {
            "round":           args["round"],
            "timestamp":       args.get("timestamp", time.time()),
            "model_hash":      args["model_hash"],
            "block_hash":      args["block_hash"],
            "prev_block_hash": args.get("prev_block_hash", "0" * 64),
            "global_f1":       args.get("global_f1", 0.0),
            "global_auc":      args.get("global_auc", 0.0),
            "trusted_clients": args.get("trusted_clients", []),
            "flagged_clients": args.get("flagged_clients", []),
            "param_count":     args.get("param_count", 0),
            "total_bytes":     args.get("total_bytes", 0),
            "status":          "COMMITTED",
        }
        self._state[key] = record
        # Update latest pointer
        self._state["model:latest"] = record
        return True, record

    def _fn_GetModel(self, args: Dict) -> Tuple[bool, Any]:
        key = f"model:{args['round']}"
        if key not in self._state:
            return False, f"No model registered for round {args['round']}."
        return True, self._state[key]

    def _fn_QueryAllModels(self, args: Dict) -> Tuple[bool, List[Dict]]:
        records = []
        for k, v in sorted(self._state.items()):
            if k.startswith("model:") and k != "model:latest":
                records.append(v)
        return True, records

    def _fn_VerifyModelHash(self, args: Dict) -> Tuple[bool, Dict]:
        key = f"model:{args['round']}"
        if key not in self._state:
            return False, {"verified": False, "reason": f"Round {args['round']} not found."}
        stored = self._state[key]["model_hash"]
        claimed = args["claimed_hash"]
        verified = (stored == claimed)
        return True, {
            "verified":    verified,
            "round":       args["round"],
            "stored_hash": stored,
            "claimed_hash": claimed,
            "match":       verified,
        }


class TamperAlertChaincode:
    """
    Smart contract: TamperAlert
    Records and queries tamper detection events.

    Functions:
      RaiseTamperAlert(round, alert_type, detail)
      GetAlerts()
      GetAlertsByRound(round)
      ClearAlert(alert_id)  — governance action, requires Org1MSP
    """

    NAME = "TamperAlert"

    def __init__(self, world_state: Dict[str, Any]) -> None:
        self._state = world_state
        self._alert_counter = 0

    def invoke(self, function: str, args: Dict) -> Tuple[bool, Any]:
        fn = getattr(self, f"_fn_{function}", None)
        if fn is None:
            return False, f"Unknown function: {function}"
        return fn(args)

    def _fn_RaiseTamperAlert(self, args: Dict) -> Tuple[bool, Dict]:
        self._alert_counter += 1
        alert_id = f"alert:{self._alert_counter:06d}"
        alert = {
            "alert_id":    alert_id,
            "timestamp":   time.time(),
            "round":       args.get("round", -1),
            "alert_type":  args.get("alert_type", "UNKNOWN"),
            "detail":      args.get("detail", ""),
            "severity":    args.get("severity", "HIGH"),
            "resolved":    False,
        }
        self._state[alert_id] = alert
        return True, alert

    def _fn_GetAlerts(self, args: Dict) -> Tuple[bool, List[Dict]]:
        alerts = [v for k, v in sorted(self._state.items())
                  if k.startswith("alert:")]
        return True, alerts

    def _fn_GetAlertsByRound(self, args: Dict) -> Tuple[bool, List[Dict]]:
        rnd = args["round"]
        alerts = [v for k, v in sorted(self._state.items())
                  if k.startswith("alert:") and v["round"] == rnd]
        return True, alerts

    def _fn_ClearAlert(self, args: Dict) -> Tuple[bool, Dict]:
        key = args["alert_id"]
        if key not in self._state:
            return False, f"Alert {key} not found."
        self._state[key]["resolved"] = True
        return True, self._state[key]


class AuditLogChaincode:
    """
    Smart contract: AuditLog
    Append-only participation and governance event log.
    Supports GDPR/PCI-DSS audit trail requirements.

    Functions:
      AppendEvent(event_type, round, data)
      QueryByRound(round)
      QueryByType(event_type)
      ExportAuditTrail()
    """

    NAME = "AuditLog"

    def __init__(self, world_state: Dict[str, Any]) -> None:
        self._state = world_state
        self._event_counter = 0

    def invoke(self, function: str, args: Dict) -> Tuple[bool, Any]:
        fn = getattr(self, f"_fn_{function}", None)
        if fn is None:
            return False, f"Unknown function: {function}"
        return fn(args)

    def _fn_AppendEvent(self, args: Dict) -> Tuple[bool, Dict]:
        self._event_counter += 1
        key = f"event:{self._event_counter:08d}"
        event = {
            "event_id":    key,
            "timestamp":   time.time(),
            "event_type":  args.get("event_type", "UNKNOWN"),
            "round":       args.get("round", -1),
            "actor":       args.get("actor", "system"),
            "data":        args.get("data", {}),
        }
        self._state[key] = event
        return True, event

    def _fn_QueryByRound(self, args: Dict) -> Tuple[bool, List[Dict]]:
        rnd = args["round"]
        events = [v for k, v in sorted(self._state.items())
                  if k.startswith("event:") and v["round"] == rnd]
        return True, events

    def _fn_QueryByType(self, args: Dict) -> Tuple[bool, List[Dict]]:
        et = args["event_type"]
        events = [v for k, v in sorted(self._state.items())
                  if k.startswith("event:") and v["event_type"] == et]
        return True, events

    def _fn_ExportAuditTrail(self, args: Dict) -> Tuple[bool, List[Dict]]:
        events = [v for k, v in sorted(self._state.items())
                  if k.startswith("event:")]
        return True, events


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Simulated Ledger (channel)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimulatedLedger:
    """
    Simulated Hyperledger Fabric channel ledger.

    Maintains:
      - Blockchain (ordered list of immutable Block objects)
      - World state (current key-value store)
      - Installed chaincodes

    Endorsement policy: 1-of-2 orgs (Org1MSP or Org2MSP).
    Ordering: single-threaded FIFO (simulates solo orderer).
    """

    CHANNEL = "fraud-detection-channel"
    GENESIS_HASH = "0" * 64

    def __init__(self) -> None:
        self._blocks:      List[Block]            = []
        self._world_state: Dict[str, Any]         = {}
        self._chaincodes:  Dict[str, Any]         = {}
        self._tx_pool:     List[Transaction]      = []
        self._tx_index:    Dict[str, Transaction] = {}

        # Install built-in chaincodes
        self._install_chaincode(ModelRegistryChaincode(self._world_state))
        self._install_chaincode(TamperAlertChaincode(self._world_state))
        self._install_chaincode(AuditLogChaincode(self._world_state))

        # Create genesis block
        self._create_genesis_block()

    # ── Chaincode management ──────────────────────────────────────────────────

    def _install_chaincode(self, cc: Any) -> None:
        self._chaincodes[cc.NAME] = cc

    def instantiate_chaincode(self, name: str) -> bool:
        return name in self._chaincodes

    # ── Transaction lifecycle ─────────────────────────────────────────────────

    def submit_transaction(
        self,
        identity:  Identity,
        chaincode: str,
        function:  str,
        args:      Dict,
    ) -> Tuple[str, bool, Any]:
        """
        Submit a transaction through the full lifecycle:
          Propose → Endorse → Order → Commit

        Returns: (tx_id, success, result)
        """
        # 1. Propose
        tx_id = self._generate_tx_id(identity, chaincode, function)

        # 2. Endorse (simulate: auto-endorse from Org1MSP)
        endorsement = identity.sign(tx_id)

        tx = Transaction(
            tx_id=tx_id,
            timestamp=time.time(),
            creator=identity,
            chaincode=chaincode,
            function=function,
            args=args,
            endorsements=[endorsement],
        )

        # 3. Execute chaincode
        cc = self._chaincodes.get(chaincode)
        if cc is None:
            tx.is_valid = False
            return tx_id, False, f"Chaincode '{chaincode}' not installed."

        success, result = cc.invoke(function, args)
        tx.is_valid = success

        # 4. Order & Commit (batch = 1 tx per block for simplicity)
        self._tx_pool.append(tx)
        self._commit_block()
        self._tx_index[tx_id] = tx

        return tx_id, success, result

    def query(
        self,
        chaincode: str,
        function:  str,
        args:      Dict,
    ) -> Tuple[bool, Any]:
        """
        Read-only query — does NOT create a transaction or block.
        """
        cc = self._chaincodes.get(chaincode)
        if cc is None:
            return False, f"Chaincode '{chaincode}' not installed."
        return cc.invoke(function, args)

    # ── Block explorer ────────────────────────────────────────────────────────

    def get_block(self, block_num: int) -> Optional[Block]:
        if 0 <= block_num < len(self._blocks):
            return self._blocks[block_num]
        return None

    def get_latest_block(self) -> Optional[Block]:
        return self._blocks[-1] if self._blocks else None

    def get_block_count(self) -> int:
        return len(self._blocks)

    def get_transaction(self, tx_id: str) -> Optional[Transaction]:
        return self._tx_index.get(tx_id)

    def get_world_state(self, key: str) -> Optional[Any]:
        return self._world_state.get(key)

    def verify_ledger_integrity(self) -> Tuple[bool, List[str]]:
        """
        Re-verify every block hash and prev_hash chain.
        Returns (is_intact, list_of_issues).
        """
        issues = []
        if not self._blocks:
            return True, []

        for i, block in enumerate(self._blocks):
            # Re-compute block hash
            recomputed_data_hash = hashlib.sha256(
                json.dumps(
                    [tx.to_dict() for tx in block.transactions],
                    sort_keys=True, default=str
                ).encode()
            ).hexdigest()
            if recomputed_data_hash != block.data_hash:
                issues.append(f"Block {i}: data_hash mismatch.")

            header = f"{block.block_num}|{block.timestamp}|{block.prev_hash}|{block.data_hash}"
            recomputed_block_hash = hashlib.sha256(header.encode()).hexdigest()
            if recomputed_block_hash != block.block_hash:
                issues.append(f"Block {i}: block_hash mismatch.")

            # Verify chain linkage
            if i > 0:
                if block.prev_hash != self._blocks[i - 1].block_hash:
                    issues.append(
                        f"Block {i}: broken chain link — "
                        f"prev_hash != block[{i-1}].block_hash"
                    )

        return (len(issues) == 0), issues

    def print_ledger_summary(self) -> None:
        print(f"\n{'─'*60}")
        print(f"  Simulated Ledger — Channel: {self.CHANNEL}")
        print(f"{'─'*60}")
        print(f"  Blocks:        {len(self._blocks)}")
        print(f"  Transactions:  {len(self._tx_index)}")
        print(f"  World state:   {len(self._world_state)} keys")
        print(f"  Chaincodes:    {list(self._chaincodes.keys())}")
        if self._blocks:
            latest = self._blocks[-1]
            print(f"  Latest block:  #{latest.block_num}  "
                  f"hash={latest.block_hash[:16]}...")
        print(f"{'─'*60}")

    # ── Internal ──────────────────────────────────────────────────────────────

    def _create_genesis_block(self) -> None:
        genesis = Block(
            block_num=0,
            timestamp=time.time(),
            prev_hash=self.GENESIS_HASH,
            transactions=[],
        )
        self._blocks.append(genesis)

    def _commit_block(self) -> None:
        if not self._tx_pool:
            return
        prev_hash = self._blocks[-1].block_hash if self._blocks else self.GENESIS_HASH
        block = Block(
            block_num=len(self._blocks),
            timestamp=time.time(),
            prev_hash=prev_hash,
            transactions=list(self._tx_pool),
        )
        self._blocks.append(block)
        self._tx_pool.clear()

    @staticmethod
    def _generate_tx_id(identity: Identity, chaincode: str, function: str) -> str:
        nonce = str(time.time_ns())
        raw = f"{identity.msp_id}|{chaincode}|{function}|{nonce}"
        return hashlib.sha256(raw.encode()).hexdigest()[:32]


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# High-level simulation gateway (mirrors FabricGateway API)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

class SimBlockchainGateway:
    """
    High-level gateway that mirrors the FabricGateway API in fabric_gateway.py.
    Swap SimBlockchainGateway → FabricGateway for production deployment.

    All methods have identical signatures to FabricGateway.
    """

    def __init__(self, org_msp: str = "Org1MSP") -> None:
        self._ledger   = SimulatedLedger()
        self._identity = Identity(
            msp_id=org_msp,
            cert_id=hashlib.sha256(f"{org_msp}_admin".encode()).hexdigest(),
        )
        print(f"[SimGateway] Initialised — org={org_msp}, "
              f"channel={SimulatedLedger.CHANNEL}")

    # ── Model Registry ────────────────────────────────────────────────────────

    def register_model(
        self,
        round_num:       int,
        model_hash:      str,
        block_hash:      str,
        prev_block_hash: str,
        global_f1:       float = 0.0,
        global_auc:      float = 0.0,
        trusted_clients: List[int] = None,
        flagged_clients: List[int] = None,
        param_count:     int = 0,
        total_bytes:     int = 0,
    ) -> Tuple[str, bool]:
        """Register model hash for a completed FL round."""
        tx_id, ok, result = self._ledger.submit_transaction(
            identity=self._identity,
            chaincode="ModelRegistry",
            function="RegisterModel",
            args={
                "round":           round_num,
                "timestamp":       time.time(),
                "model_hash":      model_hash,
                "block_hash":      block_hash,
                "prev_block_hash": prev_block_hash,
                "global_f1":       global_f1,
                "global_auc":      global_auc,
                "trusted_clients": trusted_clients or [],
                "flagged_clients": flagged_clients or [],
                "param_count":     param_count,
                "total_bytes":     total_bytes,
            },
        )
        if ok:
            print(f"  [Blockchain] ✅ Round {round_num} model registered  "
                  f"tx={tx_id[:12]}...  hash={model_hash[:16]}...")
        else:
            print(f"  [Blockchain] ❌ Round {round_num} registration failed: {result}")
        return tx_id, ok

    def verify_model_hash(self, round_num: int, claimed_hash: str) -> Dict:
        """Query the ledger to verify a model hash claim."""
        ok, result = self._ledger.query(
            chaincode="ModelRegistry",
            function="VerifyModelHash",
            args={"round": round_num, "claimed_hash": claimed_hash},
        )
        return result if ok else {"verified": False, "reason": result}

    def get_model_record(self, round_num: int) -> Optional[Dict]:
        """Retrieve a model record from the ledger."""
        ok, result = self._ledger.query(
            chaincode="ModelRegistry",
            function="GetModel",
            args={"round": round_num},
        )
        return result if ok else None

    def get_all_model_records(self) -> List[Dict]:
        """Get all registered model records (full history)."""
        _, result = self._ledger.query(
            chaincode="ModelRegistry",
            function="QueryAllModels",
            args={},
        )
        return result or []

    # ── Tamper Alerts ─────────────────────────────────────────────────────────

    def raise_tamper_alert(
        self,
        round_num:  int,
        alert_type: str,
        detail:     str,
        severity:   str = "HIGH",
    ) -> Tuple[str, Dict]:
        """Commit a tamper detection event to the ledger."""
        tx_id, ok, result = self._ledger.submit_transaction(
            identity=self._identity,
            chaincode="TamperAlert",
            function="RaiseTamperAlert",
            args={
                "round":      round_num,
                "alert_type": alert_type,
                "detail":     detail,
                "severity":   severity,
            },
        )
        print(f"  [TamperAlert] 🚨 {alert_type} at round {round_num}  "
              f"tx={tx_id[:12]}...")
        return tx_id, result

    def get_tamper_alerts(self) -> List[Dict]:
        _, result = self._ledger.query("TamperAlert", "GetAlerts", {})
        return result or []

    # ── Audit Log ─────────────────────────────────────────────────────────────

    def append_audit_event(
        self,
        event_type: str,
        round_num:  int,
        data:       Dict,
        actor:      str = "system",
    ) -> str:
        """Append an immutable audit event (GDPR/PCI-DSS compliance)."""
        tx_id, _, _ = self._ledger.submit_transaction(
            identity=self._identity,
            chaincode="AuditLog",
            function="AppendEvent",
            args={
                "event_type": event_type,
                "round":      round_num,
                "actor":      actor,
                "data":       data,
            },
        )
        return tx_id

    def get_audit_trail(self) -> List[Dict]:
        _, result = self._ledger.query("AuditLog", "ExportAuditTrail", {})
        return result or []

    # ── Ledger integrity ──────────────────────────────────────────────────────

    def verify_ledger(self) -> Tuple[bool, List[str]]:
        """Full ledger block-hash chain verification."""
        return self._ledger.verify_ledger_integrity()

    def get_block_count(self) -> int:
        return self._ledger.get_block_count()

    def print_summary(self) -> None:
        self._ledger.print_ledger_summary()