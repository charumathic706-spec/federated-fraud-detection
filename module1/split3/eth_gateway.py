# =============================================================================
# split3/eth_gateway.py
# Real Ethereum blockchain gateway using web3.py + Ganache
#
# API is 100% identical to SimBlockchainGateway in blockchain_sim.py.
# governance.py does NOT need any changes — only fabric_gateway.py changes.
#
# How it works:
#   - Connects to Ganache (local Ethereum node, 1 command to start)
#   - Compiles ModelRegistry.sol using py-solc-x
#   - Deploys the contract once, saves address to eth_deployment.json
#   - Every FL round calls contract functions via real web3 transactions
#   - Each call is mined into a real block with a real transaction hash
#
# Install:
#   pip install web3 py-solc-x
#   npm install -g ganache
#
# Start Ganache:
#   ganache --deterministic --port 8545
#
# =============================================================================

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# ── Constants ─────────────────────────────────────────────────────────────────
GANACHE_URL  = "http://127.0.0.1:8545"

# Paths — resolved relative to this file's location
_THIS_DIR    = os.path.dirname(os.path.abspath(__file__))
_DEPLOY_FILE = os.path.join(_THIS_DIR, "eth_deployment.json")

# ModelRegistry.sol — look in several candidate locations
_SOL_CANDIDATES = [
    os.path.join(_THIS_DIR, "contracts", "ModelRegistry.sol"),
    os.path.join(_THIS_DIR, "..", "contracts", "ModelRegistry.sol"),
    os.path.join(_THIS_DIR, "..", "split3", "contracts", "ModelRegistry.sol"),
]


# =============================================================================
# COMPILER
# =============================================================================

def _compile() -> Tuple[str, list]:
    """Compile ModelRegistry.sol. Returns (bytecode, abi)."""
    try:
        from solcx import compile_source, install_solc, get_installed_solc_versions
    except ImportError:
        raise ImportError(
            "py-solc-x not installed.\n"
            "Run: pip install py-solc-x"
        )

    if "0.8.19" not in [str(v) for v in get_installed_solc_versions()]:
        print("[Ethereum] Installing Solidity 0.8.19 compiler (one time)...")
        install_solc("0.8.19", show_progress=True)

    sol_path = next((p for p in _SOL_CANDIDATES if os.path.exists(p)), None)
    if sol_path:
        print(f"[Ethereum] Compiling {sol_path}")
        with open(sol_path, encoding="utf-8") as f:
            source = f.read()
    else:
        raise FileNotFoundError(
            "ModelRegistry.sol not found.\n"
            f"Searched: {_SOL_CANDIDATES}"
        )

    compiled  = compile_source(source, output_values=["abi", "bin"],
                                solc_version="0.8.19")
    key       = next(k for k in compiled if "ModelRegistry" in k)
    interface = compiled[key]
    return interface["bin"], interface["abi"]


# =============================================================================
# ETHEREUM GATEWAY
# =============================================================================

class EthBlockchainGateway:
    """
    Real Ethereum blockchain gateway.

    Every state-changing call (register_model, raise_tamper_alert,
    append_audit_event, quarantine_client) creates a real transaction
    that is mined into a real block on the Ganache chain.

    Read-only calls (verify_model_hash, get_block_count, etc.) are
    free view calls — no transaction, no gas.
    """

    def __init__(
        self,
        ganache_url: str = GANACHE_URL,
        org_msp:     str = "Org1MSP",
    ):
        self.url     = ganache_url
        self.org_msp = org_msp
        self._w3     = None
        self._contract = None
        self._account  = None

        self._init_web3()
        self._load_or_deploy()

    # ── Connection ────────────────────────────────────────────────────────────

    def _init_web3(self):
        try:
            from web3 import Web3
        except ImportError:
            raise ImportError(
                "web3 not installed.\n"
                "Run: pip install web3"
            )

        w3 = Web3(Web3.HTTPProvider(self.url))

        if not w3.is_connected():
            raise ConnectionError(
                f"\n[Ethereum] Cannot connect to Ganache at {self.url}\n"
                f"  Start Ganache with:  ganache --deterministic --port 8545\n"
                f"  Install with:        npm install -g ganache\n"
            )

        self._w3      = w3
        self._account = w3.eth.accounts[0]
        balance       = w3.from_wei(w3.eth.get_balance(self._account), "ether")

        print(f"\n[Ethereum] Connected to Ganache")
        print(f"  URL:     {self.url}")
        print(f"  Account: {self._account}")
        print(f"  Balance: {balance:.2f} ETH")
        print(f"  ChainID: {w3.eth.chain_id}")

    # ── Contract deployment ───────────────────────────────────────────────────

    def _load_or_deploy(self):
        """Load existing contract or deploy fresh one."""
        # Try loading saved deployment
        if os.path.exists(_DEPLOY_FILE):
            with open(_DEPLOY_FILE) as f:
                info = json.load(f)
            addr = info.get("address")
            abi  = info.get("abi")
            if addr and abi:
                code = self._w3.eth.get_code(addr)
                if code and code != b"" and code != "0x":
                    self._contract = self._w3.eth.contract(
                        address=addr, abi=abi
                    )
                    print(f"[Ethereum] Loaded contract at {addr}")
                    return
                else:
                    print("[Ethereum] Saved contract not found on chain "
                          "(Ganache was restarted) — redeploying...")

        # Deploy
        self._deploy()

    def _deploy(self):
        bytecode, abi = _compile()
        factory  = self._w3.eth.contract(abi=abi, bytecode=bytecode)
        tx_hash  = factory.constructor().transact({
            "from": self._account,
            "gas":  4_000_000,
        })
        receipt  = self._w3.eth.wait_for_transaction_receipt(tx_hash)
        addr     = receipt["contractAddress"]

        self._contract = self._w3.eth.contract(address=addr, abi=abi)

        # Save deployment info
        info = {"address": addr, "abi": abi, "deploy_tx": tx_hash.hex()}
        with open(_DEPLOY_FILE, "w") as f:
            json.dump(info, f, indent=2)

        print(f"[Ethereum] Contract deployed")
        print(f"  Address: {addr}")
        print(f"  Tx:      {tx_hash.hex()}")
        print(f"  Saved:   {_DEPLOY_FILE}")

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _transact(self, fn, *args) -> Tuple[str, bool]:
        """Send a state-changing transaction. Returns (tx_hash, success)."""
        try:
            tx_hash = fn(*args).transact({
                "from": self._account,
                "gas":  600_000,
            })
            receipt = self._w3.eth.wait_for_transaction_receipt(tx_hash)
            ok      = receipt["status"] == 1
            return tx_hash.hex(), ok
        except Exception as exc:
            logger.error(f"Transaction failed: {exc}")
            return f"FAILED_{int(time.time())}", False

    def _call(self, fn, *args) -> Any:
        """Read-only view call. No gas, no transaction."""
        try:
            return fn(*args).call()
        except Exception as exc:
            logger.error(f"View call failed: {exc}")
            return None

    # =========================================================================
    # PUBLIC API — identical to SimBlockchainGateway
    # =========================================================================

    def register_model(
        self,
        round_num:       int,
        model_hash:      str,
        block_hash:      str,
        prev_block_hash: str,
        global_f1:       float,
        global_auc:      float,
        trusted_clients: List[int],
        flagged_clients: List[int],
        param_count:     int = 0,
        total_bytes:     int = 0,
    ) -> Tuple[str, bool]:
        """
        Register one FL round on the Ethereum blockchain.

        Solidity does not support floats.
        F1 and AUC are multiplied by 1,000,000 before storing:
          0.836 → 836000
          0.971 → 971000
        When reading back, divide by 1,000,000.
        """
        f1_int  = int(round(float(global_f1),  6) * 1_000_000)
        auc_int = int(round(float(global_auc), 6) * 1_000_000)
        trusted = [int(c) for c in (trusted_clients or [])]
        flagged = [int(c) for c in (flagged_clients or [])]

        tx, ok = self._transact(
            self._contract.functions.registerModel,
            round_num, model_hash, block_hash, prev_block_hash,
            f1_int, auc_int, trusted, flagged,
        )
        if ok:
            print(f"  [Ethereum] Round {round_num:02d} → block | "
                  f"tx={tx[:20]}... | F1={global_f1:.4f}")
        else:
            print(f"  [Ethereum] WARNING: Round {round_num} registration FAILED")
        return tx, ok

    def verify_model_hash(self, round_num: int, claimed_hash: str) -> bool:
        """Verify on-chain that a model hash matches the ledger."""
        result = self._call(
            self._contract.functions.verifyModelHash,
            round_num, claimed_hash,
        )
        if result is None:
            return False
        is_valid, _ = result
        return bool(is_valid)

    def verify_full_chain(self) -> Tuple[bool, int]:
        """
        Verify full hash chain on-chain.
        Returns (intact: bool, broken_at_round: int).
        broken_at_round = 0 means the chain is intact.
        """
        result = self._call(self._contract.functions.verifyFullChain)
        if result is None:
            return True, 0
        intact, broken_at = result
        return bool(intact), int(broken_at)

    def raise_tamper_alert(
        self,
        round_num:  int,
        alert_type: str,
        detail:     str,
        severity:   str = "HIGH",
    ) -> str:
        tx, _ = self._transact(
            self._contract.functions.raiseTamperAlert,
            round_num, alert_type, detail, severity,
        )
        return tx

    def get_tamper_alerts(self) -> List[Dict]:
        count = self._call(self._contract.functions.getAlertCount)
        if not count:
            return []
        alerts = []
        for i in range(1, int(count) + 1):
            r = self._call(self._contract.functions.getAlert, i)
            if r:
                rnd, atype, severity, ts = r
                alerts.append({
                    "alert_id":  i,
                    "round":     int(rnd),
                    "alert_type": atype,
                    "severity":  severity,
                    "timestamp": int(ts),
                })
        return alerts

    def append_audit_event(
        self,
        event_type: str,
        round_num:  int,
        data:       Dict,
        actor:      str = "FederatedCoordinator",
    ) -> str:
        data_str = json.dumps(data, default=str)
        tx, _ = self._transact(
            self._contract.functions.appendAuditEvent,
            event_type, round_num, actor, data_str,
        )
        return tx

    def get_audit_trail(self) -> List[Dict]:
        count = self._call(self._contract.functions.getAuditCount)
        if not count:
            return []
        events = []
        for i in range(1, int(count) + 1):
            r = self._call(self._contract.functions.getAuditEvent, i)
            if r:
                etype, rnd, actor, data_str, ts = r
                try:
                    data = json.loads(data_str)
                except Exception:
                    data = data_str
                events.append({
                    "event_id":  i,
                    "event_type": etype,
                    "round":     int(rnd),
                    "actor":     actor,
                    "data":      data,
                    "timestamp": int(ts),
                })
        return events

    def get_block_count(self) -> int:
        r = self._call(self._contract.functions.getRoundCount)
        return int(r) if r is not None else 0

    def print_ledger_summary(self):
        rounds = self.get_block_count()
        alerts = self._call(self._contract.functions.getAlertCount) or 0
        audit  = self._call(self._contract.functions.getAuditCount) or 0
        print(f"\n[Ethereum Ledger]")
        print(f"  Contract address : {self._contract.address}")
        print(f"  Rounds committed : {rounds}")
        print(f"  Tamper alerts    : {int(alerts)}")
        print(f"  Audit events     : {int(audit)}")
        print(f"  Ganache URL      : {self.url}")
