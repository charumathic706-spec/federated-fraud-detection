"""
model_hasher.py
---------------
SHA-256 model hashing, hash chain construction, and tamper detection
for Split 3 — Blockchain Governance Layer.

Responsibilities:
  1. Hash individual model parameter tensors (SHA-256)
  2. Build a hash chain: each round's hash includes the previous round's hash
     → any retroactive tampering breaks the chain
  3. Verify hash chain integrity across all rounds
  4. Detect parameter-level tampering by re-hashing and comparing
  5. Produce a canonical model fingerprint from numpy arrays or raw bytes

Hash chain structure:
  block_hash(t) = SHA256( prev_hash(t-1) || model_hash(t) || round(t) || timestamp(t) )

This mirrors how blockchain headers chain blocks — a tampered round t
invalidates every subsequent block hash t+1, t+2, ...
"""

from __future__ import annotations

import hashlib
import json
import struct
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Data structures
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

@dataclass
class ModelHashRecord:
    """Single round's hash record — one link in the chain."""
    round_num:       int
    timestamp:       float
    model_hash:      str          # SHA-256 of model params only
    block_hash:      str          # SHA-256 of (prev_block_hash + model_hash + metadata)
    prev_block_hash: str          # links to previous round
    param_count:     int          # number of parameter tensors hashed
    total_bytes:     int          # total bytes of all parameters
    client_hashes:   Dict[int, str] = field(default_factory=dict)  # per-client hashes

    def to_dict(self) -> Dict:
        return {
            "round":           self.round_num,
            "timestamp":       self.timestamp,
            "model_hash":      self.model_hash,
            "block_hash":      self.block_hash,
            "prev_block_hash": self.prev_block_hash,
            "param_count":     self.param_count,
            "total_bytes":     self.total_bytes,
            "client_hashes":   {str(k): v for k, v in self.client_hashes.items()},
        }


@dataclass
class TamperReport:
    """Result of a hash chain integrity verification."""
    is_intact:          bool
    verified_rounds:    int
    tampered_rounds:    List[int] = field(default_factory=list)
    broken_links:       List[Tuple[int, int]] = field(default_factory=list)
    details:            List[str] = field(default_factory=list)

    def summary(self) -> str:
        if self.is_intact:
            return (f"✅ Hash chain INTACT — {self.verified_rounds} rounds verified, "
                    f"no tampering detected.")
        else:
            return (f"🚨 TAMPERING DETECTED — {len(self.tampered_rounds)} compromised "
                    f"rounds: {self.tampered_rounds}")


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Core hasher
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

GENESIS_HASH = "0" * 64  # Genesis block — no previous hash


class ModelHasher:
    """
    Stateful SHA-256 model hasher with hash chain support.

    Usage:
        hasher = ModelHasher()
        record = hasher.hash_round(round_num=1, params=model_params)
        report = hasher.verify_chain()
    """

    def __init__(self) -> None:
        self._chain: List[ModelHashRecord] = []
        self._last_block_hash: str = GENESIS_HASH

    # ── Public API ────────────────────────────────────────────────────────────

    def hash_round(
        self,
        round_num:      int,
        params:         List[np.ndarray],
        client_params:  Optional[Dict[int, List[np.ndarray]]] = None,
        timestamp:      Optional[float] = None,
    ) -> ModelHashRecord:
        """
        Hash a full model (list of param tensors) for one FL round.
        Extends the hash chain with the resulting block.

        Args:
            round_num:      Federated round number (1-indexed)
            params:         Global model parameters as list of numpy arrays
            client_params:  Optional per-client params for individual hashes
            timestamp:      Unix timestamp (defaults to now)

        Returns:
            ModelHashRecord appended to the internal chain
        """
        ts = timestamp or time.time()

        # 1. Hash model parameters
        model_hash, param_count, total_bytes = self._hash_params(params)

        # 2. Per-client hashes (optional)
        client_hashes: Dict[int, str] = {}
        if client_params:
            for cid, cp in client_params.items():
                ch, _, _ = self._hash_params(cp)
                client_hashes[cid] = ch

        # 3. Build block hash — chains to previous block
        block_hash = self._build_block_hash(
            prev_hash=self._last_block_hash,
            model_hash=model_hash,
            round_num=round_num,
            timestamp=ts,
        )

        record = ModelHashRecord(
            round_num=round_num,
            timestamp=ts,
            model_hash=model_hash,
            block_hash=block_hash,
            prev_block_hash=self._last_block_hash,
            param_count=param_count,
            total_bytes=total_bytes,
            client_hashes=client_hashes,
        )

        self._chain.append(record)
        self._last_block_hash = block_hash
        return record

    def hash_bytes(self, raw_bytes: bytes) -> str:
        """Hash arbitrary bytes — used for raw serialised model weights."""
        return hashlib.sha256(raw_bytes).hexdigest()

    def hash_trust_log_entry(self, log_entry: Dict) -> str:
        """
        Produce a deterministic SHA-256 fingerprint of a trust_training_log
        entry from Split 2.  Non-float fields are JSON-serialised; floats
        are normalised to 8 decimal places to avoid platform rounding drift.
        """
        canonical = json.dumps(log_entry, sort_keys=True, default=str)
        return hashlib.sha256(canonical.encode()).hexdigest()

    def verify_chain(self) -> TamperReport:
        """
        Verify the entire hash chain from genesis to latest round.
        Re-derives each block_hash from stored fields and checks against
        the recorded block_hash.  Also validates the prev_block_hash linkage.

        Returns:
            TamperReport with full detail on any broken links.
        """
        if not self._chain:
            return TamperReport(is_intact=True, verified_rounds=0,
                                details=["Chain is empty."])

        tampered: List[int] = []
        broken_links: List[Tuple[int, int]] = []
        details: List[str] = []
        expected_prev = GENESIS_HASH

        for rec in self._chain:
            # Check prev_block_hash linkage
            if rec.prev_block_hash != expected_prev:
                broken_links.append((rec.round_num - 1, rec.round_num))
                tampered.append(rec.round_num)
                details.append(
                    f"Round {rec.round_num}: broken link — "
                    f"expected prev={expected_prev[:12]}... "
                    f"got {rec.prev_block_hash[:12]}..."
                )

            # Re-derive block_hash from stored fields
            recomputed = self._build_block_hash(
                prev_hash=rec.prev_block_hash,
                model_hash=rec.model_hash,
                round_num=rec.round_num,
                timestamp=rec.timestamp,
            )
            if recomputed != rec.block_hash:
                tampered.append(rec.round_num)
                details.append(
                    f"Round {rec.round_num}: block_hash mismatch — "
                    f"stored={rec.block_hash[:12]}... "
                    f"recomputed={recomputed[:12]}..."
                )

            expected_prev = rec.block_hash

        return TamperReport(
            is_intact=(len(tampered) == 0),
            verified_rounds=len(self._chain),
            tampered_rounds=sorted(set(tampered)),
            broken_links=broken_links,
            details=details,
        )

    def verify_single_round(
        self,
        round_num: int,
        params:    List[np.ndarray],
    ) -> Tuple[bool, str]:
        """
        Re-hash a live model and compare against the stored record for that round.
        Used at inference time to confirm model integrity before predictions.

        Returns:
            (is_intact: bool, message: str)
        """
        record = self._get_record(round_num)
        if record is None:
            return False, f"No hash record found for round {round_num}."

        live_hash, _, _ = self._hash_params(params)
        if live_hash == record.model_hash:
            return True, (f"✅ Round {round_num} model VERIFIED — "
                          f"hash={live_hash[:16]}...")
        else:
            return False, (f"🚨 Round {round_num} TAMPERED — "
                           f"stored={record.model_hash[:16]}... "
                           f"live={live_hash[:16]}...")

    def get_chain(self) -> List[ModelHashRecord]:
        return list(self._chain)

    def get_latest_block_hash(self) -> str:
        return self._last_block_hash

    def export_chain(self) -> List[Dict]:
        return [r.to_dict() for r in self._chain]

    def import_chain(self, chain_data: List[Dict]) -> None:
        """
        Reconstruct chain from exported dict list (e.g. loaded from JSON).
        Does NOT re-verify — call verify_chain() after importing.
        """
        self._chain = []
        for d in chain_data:
            rec = ModelHashRecord(
                round_num=d["round"],
                timestamp=d["timestamp"],
                model_hash=d["model_hash"],
                block_hash=d["block_hash"],
                prev_block_hash=d["prev_block_hash"],
                param_count=d["param_count"],
                total_bytes=d["total_bytes"],
                client_hashes={int(k): v for k, v in d.get("client_hashes", {}).items()},
            )
            self._chain.append(rec)
        self._last_block_hash = self._chain[-1].block_hash if self._chain else GENESIS_HASH

    # ── Private helpers ───────────────────────────────────────────────────────

    @staticmethod
    def _hash_params(
        params: List[np.ndarray],
    ) -> Tuple[str, int, int]:
        """
        Canonical SHA-256 of a list of numpy parameter arrays.
        Arrays are serialised in float32 little-endian for platform consistency.

        Returns: (hex_digest, param_count, total_bytes)
        """
        hasher = hashlib.sha256()
        total_bytes = 0
        for arr in params:
            # Normalise dtype for cross-platform consistency
            data = arr.astype(np.float32).flatten().tobytes()
            # Prefix each tensor with its shape descriptor
            shape_tag = struct.pack(f">{len(arr.shape)}I", *arr.shape)
            hasher.update(shape_tag)
            hasher.update(data)
            total_bytes += len(data)
        return hasher.hexdigest(), len(params), total_bytes

    @staticmethod
    def _build_block_hash(
        prev_hash:  str,
        model_hash: str,
        round_num:  int,
        timestamp:  float,
    ) -> str:
        """
        Chained block hash:
          SHA256( prev_hash || model_hash || round_num_BE || timestamp_BE )
        """
        h = hashlib.sha256()
        h.update(prev_hash.encode())
        h.update(model_hash.encode())
        h.update(struct.pack(">I", round_num))           # 4-byte big-endian uint
        h.update(struct.pack(">d", timestamp))           # 8-byte big-endian double
        return h.hexdigest()

    def _get_record(self, round_num: int) -> Optional[ModelHashRecord]:
        for rec in self._chain:
            if rec.round_num == round_num:
                return rec
        return None


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# Convenience functions (used by main.py and the Colab notebook)
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

def hash_model_params(params: List[np.ndarray]) -> str:
    """One-shot hash of model parameters, no chain tracking."""
    digest, _, _ = ModelHasher._hash_params(params)
    return digest


def verify_hash_chain_from_log(chain_export: List[Dict]) -> TamperReport:
    """
    Reconstruct a ModelHasher from an exported chain (loaded from JSON)
    and run full verification.  Useful for offline audit tools.
    """
    hasher = ModelHasher()
    hasher.import_chain(chain_export)
    return hasher.verify_chain()


def simulate_tamper(
    hasher:    ModelHasher,
    round_num: int,
) -> ModelHasher:
    """
    Test utility: inject a tamper into a round's model_hash to verify
    that verify_chain() and verify_single_round() correctly detect it.
    Returns a NEW hasher with the tampered chain (original unchanged).
    """
    export = hasher.export_chain()
    for entry in export:
        if entry["round"] == round_num:
            # Flip first byte of model_hash
            orig = entry["model_hash"]
            flipped = format(int(orig[:2], 16) ^ 0xFF, "02x") + orig[2:]
            entry["model_hash"] = flipped
            print(f"  [SimTamper] Round {round_num} model_hash flipped: "
                  f"{orig[:12]}... → {flipped[:12]}...")
            break
    tampered_hasher = ModelHasher()
    tampered_hasher.import_chain(export)
    return tampered_hasher