# =============================================================================
# common/governance_bridge.py
# Bridges Module 1 (split2) with Module 2 (split3) cleanly.
#
# WHY THIS FILE EXISTS:
#   split3/governance.py uses bare imports:
#       from model_hasher import ModelHasher
#       from fabric_gateway import create_gateway
#   These work when running from split3/ directory but fail when called
#   from split2/ context.
#
#   This bridge adds split3/ to sys.path before importing, resolving the
#   path conflict without modifying split3's files.
#
# HOW TO USE (in split2/main.py):
#   from common.governance_bridge import build_governance_engine
#   engine = build_governance_engine(log_dir="logs_split2")
#   strategy = get_trust_strategy(..., governance_engine=engine)
#
# HOW TO DISABLE (run Module 1 only, original behaviour):
#   strategy = get_trust_strategy(..., governance_engine=None)
# =============================================================================

from __future__ import annotations

import os
import sys
from typing import Optional


def _add_split3_to_path() -> bool:
    """
    Add split3/ directory to sys.path so governance.py can resolve its imports.
    Tries multiple path patterns to work from any working directory.
    Returns True if split3/ was found and added.
    """
    # All candidate locations relative to this file (common/governance_bridge.py)
    # this file lives at:  <project>/module1/common/governance_bridge.py
    # split3 lives at:     <project>/split3/   OR   <project>/module1/split3/
    this_dir    = os.path.dirname(os.path.abspath(__file__))
    module1     = os.path.dirname(this_dir)           # <project>/module1/
    project     = os.path.dirname(module1)            # <project>/
    cwd         = os.getcwd()                         # wherever user runs from

    candidates = [
        os.path.join(project,  "split3"),             # <project>/split3/       ← main case
        os.path.join(module1,  "split3"),             # module1/split3/
        os.path.join(module1,  "split3_blockchain"),  # module1/split3_blockchain/
        os.path.join(project,  "split3_blockchain"),  # <project>/split3_blockchain/
        os.path.join(cwd,      "split3"),             # cwd/split3/
        os.path.join(cwd,      "..", "split3"),       # ../split3/ from cwd
    ]

    for path in candidates:
        if os.path.isdir(path) and os.path.exists(os.path.join(path, "governance.py")):
            if path not in sys.path:
                sys.path.insert(0, path)
            return True

    return False


def build_governance_engine(
    log_dir:              str   = "logs_split2",
    governance_output:    str   = "governance_output",
    anomaly_threshold:    float = 0.5,
    consecutive_flag_limit: int = 3,
    use_simulation:       bool  = True,
    enabled:              bool  = True,
):
    """
    Build and return a GovernanceEngine instance for Module 2 integration.

    Args:
        log_dir:                Where split2 writes trust_training_log.json
                                (governance output will sit alongside it)
        governance_output:      Subdirectory for governance reports
        anomaly_threshold:      alpha threshold for flagging clients (0.5)
        consecutive_flag_limit: rounds before auto-quarantine (3)
        use_simulation:         True = blockchain_sim.py (no Docker needed)
                                False = fabric_gateway.py (real Hyperledger)
        enabled:                False = return None immediately (disable Module 2)

    Returns:
        GovernanceEngine instance, or None if disabled or import fails.
    """
    if not enabled:
        print("[Module 2] Governance disabled — Module 1 only mode.")
        return None

    # Resolve split3 import path
    found = _add_split3_to_path()
    if not found:
        print("[Module 2] WARNING: split3/ directory not found. "
              "Governance disabled. Check your folder structure.")
        return None

    # Import governance after path is set
    try:
        from governance import GovernanceEngine, GovernanceConfig
    except ImportError as e:
        print(f"[Module 2] WARNING: Cannot import GovernanceEngine: {e}. "
              f"Governance disabled.")
        return None

    output_dir = os.path.join(log_dir, governance_output)
    os.makedirs(output_dir, exist_ok=True)

    config = GovernanceConfig(
        anomaly_threshold      = anomaly_threshold,
        consecutive_flag_limit = consecutive_flag_limit,
        use_simulation         = use_simulation,
        output_dir             = output_dir,
    )

    engine = GovernanceEngine(config=config)
    print(f"[Module 2] GovernanceEngine ready | "
          f"mode={'SIMULATION' if use_simulation else 'FABRIC'} | "
          f"output={output_dir}")
    return engine
