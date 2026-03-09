"""
attack_simulator.py
--------------------
Simulates adversarial poisoning attacks on federated clients.

Implements the two attack types specified in the project proposal:
  1. Label-Flipping Attack  -- malicious client flips fraud labels (0<->1)
                              during local training to corrupt the global model
  2. Gradient Scaling Attack -- malicious client amplifies its gradient update
                              by a large factor to dominate aggregation

These are injected at the client level -- the server's trust scoring
(trust_scoring.py) must detect and neutralise them.

Usage:
    attacker = AttackSimulator(attack_type="label_flip", malicious_clients=[1, 3])
    X_poisoned, y_poisoned = attacker.poison_data(client_id=1, X, y)
    params_poisoned = attacker.poison_params(client_id=1, params, global_params)
"""

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict, Optional


class AttackSimulator:
    """
    Injects adversarial behaviour into specified malicious clients.

    Supports:
      - "label_flip"       : Flip fraud/legit labels before local training
      - "gradient_scale"   : Amplify gradient updates after training
      - "combined"         : Both attacks simultaneously
      - "none"             : No attack (clean baseline)
    """

    def __init__(
        self,
        attack_type:       str        = "label_flip",
        malicious_clients: List[int]  = None,
        flip_fraction:     float      = 1.0,    # Fraction of labels to flip
        scale_factor:      float      = 5.0,    # Gradient amplification factor
        attack_start_round: int       = 1,      # Round to begin attacking
    ):
        self.attack_type        = attack_type.lower()
        self.malicious_clients  = set(malicious_clients or [])
        self.flip_fraction      = flip_fraction
        self.scale_factor       = scale_factor
        self.attack_start_round = attack_start_round
        self.current_round      = 0

        valid_types = {"label_flip", "gradient_scale", "combined", "none"}
        if self.attack_type not in valid_types:
            raise ValueError(f"attack_type must be one of {valid_types}")

        if self.malicious_clients:
            print(
                f"[AttackSimulator] [!]  Attack type: '{attack_type}' | "
                f"Malicious clients: {sorted(self.malicious_clients)} | "
                f"Scale factor: {scale_factor}x"
            )
        else:
            print("[AttackSimulator] Clean run -- no malicious clients.")

    def set_round(self, round_num: int) -> None:
        self.current_round = round_num

    def is_malicious(self, client_id: int) -> bool:
        return (
            client_id in self.malicious_clients
            and self.current_round >= self.attack_start_round
        )

    # -- Attack 1: Label Flipping ----------------------------------------------

    def poison_data(
        self,
        client_id: int,
        X: np.ndarray,
        y: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Label-flipping attack: flip fraud<->legit labels for malicious clients.

        Effect on gradient: reversed learning signal causes gradient to point
        in the OPPOSITE direction to the global gradient -> detectable via
        negative cosine similarity in trust_scoring.py.

        Args:
            client_id: Client being poisoned
            X:         Feature matrix (unchanged)
            y:         Labels to potentially flip

        Returns:
            (X, y_poisoned) -- X is unchanged, y may be flipped
        """
        if not self.is_malicious(client_id):
            return X, y

        if self.attack_type not in ("label_flip", "combined"):
            return X, y

        y_poisoned = y.copy()
        n_flip     = max(1, int(len(y) * self.flip_fraction))
        flip_idx   = np.random.choice(len(y), size=n_flip, replace=False)
        y_poisoned[flip_idx] = 1 - y_poisoned[flip_idx]  # flip 0<->1

        n_actual_flips = int((y_poisoned != y).sum())
        print(
            f"  [Attack] Client {client_id:02d} LABEL-FLIP | "
            f"Flipped {n_actual_flips}/{len(y)} labels "
            f"({n_actual_flips/len(y)*100:.1f}%)"
        )
        return X, y_poisoned

    # -- Attack 2: Gradient Scaling --------------------------------------------

    def poison_params(
        self,
        client_id:    int,
        params:       List[np.ndarray],
        global_params: List[np.ndarray],
    ) -> List[np.ndarray]:
        """
        Gradient scaling attack: amplify the parameter update delta.

        delta_i = params_i - global_params
        poisoned = global_params + scale_factor * delta_i

        Effect: client's update dominates aggregation due to large norm
        -> detectable via norm_ratio >> 1 in trust_scoring.py.

        Args:
            client_id:    Client being poisoned
            params:       Client's post-training parameters
            global_params: Current global model parameters

        Returns:
            Poisoned parameters (amplified delta from global)
        """
        if not self.is_malicious(client_id):
            return params

        if self.attack_type not in ("gradient_scale", "combined"):
            return params

        poisoned = []
        for p, g in zip(params, global_params):
            delta   = p - g
            poisoned.append(g + self.scale_factor * delta)

        orig_norm    = np.linalg.norm(np.concatenate([p.flatten() for p in params]))
        poison_norm  = np.linalg.norm(np.concatenate([p.flatten() for p in poisoned]))
        print(
            f"  [Attack] Client {client_id:02d} GRADIENT-SCALE | "
            f"factor={self.scale_factor}x | "
            f"norm: {orig_norm:.2f} -> {poison_norm:.2f}"
        )
        return poisoned

    def get_attack_summary(self) -> Dict:
        return {
            "attack_type":        self.attack_type,
            "malicious_clients":  sorted(self.malicious_clients),
            "flip_fraction":      self.flip_fraction,
            "scale_factor":       self.scale_factor,
            "attack_start_round": self.attack_start_round,
        }