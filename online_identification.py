#!/usr/bin/env python3
"""
Online System Identification for Control Effectiveness (b0)

Uses Recursive Least Squares (RLS) with exponential forgetting to estimate
the actual control effectiveness b0(t) in real time from the relationship:

    roll_accel = b0 * action + disturbance

The physics-based b0 estimate (b0_per_pa * q * tanh(q/200)) has residual
model error that degrades every controller dividing by b0 (ADRC, feedforward,
repetitive control). This module provides a closed-loop b0 estimate that
tracks the true value during flight.

Algorithm:
    The regression model is:
        alpha[t] = b0_hat * u[t-1] + c_hat
    where alpha = roll_acceleration (obs[4]), u = last_action (obs[8]),
    and c_hat captures the bias (damping + disturbance mean).

    RLS updates [b0_hat, c_hat] online with forgetting factor lambda to
    track b0 changes over 50-200 timesteps (0.5-2.0s), matching the
    timescale of dynamic pressure changes during flight.

Usage:
    from online_identification import B0Estimator

    estimator = B0Estimator(b0_init=130.0, forgetting=0.99)
    estimator.reset()
    b0_hat = estimator.update(roll_accel=obs[4], action=obs[8])
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class B0EstimatorConfig:
    """Configuration for the online b0 estimator.

    Attributes:
        b0_init: Initial b0 estimate (from physics model). Also used
                 as the center of the clamping range.
        forgetting: Exponential forgetting factor (0 < lambda <= 1).
                    Lower values = faster tracking but more noise.
                    0.99 tracks over ~100 steps (1 second at 100 Hz).
        clamp_ratio: b0_hat is clamped to [b0_init/ratio, b0_init*ratio].
        excitation_threshold: Minimum |action| for RLS update. Below this,
                              the regressor is too small for reliable estimation.
        p_init: Initial covariance scaling. Larger = faster initial convergence
                but more sensitive to noise.
    """

    b0_init: float = 130.0
    forgetting: float = 0.99
    clamp_ratio: float = 10.0
    excitation_threshold: float = 0.05
    p_init: float = 1000.0


class B0Estimator:
    """Online RLS estimator for control effectiveness b0.

    Estimates the 2-parameter model:
        roll_accel = b0 * action + c
    where b0 is the control effectiveness and c is a bias term
    capturing damping + disturbance mean.

    The b0 estimate can be fed into ADRC, feedforward, or any
    controller that needs to know the plant gain.
    """

    def __init__(self, config: B0EstimatorConfig = None):
        self.config = config or B0EstimatorConfig()
        self.reset()

    def reset(self):
        """Reset estimator state for a new episode."""
        cfg = self.config
        # Parameter vector: [b0_hat, c_hat]
        self._theta = np.array([cfg.b0_init, 0.0])
        # Covariance matrix (2x2, starts as scaled identity)
        self._P = np.eye(2) * cfg.p_init
        # Track number of updates for diagnostics
        self.n_updates = 0

    @property
    def b0_hat(self) -> float:
        """Current b0 estimate."""
        return float(self._theta[0])

    @property
    def c_hat(self) -> float:
        """Current bias estimate."""
        return float(self._theta[1])

    def update(self, roll_accel: float, action: float) -> float:
        """Update the b0 estimate with a new (roll_accel, action) pair.

        Args:
            roll_accel: Measured roll acceleration (rad/s^2), from obs[4].
            action: Previous control action (normalized [-1,1]), from obs[8].

        Returns:
            Updated b0 estimate (clamped to valid range).
        """
        cfg = self.config

        # Persistent excitation check: skip update when action is too small
        if abs(action) <= cfg.excitation_threshold:
            return self.b0_hat

        # Regressor vector: [action, 1]
        phi = np.array([action, 1.0])

        # Prediction error
        y = roll_accel
        y_hat = phi @ self._theta
        e = y - y_hat

        # RLS update with exponential forgetting
        lam = cfg.forgetting
        P_phi = self._P @ phi
        denom = lam + phi @ P_phi
        if abs(denom) < 1e-12:
            return self.b0_hat

        K = P_phi / denom  # Kalman gain
        self._theta = self._theta + K * e
        self._P = (self._P - np.outer(K, phi @ self._P)) / lam

        # Clamp b0 to prevent divergence
        b0_min = cfg.b0_init / cfg.clamp_ratio
        b0_max = cfg.b0_init * cfg.clamp_ratio
        self._theta[0] = np.clip(self._theta[0], b0_min, b0_max)

        self.n_updates += 1
        return self.b0_hat
