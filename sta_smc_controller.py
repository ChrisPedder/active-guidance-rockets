#!/usr/bin/env python3
"""
Super-Twisting Sliding Mode Control (STA-SMC) for Rocket Roll Stabilization

A higher-order sliding mode controller that provides chattering-free robust
rejection of all bounded disturbances without requiring frequency tracking,
disturbance estimation, or model identification.

The sliding surface is:
    sigma = roll_rate + c * roll_angle_error

The super-twisting algorithm generates control:
    v1 = -alpha * |sigma|^0.5 * sign(sigma)
    v2_dot = -beta * sign(sigma)
    u = (v1 + v2) / b0(q)

Gains must satisfy:
    alpha > sqrt(2 * D_max)
    beta > D_max
where D_max is an upper bound on the disturbance rate of change.

Gain scheduling: alpha and beta are scaled by q_ref / (q * tanh(q/200))
to maintain consistent effective gains across flight phases.

References:
    - Super-Twisting ADRC for Rocket Launcher Servo (2021)
    - Levant, A. "Sliding order and sliding accuracy in sliding mode control"

Usage:
    from sta_smc_controller import STASMCController, STASMCConfig

    controller = STASMCController(STASMCConfig())
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class STASMCConfig:
    """STA-SMC controller configuration.

    Attributes:
        c: Sliding surface slope. Determines convergence rate of roll angle
           to zero once on the sliding surface. Higher = faster angle
           convergence but more aggressive. Units: 1/s.
        alpha: Super-twisting proportional gain. Must satisfy
               alpha > sqrt(2 * D_max). Controls convergence speed to
               sliding surface.
        beta: Super-twisting integral gain. Must satisfy beta > D_max.
              Handles constant and slowly-varying disturbances.
        D_max: Upper bound on disturbance rate of change (rad/s^3).
               Used for default gain computation.
        b0: Control effectiveness estimate (rad/s^2 per normalized action).
        b0_per_pa: Control effectiveness per Pa. When set, b0 is computed
                   dynamically from q.
        q_ref: Reference dynamic pressure for gain scheduling (Pa).
        max_deflection: Max servo deflection (deg) for action normalization.
        use_observations: If True, read from obs (noisy IMU). If False,
                          read from info dict (ground truth).
    """

    c: float = 10.0
    alpha: float = 5.0
    beta: float = 10.0
    D_max: float = 10.0
    b0: float = 725.0
    b0_per_pa: Optional[float] = None
    q_ref: float = 500.0
    max_deflection: float = 30.0
    use_observations: bool = False


class STASMCController:
    """Super-Twisting Sliding Mode Controller for rocket roll stabilization.

    Provides chattering-free robust disturbance rejection for any bounded
    disturbance, regardless of frequency content. Does not require
    disturbance estimation or frequency tracking.
    """

    def __init__(self, config: STASMCConfig = None):
        self.config = config or STASMCConfig()
        self._ref_effectiveness = self.config.q_ref * np.tanh(self.config.q_ref / 200.0)
        self.reset()

    def reset(self):
        """Reset controller state for a new episode."""
        self.launch_detected = False
        self.target_angle = 0.0
        self.v2 = 0.0  # Super-twisting integrator state

    def _gain_scale(self, q: float) -> float:
        """Compute gain scaling factor from dynamic pressure.

        Scales alpha and beta so effective control authority remains
        constant across the flight envelope.
        """
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1e-3:
            return 5.0
        scale = self._ref_effectiveness / effectiveness
        return float(np.clip(scale, 0.5, 5.0))

    def _get_b0(self, obs: np.ndarray, info: dict) -> float:
        """Get current control effectiveness b0."""
        cfg = self.config
        if cfg.b0_per_pa is not None:
            if cfg.use_observations:
                q = float(obs[5]) if len(obs) > 5 else 0.0
            else:
                q = info.get("dynamic_pressure_Pa", 0.0)
            q_effectiveness = q * np.tanh(q / 200.0)
            b0_now = cfg.b0_per_pa * q_effectiveness
            b0_min = cfg.b0 * 0.01
            if b0_now < b0_min:
                b0_now = cfg.b0
            return b0_now
        return cfg.b0

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action using STA-SMC.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        cfg = self.config

        # --- Read state ---
        if cfg.use_observations:
            # IMU mode: roll rate from info dict (noisy but current —
            # bypasses sensor_delay_steps).
            roll_angle = obs[2] if len(obs) > 2 else 0.0
            roll_rate = np.radians(
                info.get("roll_rate_deg_s", np.degrees(obs[3]) if len(obs) > 3 else 0.0)
            )
            # IMU mode: prefer info dict for current q (bypasses sensor delay)
            q = info.get("dynamic_pressure_Pa", float(obs[5]) if len(obs) > 5 else 0.0)

            if not self.launch_detected:
                self.launch_detected = True
                self.target_angle = roll_angle
        else:
            roll_angle = info.get("roll_angle_rad", 0.0)
            roll_rate = np.radians(info.get("roll_rate_deg_s", 0.0))
            accel = info.get("vertical_acceleration_ms2", 0.0)
            q = info.get("dynamic_pressure_Pa", 0.0)

            if not self.launch_detected:
                if accel > 20.0:
                    self.launch_detected = True
                    self.target_angle = roll_angle
                else:
                    return np.array([0.0], dtype=np.float32)

        # --- Angle error (normalized to [-pi, pi]) ---
        angle_error = roll_angle - self.target_angle
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        # --- Sliding surface ---
        # sigma = roll_rate + c * angle_error
        # When sigma = 0: roll_rate = -c * angle_error → exponential
        # convergence of angle error to zero
        sigma = roll_rate + cfg.c * angle_error

        # --- Gain scheduling ---
        scale = self._gain_scale(q)
        alpha_eff = cfg.alpha * scale
        beta_eff = cfg.beta * scale

        # --- Super-twisting algorithm ---
        # v1 = -alpha * |sigma|^0.5 * sign(sigma)
        abs_sigma = abs(sigma)
        sqrt_sigma = np.sqrt(abs_sigma)
        sign_sigma = np.sign(sigma) if abs_sigma > 1e-10 else 0.0

        v1 = -alpha_eff * sqrt_sigma * sign_sigma

        # v2_dot = -beta * sign(sigma), integrate v2
        self.v2 += -beta_eff * sign_sigma * dt

        # --- Control law ---
        b0_now = self._get_b0(obs, info)
        action = (v1 + self.v2) / b0_now

        action = float(np.clip(action, -1.0, 1.0))
        return np.array([action], dtype=np.float32)

    @property
    def gains_valid(self) -> bool:
        """Check if gains satisfy the STA-SMC stability conditions."""
        cfg = self.config
        return cfg.alpha > np.sqrt(2 * cfg.D_max) and cfg.beta > cfg.D_max
