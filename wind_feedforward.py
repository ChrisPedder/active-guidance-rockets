#!/usr/bin/env python3
"""
Roll-Angle Feedforward for Wind Disturbance Rejection

Wind torque on a spinning rocket follows: torque ∝ A * sin(wind_dir - roll_angle).
This is a sinusoidal disturbance at the spin frequency that PID integral action
cannot track (90° phase lag). This module estimates the disturbance's sinusoidal
structure and cancels it via feedforward.

Algorithm:
    The disturbance d(t) ≈ a*cos(θ) + b*sin(θ), where θ = roll_angle.
    Expanding sin(φ - θ) = sin(φ)cos(θ) - cos(φ)sin(θ), the coefficients
    a = A*sin(φ) and b = -A*cos(φ) encode the wind amplitude A and direction φ.

    We estimate a, b using a recursive least-squares (exponential forgetting)
    update that correlates the disturbance signal with cos(θ) and sin(θ):
        a += mu * e * cos(θ)
        b += mu * e * sin(θ)
    where e = d_observed - (a*cos(θ) + b*sin(θ)) is the prediction error.

    The feedforward action cancels the predicted disturbance:
        u_ff = -K_ff * (a*cos(θ) + b*sin(θ)) / b0

    This is added to the base controller output (ADRC or GS-PID).

Usage:
    from wind_feedforward import WindFeedforwardADRC, WindFeedforwardConfig

    base_config = ADRCConfig(omega_c=15, omega_o=50, b0=100, b0_per_pa=0.5)
    ff_config = WindFeedforwardConfig(K_ff=0.5, mu=0.02)
    controller = WindFeedforwardADRC(base_config, ff_config)
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from adrc_controller import ADRCController, ADRCConfig, estimate_adrc_config


@dataclass
class WindFeedforwardConfig:
    """Configuration for wind feedforward estimator.

    Attributes:
        K_ff: Feedforward gain (0 to 1). How aggressively to cancel the
              estimated disturbance. 0 = no feedforward, 1 = full cancellation.
              Start conservative (0.5) and increase if stable.
        mu: Adaptation rate for the sinusoidal estimator. Higher = faster
            tracking of wind changes but more noise sensitivity.
            Typical range: 0.005 to 0.05.
        forgetting: Exponential forgetting factor for the coefficient estimates.
                    Closer to 1.0 = longer memory, more stable but slower to
                    adapt to wind changes. Typical range: 0.995 to 0.9999.
        warmup_steps: Number of steps before feedforward activates. Allows the
                      base controller and ESO to initialize before feedforward
                      starts adapting.
    """

    K_ff: float = 0.5
    mu: float = 0.02
    forgetting: float = 0.998
    warmup_steps: int = 50


class WindFeedforwardADRC:
    """ADRC controller with roll-angle feedforward for wind rejection.

    Wraps ADRCController and adds a sinusoidal disturbance estimator that
    uses the ADRC's z3 (total disturbance estimate) as the signal to
    decompose into roll-angle-dependent components.

    The feedforward predicts and cancels the periodic wind torque:
        d_predicted = a * cos(roll_angle) + b * sin(roll_angle)
    where a, b are adaptively estimated from z3.
    """

    def __init__(
        self,
        adrc_config: ADRCConfig = None,
        ff_config: WindFeedforwardConfig = None,
        b0_estimator=None,
    ):
        self.adrc = ADRCController(adrc_config, b0_estimator=b0_estimator)
        self.config = adrc_config or ADRCConfig()
        self.ff_config = ff_config or WindFeedforwardConfig()
        self.reset()

    def reset(self):
        """Reset controller, observer, and feedforward estimator."""
        self.adrc.reset()
        # Fourier coefficients for disturbance decomposition
        self.coeff_cos = 0.0  # a: weight on cos(roll_angle)
        self.coeff_sin = 0.0  # b: weight on sin(roll_angle)
        self._step_count = 0
        self._ff_action = 0.0  # Last feedforward action (for diagnostics)

    @property
    def launch_detected(self):
        return self.adrc.launch_detected

    @launch_detected.setter
    def launch_detected(self, value):
        self.adrc.launch_detected = value

    @property
    def target_angle(self):
        return self.adrc.target_angle

    @target_angle.setter
    def target_angle(self, value):
        self.adrc.target_angle = value

    @property
    def z1(self):
        return self.adrc.z1

    @property
    def z2(self):
        return self.adrc.z2

    @property
    def z3(self):
        return self.adrc.z3

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action: ADRC base + wind feedforward.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        ff_cfg = self.ff_config
        cfg = self.config

        # --- Run base ADRC controller ---
        base_action = self.adrc.step(obs, info, dt)

        if not self.adrc.launch_detected:
            return base_action

        self._step_count += 1

        # --- Read roll angle ---
        if cfg.use_observations:
            roll_angle = obs[2] if len(obs) > 2 else 0.0
        else:
            roll_angle = info.get("roll_angle_rad", 0.0)

        # --- Get current b0 for scaling feedforward action ---
        if cfg.b0_per_pa is not None:
            if cfg.use_observations:
                # IMU mode: prefer info dict for current q (bypasses sensor delay)
                q = info.get(
                    "dynamic_pressure_Pa", float(obs[5]) if len(obs) > 5 else 0.0
                )
            else:
                q = info.get("dynamic_pressure_Pa", 0.0)
            q_effectiveness = q * np.tanh(q / 200.0)
            b0_now = cfg.b0_per_pa * q_effectiveness
            b0_min = cfg.b0 * 0.01
            if b0_now < b0_min:
                b0_now = cfg.b0
        else:
            b0_now = cfg.b0

        # --- Update sinusoidal disturbance estimator ---
        # Use ADRC's z3 as the disturbance signal to decompose
        cos_theta = np.cos(roll_angle)
        sin_theta = np.sin(roll_angle)

        # Predicted disturbance from current estimates
        d_predicted = self.coeff_cos * cos_theta + self.coeff_sin * sin_theta

        # Prediction error: what z3 says minus what we predicted
        prediction_error = self.adrc.z3 - d_predicted

        # Adaptive update with exponential forgetting
        self.coeff_cos = (
            ff_cfg.forgetting * self.coeff_cos
            + ff_cfg.mu * prediction_error * cos_theta
        )
        self.coeff_sin = (
            ff_cfg.forgetting * self.coeff_sin
            + ff_cfg.mu * prediction_error * sin_theta
        )

        # --- Compute feedforward action ---
        if self._step_count < ff_cfg.warmup_steps:
            self._ff_action = 0.0
            return base_action

        # Feedforward cancels the predicted periodic disturbance
        # The base ADRC already cancels z3, but z3 lags the oscillation.
        # We predict AHEAD using the sinusoidal model.
        # The feedforward uses the CURRENT roll angle to predict the
        # disturbance at this instant, which is more accurate than z3
        # when the disturbance oscillates faster than the ESO bandwidth.
        ff_disturbance = self.coeff_cos * cos_theta + self.coeff_sin * sin_theta

        # Scale feedforward to action units: disturbance is in rad/s^2,
        # divide by b0 to get action units
        ff_action = -ff_cfg.K_ff * ff_disturbance / b0_now

        self._ff_action = float(ff_action)

        # Combine base ADRC action with feedforward
        total_action = float(base_action[0]) + ff_action

        # Clamp to [-1, 1]
        total_action = float(np.clip(total_action, -1.0, 1.0))

        return np.array([total_action], dtype=np.float32)

    @property
    def wind_amplitude(self) -> float:
        """Estimated wind disturbance amplitude in rad/s^2."""
        return np.sqrt(self.coeff_cos**2 + self.coeff_sin**2)

    @property
    def wind_direction_estimate(self) -> float:
        """Estimated wind direction in radians."""
        return np.arctan2(self.coeff_cos, -self.coeff_sin)
