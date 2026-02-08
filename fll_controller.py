#!/usr/bin/env python3
"""
Frequency-Locked Loop (FLL) Controller for Rocket Roll Stabilization

A discrete-time adaptive filter with PLL-like structure that tracks the
dominant disturbance frequency and generates a synchronized cancellation
signal. Wraps GS-PID and adds adaptive notch-based feedforward.

The FLL maintains an internal oscillator at an adaptive frequency:
    x1[k+1] = x1[k] + omega_hat * dt * x2[k]
    x2[k+1] = x2[k] - omega_hat * dt * x1[k]

The oscillator's phase and amplitude are adapted to minimize the
residual disturbance (roll rate that the base controller hasn't rejected).

Frequency is adapted via gradient descent on the output power:
    omega_hat += mu_f * (x1*e_dot - x2*e) / (x1^2 + x2^2)

where e is the residual error signal (roll rate after base control).

References:
    - Discrete-Time FLL Adaptive Filters (Mojiri et al.)
    - Second-Order Generalized Integrator FLL (SOGI-FLL)

Usage:
    from fll_controller import FLLController, FLLConfig

    controller = FLLController(pid_config, fll_config)
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from pid_controller import PIDConfig, GainScheduledPIDController


@dataclass
class FLLConfig:
    """Configuration for the FLL adaptive controller.

    Attributes:
        K_ff: Feedforward gain (0 to 1). How aggressively to cancel
              the tracked disturbance.
        mu_freq: Frequency adaptation rate. Higher = faster frequency
                 tracking but more noise sensitivity.
        mu_amp: Amplitude/phase adaptation rate.
        omega_init: Initial frequency estimate (rad/s).
        freq_min: Minimum trackable frequency (rad/s).
        freq_max: Maximum trackable frequency (rad/s).
        b0: Control effectiveness fallback.
        b0_per_pa: Per-Pa effectiveness for gain scheduling.
        q_ref: Reference dynamic pressure for gain scheduling.
        warmup_steps: Steps before feedforward activates.
        error_filter_alpha: Low-pass filter coefficient for error signal.
    """

    K_ff: float = 0.5
    mu_freq: float = 0.0005
    mu_amp: float = 0.03
    omega_init: float = 10.0
    freq_min: float = 1.0
    freq_max: float = 100.0
    b0: float = 725.0
    b0_per_pa: Optional[float] = None
    q_ref: float = 500.0
    warmup_steps: int = 50
    error_filter_alpha: float = 0.1


class FLLController:
    """FLL-based adaptive controller wrapping GS-PID.

    The FLL tracks the dominant disturbance frequency using a coupled
    oscillator with gradient-based frequency adaptation. Once locked,
    it generates a synchronized cancellation signal.
    """

    def __init__(
        self,
        pid_config: PIDConfig = None,
        fll_config: FLLConfig = None,
        use_observations: bool = False,
    ):
        self.pid_config = pid_config or PIDConfig()
        self.config = fll_config or FLLConfig()
        self.use_observations = use_observations

        self.base_ctrl = GainScheduledPIDController(
            self.pid_config,
            use_observations=use_observations,
        )

        q_ref = self.config.q_ref
        self._ref_effectiveness = q_ref * np.tanh(q_ref / 200.0)

        self.reset()

    def reset(self):
        """Reset all controller and oscillator state."""
        self.base_ctrl.reset()

        cfg = self.config

        # Adaptive oscillator state
        self._omega_hat = cfg.omega_init
        self._x1 = 1.0  # Cosine-like component (initialized to unit amplitude)
        self._x2 = 0.0  # Sine-like component

        # Amplitude tracking
        self._a_cos = 0.0  # Output cosine coefficient
        self._a_sin = 0.0  # Output sine coefficient

        # Error filtering
        self._error_filtered = 0.0
        self._error_prev = 0.0

        self._step_count = 0

    @property
    def launch_detected(self):
        return self.base_ctrl.launch_detected

    @launch_detected.setter
    def launch_detected(self, value):
        self.base_ctrl.launch_detected = value

    def _get_b0(self, obs: np.ndarray, info: dict) -> float:
        """Get current control effectiveness b0."""
        cfg = self.config
        if cfg.b0_per_pa is not None:
            if self.use_observations:
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

    def _gain_scale(self, q: float) -> float:
        """Compute gain scaling factor from dynamic pressure."""
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1e-3:
            return 5.0
        scale = self._ref_effectiveness / effectiveness
        return float(np.clip(scale, 0.5, 5.0))

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action: GS-PID + FLL feedforward.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        cfg = self.config

        # --- Base GS-PID action ---
        base_action = self.base_ctrl.step(obs, info, dt)

        if not self.base_ctrl.launch_detected:
            return base_action

        self._step_count += 1

        # --- Read roll rate as error signal ---
        if self.use_observations:
            # IMU mode: roll rate from info dict (noisy but current —
            # bypasses sensor_delay_steps).
            roll_rate = np.radians(
                info.get(
                    "roll_rate_deg_s",
                    np.degrees(float(obs[3])) if len(obs) > 3 else 0.0,
                )
            )
            # IMU mode: prefer info dict for current q (bypasses sensor delay)
            q = info.get("dynamic_pressure_Pa", float(obs[5]) if len(obs) > 5 else 0.0)
        else:
            roll_rate = np.radians(info.get("roll_rate_deg_s", 0.0))
            q = info.get("dynamic_pressure_Pa", 0.0)

        # The error signal is the roll rate — what the base controller
        # hasn't been able to reject. We want to identify and cancel the
        # periodic component.
        error = roll_rate

        # Low-pass filter the error to reduce noise
        alpha_f = cfg.error_filter_alpha
        self._error_filtered = (1.0 - alpha_f) * self._error_filtered + alpha_f * error
        e = self._error_filtered

        # --- Update adaptive oscillator ---
        # Discrete rotation by omega_hat * dt
        theta = self._omega_hat * dt
        cos_t = np.cos(theta)
        sin_t = np.sin(theta)

        x1_new = cos_t * self._x1 + sin_t * self._x2
        x2_new = -sin_t * self._x1 + cos_t * self._x2

        # Normalize oscillator to prevent amplitude drift
        norm = np.sqrt(x1_new**2 + x2_new**2)
        if norm > 1e-10:
            x1_new /= norm
            x2_new /= norm

        self._x1 = x1_new
        self._x2 = x2_new

        # --- Update amplitude coefficients (LMS) ---
        # Predicted disturbance
        d_pred = self._a_cos * self._x1 + self._a_sin * self._x2
        pred_error = e - d_pred

        self._a_cos += cfg.mu_amp * pred_error * self._x1
        self._a_sin += cfg.mu_amp * pred_error * self._x2

        # --- Update frequency (gradient-based FLL) ---
        # The frequency gradient uses the cross terms between the
        # error and orthogonal oscillator components
        e_dot = (e - self._error_prev) / dt if dt > 0 else 0.0

        denom = self._a_cos**2 + self._a_sin**2
        if denom > 1e-8:
            # Frequency error signal
            freq_error = self._x1 * e_dot - self._x2 * e * self._omega_hat
            self._omega_hat += cfg.mu_freq * freq_error / np.sqrt(denom)

        # Clamp frequency
        self._omega_hat = float(
            np.clip(
                self._omega_hat,
                cfg.freq_min,
                cfg.freq_max,
            )
        )

        self._error_prev = e

        # === Feedforward ===
        if self._step_count < cfg.warmup_steps:
            return base_action

        b0_now = self._get_b0(obs, info)
        scale = self._gain_scale(q)

        # Predict disturbance one step ahead
        theta_ahead = self._omega_hat * dt
        cos_a = np.cos(theta_ahead)
        sin_a = np.sin(theta_ahead)
        x1_ahead = cos_a * self._x1 + sin_a * self._x2
        x2_ahead = -sin_a * self._x1 + cos_a * self._x2

        d_predicted = self._a_cos * x1_ahead + self._a_sin * x2_ahead

        ff_action = -cfg.K_ff * d_predicted * scale / b0_now

        total_action = float(base_action[0]) + ff_action
        total_action = float(np.clip(total_action, -1.0, 1.0))

        return np.array([total_action], dtype=np.float32)

    @property
    def frequency_estimate(self) -> float:
        """Current estimated disturbance frequency (rad/s)."""
        return self._omega_hat

    @property
    def amplitude_estimate(self) -> float:
        """Current estimated disturbance amplitude."""
        return np.sqrt(self._a_cos**2 + self._a_sin**2)

    @property
    def frequency_estimate_hz(self) -> float:
        """Current estimated disturbance frequency (Hz)."""
        return self._omega_hat / (2 * np.pi)
