#!/usr/bin/env python3
"""
Cascade Disturbance Observer (CDO) for Rocket Roll Stabilization

Two disturbance observers in cascade:
1. First observer: estimates total disturbance d(t) from roll dynamics
   using a Luenberger-style disturbance observer.
2. Second observer: decomposes d(t) into sinusoidal components
   A*sin(omega*t + phi), adaptively tracking amplitude, frequency, and phase.

The decomposed disturbance is predicted ahead and cancelled via feedforward.

Unlike ADRC+FF which uses fixed frequency candidates, CDO jointly estimates
both the disturbance and its frequency in a coupled structure. The frequency
estimate converges even when the disturbance frequency varies slowly.

Architecture:
    d_hat = roll_accel_measured - b0 * action   (DOB stage 1)
    [x1, x2, omega] = adaptive_sinusoidal_tracker(d_hat)  (DOB stage 2)
    u_ff = -K_ff * (x1*cos(theta) + x2*sin(theta)) / b0  (feedforward)
    u_total = u_base + u_ff

References:
    - Cascade observer for unknown-frequency sinusoidal disturbance
      rejection (Wen Xinyu et al. 2021)

Usage:
    from cascade_dob import CascadeDOBController, CascadeDOBConfig

    controller = CascadeDOBController(pid_config, cdo_config)
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from pid_controller import PIDConfig, GainScheduledPIDController


@dataclass
class CascadeDOBConfig:
    """Configuration for the Cascade Disturbance Observer.

    Attributes:
        K_ff: Feedforward gain (0 to 1). How aggressively to cancel the
              estimated sinusoidal disturbance.
        observer_bw: First-stage DOB bandwidth (rad/s). Higher = faster
                     disturbance tracking but more noise.
        freq_adapt_rate: Adaptation rate for frequency tracking (mu_omega).
                         Higher = faster frequency convergence.
        amp_adapt_rate: Adaptation rate for amplitude/phase tracking (mu_amp).
        freq_min: Minimum trackable frequency (rad/s).
        freq_max: Maximum trackable frequency (rad/s).
        omega_init: Initial frequency estimate (rad/s). ~10 rad/s is a
                    reasonable starting point for typical spin rates.
        b0: Control effectiveness fallback.
        b0_per_pa: Per-Pa effectiveness for gain scheduling.
        q_ref: Reference dynamic pressure for gain scheduling.
        warmup_steps: Steps before feedforward activates.
        forgetting: Exponential forgetting for amplitude estimates.
    """

    K_ff: float = 0.5
    observer_bw: float = 30.0
    freq_adapt_rate: float = 0.001
    amp_adapt_rate: float = 0.05
    freq_min: float = 1.0
    freq_max: float = 100.0
    omega_init: float = 10.0
    b0: float = 725.0
    b0_per_pa: Optional[float] = None
    q_ref: float = 500.0
    warmup_steps: int = 50
    forgetting: float = 0.998


class CascadeDOBController:
    """Cascade DOB wrapping a GS-PID base controller.

    Stage 1: Luenberger-style disturbance observer estimates d(t) from
             the difference between measured roll acceleration and the
             expected acceleration from the control input.

    Stage 2: Adaptive sinusoidal tracker decomposes d(t) into
             amplitude and frequency components. The frequency is tracked
             via gradient descent on the prediction error.

    Feedforward: The predicted sinusoidal disturbance is cancelled by
                 adding a compensating control signal.
    """

    def __init__(
        self,
        pid_config: PIDConfig = None,
        cdo_config: CascadeDOBConfig = None,
        use_observations: bool = False,
    ):
        self.pid_config = pid_config or PIDConfig()
        self.config = cdo_config or CascadeDOBConfig()
        self.use_observations = use_observations

        # Base controller: GS-PID
        self.base_ctrl = GainScheduledPIDController(
            self.pid_config,
            use_observations=use_observations,
        )

        # Pre-compute reference effectiveness
        q_ref = self.config.q_ref
        self._ref_effectiveness = q_ref * np.tanh(q_ref / 200.0)

        self.reset()

    def reset(self):
        """Reset all controller and observer state."""
        self.base_ctrl.reset()

        # Stage 1: DOB state
        self._d_hat = 0.0  # Estimated disturbance (rad/s^2)
        self._d_hat_dot = 0.0  # Disturbance rate estimate
        self._prev_action = 0.0

        # Stage 2: Sinusoidal tracker state
        self._omega_hat = self.config.omega_init  # Frequency estimate (rad/s)
        self._a_cos = 0.0  # Cosine coefficient (A*cos(phi))
        self._a_sin = 0.0  # Sine coefficient (A*sin(phi))
        self._phase_acc = 0.0  # Accumulated phase for the oscillator
        self._pred_error_prev = 0.0  # Previous prediction error (for freq gradient)

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
        """Compute control action: GS-PID + cascade DOB feedforward.

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

        # --- Read state ---
        # Roll accel from obs[4] is acceptable in both modes because it is
        # derived from noisy gyro rate in the IMU wrapper (not independently
        # delayed). Dynamic pressure from info dict in IMU mode to bypass
        # sensor_delay_steps.
        roll_accel = float(obs[4]) if len(obs) > 4 else 0.0
        if self.use_observations:
            q = info.get("dynamic_pressure_Pa", float(obs[5]) if len(obs) > 5 else 0.0)
        else:
            q = info.get("dynamic_pressure_Pa", 0.0)

        b0_now = self._get_b0(obs, info)

        # === Stage 1: Disturbance Observer ===
        # Plant model: roll_accel = b0 * action + d(t)
        # d_hat = roll_accel - b0 * prev_action
        d_measured = roll_accel - b0_now * self._prev_action

        # Low-pass filter the disturbance estimate (first-order)
        bw = cfg.observer_bw
        alpha_lpf = bw * dt / (1.0 + bw * dt)
        self._d_hat = (1.0 - alpha_lpf) * self._d_hat + alpha_lpf * d_measured

        # === Stage 2: Adaptive Sinusoidal Tracker ===
        # Update accumulated phase
        self._phase_acc += self._omega_hat * dt
        # Keep phase bounded
        if self._phase_acc > 2 * np.pi:
            self._phase_acc -= 2 * np.pi
        elif self._phase_acc < -2 * np.pi:
            self._phase_acc += 2 * np.pi

        # Basis functions at current phase
        cos_phase = np.cos(self._phase_acc)
        sin_phase = np.sin(self._phase_acc)

        # Predicted sinusoidal disturbance
        d_sin_predicted = self._a_cos * cos_phase + self._a_sin * sin_phase

        # Prediction error
        pred_error = self._d_hat - d_sin_predicted

        # Update amplitude/phase coefficients (LMS-style)
        mu_a = cfg.amp_adapt_rate
        self._a_cos = cfg.forgetting * self._a_cos + mu_a * pred_error * cos_phase
        self._a_sin = cfg.forgetting * self._a_sin + mu_a * pred_error * sin_phase

        # Update frequency estimate via gradient descent
        # Gradient of prediction error squared w.r.t omega:
        # d/d_omega [pred_error^2] ≈ -2 * pred_error * d/d_omega(d_sin_predicted)
        # d/d_omega(d_sin_predicted) = t * (-a_cos*sin(omega*t) + a_sin*cos(omega*t))
        # Approximation: use accumulated time as t
        d_pred_d_omega = -self._a_cos * sin_phase + self._a_sin * cos_phase
        omega_grad = -pred_error * d_pred_d_omega

        self._omega_hat += cfg.freq_adapt_rate * omega_grad
        # Clamp frequency
        self._omega_hat = float(
            np.clip(
                self._omega_hat,
                cfg.freq_min,
                cfg.freq_max,
            )
        )

        self._pred_error_prev = pred_error

        # === Feedforward ===
        if self._step_count < cfg.warmup_steps:
            self._prev_action = float(base_action[0])
            return base_action

        # Gain-schedule feedforward
        scale = self._gain_scale(q)

        # Predict disturbance one step ahead
        phase_ahead = self._phase_acc + self._omega_hat * dt
        cos_ahead = np.cos(phase_ahead)
        sin_ahead = np.sin(phase_ahead)
        d_predicted_ahead = self._a_cos * cos_ahead + self._a_sin * sin_ahead

        ff_action = -cfg.K_ff * d_predicted_ahead * scale / b0_now

        total_action = float(base_action[0]) + ff_action
        total_action = float(np.clip(total_action, -1.0, 1.0))

        self._prev_action = total_action
        return np.array([total_action], dtype=np.float32)

    @property
    def disturbance_estimate(self) -> float:
        """Current Stage 1 disturbance estimate."""
        return self._d_hat

    @property
    def frequency_estimate(self) -> float:
        """Current estimated disturbance frequency (rad/s)."""
        return self._omega_hat

    @property
    def amplitude_estimate(self) -> float:
        """Current estimated disturbance amplitude."""
        return np.sqrt(self._a_cos**2 + self._a_sin**2)
