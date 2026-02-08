#!/usr/bin/env python3
"""
Repetitive Controller for Spin-Frequency Disturbance Rejection

Implements the Internal Model Principle: a controller containing a resonant
mode at frequency omega achieves zero steady-state error against sinusoidal
disturbances at that frequency. For a spinning rocket, wind torque is
A * sin(wind_dir - roll_angle), a sinusoidal disturbance at the spin frequency.

Architecture:
    u_total = u_GS-PID + u_resonant

    The resonant filter is:
        H(s) = K_rc * s / (s^2 + omega^2)

    Discretized via Tustin (bilinear) transform at 100 Hz. The center
    frequency omega is updated every timestep from the measured roll rate.

    The resonant mode creates infinite gain at exactly omega, forcing the
    closed-loop error to zero at that frequency — regardless of the
    disturbance amplitude or phase. This is the key theoretical advantage
    over feedforward (which requires accurate estimation) and PID (which
    has 90-degree lag at the disturbance frequency).

Usage:
    from repetitive_controller import RepetitiveGSPIDController, RepetitiveConfig

    config = RepetitiveConfig(K_rc=0.5)
    controller = RepetitiveGSPIDController(config=config, use_observations=True)
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from pid_controller import PIDConfig, GainScheduledPIDController


@dataclass
class RepetitiveConfig:
    """Configuration for the repetitive (resonant) controller.

    Attributes:
        K_rc: Resonant controller gain. Scales the resonant filter output.
              Higher = more aggressive disturbance cancellation but risks
              instability. Typical range: 0.1 to 2.0 * Kp.
        min_omega: Minimum spin frequency (rad/s) for the resonant filter.
                   Below this, the filter is disabled to avoid near-DC
                   resonance which causes integrator-like windup.
        max_omega: Maximum spin frequency (rad/s). Above this, the Tustin
                   discretization becomes inaccurate (Nyquist limit).
        damping: Small damping factor added to the resonant denominator:
                 H(s) = K_rc * s / (s^2 + 2*zeta*omega*s + omega^2).
                 Prevents infinite gain exactly at omega (which causes
                 numerical issues) and widens the rejection bandwidth.
                 Typical range: 0.01 to 0.1.
        omega_smoothing: Exponential smoothing factor for the spin frequency
                        estimate. 0 = no smoothing, 0.99 = very slow tracking.
                        Prevents the resonant frequency from jumping around
                        due to noisy gyro readings.
        warmup_steps: Number of steps before resonant action activates.
                      Allows the base GS-PID to stabilize before the
                      resonant mode starts.
    """

    K_rc: float = 0.5
    min_omega: float = 3.0  # ~0.5 Hz minimum
    max_omega: float = 150.0  # ~24 Hz maximum (below Nyquist at 50 Hz)
    damping: float = 0.05
    omega_smoothing: float = 0.9
    warmup_steps: int = 30


class RepetitiveGSPIDController:
    """GS-PID controller augmented with a resonant mode at the spin frequency.

    The resonant filter provides theoretically perfect rejection of sinusoidal
    disturbances at the spin frequency, per the Internal Model Principle.

    The filter state is maintained as two internal variables (x1, x2)
    representing the discretized second-order resonant system. The center
    frequency is updated every timestep from the measured roll rate.
    """

    def __init__(
        self,
        pid_config: PIDConfig = None,
        config: RepetitiveConfig = None,
        use_observations: bool = False,
    ):
        self.pid_config = pid_config or PIDConfig()
        self.config = config or RepetitiveConfig()
        self.use_observations = use_observations

        # Base GS-PID controller
        self._gs_pid = GainScheduledPIDController(
            self.pid_config, use_observations=use_observations
        )

        # Pre-compute reference effectiveness for gain scheduling
        q_ref = self.pid_config.q_ref
        self._ref_effectiveness = q_ref * np.tanh(q_ref / 200.0)

        self.reset()

    def reset(self):
        """Reset controller and resonant filter state."""
        self._gs_pid.reset()
        # Resonant filter state
        self._res_x1 = 0.0  # Filter state 1
        self._res_x2 = 0.0  # Filter state 2
        self._omega_est = 0.0  # Smoothed spin frequency estimate
        self._step_count = 0
        self._res_action = 0.0  # Last resonant action (diagnostics)

    @property
    def launch_detected(self):
        return self._gs_pid.launch_detected

    @launch_detected.setter
    def launch_detected(self, value):
        self._gs_pid.launch_detected = value

    @property
    def target_orient(self):
        return self._gs_pid.target_orient

    @target_orient.setter
    def target_orient(self, value):
        self._gs_pid.target_orient = value

    def _gain_scale(self, q: float) -> float:
        """Compute gain scaling factor from dynamic pressure."""
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1e-3:
            return 5.0
        scale = self._ref_effectiveness / effectiveness
        return float(np.clip(scale, 0.5, 5.0))

    def _update_resonant_filter(
        self, error_signal: float, omega: float, dt: float
    ) -> float:
        """Update the discretized resonant filter and return its output.

        The continuous-time transfer function is:
            H(s) = K_rc * s / (s^2 + 2*zeta*omega*s + omega^2)

        Discretized via Tustin (bilinear) transform:
            s = (2/T) * (z-1)/(z+1)

        We use state-space form for numerical stability:
            x1_dot = x2
            x2_dot = -omega^2 * x1 - 2*zeta*omega * x2 + error_signal
            output = K_rc * (x2)  [derivative of x1, approximating s*X(s)]

        Discretized with Euler for simplicity and robustness to
        time-varying omega.
        """
        cfg = self.config
        zeta = cfg.damping

        # State-space update (Euler integration)
        x1_dot = self._res_x2
        x2_dot = (
            -(omega**2) * self._res_x1
            - 2.0 * zeta * omega * self._res_x2
            + error_signal
        )

        self._res_x1 += x1_dot * dt
        self._res_x2 += x2_dot * dt

        # Anti-windup: clamp state magnitudes
        max_state = 100.0
        self._res_x1 = np.clip(self._res_x1, -max_state, max_state)
        self._res_x2 = np.clip(self._res_x2, -max_state, max_state)

        # Output: K_rc * x2 (the "velocity" state, which approximates s * X(s))
        return cfg.K_rc * self._res_x2

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action: GS-PID base + resonant correction.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        cfg = self.config

        # --- Run base GS-PID controller ---
        base_action = self._gs_pid.step(obs, info, dt)

        if not self._gs_pid.launch_detected:
            return base_action

        self._step_count += 1

        # --- Get roll rate and angle for resonant filter ---
        if self.use_observations:
            roll_angle_rad = obs[2] if len(obs) > 2 else 0.0
            roll_rate_rads = obs[3] if len(obs) > 3 else 0.0
            q = float(obs[5]) if len(obs) > 5 else 0.0
        else:
            roll_angle_rad = info.get("roll_angle_rad", 0.0)
            roll_rate_rads = np.radians(info.get("roll_rate_deg_s", 0.0))
            q = info.get("dynamic_pressure_Pa", 0.0)

        # --- Estimate spin frequency from roll rate ---
        omega_measured = abs(roll_rate_rads)

        # Exponential smoothing of spin frequency
        alpha = cfg.omega_smoothing
        self._omega_est = alpha * self._omega_est + (1.0 - alpha) * omega_measured

        # --- Check if spin frequency is in valid range ---
        if self._omega_est < cfg.min_omega or self._omega_est > cfg.max_omega:
            self._res_action = 0.0
            return base_action

        # --- Warmup period ---
        if self._step_count < cfg.warmup_steps:
            self._res_action = 0.0
            return base_action

        # --- Compute error signal for resonant filter ---
        # The error signal is the roll angle error (what the PID is trying to drive to zero)
        roll_angle_deg = np.degrees(roll_angle_rad)
        angle_error = roll_angle_deg - self._gs_pid.target_orient
        while angle_error > 180:
            angle_error -= 360
        while angle_error < -180:
            angle_error += 360

        # --- Update resonant filter ---
        res_output = self._update_resonant_filter(angle_error, self._omega_est, dt)

        # --- Gain schedule the resonant output ---
        scale = self._gain_scale(q)
        res_action = res_output * scale

        # Convert to normalized action (same scaling as GS-PID)
        res_action_norm = -res_action / self.pid_config.max_deflection

        self._res_action = float(res_action_norm)

        # --- Combine base + resonant ---
        total_action = float(base_action[0]) + res_action_norm

        # Clamp to [-1, 1]
        total_action = float(np.clip(total_action, -1.0, 1.0))

        return np.array([total_action], dtype=np.float32)
