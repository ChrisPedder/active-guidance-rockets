#!/usr/bin/env python3
"""
Fourier-Domain Adaptive Disturbance Model for Wind Rejection

Learns a compact Fourier basis decomposition of the disturbance online using
L1-regularized Recursive Least Squares (RLS) on ADRC's z3 estimate. Unlike
the single-frequency sinusoidal estimator in wind_feedforward.py, this
captures the full multi-frequency gust structure from the wind model's
dual-harmonic gusts.

Candidate frequencies:
    - DC (constant offset)
    - Spin frequency (|roll_rate|) and its 2nd harmonic
    - Gust frequencies: 0.5-4.6 Hz covering f and 2.3f from wind_model.py
    - Cross terms between spin and gust frequencies

The Fourier basis makes spectral structure explicit, reducing the learning
problem dramatically compared to the GRU approach (which had val loss 0.31).

Algorithm:
    1. At each step, construct feature vector:
       phi = [1, cos(omega_spin*t), sin(omega_spin*t),
              cos(2*omega_spin*t), sin(2*omega_spin*t),
              cos(omega_gust*t), sin(omega_gust*t), ...]

    2. Update weights via exponential-forgetting RLS:
       K = P * phi / (lambda + phi^T * P * phi)
       e = z3 - phi^T * w
       w = w + K * e
       P = (P - K * phi^T * P) / lambda

    3. L1 regularization via soft-thresholding after each update
       to maintain sparsity and prevent overfitting.

    4. Predict disturbance 1-5 steps ahead using the Fourier model
       and cancel via feedforward.

Usage:
    from fourier_adaptive import FourierAdaptiveADRC, FourierAdaptiveConfig

    base_config = ADRCConfig(omega_c=15, omega_o=50, b0=100, b0_per_pa=0.5)
    fourier_config = FourierAdaptiveConfig(n_gust_freqs=4, K_ff=0.5)
    controller = FourierAdaptiveADRC(base_config, fourier_config)
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, List

from adrc_controller import ADRCController, ADRCConfig, estimate_adrc_config


@dataclass
class FourierAdaptiveConfig:
    """Configuration for Fourier-domain adaptive disturbance model.

    Attributes:
        n_gust_freqs: Number of gust frequency candidates to include in the
                      basis. Each adds a cos/sin pair. Frequencies span
                      0.5-4.6 Hz to cover the wind model's f and 2.3f.
        include_spin_harmonics: Number of spin frequency harmonics (1 = just
                                fundamental, 2 = fundamental + 2nd harmonic).
        K_ff: Feedforward gain (0 to 1). How aggressively to cancel the
              predicted disturbance. Start conservative.
        rls_forgetting: RLS forgetting factor. Closer to 1 = longer memory.
        l1_lambda: L1 regularization strength for weight sparsity. Higher
                   = more aggressive pruning of unused frequencies.
        warmup_steps: Steps before feedforward activates.
        predict_steps: How many steps ahead to predict (for phase advance).
        omega_spin_smoothing: Smoothing for spin frequency estimate.
    """

    n_gust_freqs: int = 4
    include_spin_harmonics: int = 2
    K_ff: float = 0.5
    rls_forgetting: float = 0.995
    l1_lambda: float = 0.001
    warmup_steps: int = 50
    predict_steps: int = 3
    omega_spin_smoothing: float = 0.9


class FourierAdaptiveADRC:
    """ADRC controller with Fourier-domain adaptive feedforward.

    Wraps ADRCController and learns a multi-frequency decomposition of the
    disturbance using ADRC's z3. The Fourier model predicts the disturbance
    ahead by predict_steps timesteps, enabling phase-advance cancellation.
    """

    def __init__(
        self,
        adrc_config: ADRCConfig = None,
        fourier_config: FourierAdaptiveConfig = None,
    ):
        self.adrc = ADRCController(adrc_config)
        self.config = adrc_config or ADRCConfig()
        self.fourier_config = fourier_config or FourierAdaptiveConfig()

        # Pre-compute gust frequency candidates (Hz -> rad/s)
        fc = self.fourier_config
        if fc.n_gust_freqs > 0:
            # Span 0.5-4.6 Hz (covers f and 2.3f from wind model)
            self._gust_freqs_hz = np.linspace(0.5, 4.6, fc.n_gust_freqs)
        else:
            self._gust_freqs_hz = np.array([])
        self._gust_freqs_rads = 2 * np.pi * self._gust_freqs_hz

        # Feature dimension: 1 (DC) + 2*harmonics (spin) + 2*n_gust (gust cos/sin)
        self._n_features = 1 + 2 * fc.include_spin_harmonics + 2 * fc.n_gust_freqs

        self.reset()

    def reset(self):
        """Reset controller, observer, and Fourier estimator."""
        self.adrc.reset()
        fc = self.fourier_config
        n = self._n_features

        # RLS state
        self._weights = np.zeros(n)
        self._P = np.eye(n) * 100.0  # Covariance (start with high uncertainty)

        # Time tracking
        self._time = 0.0
        self._step_count = 0
        self._omega_spin_est = 0.0

        # Diagnostics
        self._ff_action = 0.0
        self._prediction_error = 0.0

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

    @property
    def weights(self) -> np.ndarray:
        """Current Fourier coefficient weights."""
        return self._weights.copy()

    @property
    def active_features(self) -> int:
        """Number of non-zero (active) features."""
        return int(np.sum(np.abs(self._weights) > 1e-6))

    def _build_features(self, t: float, omega_spin: float) -> np.ndarray:
        """Build the Fourier feature vector at time t.

        Features:
            [1, cos(omega*t), sin(omega*t), cos(2*omega*t), sin(2*omega*t),
             cos(f1*t), sin(f1*t), cos(f2*t), sin(f2*t), ...]
        """
        fc = self.fourier_config
        phi = np.zeros(self._n_features)

        idx = 0
        # DC term
        phi[idx] = 1.0
        idx += 1

        # Spin frequency harmonics
        for h in range(1, fc.include_spin_harmonics + 1):
            omega_h = h * omega_spin
            phi[idx] = np.cos(omega_h * t)
            phi[idx + 1] = np.sin(omega_h * t)
            idx += 2

        # Gust frequency terms
        for gf in self._gust_freqs_rads:
            phi[idx] = np.cos(gf * t)
            phi[idx + 1] = np.sin(gf * t)
            idx += 2

        return phi

    def _update_rls(self, phi: np.ndarray, target: float):
        """Update RLS weights with new observation.

        Uses standard exponential-forgetting RLS with optional L1
        regularization (soft-thresholding) for sparsity.
        """
        fc = self.fourier_config
        lam = fc.rls_forgetting

        # Prediction error
        prediction = phi @ self._weights
        error = target - prediction
        self._prediction_error = error

        # RLS update
        P_phi = self._P @ phi
        denom = lam + phi @ P_phi
        if abs(denom) < 1e-10:
            return  # Skip if degenerate

        K = P_phi / denom
        self._weights += K * error
        self._P = (self._P - np.outer(K, P_phi)) / lam

        # Bound P to prevent numerical explosion
        diag_max = 1e6
        np.clip(
            np.diag(self._P), 0, diag_max, out=self._P[np.diag_indices_from(self._P)]
        )

        # L1 regularization: soft-thresholding
        if fc.l1_lambda > 0:
            thresh = fc.l1_lambda
            # Don't regularize the DC term (index 0) — let it absorb bias
            self._weights[1:] = np.sign(self._weights[1:]) * np.maximum(
                np.abs(self._weights[1:]) - thresh, 0.0
            )

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action: ADRC base + Fourier feedforward.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        fc = self.fourier_config
        cfg = self.config

        # --- Run base ADRC controller ---
        base_action = self.adrc.step(obs, info, dt)

        if not self.adrc.launch_detected:
            return base_action

        self._step_count += 1
        self._time += dt

        # --- Get roll rate for spin frequency estimate ---
        if cfg.use_observations:
            # IMU mode: roll rate from info dict (noisy but current —
            # bypasses sensor_delay_steps).
            roll_rate_rads = np.radians(
                info.get("roll_rate_deg_s", np.degrees(obs[3]) if len(obs) > 3 else 0.0)
            )
        else:
            roll_rate_rads = np.radians(info.get("roll_rate_deg_s", 0.0))

        # Smooth spin frequency estimate
        omega_measured = abs(roll_rate_rads)
        alpha = fc.omega_spin_smoothing
        self._omega_spin_est = (
            alpha * self._omega_spin_est + (1.0 - alpha) * omega_measured
        )

        # --- Build Fourier features at current time ---
        phi = self._build_features(self._time, self._omega_spin_est)

        # --- Update RLS with ADRC's z3 as target ---
        self._update_rls(phi, self.adrc.z3)

        # --- Warmup: don't apply feedforward yet ---
        if self._step_count < fc.warmup_steps:
            self._ff_action = 0.0
            return base_action

        # --- Predict disturbance at future time ---
        t_future = self._time + fc.predict_steps * dt
        phi_future = self._build_features(t_future, self._omega_spin_est)
        predicted_disturbance = phi_future @ self._weights

        # --- Get b0 for scaling feedforward ---
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

        # --- Compute feedforward action ---
        ff_action = -fc.K_ff * predicted_disturbance / b0_now
        self._ff_action = float(ff_action)

        # --- Combine and clamp ---
        total_action = float(base_action[0]) + ff_action
        total_action = float(np.clip(total_action, -1.0, 1.0))

        return np.array([total_action], dtype=np.float32)

    @property
    def disturbance_estimate(self) -> float:
        """Current disturbance prediction (rad/s^2)."""
        phi = self._build_features(self._time, self._omega_spin_est)
        return float(phi @ self._weights)

    @property
    def rls_prediction_error(self) -> float:
        """Most recent RLS prediction error."""
        return self._prediction_error
