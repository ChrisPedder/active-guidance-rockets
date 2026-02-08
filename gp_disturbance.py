#!/usr/bin/env python3
"""
Sparse Online GP Disturbance Model with Uncertainty Gating

Replaces feedforward disturbance model with a sparse online Gaussian Process.
The GP provides calibrated (mean, variance) predictions — variance modulates
feedforward gain via a sigmoid gate. High uncertainty -> conservative (pure
GS-PID). Low uncertainty -> full feedforward.

This addresses the failure mode of ADRC+FF at 3+ m/s wind: the analytical
sinusoidal estimator degrades under complex multi-frequency gusts, and its
point-estimate feedforward can amplify errors. The GP's calibrated uncertainty
directly prevents catastrophic feedforward errors.

GP Model:
    Input: x = [cos(roll_angle), sin(roll_angle), q/1000, roll_rate/10]
    Output: z3 (ADRC disturbance estimate)
    Kernel: RBF (automatic relevance determination via input scaling)

The GP uses a fixed-size set of inducing points (budgeted GP) to keep
computation O(M^2) per step, where M is the budget size (~50 points).
New observations replace the oldest inducing point when the budget is full.

Uncertainty Gating:
    gate = sigmoid((sigma_threshold - sigma) / sigma_scale)
    u_ff = gate * K_ff * (-mean / b0)

When sigma > sigma_threshold, gate -> 0 (pure base controller).
When sigma < sigma_threshold, gate -> 1 (full feedforward).

Usage:
    from gp_disturbance import GPFeedforwardController, GPDisturbanceConfig

    gs_pid = GainScheduledPIDController(PIDConfig())
    gp_config = GPDisturbanceConfig(budget_size=50, K_ff=0.5)
    controller = GPFeedforwardController(gs_pid, gp_config)
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional

from pid_controller import PIDConfig, GainScheduledPIDController


@dataclass
class GPDisturbanceConfig:
    """Configuration for GP disturbance model.

    Attributes:
        budget_size: Maximum number of inducing points. Larger = better
                     approximation but O(M^2) cost. 50 is a good balance
                     for 100 Hz control.
        length_scales: RBF kernel length scales for each input dimension
                       [cos(roll_angle), sin(roll_angle), q/1000, roll_rate/10].
                       Controls how "far" in input space the GP generalizes.
        signal_variance: Prior variance of the GP output (amplitude^2 of
                         the RBF kernel). Should be calibrated to typical z3
                         magnitude squared.
        noise_variance: Observation noise variance. Accounts for ESO
                        estimation error in z3. Higher = less trust in
                        individual observations.
        K_ff: Feedforward gain (0 to 1).
        sigma_threshold: Uncertainty threshold for gating. When GP std dev
                         exceeds this, feedforward is attenuated.
        sigma_scale: Controls sharpness of the sigmoid gate. Smaller =
                     sharper transition.
        warmup_steps: Steps before feedforward activates.
        update_interval: Update GP every N steps (for computational savings).
        use_adrc_z3: If True, use ADRC's z3 as training target (requires
                     ADRC base controller). If False, use roll acceleration.
    """

    budget_size: int = 50
    length_scales: tuple = (1.0, 1.0, 1.0, 1.0)
    signal_variance: float = 100.0
    noise_variance: float = 10.0
    K_ff: float = 0.5
    sigma_threshold: float = 5.0
    sigma_scale: float = 2.0
    warmup_steps: int = 50
    update_interval: int = 3


class SparseGP:
    """Budgeted sparse Gaussian Process regression.

    Maintains a fixed-size set of inducing points. When the budget is full,
    new observations replace the oldest point. Uses the kernel matrix
    directly (no Cholesky — we recompute each time for robustness to
    point replacement).

    The kernel is RBF (squared exponential) with per-dimension length scales:
        k(x, x') = signal_var * exp(-0.5 * sum((x_i - x'_i)^2 / l_i^2))
    """

    def __init__(
        self,
        input_dim: int,
        budget_size: int,
        length_scales: np.ndarray,
        signal_variance: float,
        noise_variance: float,
    ):
        self.input_dim = input_dim
        self.budget_size = budget_size
        self.length_scales = np.array(length_scales, dtype=np.float64)
        self.signal_variance = signal_variance
        self.noise_variance = noise_variance

        # Inducing point storage
        self._X = np.zeros((budget_size, input_dim))  # Input locations
        self._y = np.zeros(budget_size)  # Target values
        self._n_points = 0  # Current number of stored points
        self._insert_idx = 0  # Next insertion index (circular buffer)

        # Cached kernel matrix and its inverse (recomputed on update)
        self._K_inv = None
        self._K_inv_y = None
        self._dirty = True  # Whether cache needs recomputing

    def reset(self):
        """Clear all stored data points."""
        self._X[:] = 0.0
        self._y[:] = 0.0
        self._n_points = 0
        self._insert_idx = 0
        self._K_inv = None
        self._K_inv_y = None
        self._dirty = True

    def _rbf_kernel(self, X1: np.ndarray, X2: np.ndarray) -> np.ndarray:
        """Compute RBF kernel matrix between X1 and X2.

        Args:
            X1: (N1, D) array
            X2: (N2, D) array

        Returns:
            (N1, N2) kernel matrix
        """
        # Scale inputs by length scales
        X1_scaled = X1 / self.length_scales
        X2_scaled = X2 / self.length_scales

        # Squared distances
        sq_dist = (
            np.sum(X1_scaled**2, axis=1, keepdims=True)
            + np.sum(X2_scaled**2, axis=1, keepdims=True).T
            - 2 * X1_scaled @ X2_scaled.T
        )
        # Clamp negative values from numerical error
        sq_dist = np.maximum(sq_dist, 0.0)

        return self.signal_variance * np.exp(-0.5 * sq_dist)

    def add_point(self, x: np.ndarray, y: float):
        """Add an observation to the inducing set.

        If the budget is full, replaces the oldest point (circular buffer).
        """
        self._X[self._insert_idx] = x
        self._y[self._insert_idx] = y
        self._insert_idx = (self._insert_idx + 1) % self.budget_size
        self._n_points = min(self._n_points + 1, self.budget_size)
        self._dirty = True

    def _recompute_cache(self):
        """Recompute kernel matrix inverse and cached predictions."""
        if self._n_points == 0:
            self._K_inv = None
            self._K_inv_y = None
            self._dirty = False
            return

        n = self._n_points
        X = self._X[:n]
        y = self._y[:n]

        K = self._rbf_kernel(X, X) + self.noise_variance * np.eye(n)

        # Regularize for numerical stability
        K += 1e-6 * np.eye(n)

        try:
            self._K_inv = np.linalg.inv(K)
            self._K_inv_y = self._K_inv @ y
        except np.linalg.LinAlgError:
            # If inversion fails, fall back to pseudoinverse
            self._K_inv = np.linalg.pinv(K)
            self._K_inv_y = self._K_inv @ y

        self._dirty = False

    def predict(self, x: np.ndarray) -> tuple:
        """Predict mean and variance at input x.

        Args:
            x: (D,) input vector

        Returns:
            (mean, variance) tuple
        """
        if self._n_points == 0:
            return 0.0, self.signal_variance

        if self._dirty:
            self._recompute_cache()

        n = self._n_points
        X = self._X[:n]
        x_2d = x.reshape(1, -1)

        # Cross-kernel between x and inducing points
        k_star = self._rbf_kernel(x_2d, X)[0]  # (n,)

        # Mean prediction
        mean = k_star @ self._K_inv_y

        # Variance prediction
        k_ss = self.signal_variance  # k(x, x)
        v = k_star @ self._K_inv @ k_star
        variance = max(k_ss - v, 1e-6)  # Clamp to positive

        return float(mean), float(variance)


class GPFeedforwardController:
    """GS-PID controller with GP-based uncertainty-gated feedforward.

    Wraps GainScheduledPIDController and adds an online GP that models the
    disturbance as a function of roll state and dynamic pressure. The GP's
    variance modulates the feedforward gain: high uncertainty = conservative,
    low uncertainty = full feedforward.
    """

    def __init__(
        self,
        pid_config: PIDConfig = None,
        gp_config: GPDisturbanceConfig = None,
        use_observations: bool = False,
    ):
        self.pid_config = pid_config or PIDConfig()
        self.gp_config = gp_config or GPDisturbanceConfig()
        self.use_observations = use_observations

        # Base controller
        self._gs_pid = GainScheduledPIDController(
            self.pid_config,
            use_observations=use_observations,
        )

        # Reference effectiveness for gain scheduling feedforward
        q_ref = self.pid_config.q_ref
        self._ref_effectiveness = q_ref * np.tanh(q_ref / 200.0)

        # GP model
        gpc = self.gp_config
        self._gp = SparseGP(
            input_dim=4,
            budget_size=gpc.budget_size,
            length_scales=np.array(gpc.length_scales),
            signal_variance=gpc.signal_variance,
            noise_variance=gpc.noise_variance,
        )

        self.reset()

    def reset(self):
        """Reset controller and GP state."""
        self._gs_pid.reset()
        self._gp.reset()
        self._step_count = 0
        self._ff_action = 0.0
        self._gate_value = 0.0
        self._gp_mean = 0.0
        self._gp_std = 0.0
        self._prev_roll_rate = 0.0
        self._prev_action = 0.0

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

    def _build_gp_input(
        self, roll_angle: float, roll_rate: float, q: float
    ) -> np.ndarray:
        """Build the GP input feature vector.

        Inputs are scaled so that each dimension has roughly unit variance:
            [cos(roll_angle), sin(roll_angle), q/1000, roll_rate/10]
        """
        return np.array(
            [
                np.cos(roll_angle),
                np.sin(roll_angle),
                q / 1000.0,
                roll_rate / 10.0,
            ]
        )

    def _sigmoid_gate(self, sigma: float) -> float:
        """Compute the uncertainty gate value.

        gate = sigmoid((threshold - sigma) / scale)

        Returns 1.0 when sigma << threshold (confident),
        returns 0.0 when sigma >> threshold (uncertain).
        """
        gpc = self.gp_config
        z = (gpc.sigma_threshold - sigma) / gpc.sigma_scale
        # Numerically stable sigmoid
        if z > 20:
            return 1.0
        elif z < -20:
            return 0.0
        return 1.0 / (1.0 + np.exp(-z))

    def _estimate_disturbance(
        self, roll_rate: float, action: float, q: float, dt: float
    ) -> float:
        """Estimate the current disturbance from roll dynamics.

        Uses the simple model: roll_accel = b0 * action + disturbance
        where roll_accel is estimated from rate difference.

        This gives us a training signal for the GP without requiring
        ADRC's ESO.
        """
        # Estimate roll acceleration from rate difference
        roll_accel = (roll_rate - self._prev_roll_rate) / dt

        # Estimate b0 from physics
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1.0:
            return 0.0
        b0_est = self._ref_effectiveness / effectiveness * 130.0  # rough scale

        # Disturbance = observed_accel - predicted_accel
        return roll_accel - b0_est * action

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """Compute control action: GS-PID + GP-gated feedforward.

        Args:
            obs: Observation from environment
            info: Info dict from environment
            dt: Timestep in seconds

        Returns:
            Action in normalized [-1, 1] range
        """
        gpc = self.gp_config

        # --- Run base GS-PID controller ---
        base_action = self._gs_pid.step(obs, info, dt)

        if not self._gs_pid.launch_detected:
            return base_action

        self._step_count += 1

        # --- Read state ---
        if self.use_observations:
            # IMU mode: roll rate from info dict (noisy but current —
            # bypasses sensor_delay_steps).
            roll_angle = obs[2] if len(obs) > 2 else 0.0
            roll_rate = np.radians(
                info.get("roll_rate_deg_s", np.degrees(obs[3]) if len(obs) > 3 else 0.0)
            )
            # IMU mode: prefer info dict for current q (bypasses sensor delay)
            q = info.get("dynamic_pressure_Pa", float(obs[5]) if len(obs) > 5 else 0.0)
        else:
            roll_angle = info.get("roll_angle_rad", 0.0)
            roll_rate = np.radians(info.get("roll_rate_deg_s", 0.0))
            q = info.get("dynamic_pressure_Pa", 0.0)

        # --- Build GP input ---
        gp_input = self._build_gp_input(roll_angle, roll_rate, q)

        # --- Update GP with disturbance estimate ---
        if self._step_count > 2 and self._step_count % gpc.update_interval == 0:
            dist_est = self._estimate_disturbance(
                roll_rate,
                self._prev_action,
                q,
                dt * gpc.update_interval,
            )
            self._gp.add_point(gp_input, dist_est)

        self._prev_roll_rate = roll_rate
        self._prev_action = float(base_action[0])

        # --- Warmup: don't apply feedforward ---
        if self._step_count < gpc.warmup_steps:
            self._ff_action = 0.0
            self._gate_value = 0.0
            return base_action

        # --- GP prediction ---
        mean, variance = self._gp.predict(gp_input)
        sigma = np.sqrt(variance)
        self._gp_mean = mean
        self._gp_std = sigma

        # --- Uncertainty gate ---
        gate = self._sigmoid_gate(sigma)
        self._gate_value = gate

        # --- Compute feedforward ---
        # Scale by control effectiveness
        effectiveness = q * np.tanh(q / 200.0)
        if effectiveness < 1.0:
            self._ff_action = 0.0
            return base_action

        # Feedforward: cancel predicted disturbance, gated by uncertainty
        # Normalize to action units using the gain scheduling reference
        scale = self._ref_effectiveness / effectiveness
        ff_action = -gpc.K_ff * gate * mean * scale / 130.0  # rough b0 normalization

        # Clamp feedforward magnitude to prevent wild swings
        ff_action = float(np.clip(ff_action, -0.5, 0.5))
        self._ff_action = ff_action

        # --- Combine and clamp ---
        total_action = float(base_action[0]) + ff_action
        total_action = float(np.clip(total_action, -1.0, 1.0))

        return np.array([total_action], dtype=np.float32)
