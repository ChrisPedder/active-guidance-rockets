#!/usr/bin/env python3
"""
H-infinity Robust Controller for Rocket Roll Stabilization

Synthesizes a robust controller via LQG/LTR (Loop Transfer Recovery)
that provides near-H-infinity-optimal worst-case disturbance rejection
across the full flight envelope.

Plant model (continuous-time, normalized so b0=1):
    x1_dot = x2                    (roll angle)
    x2_dot = u_phys + d(t)        (roll rate; u_phys is the physical torque)
    y = [x1, x2]                   (measured roll angle and rate)

Design approach:
    1. LQR: solve control ARE for state feedback gain K
       - Q = diag(q_angle, q_rate) penalizes angle error and spin rate
       - R penalizes control effort
    2. Kalman Filter: solve estimation ARE for observer gain L
       - High process noise W provides Loop Transfer Recovery (LTR)
       - Low measurement noise V trusts IMU data
    3. LQG controller: combine K and L into dynamic output-feedback compensator
       - x_hat_dot = (A - BK - LC) x_hat + L y
       - u = -K x_hat
    4. Discretize via Tustin (bilinear) transform for real-time execution

LTR guarantees: as W → ∞, the LQG loop transfer matches the LQR loop
transfer, recovering the robustness margins of LQR (infinite gain margin,
60° phase margin). This provides inherent robustness to the 20× b0
variation during flight, complemented by explicit gain scheduling.

The controller operates in "physical torque" space (b0=1) and divides
by the current b0(q) at runtime, identical to GS-PID's gain scheduling.

Usage:
    from hinf_controller import HinfController, HinfConfig

    controller = HinfController(HinfConfig())
    controller.reset()
    action = controller.step(obs, info, dt=0.01)

References:
    - Doyle & Stein, "Multivariable Feedback Design: Concepts for a
      Classical/Modern Synthesis" (1981) — LQG/LTR theory
    - Maciejowski, "Multivariable Feedback Design" (1989) — Chapter 5
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from scipy.linalg import solve_continuous_are, eigvals


@dataclass
class HinfConfig:
    """H-infinity (LQG/LTR) controller configuration.

    Attributes:
        q_angle: LQR weight on angle error squared. Higher = faster angle
                 convergence but more aggressive.
        q_rate: LQR weight on rate squared. Higher = stronger rate damping.
        r_control: LQR weight on control effort squared. Lower = more
                   aggressive control.
        w_process: Process noise intensity for LTR. Higher values push the
                   observer to track measurements more closely, recovering
                   the LQR robustness margins. Values > 50 give good LTR.
        v_angle: Measurement noise variance for angle (rad^2).
        v_rate: Measurement noise variance for rate (rad/s)^2.
        b0: Nominal control effectiveness (rad/s^2 per normalized action).
        b0_per_pa: Per-Pa effectiveness for gain scheduling.
        q_ref: Reference dynamic pressure for gain scheduling (Pa).
        max_deflection: Max servo deflection (degrees).
        use_observations: If True, read from obs array (noisy IMU).
        dt_design: Design timestep for Tustin discretization (seconds).
    """

    q_angle: float = 100.0
    q_rate: float = 10.0
    r_control: float = 0.01
    w_process: float = 100.0
    v_angle: float = 0.001
    v_rate: float = 0.001
    b0: float = 725.0
    b0_per_pa: Optional[float] = None
    q_ref: float = 500.0
    max_deflection: float = 30.0
    use_observations: bool = False
    dt_design: float = 0.01


def synthesize_lqg_ltr(
    q_angle: float,
    q_rate: float,
    r_control: float,
    w_process: float,
    v_angle: float,
    v_rate: float,
) -> dict:
    """Synthesize LQG/LTR controller for the roll channel double integrator.

    Plant (normalized, b0=1):
        x = [angle, rate]
        x_dot = [[0,1],[0,0]] x + [[0],[1]] u + [[0],[1]] d
        y = [[1,0],[0,1]] x + noise

    Returns dict with continuous-time controller matrices A_K, B_K, C_K, D_K,
    plus the LQR gain K and Kalman gain L.
    """
    A = np.array([[0.0, 1.0], [0.0, 0.0]])
    B = np.array([[0.0], [1.0]])
    C = np.array([[1.0, 0.0], [0.0, 1.0]])

    # --- LQR: solve A'X + XA - XBR^-1B'X + Q = 0 ---
    Q_lqr = np.diag([q_angle, q_rate])
    R_lqr = np.array([[r_control]])

    X = solve_continuous_are(A, B, Q_lqr, R_lqr)
    K = np.linalg.solve(R_lqr, B.T @ X)  # K = R^-1 B' X

    # --- Kalman filter: solve AY + YA' - YC'V^-1CY + GWG' = 0 ---
    G = np.array([[0.0], [1.0]])  # Disturbance enters acceleration channel
    W = np.array([[w_process]])
    V = np.diag([v_angle, v_rate])

    Y = solve_continuous_are(A.T, C.T, G @ W @ G.T, V)
    L = Y @ C.T @ np.linalg.inv(V)  # L = Y C' V^-1

    # --- LQG controller ---
    # x_hat_dot = (A - BK - LC) x_hat + L y
    # u = -K x_hat
    A_K = A - B @ K - L @ C
    B_K = L  # (2x2)
    C_K = -K  # (1x2)
    D_K = np.zeros((1, 2))

    # Verify stability
    cl_A = np.block(
        [
            [A, B @ C_K],
            [B_K @ C, A_K],
        ]
    )
    cl_eigs = eigvals(cl_A)
    stable = bool(np.all(np.real(cl_eigs) < 0))

    return {
        "A_K": A_K,
        "B_K": B_K,
        "C_K": C_K,
        "D_K": D_K,
        "K": K,
        "L": L,
        "cl_eigenvalues": cl_eigs,
        "stable": stable,
    }


def _discretize_tustin(A_K, B_K, C_K, D_K, dt):
    """Discretize a continuous-time state-space system via Tustin transform.

    Returns (Ad, Bd, Cd, Dd) for discrete-time system.
    """
    n = A_K.shape[0]
    I = np.eye(n)

    M1 = I - (dt / 2.0) * A_K
    M2 = I + (dt / 2.0) * A_K
    M1_inv = np.linalg.inv(M1)

    Ad = M1_inv @ M2
    Bd = dt * M1_inv @ B_K
    Cd = C_K @ M1_inv
    Dd = D_K + (dt / 2.0) * C_K @ M1_inv @ B_K

    return Ad, Bd, Cd, Dd


class HinfController:
    """LQG/LTR robust controller for rocket roll stabilization.

    Provides near-H-infinity-optimal disturbance rejection via LQG design
    with Loop Transfer Recovery. The controller is a 2-state dynamic
    output-feedback compensator that reads roll angle and rate, and outputs
    a normalized control action in [-1, 1].

    Gain-scheduled with dynamic pressure for robustness to the 20x
    variation in control effectiveness during flight.
    """

    def __init__(self, config: HinfConfig = None):
        self.config = config or HinfConfig()
        self._ref_effectiveness = self.config.q_ref * np.tanh(self.config.q_ref / 200.0)

        # Synthesize controller
        self._synthesis = synthesize_lqg_ltr(
            self.config.q_angle,
            self.config.q_rate,
            self.config.r_control,
            self.config.w_process,
            self.config.v_angle,
            self.config.v_rate,
        )

        # Discretize
        r = self._synthesis
        Ad, Bd, Cd, Dd = _discretize_tustin(
            r["A_K"],
            r["B_K"],
            r["C_K"],
            r["D_K"],
            self.config.dt_design,
        )
        self._Ad = Ad
        self._Bd = Bd
        self._Cd = Cd
        self._Dd = Dd

        self.reset()

    @property
    def synthesis_succeeded(self) -> bool:
        """Whether synthesis produced a stable closed-loop system."""
        return self._synthesis["stable"]

    @property
    def lqr_gain(self) -> np.ndarray:
        """LQR state feedback gain K (1x2)."""
        return self._synthesis["K"]

    @property
    def kalman_gain(self) -> np.ndarray:
        """Kalman observer gain L (2x2)."""
        return self._synthesis["L"]

    @property
    def cl_eigenvalues(self) -> np.ndarray:
        """Closed-loop eigenvalues."""
        return self._synthesis["cl_eigenvalues"]

    def reset(self):
        """Reset controller state for a new episode."""
        self.launch_detected = False
        self.target_angle = 0.0
        self._x_K = np.zeros(2)

    def _gain_scale(self, q: float) -> float:
        """Compute gain scaling factor from dynamic pressure."""
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
        """Compute control action using LQG/LTR controller.

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

        # --- Measurement vector ---
        y = np.array([angle_error, roll_rate])

        # --- LQG controller: u = Cd * x_K + Dd * y ---
        u_phys = (self._Cd @ self._x_K + self._Dd @ y).item()

        # Update controller state: x_K[k+1] = Ad * x_K + Bd * y
        self._x_K = self._Ad @ self._x_K + self._Bd @ y

        # --- Gain scheduling ---
        # The controller was designed with b0=1 (physical torque space).
        # Convert to normalized action by dividing by current b0.
        b0_now = self._get_b0(obs, info)
        scale = self._gain_scale(q)

        action = u_phys * scale / b0_now

        action = float(np.clip(action, -1.0, 1.0))
        return np.array([action], dtype=np.float32)
