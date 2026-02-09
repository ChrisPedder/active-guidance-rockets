#!/usr/bin/env python3
"""
Active Disturbance Rejection Control (ADRC) for Rocket Roll Stabilization

ADRC uses an Extended State Observer (ESO) to estimate the "total disturbance"
acting on the roll axis — including wind, model mismatch, and unmodeled dynamics —
and cancels it in the control law.

Architecture:
    1. Extended State Observer (ESO):
       - Estimates [roll_angle, roll_rate, total_disturbance]
       - Gains set by observer bandwidth omega_o (higher = faster tracking, more noise)

    2. Control law:
       - PD controller on angle/rate error + disturbance cancellation
       - u = (kp * angle_error + kd * rate_error - z3_disturbance) / b0
       - Gains set by controller bandwidth omega_c

    Key advantage over PID: the ESO estimates and cancels the disturbance in
    real time, rather than relying on integral action to slowly accumulate
    a correction.

References:
    - Han, J. "From PID to ADRC" IEEE Trans. Ind. Electronics, 2009
    - Roll-motion stabilizer for sounding rockets using ADRC (IEEE, 2023)

Usage:
    from adrc_controller import ADRCController, ADRCConfig

    controller = ADRCController(ADRCConfig(omega_c=15, omega_o=50))
    controller.reset()
    action = controller.step(obs, info, dt=0.01)
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class ADRCConfig:
    """ADRC controller configuration.

    Attributes:
        omega_c: Controller bandwidth (rad/s). Higher = more aggressive tracking.
                 Sets PD gains: kp = omega_c^2, kd = 2*omega_c.
        omega_o: Observer bandwidth (rad/s). Higher = faster disturbance estimation
                 but more noise sensitivity. Rule of thumb: omega_o = 3-5 * omega_c.
                 Sets ESO gains: beta1 = 3*omega_o, beta2 = 3*omega_o^2, beta3 = omega_o^3.
        b0: Control effectiveness estimate (rad/s^2 per normalized action).
            How much angular acceleration one unit of control input produces.
            If b0_per_pa is set, b0 is used as a fallback when dynamic pressure
            is unavailable.
        b0_per_pa: Control effectiveness per unit dynamic pressure (rad/s^2 per
                   action per Pa). When set, b0 is computed dynamically as
                   b0_per_pa * q * tanh(q/200), matching the environment's
                   speed-dependent effectiveness model.
        max_deflection: Max servo deflection (deg) for normalization to [-1, 1].
        use_observations: If True, read from obs array (noisy IMU). If False, read
                          from info dict (ground truth).
    """

    omega_c: float = 15.0  # Tuned: matches PID at 0 wind with b0=725
    omega_o: float = 50.0  # 3.3x omega_c, conservative for 100Hz control
    b0: float = 725.0  # Typical value for Estes Alpha at q=500 Pa
    b0_per_pa: Optional[float] = None  # Set by estimate_adrc_config()
    max_deflection: float = 30.0
    use_observations: bool = False


class ADRCController:
    """
    ADRC controller for rocket roll stabilization.

    The controller maintains an Extended State Observer (ESO) that tracks
    three states:
        z1: estimated roll angle (rad)
        z2: estimated roll rate (rad/s)
        z3: estimated total disturbance (rad/s^2)

    The control law cancels the estimated disturbance and applies PD control:
        u = (kp * (target - z1) + kd * (0 - z2) - z3) / b0
    """

    def __init__(self, config: ADRCConfig = None, b0_estimator=None):
        self.config = config or ADRCConfig()
        self.b0_estimator = b0_estimator
        self._compute_gains()
        self.reset()

    def _compute_gains(self):
        """Compute ESO and controller gains from bandwidths."""
        cfg = self.config

        # Controller gains (PD from bandwidth parameterization)
        self.kp = cfg.omega_c**2
        self.kd = 2.0 * cfg.omega_c

        # ESO gains (3rd-order observer from bandwidth parameterization)
        self.beta1 = 3.0 * cfg.omega_o
        self.beta2 = 3.0 * cfg.omega_o**2
        self.beta3 = cfg.omega_o**3

    def reset(self):
        """Reset controller and observer state for a new episode."""
        self.z1 = 0.0  # Estimated roll angle (rad)
        self.z2 = 0.0  # Estimated roll rate (rad/s)
        self.z3 = 0.0  # Estimated total disturbance (rad/s^2)
        self.target_angle = 0.0  # Target roll angle (rad)
        self.launch_detected = False
        self.prev_action = 0.0  # Previous control output (normalized)
        if self.b0_estimator is not None:
            self.b0_estimator.reset()

    def step(self, obs: np.ndarray, info: dict, dt: float = 0.01) -> np.ndarray:
        """
        Compute control action using ADRC.

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
            # IMU mode: roll angle from obs (not affected by gyro noise),
            # roll rate from info dict (noisy but current — bypasses
            # sensor_delay_steps which is an RL-specific feature).
            roll_angle = obs[2] if len(obs) > 2 else 0.0  # rad
            roll_rate = np.radians(
                info.get("roll_rate_deg_s", np.degrees(obs[3]) if len(obs) > 3 else 0.0)
            )

            if not self.launch_detected:
                self.launch_detected = True
                self.target_angle = roll_angle
                self.z1 = roll_angle
                self.z2 = roll_rate
        else:
            roll_angle = info.get("roll_angle_rad", 0.0)
            roll_rate = np.radians(info.get("roll_rate_deg_s", 0.0))
            accel = info.get("vertical_acceleration_ms2", 0.0)

            if not self.launch_detected:
                if accel > 20.0:
                    self.launch_detected = True
                    self.target_angle = roll_angle
                    self.z1 = roll_angle
                    self.z2 = roll_rate
                else:
                    return np.array([0.0], dtype=np.float32)

        # --- Effective b0 ---
        # When b0_per_pa is set, compute b0 dynamically from dynamic pressure
        # to match the environment's speed-dependent control effectiveness:
        #   effectiveness ∝ q * tanh(q/200)
        # This prevents the ESO from conflating model mismatch with actual
        # disturbance, which was the root cause of ADRC underperformance.
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
            # Clamp to minimum to prevent division by zero in control law;
            # fall back to fixed b0 when q is too low for aero control.
            b0_min = cfg.b0 * 0.01
            if b0_now < b0_min:
                b0_now = cfg.b0
        else:
            b0_now = cfg.b0

        # --- Online b0 identification (optional) ---
        # When a B0Estimator is attached, update it with the latest
        # (roll_accel, action) pair and use its estimate for b0_now.
        if self.b0_estimator is not None:
            if cfg.use_observations:
                roll_accel = float(obs[4]) if len(obs) > 4 else 0.0
            else:
                roll_accel = info.get("roll_acceleration_rad_s2", 0.0)
            b0_rls = self.b0_estimator.update(roll_accel, self.prev_action)
            if b0_rls > 0:
                b0_now = b0_rls

        # --- Extended State Observer (ESO) update ---
        # Observation error: difference between measured angle and estimated angle
        e_obs = roll_angle - self.z1

        # ESO state update (Euler integration)
        z1_dot = self.z2 + self.beta1 * e_obs
        z2_dot = self.z3 + self.beta2 * e_obs + b0_now * self.prev_action
        z3_dot = self.beta3 * e_obs

        self.z1 += z1_dot * dt
        self.z2 += z2_dot * dt
        self.z3 += z3_dot * dt

        # --- Control law ---
        angle_error = self.target_angle - self.z1

        # Normalize angle error to [-pi, pi]
        angle_error = (angle_error + np.pi) % (2 * np.pi) - np.pi

        rate_error = 0.0 - self.z2  # Target rate is zero

        # ADRC control: PD + disturbance cancellation
        # u0 is the desired acceleration (in the direction to reduce error)
        u0 = self.kp * angle_error + self.kd * rate_error

        # The control law cancels the estimated disturbance and commands
        # the desired acceleration via the plant input.
        # Plant convention: alpha = b0 * action + disturbance
        # where positive action = positive torque = positive acceleration.
        action = (u0 - self.z3) / b0_now

        # Clamp to [-1, 1]
        action = float(np.clip(action, -1.0, 1.0))

        self.prev_action = action

        return np.array([action], dtype=np.float32)


def estimate_adrc_config(
    airframe,
    rocket_config,
    omega_c: float = 15.0,
    omega_o: float = 50.0,
) -> ADRCConfig:
    """
    Estimate ADRC parameters from airframe physics.

    Computes b0 (control effectiveness in rad/s^2 per normalized action)
    from the airframe geometry, matching the approach in disturbance_observer.py.

    Args:
        airframe: RocketAirframe instance
        rocket_config: RocketConfig with physics settings
        omega_c: Controller bandwidth (rad/s)
        omega_o: Observer bandwidth (rad/s)

    Returns:
        ADRCConfig with estimated b0 and specified bandwidths
    """
    # Roll inertia at typical flight mass (dry + half propellant)
    # Use `or` to handle None values (attribute exists but is None)
    additional_mass = (getattr(rocket_config, "propellant_mass", None) or 0.012) / 2
    I_roll = airframe.get_roll_inertia(additional_mass)

    # Control effectiveness scales linearly with dynamic pressure.
    # Get effectiveness at a reference q to extract the per-Pa coefficient.
    ref_q = 500.0
    effectiveness_at_ref = airframe.get_control_effectiveness(
        ref_q,
        tab_chord_fraction=getattr(rocket_config, "tab_chord_fraction", None) or 0.25,
        tab_span_fraction=getattr(rocket_config, "tab_span_fraction", None) or 0.5,
        num_controlled_fins=getattr(rocket_config, "num_controlled_fins", None) or 2,
    )
    # effectiveness_at_ref is proportional to q, so per-Pa rate:
    effectiveness_per_pa = effectiveness_at_ref / ref_q

    max_deflection_rad = np.deg2rad(
        getattr(rocket_config, "max_tab_deflection", None) or 30.0
    )

    # b0_per_pa: angular acceleration per unit action per Pa (before tanh scaling)
    # Full b0 at runtime = b0_per_pa * q * tanh(q/200)
    b0_per_pa = effectiveness_per_pa * max_deflection_rad / I_roll

    # Also compute a typical b0 for the fixed fallback
    b0_typical = b0_per_pa * ref_q * np.tanh(ref_q / 200.0)

    return ADRCConfig(
        omega_c=omega_c,
        omega_o=omega_o,
        b0=b0_typical,
        b0_per_pa=b0_per_pa,
        max_deflection=getattr(rocket_config, "max_tab_deflection", None) or 30.0,
    )


if __name__ == "__main__":
    print("=" * 60)
    print("ADRC Controller Test")
    print("=" * 60)

    # Test with default config
    config = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=100.0)
    ctrl = ADRCController(config)

    print(
        f"\nConfig: omega_c={config.omega_c}, omega_o={config.omega_o}, b0={config.b0}"
    )
    print(f"PD gains: kp={ctrl.kp:.1f}, kd={ctrl.kd:.1f}")
    print(
        f"ESO gains: beta1={ctrl.beta1:.1f}, beta2={ctrl.beta2:.1f}, beta3={ctrl.beta3:.1f}"
    )

    # Simulate step response
    print("\n--- Step response (30 deg/s initial spin) ---")
    dt = 0.01
    roll_angle = 0.0
    roll_rate = np.radians(30.0)  # 30 deg/s initial spin

    info = {
        "roll_angle_rad": roll_angle,
        "roll_rate_deg_s": np.degrees(roll_rate),
        "vertical_acceleration_ms2": 50.0,
    }

    for step in range(200):
        obs = np.zeros(10)
        action = ctrl.step(obs, info, dt)

        # Simple dynamics: alpha = b0 * action (no disturbance)
        alpha = config.b0 * action[0]
        roll_rate += alpha * dt
        roll_angle += roll_rate * dt

        info = {
            "roll_angle_rad": roll_angle,
            "roll_rate_deg_s": np.degrees(roll_rate),
            "vertical_acceleration_ms2": 50.0,
        }

        if step % 40 == 0:
            print(
                f"Step {step:3d}: angle={np.degrees(roll_angle):+.1f} deg, "
                f"rate={np.degrees(roll_rate):+.1f} deg/s, "
                f"action={action[0]:+.3f}, z3={ctrl.z3:+.2f}"
            )

    print(
        f"\nFinal: angle={np.degrees(roll_angle):+.2f} deg, "
        f"rate={np.degrees(roll_rate):+.2f} deg/s"
    )
