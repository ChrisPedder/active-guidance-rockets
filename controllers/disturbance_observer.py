#!/usr/bin/env python3
"""
Disturbance Observer for Rocket Roll Control

Estimates external disturbance torque from the discrepancy between
expected and actual roll dynamics. This makes wind disturbances
observable to the RL policy, enabling conditional behavior:
- Low disturbance -> SAC stays passive, PID handles control
- High disturbance -> SAC actively contributes corrections

Physics basis:
    tau_total = I * alpha
    tau_total = tau_control + tau_damping + tau_disturbance
    tau_disturbance = I * alpha - tau_control - tau_damping

The disturbance estimate reveals external forces (wind) that the
policy cannot directly observe but can learn to react to.

Usage:
    dob = DisturbanceObserver(
        I_roll=0.0001,
        control_effectiveness=0.001,
        damping_coeff=0.0005,
        filter_alpha=0.1
    )

    disturbance = dob.update(
        roll_rate=obs[3],
        roll_accel=obs[4],
        action=action[0],
        dynamic_pressure=info['dynamic_pressure_Pa'],
        velocity=info['vertical_velocity_ms']
    )
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class DOBConfig:
    """Configuration for Disturbance Observer"""

    # Physics parameters (estimated from rocket/environment)
    I_roll: float = 0.0001  # Roll moment of inertia (kg*m^2)
    control_effectiveness: float = 0.001  # Torque per unit action per Pa
    damping_coeff: float = 0.0005  # Aerodynamic damping coefficient

    # Filter parameters
    filter_alpha: float = 0.1  # Low-pass filter coefficient (0-1, lower = smoother)

    # Normalization for observation space
    max_disturbance: float = 0.01  # Expected max disturbance torque (N*m)


class DisturbanceObserver:
    """
    Estimates external disturbance torque acting on rocket roll axis.

    Uses a simple physics-based observer that compares expected
    roll acceleration (from control + damping) with actual acceleration.
    The discrepancy reveals external disturbances like wind.

    The estimate is low-pass filtered to reduce noise from:
    - IMU measurement noise
    - Model mismatch
    - High-frequency dynamics
    """

    def __init__(self, config: DOBConfig = None):
        """
        Initialize disturbance observer.

        Args:
            config: DOBConfig with physics and filter parameters
        """
        self.config = config or DOBConfig()
        self.reset()

    def reset(self):
        """Reset observer state for new episode."""
        self.estimate = 0.0
        self.estimate_filtered = 0.0
        self.estimate_magnitude = 0.0
        self._prev_roll_rate = 0.0
        self._step_count = 0

    def update(
        self,
        roll_rate: float,
        roll_accel: float,
        action: float,
        dynamic_pressure: float,
        velocity: float,
    ) -> Tuple[float, float]:
        """
        Update disturbance estimate based on current state.

        Args:
            roll_rate: Current roll rate (rad/s)
            roll_accel: Current roll acceleration (rad/s^2)
            action: Control action in [-1, 1]
            dynamic_pressure: Dynamic pressure (Pa)
            velocity: Vertical velocity (m/s)

        Returns:
            Tuple of (disturbance_estimate, disturbance_magnitude)
            - disturbance_estimate: Signed estimate normalized to [-1, 1]
            - disturbance_magnitude: Unsigned magnitude normalized to [0, 1]
        """
        cfg = self.config
        self._step_count += 1

        # Skip first step (no valid derivative)
        if self._step_count == 1:
            self._prev_roll_rate = roll_rate
            return 0.0, 0.0

        # Expected torques from control and damping
        # Control torque: effectiveness * action * dynamic_pressure
        tau_control = cfg.control_effectiveness * action * dynamic_pressure

        # Damping torque: -damping * roll_rate * q / v
        # Velocity clamped to avoid division issues
        v_safe = max(abs(velocity), 1.0)
        tau_damping = -cfg.damping_coeff * roll_rate * dynamic_pressure / v_safe

        # Expected acceleration from known torques
        alpha_expected = (tau_control + tau_damping) / cfg.I_roll

        # Actual acceleration - expected = acceleration from disturbance
        alpha_disturbance = roll_accel - alpha_expected

        # Convert back to torque
        tau_disturbance = cfg.I_roll * alpha_disturbance

        # Low-pass filter to reduce noise
        # estimate = alpha * new + (1 - alpha) * old
        self.estimate_filtered = (
            cfg.filter_alpha * tau_disturbance
            + (1 - cfg.filter_alpha) * self.estimate_filtered
        )

        # Store unfiltered estimate for debugging
        self.estimate = tau_disturbance

        # Normalize for observation space
        # Signed estimate in [-1, 1]
        normalized_estimate = np.clip(
            self.estimate_filtered / cfg.max_disturbance, -1.0, 1.0
        )

        # Magnitude in [0, 1] for gating
        magnitude = np.clip(abs(self.estimate_filtered) / cfg.max_disturbance, 0.0, 1.0)
        self.estimate_magnitude = magnitude

        self._prev_roll_rate = roll_rate

        return float(normalized_estimate), float(magnitude)

    def get_state(self) -> dict:
        """Get current observer state for debugging."""
        return {
            "disturbance_estimate_raw": self.estimate,
            "disturbance_estimate_filtered": self.estimate_filtered,
            "disturbance_magnitude": self.estimate_magnitude,
            "step_count": self._step_count,
        }


def estimate_dob_parameters(
    airframe,
    rocket_config,
    dt: float = 0.01,
) -> DOBConfig:
    """
    Estimate DOB parameters from airframe and config.

    This function extracts the physics parameters needed by the DOB
    from the rocket airframe and simulation config, ensuring the
    observer uses consistent values with the environment.

    Args:
        airframe: RocketAirframe instance
        rocket_config: RocketConfig with physics settings
        dt: Timestep for any dynamic calculations

    Returns:
        DOBConfig with estimated parameters
    """
    # Roll inertia from airframe (at typical flight mass)
    # Use dry mass + half propellant as representative
    additional_mass = getattr(rocket_config, "propellant_mass", 0.012) / 2
    I_roll = airframe.get_roll_inertia(additional_mass)

    # Control effectiveness from airframe at typical dynamic pressure
    # q = 0.5 * rho * v^2, typical v = 30 m/s -> q ~ 550 Pa
    typical_q = 500.0
    effectiveness_total = airframe.get_control_effectiveness(
        typical_q,
        tab_chord_fraction=getattr(rocket_config, "tab_chord_fraction", 0.25),
        tab_span_fraction=getattr(rocket_config, "tab_span_fraction", 0.5),
        num_controlled_fins=getattr(rocket_config, "num_controlled_fins", 2),
    )
    # Normalize to effectiveness per unit action per Pa
    # Action range is [-1, 1] with max_deflection scaling
    max_deflection_rad = np.deg2rad(getattr(rocket_config, "max_tab_deflection", 30.0))
    control_effectiveness = effectiveness_total * max_deflection_rad / typical_q

    # Damping coefficient from airframe
    damping_coeff = airframe.get_aerodynamic_damping_coeff()
    damping_coeff *= getattr(rocket_config, "damping_scale", 1.0)

    # Max disturbance: estimate from wind model
    # At 3 m/s wind, typical roll torque magnitude
    # Scale with airframe size
    size_factor = (airframe.body_diameter / 0.054) ** 3
    max_disturbance = 0.005 * size_factor  # Rough estimate

    return DOBConfig(
        I_roll=I_roll,
        control_effectiveness=control_effectiveness,
        damping_coeff=damping_coeff,
        filter_alpha=0.1,
        max_disturbance=max_disturbance,
    )


# Test the observer
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Disturbance Observer")
    print("=" * 60)

    # Create observer with default config
    config = DOBConfig(
        I_roll=0.0001,
        control_effectiveness=0.001,
        damping_coeff=0.0005,
        filter_alpha=0.1,
        max_disturbance=0.01,
    )
    dob = DisturbanceObserver(config)

    print(
        f"\nConfig: I_roll={config.I_roll}, effectiveness={config.control_effectiveness}"
    )
    print(f"        damping={config.damping_coeff}, filter_alpha={config.filter_alpha}")

    # Simulate with synthetic data
    print("\n--- Simulating with wind disturbance ---")
    np.random.seed(42)

    dt = 0.01
    roll_rate = 0.0
    wind_torque = 0.0

    for step in range(100):
        # Simulate wind ramp-up
        if step > 20:
            wind_torque = 0.002 * min((step - 20) / 30, 1.0)

        # Compute actual acceleration (control=0, just wind + damping)
        tau_damping = -config.damping_coeff * roll_rate * 500 / 30
        tau_total = tau_damping + wind_torque
        roll_accel = tau_total / config.I_roll

        # Update roll rate
        roll_rate += roll_accel * dt

        # Update DOB
        est, mag = dob.update(
            roll_rate=roll_rate,
            roll_accel=roll_accel,
            action=0.0,
            dynamic_pressure=500.0,
            velocity=30.0,
        )

        if step % 20 == 0:
            print(
                f"Step {step:3d}: wind_torque={wind_torque:.4f}, "
                f"estimate={est:+.3f}, magnitude={mag:.3f}, "
                f"roll_rate={np.rad2deg(roll_rate):.1f} deg/s"
            )

    print("\n--- Observer should track wind disturbance ---")
    print(f"Final: actual wind={wind_torque:.4f}, estimated magnitude={mag:.3f}")
