"""
Spin-Stabilized Camera Rocket Environment

A Gymnasium environment for training RL agents to control roll/spin rate
on model rockets using small deflectable tabs on fins. The goal is to
maintain stable camera footage by minimizing spin rate.

Physics modeling:
- Roll inertia calculated from RocketAirframe component geometry
- Control effectiveness from fin and tab geometry
- Aerodynamic damping from fin area and moment arm
- Disturbance torque scaled by rocket diameter³ (volume scaling)

REQUIRES: A RocketAirframe instance must be provided for physics calculations.
"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any
from dataclasses import dataclass

from airframe import RocketAirframe


@dataclass
class RocketConfig:
    """
    Configuration for spin-stabilized camera rocket simulation.

    This contains simulation and physics tuning parameters.
    Rocket geometry is defined separately via RocketAirframe.

    Parameters:
    - Control tab geometry (applied to airframe fins)
    - Physics tuning (disturbance, damping scales)
    - Simulation settings (timestep, termination thresholds)
    - Motor parameters (overridden when using RealisticMotorRocket)
    """

    # === Control Tab Geometry ===
    tab_chord_fraction: float = 0.25  # Fraction of fin chord that is tab
    tab_span_fraction: float = 0.5  # Fraction of fin span with tab
    max_tab_deflection: float = 15.0  # Maximum deflection (degrees)
    num_controlled_fins: int = 2  # Number of fins with active tabs

    # === Physics Tuning ===
    disturbance_scale: float = 0.0001  # Random torque magnitude scaling
    damping_scale: float = 1.0  # Multiplier for aerodynamic damping
    initial_spin_std: float = 15.0  # Initial spin disturbance (deg/s std)

    # === Motor (defaults for simple thrust model) ===
    # These are overridden when using RealisticMotorRocket with real motor data
    average_thrust: float = 5.4  # N - Estes C6 average
    burn_time: float = 1.85  # s
    propellant_mass: float = 0.012  # kg
    thrust_curve: str = "neutral"  # "neutral", "progressive", "regressive"

    # === Simulation ===
    dt: float = 0.01  # Time step (100 Hz)
    max_altitude: float = 500.0  # m - for observation normalization
    max_roll_rate: float = 360.0  # deg/s - termination threshold
    max_episode_time: float = 15.0  # seconds


class SpinStabilizedCameraRocket(gym.Env):
    """
    Gymnasium environment for spin-stabilized rocket control.

    Physics are calculated from the provided RocketAirframe, which defines
    the rocket's geometry, mass distribution, and aerodynamic properties.

    Args:
        airframe: RocketAirframe instance defining rocket geometry (REQUIRED)
        config: RocketConfig with simulation and physics tuning parameters
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, airframe: RocketAirframe, config: RocketConfig = None):
        super().__init__()

        if airframe is None:
            raise ValueError(
                "RocketAirframe is required. Create one with:\n"
                "  airframe = RocketAirframe.load('my_rocket.ork')  # From OpenRocket\n"
                "  airframe = RocketAirframe.estes_alpha()          # Factory method\n"
                "  airframe = RocketAirframe.load('airframe.yaml')  # From YAML"
            )

        self.airframe = airframe
        self.config = config or RocketConfig()

        # Action space: normalized tab deflection [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,  # altitude (m)
                    -100.0,  # vertical velocity (m/s)
                    -np.pi,  # roll angle (rad)
                    -np.deg2rad(self.config.max_roll_rate),  # roll rate (rad/s)
                    -50.0,  # roll acceleration (rad/s²)
                    0.0,  # dynamic pressure (Pa)
                    0.0,  # time (s)
                    0.0,  # thrust fraction
                    -1.0,  # last action
                    0.0,  # camera shake
                ],
                dtype=np.float32,
            ),
            high=np.array(
                [
                    self.config.max_altitude,
                    100.0,
                    np.pi,
                    np.deg2rad(self.config.max_roll_rate),
                    50.0,
                    3000.0,
                    self.config.max_episode_time,
                    1.0,
                    1.0,
                    50.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # State variables
        self.altitude = 0.0
        self.vertical_velocity = 0.0
        self.vertical_acceleration = 0.0

        # Roll state - use configurable initial disturbance
        self.roll_angle = 0.0
        self.roll_rate = np.random.normal(0, np.deg2rad(self.config.initial_spin_std))
        self.roll_acceleration = 0.0

        # Time and mass
        self.time = 0.0
        self.propellant_remaining = self.config.propellant_mass

        # Control
        self.last_action = 0.0
        self.previous_action = 0.0
        self.tab_deflection = 0.0

        # Tracking
        self.camera_shake_history = []
        self.total_rotation = 0.0
        self.max_altitude_reached = 0.0
        self.integrated_roll_error = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Process action
        self.previous_action = self.last_action
        self.last_action = float(np.clip(action[0], -1.0, 1.0))
        self.tab_deflection = self.last_action * np.deg2rad(
            self.config.max_tab_deflection
        )

        dt = self.config.dt
        self.time += dt

        # Update propulsion
        thrust, mass = self._update_propulsion()

        # Aerodynamics
        rho = self._get_air_density()
        v = max(self.vertical_velocity, 0.1)  # Avoid division issues
        q = 0.5 * rho * v**2

        # Vertical dynamics - use airframe frontal area
        frontal_area = self.airframe.get_frontal_area()
        cd = 0.4 if self.time < self.config.burn_time else 0.5
        drag = cd * q * frontal_area

        self.vertical_acceleration = (thrust - drag - mass * 9.81) / mass
        self.vertical_velocity += self.vertical_acceleration * dt
        self.altitude += self.vertical_velocity * dt
        self.altitude = max(0, self.altitude)  # Can't go below ground

        # Roll dynamics - using airframe geometry
        roll_torque = self._calculate_roll_torque(q)
        I_roll = self._calculate_roll_inertia(mass)

        self.roll_acceleration = roll_torque / I_roll

        # Clamp acceleration to prevent numerical instability
        max_accel = 100.0  # rad/s² - reasonable physical limit
        self.roll_acceleration = np.clip(self.roll_acceleration, -max_accel, max_accel)

        self.roll_rate += self.roll_acceleration * dt
        self.roll_angle += self.roll_rate * dt
        self.roll_angle = np.arctan2(np.sin(self.roll_angle), np.cos(self.roll_angle))

        # Tracking
        self.total_rotation += abs(self.roll_rate) * dt
        self.max_altitude_reached = max(self.max_altitude_reached, self.altitude)
        self.integrated_roll_error += abs(self.roll_rate) * dt

        # Camera shake
        camera_shake = self._calculate_camera_shake()
        self.camera_shake_history.append(camera_shake)

        # Reward
        reward = self._calculate_reward(camera_shake)

        # Termination
        terminated = self._is_terminated()
        truncated = self.time > self.config.max_episode_time

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _calculate_roll_torque(self, dynamic_pressure: float) -> float:
        """
        Calculate roll torque using airframe geometry.

        Components:
        - Control torque from deflectable tabs
        - Aerodynamic damping from fins
        - Random disturbance torque
        """
        control_torque = 0.0
        damping_torque = 0.0
        disturbance = 0.0

        if dynamic_pressure > 1.0:
            # Control torque using airframe geometry
            effectiveness = self.airframe.get_control_effectiveness(
                dynamic_pressure,
                tab_chord_fraction=self.config.tab_chord_fraction,
                tab_span_fraction=self.config.tab_span_fraction,
                num_controlled_fins=self.config.num_controlled_fins,
            )
            control_torque = effectiveness * self.tab_deflection

            # Velocity-dependent effectiveness
            speed_effectiveness = np.tanh(dynamic_pressure / 200.0)
            control_torque *= speed_effectiveness

            # Aerodynamic damping from airframe
            damping_coef = self.airframe.get_aerodynamic_damping_coeff()
            damping_coef *= self.config.damping_scale
            damping_torque = (
                -damping_coef
                * self.roll_rate
                * dynamic_pressure
                / max(self.vertical_velocity, 1.0)
            )

            # Disturbance scaled by airframe diameter (volume scaling)
            size_factor = (self.airframe.body_diameter / 0.054) ** 3
            disturbance_std = (
                self.config.disturbance_scale * np.sqrt(dynamic_pressure) * size_factor
            )
            disturbance = np.random.normal(0, disturbance_std)

        return control_torque + damping_torque + disturbance

    def _calculate_roll_inertia(self, mass: float) -> float:
        """
        Calculate roll inertia using airframe component geometry.

        The airframe calculates inertia from its components (nose cone,
        body tube, fins) using the parallel axis theorem.
        """
        # Additional mass is motor/propellant (total mass minus airframe dry mass)
        additional_mass = mass - self.airframe.dry_mass
        additional_mass = max(0, additional_mass)  # Ensure non-negative
        return self.airframe.get_roll_inertia(additional_mass)

    def _update_propulsion(self) -> Tuple[float, float]:
        """Update thrust and mass using simple thrust model."""
        if self.time < self.config.burn_time:
            burn_fraction = self.config.dt / self.config.burn_time
            self.propellant_remaining -= self.config.propellant_mass * burn_fraction
            self.propellant_remaining = max(0, self.propellant_remaining)

            if self.config.thrust_curve == "progressive":
                thrust_mult = 0.7 + 0.6 * (self.time / self.config.burn_time)
            elif self.config.thrust_curve == "regressive":
                thrust_mult = 1.3 - 0.6 * (self.time / self.config.burn_time)
            else:
                thrust_mult = 1.0

            thrust = self.config.average_thrust * thrust_mult
        else:
            thrust = 0.0

        mass = self.airframe.dry_mass + self.propellant_remaining
        return thrust, mass

    def _get_air_density(self) -> float:
        """Atmospheric density model (ISA)."""
        return 1.225 * np.exp(-self.altitude / 8000)

    def _calculate_camera_shake(self) -> float:
        """Camera shake metric based on roll rate and acceleration."""
        roll_rate_contribution = abs(self.roll_rate) * 10.0
        accel_contribution = abs(self.roll_acceleration) * 0.5
        return roll_rate_contribution + accel_contribution

    def _calculate_reward(self, camera_shake: float) -> float:
        """Reward function for camera stability."""
        reward = 0.0
        roll_rate_deg = np.rad2deg(abs(self.roll_rate))

        # 1. Roll rate reward (primary objective)
        if roll_rate_deg < 5:
            reward += 10.0  # Excellent
        elif roll_rate_deg < 15:
            reward += 5.0  # Good
        elif roll_rate_deg < 30:
            reward += 2.0  # Acceptable
        else:
            reward -= roll_rate_deg * 0.05  # Penalty

        # 2. Camera shake penalty
        reward -= camera_shake * 0.1

        # 3. Control effort penalty
        reward -= abs(self.last_action) * 0.1

        # 4. Control smoothness
        action_change = abs(self.last_action - self.previous_action)
        reward -= action_change * 0.5

        # 5. Altitude progress bonus
        if self.altitude > 0:
            reward += min(self.altitude / 100, 2.0)

        # 6. Survival bonus (encourage longer episodes)
        reward += 0.1

        return float(np.clip(reward, -20, 20))

    def _is_terminated(self) -> bool:
        """Check termination conditions."""
        # Ground impact
        if self.altitude < -0.1 and self.time > 0.5:
            return True

        # Excessive spin
        if abs(self.roll_rate) > np.deg2rad(self.config.max_roll_rate):
            return True

        # Past apogee and descending
        if (
            self.altitude < self.max_altitude_reached - 30
            and self.time > self.config.burn_time + 1.0
        ):
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get observation vector."""
        rho = self._get_air_density()
        q = 0.5 * rho * max(self.vertical_velocity, 0) ** 2

        thrust_frac = max(
            0, (self.config.burn_time - self.time) / self.config.burn_time
        )

        recent_shake = (
            np.mean(self.camera_shake_history[-10:])
            if self.camera_shake_history
            else 0.0
        )

        obs = np.array(
            [
                self.altitude,
                self.vertical_velocity,
                self.roll_angle,
                self.roll_rate,
                self.roll_acceleration,
                q,
                self.time,
                thrust_frac,
                self.last_action,
                recent_shake,
            ],
            dtype=np.float32,
        )

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict with flight telemetry."""
        roll_rate_deg = np.rad2deg(abs(self.roll_rate))

        if roll_rate_deg < 10:
            h_quality = "Excellent - Stable footage"
        elif roll_rate_deg < 30:
            h_quality = "Good - Minor blur"
        elif roll_rate_deg < 60:
            h_quality = "Fair - Noticeable blur"
        else:
            h_quality = "Poor - Severe blur"

        return {
            "altitude_m": self.altitude,
            "vertical_velocity_ms": self.vertical_velocity,
            "roll_rate_deg_s": np.rad2deg(self.roll_rate),
            "roll_total_rotations": self.total_rotation / (2 * np.pi),
            "tab_deflection_deg": np.rad2deg(self.tab_deflection),
            "time_s": self.time,
            "phase": "boost" if self.time < self.config.burn_time else "coast",
            "mass_kg": self.airframe.dry_mass + self.propellant_remaining,
            "camera_shake": (
                self.camera_shake_history[-1] if self.camera_shake_history else 0
            ),
            "max_altitude_m": self.max_altitude_reached,
            "horizontal_camera_quality": h_quality,
            "downward_camera_quality": h_quality,
            "airframe": self.airframe.name,
        }

    def render(self, mode="human"):
        if mode == "human":
            info = self._get_info()
            print(
                f"T={self.time:.2f}s | Alt={self.altitude:.1f}m | "
                f"Roll={info['roll_rate_deg_s']:.1f}°/s | "
                f"Phase={info['phase']} | Quality={info['horizontal_camera_quality']}"
            )


# Test the environment
if __name__ == "__main__":
    print("=" * 60)
    print("Testing Spin-Stabilized Rocket Environment")
    print("=" * 60)

    # Create airframe (required)
    airframe = RocketAirframe.estes_alpha()
    print(f"\nAirframe: {airframe.name}")
    print(airframe.summary())

    # Create config with physics tuning
    config = RocketConfig(
        max_tab_deflection=15.0,
        initial_spin_std=15.0,
        disturbance_scale=0.0001,
    )

    # Create environment
    env = SpinStabilizedCameraRocket(airframe=airframe, config=config)

    print(f"\nConfig: max_tab_deflection={config.max_tab_deflection}°")
    print(f"Disturbance scale: {config.disturbance_scale}")

    # Test with zero control
    print("\n--- Testing with ZERO control ---")
    obs, info = env.reset()
    print(f"Initial roll rate: {info['roll_rate_deg_s']:.1f}°/s")

    for step in range(200):
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0 or terminated or truncated:
            print(
                f"Step {step:3d}: alt={info['altitude_m']:6.1f}m, "
                f"roll={info['roll_rate_deg_s']:7.1f}°/s, "
                f"phase={info['phase']}"
            )

        if terminated or truncated:
            print(f"\n*** Episode ended at step {step} ***")
            break

    print(
        f"\nFinal: altitude={info['altitude_m']:.1f}m, "
        f"max_alt={info['max_altitude_m']:.1f}m"
    )
    print(f"Camera quality: {info['horizontal_camera_quality']}")

    # Test with simple P-controller
    print("\n--- Testing with P-controller ---")
    obs, info = env.reset()
    total_reward = 0
    Kp = 0.5

    for step in range(500):
        roll_rate = obs[3]  # rad/s
        action = np.array([-Kp * roll_rate])
        action = np.clip(action, -1, 1)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 50 == 0:
            print(
                f"Step {step:3d}: alt={info['altitude_m']:6.1f}m, "
                f"roll={info['roll_rate_deg_s']:6.1f}°/s, reward={reward:.2f}"
            )

        if terminated or truncated:
            break

    print(f"\n=== Episode Summary ===")
    print(f"Duration: {info['time_s']:.2f}s")
    print(f"Max altitude: {info['max_altitude_m']:.1f}m")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Camera: {info['horizontal_camera_quality']}")
