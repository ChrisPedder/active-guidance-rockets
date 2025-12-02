"""
Patched Spin-Stabilized Camera Rocket Environment

This file patches the physics bugs in the original spin_stabilized_control_env.py:

Bug 1: Disturbance torque was ~1000x too large for small rockets
Bug 2: Roll inertia calculation didn't account for rocket size properly
Bug 3: Damping was insufficient for the disturbance magnitude

These bugs caused:
- Spin rates of 4000+ °/s within 10 timesteps
- Episodes terminating in <0.2 seconds
- Impossible control task for RL agent

"""

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any
from dataclasses import dataclass, field


@dataclass
class RocketConfig:
    """
    Configuration for spin-stabilized camera rocket.

    Key changes from original:
    - Added disturbance_scale parameter (default much lower)
    - Added damping_scale parameter
    - Better documentation of expected ranges
    """
    # === Mass Properties ===
    # IMPORTANT: Must give TWR > 2.0 with your motor!
    dry_mass: float = 0.150          # kg - for Estes C6, use 0.08-0.15 kg
    propellant_mass: float = 0.012   # kg - will be overridden by motor

    # === Geometry ===
    diameter: float = 0.024          # m - 24mm for C motors, 29mm for D-F, 38mm for G+
    length: float = 0.45             # m
    wall_thickness: float = 0.001    # m

    # === Motor (defaults for simple thrust model) ===
    thrust_curve: str = "neutral"
    average_thrust: float = 5.4      # N - Estes C6 average
    burn_time: float = 1.85          # s

    # === Fins ===
    num_fins: int = 4
    fin_span: float = 0.04           # m
    fin_root_chord: float = 0.05     # m
    fin_thickness: float = 0.002     # m
    fin_position: float = 0.35       # m from nose

    # === Control Tabs ===
    tab_chord_fraction: float = 0.25
    tab_span_fraction: float = 0.5
    max_tab_deflection: float = 15.0  # degrees
    num_controlled_fins: int = 2

    # === Physics Tuning (NEW - the key fixes) ===
    disturbance_scale: float = 0.0001    # REDUCED from 0.01 (100x smaller)
    damping_scale: float = 1.0           # Multiplier for aerodynamic damping
    initial_spin_std: float = 15.0       # degrees/s std for initial spin

    # === Camera ===
    horizontal_camera_fov: float = 120.0
    downward_camera_fov: float = 90.0

    # === Simulation ===
    dt: float = 0.01                 # 100 Hz
    max_altitude: float = 500.0      # m
    max_roll_rate: float = 360.0     # deg/s - reduced from 720 for earlier termination
    max_episode_time: float = 15.0   # seconds


class SpinStabilizedCameraRocket(gym.Env):
    """
    Patched rocket environment with fixed physics for small model rockets.

    Key fixes:
    1. Disturbance torque scaled by rocket diameter³ (volume scaling)
    2. Damping coefficient scaled appropriately
    3. Roll inertia uses more realistic mass distribution
    4. Configurable physics parameters for tuning
    """

    metadata = {'render_modes': ['human']}

    def __init__(self, config: RocketConfig = None):
        super().__init__()
        self.config = config or RocketConfig()

        # Action space: normalized tab deflection [-1, 1]
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(1,), dtype=np.float32
        )

        # Observation space
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,       # altitude (m)
                -100.0,    # vertical velocity (m/s)
                -np.pi,    # roll angle (rad)
                -np.deg2rad(self.config.max_roll_rate),  # roll rate (rad/s)
                -50.0,     # roll acceleration (rad/s²)
                0.0,       # dynamic pressure (Pa)
                0.0,       # time (s)
                0.0,       # thrust fraction
                -1.0,      # last action
                0.0,       # camera shake
            ], dtype=np.float32),
            high=np.array([
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
            ], dtype=np.float32),
            dtype=np.float32
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
        self.tab_deflection = self.last_action * np.deg2rad(self.config.max_tab_deflection)

        dt = self.config.dt
        self.time += dt

        # Update propulsion
        thrust, mass = self._update_propulsion()

        # Aerodynamics
        rho = self._get_air_density()
        v = max(self.vertical_velocity, 0.1)  # Avoid division issues
        q = 0.5 * rho * v**2

        # Vertical dynamics
        frontal_area = np.pi * (self.config.diameter / 2)**2
        cd = 0.4 if self.time < self.config.burn_time else 0.5
        drag = cd * q * frontal_area

        self.vertical_acceleration = (thrust - drag - mass * 9.81) / mass
        self.vertical_velocity += self.vertical_acceleration * dt
        self.altitude += self.vertical_velocity * dt
        self.altitude = max(0, self.altitude)  # Can't go below ground

        # Roll dynamics (FIXED)
        roll_torque = self._calculate_roll_torque_fixed(q)
        I_roll = self._calculate_roll_inertia_fixed(mass)

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

    def _calculate_roll_torque_fixed(self, dynamic_pressure: float) -> float:
        """
        FIXED roll torque calculation with proper scaling for small rockets.
        """
        control_torque = 0.0
        damping_torque = 0.0
        disturbance = 0.0

        if dynamic_pressure > 1.0:
            # === Control torque from tabs ===
            tab_area = (self.config.tab_chord_fraction * self.config.fin_root_chord *
                       self.config.tab_span_fraction * self.config.fin_span)

            # Lift coefficient (thin airfoil theory, valid for small angles)
            Cl_tab = 2 * np.pi * np.sin(self.tab_deflection)

            # Force per tab
            tab_force = 0.5 * Cl_tab * dynamic_pressure * tab_area

            # Moment arm
            moment_arm = self.config.diameter / 2 + 0.5 * self.config.fin_span

            # Two tabs in differential mode
            control_torque = 2 * tab_force * moment_arm

            # Effectiveness factor
            speed_effectiveness = np.tanh(dynamic_pressure / 200.0)
            control_torque *= speed_effectiveness

            # === Aerodynamic damping (INCREASED) ===
            # Damping scales with fin area and moment arm
            fin_area = self.config.fin_span * self.config.fin_root_chord
            total_fin_area = self.config.num_fins * fin_area

            # Damping coefficient based on fin geometry
            # This creates a torque opposing rotation
            damping_coef = 0.5 * total_fin_area * moment_arm**2 * self.config.damping_scale
            damping_torque = -damping_coef * self.roll_rate * dynamic_pressure / max(self.vertical_velocity, 1.0)

            # === Disturbance torque (FIXED - scaled by rocket size) ===
            # Disturbance scales with diameter³ (volume/inertia scaling)
            # This is the key fix - small rockets get small disturbances
            size_factor = (self.config.diameter / 0.054)**3  # Normalized to 54mm rocket
            disturbance_std = self.config.disturbance_scale * np.sqrt(dynamic_pressure) * size_factor
            disturbance = np.random.normal(0, disturbance_std)

        total_torque = control_torque + damping_torque + disturbance

        return total_torque

    def _calculate_roll_inertia_fixed(self, mass: float) -> float:
        """
        FIXED roll inertia with better mass distribution model.
        """
        radius = self.config.diameter / 2

        # Body tube - thin walled cylinder
        # Assume tube is ~20% of total mass
        tube_mass_fraction = 0.2
        tube_inertia = tube_mass_fraction * mass * radius**2

        # Internal components (motor, nose, etc.) - roughly cylindrical, centered
        # These contribute less to roll inertia
        internal_mass_fraction = 0.8
        internal_radius = radius * 0.5  # Effective radius of internal mass
        internal_inertia = 0.5 * internal_mass_fraction * mass * internal_radius**2

        # Fins
        fin_mass_each = (self.config.fin_span * self.config.fin_root_chord *
                        self.config.fin_thickness * 1800)  # kg/m³ for fiberglass
        fin_distance = radius + self.config.fin_span / 2
        fins_inertia = self.config.num_fins * fin_mass_each * fin_distance**2

        total_inertia = tube_inertia + internal_inertia + fins_inertia

        # Ensure minimum inertia to prevent numerical issues
        min_inertia = 1e-6  # kg·m²
        return max(total_inertia, min_inertia)

    def _update_propulsion(self) -> Tuple[float, float]:
        """Update thrust and mass"""
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

        mass = self.config.dry_mass + self.propellant_remaining
        return thrust, mass

    def _get_air_density(self) -> float:
        """Atmospheric density model"""
        return 1.225 * np.exp(-self.altitude / 8000)

    def _calculate_camera_shake(self) -> float:
        """Camera shake metric"""
        roll_rate_contribution = abs(self.roll_rate) * 10.0
        accel_contribution = abs(self.roll_acceleration) * 0.5
        return roll_rate_contribution + accel_contribution

    def _calculate_reward(self, camera_shake: float) -> float:
        """Reward function for camera stability"""
        reward = 0.0
        roll_rate_deg = np.rad2deg(abs(self.roll_rate))

        # 1. Roll rate reward (primary objective)
        if roll_rate_deg < 5:
            reward += 10.0  # Excellent
        elif roll_rate_deg < 15:
            reward += 5.0   # Good
        elif roll_rate_deg < 30:
            reward += 2.0   # Acceptable
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
        """Check termination conditions"""
        # Ground impact
        if self.altitude < -0.1 and self.time > 0.5:
            return True

        # Excessive spin
        if abs(self.roll_rate) > np.deg2rad(self.config.max_roll_rate):
            return True

        # Past apogee and descending
        if (self.altitude < self.max_altitude_reached - 30 and
            self.time > self.config.burn_time + 1.0):
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get observation vector"""
        rho = self._get_air_density()
        q = 0.5 * rho * max(self.vertical_velocity, 0)**2

        thrust_frac = max(0, (self.config.burn_time - self.time) / self.config.burn_time)

        recent_shake = np.mean(self.camera_shake_history[-10:]) if self.camera_shake_history else 0.0

        obs = np.array([
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
        ], dtype=np.float32)

        return np.clip(obs, self.observation_space.low, self.observation_space.high)

    def _get_info(self) -> Dict[str, Any]:
        """Get info dict"""
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
            'altitude_m': self.altitude,
            'vertical_velocity_ms': self.vertical_velocity,
            'roll_rate_deg_s': np.rad2deg(self.roll_rate),
            'roll_total_rotations': self.total_rotation / (2 * np.pi),
            'tab_deflection_deg': np.rad2deg(self.tab_deflection),
            'time_s': self.time,
            'phase': 'boost' if self.time < self.config.burn_time else 'coast',
            'mass_kg': self.config.dry_mass + self.propellant_remaining,
            'camera_shake': self.camera_shake_history[-1] if self.camera_shake_history else 0,
            'max_altitude_m': self.max_altitude_reached,
            'horizontal_camera_quality': h_quality,
            'downward_camera_quality': h_quality,  # Simplified
        }

    def render(self, mode='human'):
        if mode == 'human':
            info = self._get_info()
            print(f"T={self.time:.2f}s | Alt={self.altitude:.1f}m | "
                  f"Roll={info['roll_rate_deg_s']:.1f}°/s | "
                  f"Phase={info['phase']} | Quality={info['horizontal_camera_quality']}")


# Test the fix
if __name__ == "__main__":
    print("=" * 60)
    print("Testing PATCHED Spin-Stabilized Rocket Environment")
    print("=" * 60)

    # Create environment with small rocket config (like Estes C6)
    config = RocketConfig(
        dry_mass=0.100,         # 100g rocket
        propellant_mass=0.012,  # C6 motor
        diameter=0.024,         # 24mm
        length=0.40,
        average_thrust=5.4,     # C6 average
        burn_time=1.85,
        max_tab_deflection=15.0,
        initial_spin_std=15.0,  # Moderate initial disturbance
        disturbance_scale=0.0001,  # Key fix - much smaller
    )

    env = SpinStabilizedCameraRocket(config)

    print(f"\nConfig: {config.dry_mass*1000:.0f}g rocket, {config.diameter*1000:.0f}mm diameter")
    print(f"Motor: {config.average_thrust:.1f}N for {config.burn_time:.2f}s")
    print(f"Disturbance scale: {config.disturbance_scale}")

    # Test with zero control
    print("\n--- Testing with ZERO control ---")
    obs, info = env.reset()
    print(f"Initial roll rate: {info['roll_rate_deg_s']:.1f}°/s")

    for step in range(200):
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        if step % 20 == 0 or terminated or truncated:
            print(f"Step {step:3d}: alt={info['altitude_m']:6.1f}m, "
                  f"roll={info['roll_rate_deg_s']:7.1f}°/s, "
                  f"phase={info['phase']}")

        if terminated or truncated:
            print(f"\n*** Episode ended at step {step} ***")
            break

    print(f"\nFinal: altitude={info['altitude_m']:.1f}m, "
          f"max_alt={info['max_altitude_m']:.1f}m")
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
            print(f"Step {step:3d}: alt={info['altitude_m']:6.1f}m, "
                  f"roll={info['roll_rate_deg_s']:6.1f}°/s, reward={reward:.2f}")

        if terminated or truncated:
            break

    print(f"\n=== Episode Summary ===")
    print(f"Duration: {info['time_s']:.2f}s")
    print(f"Max altitude: {info['max_altitude_m']:.1f}m")
    print(f"Total reward: {total_reward:.1f}")
    print(f"Camera: {info['horizontal_camera_quality']}")
