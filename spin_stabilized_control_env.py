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
from typing import Tuple, Dict, Any, Optional, List
from dataclasses import dataclass, field
from collections import deque

from airframe import RocketAirframe
from wind_model import WindModel, WindConfig


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
    max_tab_deflection: float = 30.0  # Maximum deflection (degrees)
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

    # === Wind ===
    enable_wind: bool = False
    base_wind_speed: float = 0.0  # m/s mean wind speed
    max_gust_speed: float = 0.0  # m/s gust amplitude
    wind_variability: float = 0.3  # direction change rate
    wind_altitude_gradient: float = 0.0  # speed increase per 100m altitude
    use_dryden: bool = False  # Use Dryden turbulence model
    turbulence_severity: str = "light"  # "light", "moderate", "severe"
    altitude_profile_alpha: float = 0.14  # Power-law exponent for wind profile
    reference_altitude: float = 10.0  # Reference altitude for power-law (m)
    body_shadow_factor: float = (
        0.90  # Leeward fin q fraction; K_shadow = 1 - this (see docs/wind_roll_torque_analysis.md)
    )

    # === Mach-dependent aerodynamics ===
    use_mach_aero: bool = False  # Enable Mach-dependent Cd and Cl_alpha
    use_isa_full: bool = False  # Use full ISA atmosphere (temp, speed of sound)
    cd_mach_table: Optional[Dict] = (
        None  # {mach: [...], cd_boost: [...], cd_coast: [...]}
    )

    # === Servo dynamics ===
    servo_time_constant: float = 0.0  # seconds, 0 = instantaneous
    servo_rate_limit: float = 0.0  # deg/s, 0 = unlimited
    servo_deadband: float = 0.0  # degrees, 0 = none

    # === Sensor latency ===
    sensor_delay_steps: int = 0  # 0 = no delay

    # === Observation space bounds (for larger/faster vehicles) ===
    max_velocity: float = 100.0  # m/s
    max_dynamic_pressure: float = 3000.0  # Pa

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

        # Wind model
        if self.config.enable_wind:
            wind_cfg = WindConfig(
                enable=True,
                base_speed=self.config.base_wind_speed,
                max_gust_speed=self.config.max_gust_speed,
                variability=self.config.wind_variability,
                altitude_gradient=self.config.wind_altitude_gradient,
                use_dryden=self.config.use_dryden,
                turbulence_severity=self.config.turbulence_severity,
                altitude_profile_alpha=self.config.altitude_profile_alpha,
                reference_altitude=self.config.reference_altitude,
                body_shadow_factor=self.config.body_shadow_factor,
            )
            self.wind_model: Optional[WindModel] = WindModel(wind_cfg)
        else:
            self.wind_model = None

        # Action space: normalized tab deflection [-1, 1]
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

        # Observation space (bounds from config for larger/faster vehicles)
        max_vel = self.config.max_velocity
        max_q = self.config.max_dynamic_pressure
        self.observation_space = spaces.Box(
            low=np.array(
                [
                    0.0,  # altitude (m)
                    -max_vel,  # vertical velocity (m/s)
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
                    max_vel,
                    np.pi,
                    np.deg2rad(self.config.max_roll_rate),
                    50.0,
                    max_q,
                    self.config.max_episode_time,
                    1.0,
                    1.0,
                    50.0,
                ],
                dtype=np.float32,
            ),
            dtype=np.float32,
        )

        # Sensor delay buffer
        self._obs_buffer = deque(maxlen=max(self.config.sensor_delay_steps + 1, 1))

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

        # Spin rate tracking for metrics
        self.max_spin_rate = abs(self.roll_rate)  # Track max spin rate
        self.spin_rate_sum = 0.0  # For computing mean
        self.spin_rate_count = 0

        # Atmospheric disturbance tracking
        self._last_disturbance_torque = 0.0
        self._last_dynamic_pressure = 0.0

        # Wind tracking
        self._last_wind_speed = 0.0
        self._last_wind_direction = 0.0
        self._last_wind_torque = 0.0

        # Servo state
        self._servo_position = 0.0

        # Mach-dependent state (cached per step)
        self._last_mach = 0.0
        self._last_speed_of_sound = 340.3
        self._last_cd = 0.4
        self._last_cl_alpha = 2.0 * np.pi

        # Sensor delay buffer
        self._obs_buffer.clear()

        # Reset wind model for new episode
        if self.wind_model is not None:
            self.wind_model.reset()

        return self._get_observation(), self._get_info()

    def _get_atmosphere(self):
        """Full ISA atmosphere model. Returns (rho, temperature, speed_of_sound)."""
        if not self.config.use_isa_full:
            rho = 1.225 * np.exp(-self.altitude / 8000)
            return rho, 288.15, 340.3  # approximate, backward compat

        h = max(self.altitude, 0.0)
        T = 288.15 - 0.0065 * min(h, 11000.0)  # troposphere lapse rate
        p = 101325.0 * (T / 288.15) ** 5.2561
        rho = p / (287.05 * T)
        a = np.sqrt(1.4 * 287.05 * T)
        return rho, T, a

    def _get_cd(self, mach: float, is_boost: bool) -> float:
        """Get drag coefficient, optionally Mach-dependent."""
        if not self.config.use_mach_aero:
            return 0.4 if is_boost else 0.5  # original behavior

        if self.config.cd_mach_table is not None:
            table = self.config.cd_mach_table
            cd_col = table["cd_boost"] if is_boost else table["cd_coast"]
            return float(np.interp(mach, table["mach"], cd_col))

        # Analytical fallback: constant subsonic
        return 0.4 if is_boost else 0.5

    def _get_cl_alpha(self, mach: float) -> float:
        """Get lift curve slope, optionally Mach-dependent."""
        if not self.config.use_mach_aero:
            return 2.0 * np.pi  # original behavior

        cl0 = 2.0 * np.pi
        if mach < 0.8:
            return cl0 / np.sqrt(max(1.0 - mach**2, 0.04))  # Prandtl-Glauert
        elif mach <= 1.2:
            cl_sub = cl0 / np.sqrt(1.0 - 0.8**2)  # value at M=0.8
            cl_sup = 4.0 / np.sqrt(1.2**2 - 1.0)  # value at M=1.2 (Ackeret)
            t = (mach - 0.8) / 0.4
            return cl_sub + (cl_sup - cl_sub) * t  # linear interp
        else:
            return 4.0 / np.sqrt(max(mach**2 - 1.0, 0.01))  # Ackeret

    def _update_servo(self, commanded: float, dt: float) -> float:
        """Apply servo dynamics (lag, rate limit, deadband) to commanded position."""
        tau = self.config.servo_time_constant
        has_dynamics = (
            tau > 0
            or self.config.servo_rate_limit > 0
            or self.config.servo_deadband > 0
        )
        if not has_dynamics:
            self._servo_position = np.clip(
                commanded, -1.0, 1.0
            )  # instantaneous (backward compat)
            return self._servo_position

        error = commanded - self._servo_position
        deadband_norm = self.config.servo_deadband / max(
            self.config.max_tab_deflection, 1.0
        )
        if abs(error) < deadband_norm:
            return self._servo_position

        # First-order lag
        if tau > 0:
            alpha = 1.0 - np.exp(-dt / tau)
        else:
            alpha = 1.0
        desired_delta = error * alpha

        # Rate limiting (convert deg/s to normalized/s)
        if self.config.servo_rate_limit > 0:
            max_rate_norm = self.config.servo_rate_limit / max(
                self.config.max_tab_deflection, 1.0
            )
            max_delta = max_rate_norm * dt
            desired_delta = np.clip(desired_delta, -max_delta, max_delta)

        self._servo_position = np.clip(self._servo_position + desired_delta, -1.0, 1.0)
        return self._servo_position

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        # Process action
        self.previous_action = self.last_action
        self.last_action = float(np.clip(action[0], -1.0, 1.0))

        dt = self.config.dt

        # Apply servo dynamics
        actual_pos = self._update_servo(self.last_action, dt)
        self.tab_deflection = actual_pos * np.deg2rad(self.config.max_tab_deflection)

        # Update propulsion before advancing time so thrust is evaluated
        # at the current time point, not at t+dt.
        thrust, mass = self._update_propulsion()

        self.time += dt

        # Atmosphere
        rho, temperature, a = self._get_atmosphere()
        v = max(self.vertical_velocity, 0.1)  # Avoid division issues
        q = 0.5 * rho * v**2

        # Mach number and Mach-dependent aerodynamics
        mach = v / max(a, 1.0) if v > 0 else 0.0
        self._last_mach = mach
        self._last_speed_of_sound = a

        is_boost = self.time < self.config.burn_time
        cd = self._get_cd(mach, is_boost)
        self._last_cd = cd

        cl_alpha = self._get_cl_alpha(mach)
        self._last_cl_alpha = cl_alpha

        # Vertical dynamics - use airframe frontal area
        frontal_area = self.airframe.get_frontal_area()
        drag = cd * q * frontal_area

        self.vertical_acceleration = (thrust - drag - mass * 9.81) / mass
        self.vertical_velocity += self.vertical_acceleration * dt
        self.altitude += self.vertical_velocity * dt
        self.altitude = max(0, self.altitude)  # Can't go below ground

        # Roll dynamics - using airframe geometry (with Mach-corrected cl_alpha)
        roll_torque = self._calculate_roll_torque(q, cl_alpha)
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

        # Spin rate metrics
        abs_spin = abs(self.roll_rate)
        self.max_spin_rate = max(self.max_spin_rate, abs_spin)
        self.spin_rate_sum += abs_spin
        self.spin_rate_count += 1

        # Camera shake
        camera_shake = self._calculate_camera_shake()
        self.camera_shake_history.append(camera_shake)

        # Reward
        reward = self._calculate_reward(camera_shake)

        # Termination
        terminated = self._is_terminated()
        truncated = self.time > self.config.max_episode_time

        # Observation with optional sensor delay
        obs = self._get_observation()
        self._obs_buffer.append(obs.copy())
        if (
            self.config.sensor_delay_steps > 0
            and len(self._obs_buffer) > self.config.sensor_delay_steps
        ):
            obs = self._obs_buffer[-1 - self.config.sensor_delay_steps].copy()

        return obs, reward, terminated, truncated, self._get_info()

    def _calculate_roll_torque(
        self, dynamic_pressure: float, cl_alpha: float = 2 * np.pi
    ) -> float:
        """
        Calculate roll torque using airframe geometry.

        Components:
        - Control torque from deflectable tabs
        - Aerodynamic damping from fins
        - Random disturbance torque

        Args:
            dynamic_pressure: Dynamic pressure (Pa)
            cl_alpha: Lift curve slope (rad^-1), may be Mach-corrected
        """
        control_torque = 0.0
        damping_torque = 0.0
        disturbance = 0.0

        if dynamic_pressure > 1.0:
            # Control torque using airframe geometry (with Mach-corrected cl_alpha)
            effectiveness = self.airframe.get_control_effectiveness(
                dynamic_pressure,
                tab_chord_fraction=self.config.tab_chord_fraction,
                tab_span_fraction=self.config.tab_span_fraction,
                num_controlled_fins=self.config.num_controlled_fins,
                cl_alpha=cl_alpha,
            )
            control_torque = effectiveness * self.tab_deflection

            # Velocity-dependent effectiveness
            speed_effectiveness = np.tanh(dynamic_pressure / 200.0)
            control_torque *= speed_effectiveness

            # Aerodynamic damping from airframe (with Mach-corrected cl_alpha)
            damping_coef = self.airframe.get_aerodynamic_damping_coeff(
                cl_alpha=cl_alpha
            )
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

        # Wind torque (pass cl_alpha scaled to ~2.0 convention used by wind_model)
        wind_torque = 0.0
        if self.wind_model is not None and dynamic_pressure > 1.0:
            wind_speed, wind_dir = self.wind_model.get_wind(
                self.time,
                self.altitude,
                rocket_velocity=max(self.vertical_velocity, 1.0),
            )
            wind_cl_alpha = (
                cl_alpha / np.pi
            )  # wind_model uses cl_alpha ~ 2.0 convention
            wind_torque = self.wind_model.get_roll_torque(
                wind_speed,
                wind_dir,
                self.roll_angle,
                self.vertical_velocity,
                dynamic_pressure,
                self.airframe,
                cl_alpha=wind_cl_alpha,
            )
            self._last_wind_speed = wind_speed
            self._last_wind_direction = wind_dir
            self._last_wind_torque = wind_torque

        # Store for info dict
        self._last_disturbance_torque = disturbance
        self._last_dynamic_pressure = dynamic_pressure

        return control_torque + damping_torque + disturbance + wind_torque

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
        rho, _, _ = self._get_atmosphere()
        return rho

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
        # Use same velocity floor as step() (v=0.1) so obs q matches physics q
        q = 0.5 * rho * max(self.vertical_velocity, 0.1) ** 2

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

        if roll_rate_deg < 5:
            h_quality = "Excellent - Stable footage"
        elif roll_rate_deg < 10:
            h_quality = "Good - Minor blur"
        elif roll_rate_deg < 20:
            h_quality = "Fair - Noticeable blur"
        else:
            h_quality = "Poor - Severe blur"

        # Compute mean spin rate
        mean_spin_rate = (
            self.spin_rate_sum / self.spin_rate_count
            if self.spin_rate_count > 0
            else abs(self.roll_rate)
        )

        # Air density at current altitude (use the same model as physics)
        air_density = self._get_air_density()

        return {
            "altitude_m": self.altitude,
            "vertical_velocity_ms": self.vertical_velocity,
            "roll_angle_rad": self.roll_angle,
            "vertical_acceleration_ms2": self.vertical_acceleration,
            "roll_rate_deg_s": np.rad2deg(self.roll_rate),
            "mean_spin_rate_deg_s": np.rad2deg(mean_spin_rate),
            "max_spin_rate_deg_s": np.rad2deg(self.max_spin_rate),
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
            # Atmospheric conditions
            "dynamic_pressure_Pa": self._last_dynamic_pressure,
            "disturbance_torque_Nm": self._last_disturbance_torque,
            "air_density_kg_m3": air_density,
            # Wind state
            "wind_speed_ms": self._last_wind_speed,
            "wind_direction_rad": self._last_wind_direction,
            "wind_torque_Nm": self._last_wind_torque,
            # Mach-dependent aero state
            "mach_number": self._last_mach,
            "speed_of_sound_ms": self._last_speed_of_sound,
            "cd": self._last_cd,
            "cl_alpha": self._last_cl_alpha,
            # Servo state
            "servo_position": self._servo_position,
            "servo_lag_deg": (self.last_action - self._servo_position)
            * self.config.max_tab_deflection,
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
        max_tab_deflection=30.0,
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
