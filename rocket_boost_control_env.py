# realistic_rocket_control_env.py

import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any, List
import math
from dataclasses import dataclass


@dataclass
class FinConfiguration:
    """Configuration for a single fin"""
    root_chord: float  # m - chord length at rocket body
    tip_chord: float   # m - chord length at fin tip
    span: float        # m - distance from body to tip
    sweep_angle: float # rad - sweep angle of leading edge
    thickness: float   # m - fin thickness
    position_x: float  # m - axial position from nose
    azimuth_angle: float # rad - angular position around rocket body
    has_control_surface: bool = False
    control_surface_fraction: float = 0.3  # fraction of chord that's controllable


@dataclass
class RocketGeometry:
    """Complete rocket geometry definition"""
    total_length: float = 1.2       # m
    diameter: float = 0.067         # m
    nose_length: float = 0.3        # m
    body_length: float = 0.9        # m
    boat_tail_length: float = 0.0   # m

    # Mass properties
    dry_mass: float = 5.0          # kg
    propellant_mass: float = 1.2   # kg (decreases over time)
    cg_empty: float = 0.8          # m from nose when empty
    cg_full: float = 0.9           # m from nose when full

    # Fins
    fins: List[FinConfiguration] = None

    def __post_init__(self):
        if self.fins is None:
            # Default 4-fin configuration
            self.fins = [
                FinConfiguration(
                    root_chord=0.08, tip_chord=0.04, span=0.07, sweep_angle=np.pi/6,
                    thickness=0.005, position_x=8.0, azimuth_angle=i*np.pi/2,
                    has_control_surface=(i == 0), control_surface_fraction=0.3
                ) for i in range(4)
            ]


class RocketBoostControlEnv(gym.Env):
    """
    Realistic OpenAI Gym environment for rocket directional control during boost phase.

    Features:
    - Proper fin geometry and aerodynamic modeling
    - Center of gravity and center of pressure calculations
    - Realistic moment arm calculations
    - Individual fin control surfaces
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, config: Dict[str, Any] = None, rocket_config: RocketGeometry = None):
        super().__init__()

        # Default simulation configuration
        default_config = {
            'max_episode_steps': 1000,
            'dt': 0.02,  # 50 Hz simulation
            'max_wind_speed': 10.0,  # m/s
            'max_gust_speed': 20.0,  # m/s
            'max_flap_angle': 30.0,  # degrees
            'target_altitude': 300.0,  # meters
            'thrust': 90.0,  # N (constant during boost)
            'air_density': 1.225,  # kg/m³ (sea level)
            'burn_time': 3.0,  # seconds
        }

        self.config = {**default_config, **(config or {})}
        self.rocket = rocket_config or RocketGeometry()

        # Find controllable fin
        self.control_fin_idx = None
        for i, fin in enumerate(self.rocket.fins):
            if fin.has_control_surface:
                self.control_fin_idx = i
                break

        if self.control_fin_idx is None:
            raise ValueError("No controllable fin found in rocket configuration")

        # Action space: control surface deflection angle in degrees
        self.action_space = spaces.Box(
            low=-self.config['max_flap_angle'],
            high=self.config['max_flap_angle'],
            shape=(1,),
            dtype=np.float32
        )

        # Extended observation space including CG/CP info
        obs_high = np.array([
            1000,   # x position (m)
            1000,   # y position (m)
            1000,   # z position (m)
            200,    # vx velocity (m/s)
            200,    # vy velocity (m/s)
            200,    # vz velocity (m/s)
            np.pi,  # roll angle (rad)
            np.pi,  # pitch angle (rad)
            np.pi,  # yaw angle (rad)
            10,     # roll rate (rad/s)
            10,     # pitch rate (rad/s)
            10,     # yaw rate (rad/s)
            30,     # wind_x (m/s)
            30,     # wind_y (m/s)
            30,     # wind_z (m/s)
            2.0,    # static margin (CP-CG)/diameter
            1000,   # dynamic pressure (Pa)
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        # Initialize state
        self.reset()

        # Physical constants
        self.g = 9.81

        # Calculate moments of inertia based on realistic geometry
        self._calculate_moments_of_inertia()

    def _calculate_moments_of_inertia(self):
        """Calculate moments of inertia for rocket + fins"""
        # Body (cylinder)
        m_body = self.rocket.dry_mass + self.rocket.propellant_mass
        r = self.rocket.diameter / 2
        l = self.rocket.total_length

        # Cylinder moments of inertia
        self.Ixx_body = m_body * r**2 / 2  # roll
        self.Iyy_body = m_body * (3*r**2 + l**2) / 12  # pitch
        self.Izz_body = self.Iyy_body  # yaw

        # Add fin contributions
        self.Ixx_fins = 0
        self.Iyy_fins = 0
        self.Izz_fins = 0

        for fin in self.rocket.fins:
            # Simplified fin mass (assuming uniform density)
            fin_volume = fin.root_chord * fin.span * fin.thickness
            fin_mass = fin_volume * 2700  # aluminum density kg/m³

            # Distance from rocket centerline
            r_fin = self.rocket.diameter/2 + fin.span/2

            # Add to moments of inertia (parallel axis theorem)
            self.Ixx_fins += fin_mass * (fin.span**2 / 12)
            self.Iyy_fins += fin_mass * (r_fin**2 + (fin.position_x - self.get_current_cg())**2)
            self.Izz_fins += fin_mass * (r_fin**2 + (fin.position_x - self.get_current_cg())**2)

        # Total moments of inertia
        self.Ixx = self.Ixx_body + self.Ixx_fins
        self.Iyy = self.Iyy_body + self.Iyy_fins
        self.Izz = self.Izz_body + self.Izz_fins

    def get_current_cg(self) -> float:
        """Calculate current center of gravity based on propellant consumption"""
        # Linear interpolation based on remaining propellant
        prop_fraction = max(0, (self.config['burn_time'] - self.burn_time) / self.config['burn_time'])
        current_prop_mass = self.rocket.propellant_mass * prop_fraction

        if current_prop_mass > 0:
            # Weighted average of empty and full CG
            total_mass = self.rocket.dry_mass + current_prop_mass
            cg = ((self.rocket.dry_mass * self.rocket.cg_empty +
                   current_prop_mass * self.rocket.cg_full) / total_mass)
        else:
            cg = self.rocket.cg_empty

        return cg

    def get_current_mass(self) -> float:
        """Calculate current total mass"""
        prop_fraction = max(0, (self.config['burn_time'] - self.burn_time) / self.config['burn_time'])
        current_prop_mass = self.rocket.propellant_mass * prop_fraction
        return self.rocket.dry_mass + current_prop_mass

    def calculate_center_of_pressure(self, alpha: float, beta: float) -> float:
        """Calculate center of pressure based on current angle of attack"""
        # Simplified CP calculation - more sophisticated models would use CFD data

        # Body contribution (typically around 2/3 of body length from nose)
        cp_body = self.rocket.nose_length + 2 * self.rocket.body_length / 3

        # Fin contribution - fins dominate at higher angles of attack
        total_fin_area = 0
        weighted_fin_cp = 0

        for fin in self.rocket.fins:
            fin_area = (fin.root_chord + fin.tip_chord) * fin.span / 2
            total_fin_area += fin_area

            # Fin CP is approximately at 25% of mean aerodynamic chord
            mac = (fin.root_chord + fin.tip_chord) / 2
            fin_cp = fin.position_x + 0.25 * mac
            weighted_fin_cp += fin_area * fin_cp

        if total_fin_area > 0:
            avg_fin_cp = weighted_fin_cp / total_fin_area
        else:
            avg_fin_cp = cp_body

        # Body area (cross-sectional for angle of attack effects)
        body_area = np.pi * (self.rocket.diameter/2)**2

        # Weight CP based on relative contributions
        angle_magnitude = np.sqrt(alpha**2 + beta**2)
        if angle_magnitude > 0.01:  # Fins dominate at angle
            fin_weight = min(angle_magnitude / 0.2, 0.8)  # Max 80% fin influence
        else:
            fin_weight = 0.1  # Small fin influence at zero angle

        cp = cp_body * (1 - fin_weight) + avg_fin_cp * fin_weight

        return cp

    def calculate_static_margin(self, alpha: float, beta: float) -> float:
        """Calculate static margin (CP-CG)/diameter"""
        cg = self.get_current_cg()
        cp = self.calculate_center_of_pressure(alpha, beta)
        return (cp - cg) / self.rocket.diameter

    def _aerodynamic_forces_and_moments(self, velocity: np.ndarray, wind: np.ndarray,
                                        attitude: np.ndarray, angular_velocity: np.ndarray,
                                        control_deflection: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate realistic aerodynamic forces and moments with proper control surface and damping"""

        # Relative velocity
        v_rel = velocity - wind
        v_mag = np.linalg.norm(v_rel)

        if v_mag < 0.1:
            return np.zeros(3), np.zeros(3)

        # Dynamic pressure
        q = 0.5 * self.config['air_density'] * v_mag**2

        # Rotation matrix from body to world frame
        R = self._rotation_matrix(attitude[0], attitude[1], attitude[2])

        # Velocity in body frame (assuming body frame: x=forward, y=right, z=down)
        v_body = R.T @ v_rel

        # Angle of attack and sideslip (FIXED)
        # For rockets, typically x is along the longitudinal axis
        if abs(v_body[0]) > 0.1:  # Forward velocity check
            alpha = np.arctan2(-v_body[2], v_body[0])  # Angle of attack (pitch plane)
            beta = np.arctan2(v_body[1], v_body[0])    # Sideslip angle (yaw plane)
        else:
            alpha = beta = 0.0

        # Reference areas
        body_area = np.pi * (self.rocket.diameter/2)**2
        fin_area = sum((fin.root_chord + fin.tip_chord) * fin.span / 2 for fin in self.rocket.fins)

        # ============== AERODYNAMIC COEFFICIENTS ==============

        # Drag coefficient (function of angle of attack)
        cd_0 = 0.15  # Zero-angle drag
        cd_alpha = 1.5  # Additional drag due to angle of attack
        cd = cd_0 + cd_alpha * (alpha**2 + beta**2)

        # Lift/Normal force coefficients
        cn_alpha_body = 2.0  # Normal force derivative for body
        cn_alpha_fins = 0.1 * len(self.rocket.fins)  # Total fin contribution

        # ============== BASIC AERODYNAMIC FORCES ==============

        # Forces in body frame
        # Axial force (drag along x-axis)
        axial_force = -cd * q * body_area

        # Normal forces (perpendicular to body axis)
        normal_force_z = cn_alpha_body * alpha * q * body_area + cn_alpha_fins * alpha * q * fin_area
        normal_force_y = cn_alpha_body * beta * q * body_area + cn_alpha_fins * beta * q * fin_area

        # ============== CONTROL SURFACE FORCES AND MOMENTS ==============

        control_fin = self.rocket.fins[self.control_fin_idx]
        control_area = control_fin.root_chord * control_fin.span * control_fin.control_surface_fraction

        # Control surface deflection in radians
        control_rad = np.radians(control_deflection)
        control_effectiveness = 0.8

        # Control surface generates a normal force perpendicular to fin
        control_normal_force = control_effectiveness * 0.1 * control_rad * q * control_area

        # The control force acts at the control surface center (simplified: 75% of fin position)
        control_surface_x = control_fin.position_x + 0.25 * (control_fin.root_chord + control_fin.tip_chord) / 2
        control_surface_y = (self.rocket.diameter/2 + control_fin.span/2) * np.cos(control_fin.azimuth_angle)
        control_surface_z = (self.rocket.diameter/2 + control_fin.span/2) * np.sin(control_fin.azimuth_angle)

        # Control force vector in body frame (perpendicular to fin surface)
        # For a fin with control surface, deflection creates force normal to fin
        if abs(control_fin.azimuth_angle) < 0.1:  # Fin at 0° (top)
            control_force_body = np.array([0, 0, control_normal_force])
        elif abs(control_fin.azimuth_angle - np.pi/2) < 0.1:  # Fin at 90° (right)
            control_force_body = np.array([0, -control_normal_force, 0])
        elif abs(control_fin.azimuth_angle - np.pi) < 0.1:  # Fin at 180° (bottom)
            control_force_body = np.array([0, 0, -control_normal_force])
        else:  # Fin at 270° (left)
            control_force_body = np.array([0, control_normal_force, 0])

        # ============== TOTAL FORCES ==============

        # Total forces in body frame (before control surface)
        force_body = np.array([
            axial_force,
            normal_force_y,
            normal_force_z
        ])

        # Add control surface force
        force_body += control_force_body

        # Transform to world frame
        force_world = R @ force_body

        # ============== MOMENTS CALCULATION ==============

        # Get current CG and CP
        cg = self.get_current_cg()
        cp = self.calculate_center_of_pressure(alpha, beta)

        # Position vectors from CG
        cp_position = np.array([cp, 0, 0])  # CP is along x-axis
        cg_position = np.array([cg, 0, 0])

        # 1. Moment from normal forces acting at CP
        normal_force_vector = np.array([0, normal_force_y, normal_force_z])
        moment_arm_cp = cp - cg

        # Basic aerodynamic moments
        aero_moment = np.array([
            0,  # No roll moment from symmetric forces
            -normal_force_z * moment_arm_cp,  # Pitch moment
            normal_force_y * moment_arm_cp    # Yaw moment
        ])

        # 2. Control surface moment (properly calculated using cross product)
        control_position = np.array([
            control_surface_x - cg,
            control_surface_y,
            control_surface_z
        ])

        # Moment = r × F (cross product of position vector and force)
        control_moment = np.cross(control_position, control_force_body)

        # 3. Aerodynamic damping moments
        # These oppose rotation and provide stability
        damping_moment = self._calculate_damping_moments(angular_velocity, q)

        # ============== TOTAL MOMENTS ==============

        moment_body = aero_moment + control_moment + damping_moment

        return force_world, moment_body


    def _calculate_damping_moments(self, angular_velocity: np.ndarray, q: float) -> np.ndarray:
        """
        Calculate aerodynamic damping moments that oppose rotation.
        These are crucial for stability and preventing uncontrolled spinning.
        """

        # Reference dimensions
        ref_area = np.pi * (self.rocket.diameter/2)**2
        ref_length = self.rocket.total_length

        # Damping derivatives (these would ideally come from wind tunnel or CFD data)
        # Negative values mean damping opposes rotation

        # Roll damping - mainly from fins
        total_fin_area = sum((fin.root_chord + fin.tip_chord) * fin.span / 2
                            for fin in self.rocket.fins)
        Clp = -0.5 * (total_fin_area / ref_area)  # Roll damping coefficient

        # Pitch damping - from body and fins
        # Larger value = more damping
        Cmq = -10.0 * (1.0 + total_fin_area / ref_area)  # Pitch damping coefficient

        # Yaw damping - similar to pitch for symmetric rocket
        Cnr = Cmq  # Yaw damping coefficient

        # Calculate damping moments
        # M_damping = 0.5 * ρ * V * S * L * C * ω
        # But since q = 0.5 * ρ * V², we have:
        # M_damping = (q/V) * S * L * C * ω

        v_mag = np.sqrt(2 * q / self.config['air_density'])

        if v_mag > 0.1:
            damping_moments = np.array([
                Clp * angular_velocity[0] * q * self.rocket.diameter**2 / v_mag,
                Cmq * angular_velocity[1] * q * ref_area * ref_length / v_mag,
                Cnr * angular_velocity[2] * q * ref_area * ref_length / v_mag
            ])
        else:
            # Simple velocity-independent damping at low speeds
            damping_moments = np.array([
                -0.1 * angular_velocity[0],
                -1.0 * angular_velocity[1],
                -1.0 * angular_velocity[2]
            ])

        return damping_moments

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Create rotation matrix using aerospace Z-Y-X Euler sequence"""
        # First yaw, then pitch, then roll
        Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                       [np.sin(yaw), np.cos(yaw), 0],
                       [0, 0, 1]])
        Ry = np.array([[np.cos(pitch), 0, np.sin(pitch)],
                       [0, 1, 0],
                       [-np.sin(pitch), 0, np.cos(pitch)]])
        Rx = np.array([[1, 0, 0],
                       [0, np.cos(roll), -np.sin(roll)],
                       [0, np.sin(roll), np.cos(roll)]])
        return Rz @ Ry @ Rx

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Initial state
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.attitude = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)

        self.steps = 0
        self.burn_time = 0.0  # Time since ignition
        self.control_deflection = 0.0

        # Generate wind conditions
        self._generate_wind()

        # Recalculate moments of inertia for current mass
        self._calculate_moments_of_inertia()

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _generate_wind(self):
        """Generate stochastic wind conditions"""
        wind_magnitude = np.random.uniform(0, self.config['max_wind_speed'])
        wind_direction = np.random.uniform(0, 2 * np.pi)
        self.base_wind = np.array([
            wind_magnitude * np.cos(wind_direction),
            wind_magnitude * np.sin(wind_direction),
            np.random.uniform(-5, 5)
        ])

        self.gust_frequency = np.random.uniform(0.1, 2.0)
        self.gust_amplitude = np.random.uniform(0, self.config['max_gust_speed'])
        self.gust_phase = np.random.uniform(0, 2 * np.pi, 3)

    def _get_current_wind(self, t: float) -> np.ndarray:
        """Calculate current wind including gusts"""
        gust = self.gust_amplitude * np.array([
            np.sin(2 * np.pi * self.gust_frequency * t + self.gust_phase[0]),
            np.sin(2 * np.pi * self.gust_frequency * t + self.gust_phase[1]),
            np.sin(2 * np.pi * self.gust_frequency * t + self.gust_phase[2]) * 0.3
        ])
        return self.base_wind + gust

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        self.control_deflection = np.clip(action[0], -self.config['max_flap_angle'],
                                        self.config['max_flap_angle'])

        dt = self.config['dt']
        t = self.steps * dt
        self.burn_time = t

        # Update mass properties
        current_mass = self.get_current_mass()
        self._calculate_moments_of_inertia()

        # Get current wind
        current_wind = self._get_current_wind(t)

        # Calculate forces
        # Thrust (along rocket's z-axis in body frame)
        if self.burn_time < self.config['burn_time']:
            thrust_magnitude = self.config['thrust']
        else:
            thrust_magnitude = 0.0  # Burnout

        R = self._rotation_matrix(self.attitude[0], self.attitude[1], self.attitude[2])
        thrust_world = R @ np.array([0, 0, thrust_magnitude])

        # Gravity
        gravity_force = np.array([0, 0, -current_mass * self.g])

        # Aerodynamic forces and moments
        aero_force, aero_moment = self._aerodynamic_forces_and_moments(
            self.velocity, current_wind, self.attitude, self.angular_velocity, self.control_deflection
        )
        # Total forces
        total_force = thrust_world + gravity_force + aero_force

        # Linear dynamics
        acceleration = total_force / current_mass
        acceleration = np.nan_to_num(acceleration, nan=0.0, posinf=100.0, neginf=-100.0)
        acceleration = np.clip(acceleration, -200, 200)

        self.velocity += acceleration * dt
        self.velocity = np.clip(self.velocity, -500, 500)
        self.position += self.velocity * dt

        # Angular dynamics
        I = np.array([self.Ixx, self.Iyy, self.Izz])
        angular_acceleration = aero_moment / I
        angular_acceleration = np.nan_to_num(angular_acceleration, nan=0.0, posinf=10.0, neginf=-10.0)
        angular_acceleration = np.clip(angular_acceleration, -20, 20)

        self.angular_velocity += angular_acceleration * dt
        self.angular_velocity = np.clip(self.angular_velocity, -10, 10)
        self.attitude += self.angular_velocity * dt

        # Wrap angles
        self.attitude = np.arctan2(np.sin(self.attitude), np.cos(self.attitude))

        self.steps += 1

        # Calculate reward
        reward = self._calculate_reward(current_wind)

        # Check termination
        terminated = self._is_terminated()
        truncated = self.steps >= self.config['max_episode_steps']

        observation = self._get_observation()
        info = self._get_info()
        info.update({
            'wind': current_wind,
            'control_deflection': self.control_deflection,
            'static_margin': self.calculate_static_margin(0, 0),  # Simplified for info
            'cg': self.get_current_cg(),
            'mass': current_mass
        })

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, wind: np.ndarray) -> float:
        """Calculate reward with stability considerations"""
        if (np.any(np.isnan(self.position)) or np.any(np.isnan(self.velocity)) or
            np.any(np.isnan(self.attitude)) or np.any(np.isnan(self.angular_velocity))):
            return -100.0

        # Get current angle of attack
        R = self._rotation_matrix(self.attitude[0], self.attitude[1], self.attitude[2])
        v_body = R.T @ self.velocity
        if np.linalg.norm(v_body) > 0.1:
            alpha = np.arctan2(v_body[2], np.sqrt(v_body[0]**2 + v_body[1]**2))
            beta = np.arctan2(v_body[1], v_body[2])
        else:
            alpha = beta = 0.0

        # Static stability reward (positive static margin is good)
        static_margin = self.calculate_static_margin(alpha, beta)
        stability_reward = np.clip(static_margin - 0.5, -2, 2)  # Target margin of 0.5

        # Attitude penalties
        attitude_penalty = -np.sum(np.abs(self.attitude)) * 2

        # Angular velocity penalty
        angular_penalty = -np.sum(np.abs(self.angular_velocity)) * 1

        # Vertical velocity reward
        vertical_reward = np.clip(self.velocity[2] / 100, -1, 2)

        # Altitude progress
        altitude_reward = np.clip(self.position[2] / self.config['target_altitude'], 0, 1) * 3

        # Control effort penalty
        control_penalty = -abs(self.control_deflection) / self.config['max_flap_angle'] * 0.5

        total_reward = (stability_reward + attitude_penalty + angular_penalty +
                       vertical_reward + altitude_reward + control_penalty)

        return np.clip(float(total_reward), -50, 50)

    def _is_terminated(self) -> bool:
        """Check termination conditions"""
        if self.position[2] < 0:  # Ground impact
            return True
        if np.any(np.abs(self.attitude) > np.pi/3):  # Excessive attitude
            return True
        if np.sqrt(self.position[0]**2 + self.position[1]**2) > 2000:  # Excessive drift
            return True
        if self.position[2] > self.config['target_altitude']:  # Success
            return True
        return False

    def _get_observation(self) -> np.ndarray:
        """Get current observation including stability metrics"""
        t = self.steps * self.config['dt']
        current_wind = self._get_current_wind(t)

        # Calculate angle of attack for static margin
        R = self._rotation_matrix(self.attitude[0], self.attitude[1], self.attitude[2])
        v_body = R.T @ self.velocity
        if np.linalg.norm(v_body) > 0.1:
            alpha = np.arctan2(v_body[2], np.sqrt(v_body[0]**2 + v_body[1]**2))
            beta = np.arctan2(v_body[1], v_body[2])
        else:
            alpha = beta = 0.0

        static_margin = self.calculate_static_margin(alpha, beta)
        dynamic_pressure = 0.5 * self.config['air_density'] * np.linalg.norm(self.velocity)**2

        obs = np.concatenate([
            self.position,
            self.velocity,
            self.attitude,
            self.angular_velocity,
            current_wind,
            [static_margin, dynamic_pressure]
        ]).astype(np.float32)

        obs = np.nan_to_num(obs, nan=0.0, posinf=1e6, neginf=-1e6)
        return np.clip(obs, -1e6, 1e6)

    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'attitude_degrees': np.degrees(self.attitude).copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'altitude': float(self.position[2]),
            'horizontal_distance': float(np.sqrt(self.position[0]**2 + self.position[1]**2)),
            'steps': self.steps,
            'burn_time': self.burn_time,
            'fin_configuration': f"{len(self.rocket.fins)} fins"
        }

    def render(self, mode='human'):
        """Render the environment"""
        if mode == 'human':
            print(f"Step: {self.steps}, Burn time: {self.burn_time:.1f}s")
            print(f"Position: {self.position}")
            print(f"Attitude (deg): {np.degrees(self.attitude)}")
            print(f"Control deflection: {self.control_deflection:.1f}°")
            print(f"CG: {self.get_current_cg():.2f}m, Mass: {self.get_current_mass():.0f}kg")
            print(f"Static margin: {self.calculate_static_margin(0, 0):.2f}")
            print("---")

    def close(self):
        """Clean up resources"""
        pass


# Example usage
if __name__ == "__main__":
    # Create custom rocket configuration
    custom_rocket = RocketGeometry(
        total_length=12.0,
        diameter=0.6,
        dry_mass=600.0,
        propellant_mass=300.0,
        cg_empty=7.0,
        cg_full=6.2,
        fins=[
            FinConfiguration(
                root_chord=1.0, tip_chord=0.5, span=0.4, sweep_angle=np.pi/4,
                thickness=0.008, position_x=10.0, azimuth_angle=0,
                has_control_surface=True, control_surface_fraction=0.4
            ),
            FinConfiguration(
                root_chord=1.0, tip_chord=0.5, span=0.4, sweep_angle=np.pi/4,
                thickness=0.008, position_x=10.0, azimuth_angle=np.pi/2,
                has_control_surface=False
            ),
            FinConfiguration(
                root_chord=1.0, tip_chord=0.5, span=0.4, sweep_angle=np.pi/4,
                thickness=0.008, position_x=10.0, azimuth_angle=np.pi,
                has_control_surface=False
            ),
            FinConfiguration(
                root_chord=1.0, tip_chord=0.5, span=0.4, sweep_angle=np.pi/4,
                thickness=0.008, position_x=10.0, azimuth_angle=3*np.pi/2,
                has_control_surface=False
            )
        ]
    )

    # Create environment
    env = RocketBoostControlEnv(rocket_config=custom_rocket)

    print("Realistic Rocket Boost Control Environment")
    print(f"Rocket configuration: {len(env.rocket.fins)} fins")
    print(f"Control fin at azimuth: {np.degrees(env.rocket.fins[env.control_fin_idx].azimuth_angle):.1f}°")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test episode
    obs, info = env.reset()
    total_reward = 0

    for step in range(100):
        # Simple control law (try to minimize pitch angle)
        pitch_angle = obs[7]  # pitch angle from observation
        action = np.array([-pitch_angle * 10])  # Simple proportional control
        action = np.clip(action, env.action_space.low, env.action_space.high)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 25 == 0:
            env.render()

        if terminated or truncated:
            print(f"Episode finished after {step} steps")
            print(f"Final altitude: {info['altitude']:.1f}m")
            print(f"Final static margin: {info.get('static_margin', 'N/A')}")
            print(f"Total reward: {total_reward:.2f}")
            break

    env.close()
