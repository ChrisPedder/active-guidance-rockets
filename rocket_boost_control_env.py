import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any
import math


class RocketBoostControlEnv(gym.Env):
    """
    OpenAI Gym environment for rocket directional control during boost phase.

    The agent controls a single flap on one fin to minimize rotation and maintain
    vertical flight profile despite wind conditions.
    """

    metadata = {'render_modes': ['human', 'rgb_array'], 'render_fps': 50}

    def __init__(self, config: Dict[str, Any] = None):
        super().__init__()

        # Default configuration
        default_config = {
            'max_episode_steps': 1000,
            'dt': 0.02,  # 50 Hz simulation
            'max_wind_speed': 20.0,  # m/s
            'max_gust_speed': 10.0,  # m/s
            'max_flap_angle': 30.0,  # degrees
            'target_altitude': 3000.0,  # meters
            'rocket_mass': 500.0,  # kg
            'rocket_length': 10.0,  # meters
            'rocket_diameter': 0.5,  # meters
            'thrust': 7500.0,  # N (constant during boost)
            'air_density': 1.225,  # kg/m³ (sea level, constant)
        }

        self.config = {**default_config, **(config or {})}

        # Action space: flap angle in degrees
        self.action_space = spaces.Box(
            low=-self.config['max_flap_angle'],
            high=self.config['max_flap_angle'],
            shape=(1,),
            dtype=np.float32
        )

        # Observation space: [x, y, z, vx, vy, vz, roll, pitch, yaw, wx, wy, wz, wind_x, wind_y, wind_z]
        obs_high = np.array([
            5000,   # x position (m)
            5000,   # y position (m)
            4000,   # z position (m)
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
        ], dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-obs_high,
            high=obs_high,
            dtype=np.float32
        )

        # Initialize state
        self.reset()

        # Physical constants
        self.g = 9.81  # gravity

        # Rocket properties
        self.mass = self.config['rocket_mass']
        self.length = self.config['rocket_length']
        self.diameter = self.config['rocket_diameter']
        self.area = np.pi * (self.config['rocket_diameter'] / 2) ** 2

        # Moments of inertia (simplified as cylinder)
        self.Ixx = self.mass * (3 * (self.diameter/2)**2 + self.length**2) / 12
        self.Iyy = self.Ixx
        self.Izz = self.mass * (self.diameter/2)**2 / 2

        # Aerodynamic coefficients (simplified)
        self.Cd = 0.5  # drag coefficient
        self.Cl_alpha = 2.0  # lift coefficient per radian
        self.Cm_alpha = 0.1  # moment coefficient per radian

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Initial rocket state (launching vertically)
        self.position = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # x, y, z
        self.velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # vx, vy, vz
        self.attitude = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # roll, pitch, yaw
        self.angular_velocity = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # wx, wy, wz

        # Generate random wind conditions
        self._generate_wind()

        self.steps = 0
        self.flap_angle = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def _generate_wind(self):
        """Generate stochastic wind conditions"""
        # Base wind (steady state)
        wind_magnitude = np.random.uniform(0, self.config['max_wind_speed'])
        wind_direction = np.random.uniform(0, 2 * np.pi)
        self.base_wind = np.array([
            wind_magnitude * np.cos(wind_direction),
            wind_magnitude * np.sin(wind_direction),
            np.random.uniform(-5, 5)  # some vertical wind component
        ])

        # Gust parameters
        self.gust_frequency = np.random.uniform(0.1, 2.0)  # Hz
        self.gust_amplitude = np.random.uniform(0, self.config['max_gust_speed'])
        self.gust_phase = np.random.uniform(0, 2 * np.pi, 3)

    def _get_current_wind(self, t: float) -> np.ndarray:
        """Calculate current wind including gusts"""
        # Add sinusoidal gusts
        gust = self.gust_amplitude * np.array([
            np.sin(2 * np.pi * self.gust_frequency * t + self.gust_phase[0]),
            np.sin(2 * np.pi * self.gust_frequency * t + self.gust_phase[1]),
            np.sin(2 * np.pi * self.gust_frequency * t + self.gust_phase[2]) * 0.3
        ])

        return self.base_wind + gust

    def _rotation_matrix(self, roll: float, pitch: float, yaw: float) -> np.ndarray:
        """Create rotation matrix from Euler angles"""
        cr, sr = np.cos(roll), np.sin(roll)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cy, sy = np.cos(yaw), np.sin(yaw)

        R = np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])
        return R

    def _aerodynamic_forces(self, velocity: np.ndarray, wind: np.ndarray,
                           attitude: np.ndarray, flap_angle: float) -> Tuple[np.ndarray, np.ndarray]:
        """Calculate aerodynamic forces and moments"""
        # Relative velocity (rocket velocity relative to air)
        v_rel = velocity - wind
        v_mag = np.linalg.norm(v_rel)

        if v_mag < 0.1:  # Avoid division by zero
            return np.zeros(3), np.zeros(3)

        # Dynamic pressure
        q = 0.5 * self.config['air_density'] * v_mag**2

        # Rotation matrix from body to world frame
        R = self._rotation_matrix(attitude[0], attitude[1], attitude[2])

        # Velocity in body frame
        v_body = R.T @ v_rel

        # Angle of attack and sideslip
        if abs(v_body[2]) > 0.1:
            alpha = np.arctan2(v_body[0], v_body[2])  # pitch angle of attack
            beta = np.arctan2(v_body[1], v_body[2])   # yaw angle of attack
        else:
            alpha = beta = 0.0

        # Basic aerodynamic forces in body frame
        drag_body = -self.Cd * q * self.area * v_body / v_mag

        # Lift forces due to angle of attack
        lift_pitch = -self.Cl_alpha * alpha * q * self.area
        lift_yaw = -self.Cl_alpha * beta * q * self.area

        # Additional control force from flap (simplified)
        flap_rad = np.radians(flap_angle)
        flap_force_y = self.Cl_alpha * flap_rad * q * self.area * 0.1  # reduced effectiveness

        force_body = np.array([
            drag_body[0] + lift_pitch,
            drag_body[1] + lift_yaw + flap_force_y,
            drag_body[2]
        ])

        # Transform forces to world frame
        force_world = R @ force_body

        # Aerodynamic moments (simplified)
        moment_body = np.array([
            # Roll moment from flap
            flap_force_y * self.length * 0.3,
            # Pitch moment from angle of attack
            -self.Cm_alpha * alpha * q * self.area * self.length,
            # Yaw moment from sideslip
            -self.Cm_alpha * beta * q * self.area * self.length
        ])

        return force_world, moment_body

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one environment step"""
        # Clip and set flap angle
        self.flap_angle = np.clip(action[0], -self.config['max_flap_angle'],
                                 self.config['max_flap_angle'])

        dt = self.config['dt']
        t = self.steps * dt

        # Get current wind
        current_wind = self._get_current_wind(t)

        # Calculate forces
        # Thrust (along rocket's z-axis in body frame)
        R = self._rotation_matrix(self.attitude[0], self.attitude[1], self.attitude[2])
        thrust_world = R @ np.array([0, 0, self.config['thrust']])

        # Gravity
        gravity_force = np.array([0, 0, -self.mass * self.g])

        # Aerodynamic forces and moments
        aero_force, aero_moment = self._aerodynamic_forces(
            self.velocity, current_wind, self.attitude, self.flap_angle
        )

        # Total forces
        total_force = thrust_world + gravity_force + aero_force

        # Linear dynamics
        acceleration = total_force / self.mass
        self.velocity += acceleration * dt
        self.position += self.velocity * dt

        # Angular dynamics (simplified)
        I = np.array([self.Ixx, self.Iyy, self.Izz])
        angular_acceleration = aero_moment / I
        self.angular_velocity += angular_acceleration * dt
        self.attitude += self.angular_velocity * dt

        # Keep angles in reasonable range
        self.attitude = np.clip(self.attitude, -np.pi, np.pi)

        self.steps += 1

        # Calculate reward
        reward = self._calculate_reward(current_wind)

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.steps >= self.config['max_episode_steps']

        observation = self._get_observation()
        info = self._get_info()
        info['wind'] = current_wind
        info['flap_angle'] = self.flap_angle

        return observation, reward, terminated, truncated, info

    def _calculate_reward(self, wind: np.ndarray) -> float:
        """Calculate dense reward function"""
        # Attitude control reward (minimize rotation)
        attitude_penalty = -np.sum(np.abs(self.attitude)) * 10
        angular_velocity_penalty = -np.sum(np.abs(self.angular_velocity)) * 5

        # Vertical flight reward (minimize horizontal velocity)
        horizontal_velocity = np.sqrt(self.velocity[0]**2 + self.velocity[1]**2)
        horizontal_penalty = -horizontal_velocity * 2

        # Vertical velocity reward (should be positive)
        vertical_reward = max(0, self.velocity[2]) * 0.5

        # Altitude progress reward
        altitude_reward = self.position[2] * 0.01

        # Control effort penalty (smooth control)
        control_penalty = -abs(self.flap_angle) * 0.01

        # Bonus for staying within reasonable bounds
        stability_bonus = 0
        if (abs(self.attitude[0]) < 0.1 and abs(self.attitude[1]) < 0.1 and
            abs(self.attitude[2]) < 0.1 and horizontal_velocity < 10):
            stability_bonus = 5

        total_reward = (attitude_penalty + angular_velocity_penalty +
                       horizontal_penalty + vertical_reward + altitude_reward +
                       control_penalty + stability_bonus)

        return float(total_reward)

    def _is_terminated(self) -> bool:
        """Check if episode should terminate"""
        # Crash conditions
        if self.position[2] < 0:  # Ground impact
            return True

        # Excessive attitude deviation
        if np.any(np.abs(self.attitude) > np.pi/3):  # 60 degrees
            return True

        # Excessive horizontal drift
        if np.sqrt(self.position[0]**2 + self.position[1]**2) > 1000:
            return True

        # Success condition
        if self.position[2] > self.config['target_altitude']:
            return True

        return False

    def _get_observation(self) -> np.ndarray:
        """Get current observation"""
        t = self.steps * self.config['dt']
        current_wind = self._get_current_wind(t)

        obs = np.concatenate([
            self.position,
            self.velocity,
            self.attitude,
            self.angular_velocity,
            current_wind
        ]).astype(np.float32)

        return obs

    def _get_info(self) -> Dict:
        """Get additional info"""
        return {
            'position': self.position.copy(),
            'velocity': self.velocity.copy(),
            'attitude_degrees': np.degrees(self.attitude).copy(),
            'angular_velocity': self.angular_velocity.copy(),
            'altitude': float(self.position[2]),
            'horizontal_distance': float(np.sqrt(self.position[0]**2 + self.position[1]**2)),
            'steps': self.steps
        }

    def render(self, mode='human'):
        """Render the environment (basic implementation)"""
        if mode == 'human':
            print(f"Step: {self.steps}")
            print(f"Position: {self.position}")
            print(f"Attitude (deg): {np.degrees(self.attitude)}")
            print(f"Flap angle: {self.flap_angle:.1f}°")
            print(f"Altitude: {self.position[2]:.1f}m")
            print("---")

    def close(self):
        """Clean up resources"""
        pass


# Example usage and testing
if __name__ == "__main__":
    # Create environment
    env = RocketBoostControlEnv()

    print("Rocket Boost Control Environment")
    print(f"Action space: {env.action_space}")
    print(f"Observation space: {env.observation_space}")

    # Test random agent
    obs, info = env.reset()
    total_reward = 0

    for step in range(100):
        # Random action
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward

        if step % 20 == 0:
            env.render()

        if terminated or truncated:
            print(f"Episode finished after {step} steps")
            print(f"Final altitude: {info['altitude']:.1f}m")
            print(f"Total reward: {total_reward:.2f}")
            break

    env.close()
