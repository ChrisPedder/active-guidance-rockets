import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Dict, Any
from dataclasses import dataclass
from enum import Enum

class CameraView(Enum):
    HORIZONTAL = "horizontal"  # Looking radially outward
    DOWNWARD = "downward"      # Looking down along rocket body

@dataclass
class CompositeRocketConfig:
    """Configuration for small composite rocket with camera system"""
    # Rocket parameters (typical small composite rocket)
    dry_mass: float = 1.8          # kg - fiberglass body
    propellant_mass: float = 0.25   # kg - small solid motor
    diameter: float = 0.054         # m - 54mm standard
    length: float = 0.75            # m
    wall_thickness: float = 0.002   # m - typical composite thickness

    # Motor parameters
    thrust_curve: str = "progressive"  # progressive, regressive, or neutral
    average_thrust: float = 120.0      # N
    burn_time: float = 1.8             # seconds

    # Fin configuration for roll control
    num_fins: int = 4
    fin_span: float = 0.06              # m - from body
    fin_root_chord: float = 0.08        # m
    fin_thickness: float = 0.003        # m - G10 fiberglass
    fin_position: float = 0.65          # m - from nose tip

    # Roll tabs on fins (small controllable surfaces)
    tab_chord_fraction: float = 0.25    # fraction of fin chord
    tab_span_fraction: float = 0.5      # fraction of fin span
    max_tab_deflection: float = 20.0    # degrees
    num_controlled_fins: int = 2        # opposite fins for roll control

    # Camera configuration
    horizontal_camera_fov: float = 120.0  # degrees
    downward_camera_fov: float = 90.0     # degrees
    camera_sample_rate: float = 30.0      # fps

    # Simulation parameters
    dt: float = 0.01                    # 100 Hz simulation
    max_altitude: float = 1000.0        # m - expected apogee
    max_roll_rate: float = 720.0        # deg/s - 2 revs/second max

class SpinStabilizedCameraRocket(gym.Env):
    """
    Rocket environment optimized for camera stabilization using fin tab control.
    Goals:
    1. Minimize roll rate for stable horizontal camera footage
    2. Maintain aerodynamic stability during boost and coast
    3. Work effectively with composite material dynamics
    """

    metadata = {'render_modes': ['human', 'camera_view']}

    def __init__(self, config: CompositeRocketConfig = None):
        super().__init__()

        self.config = config or CompositeRocketConfig()

        # Action space: deflection angle for roll tabs (normalized -1 to 1)
        # Positive action -> positive roll torque
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(1,),
            dtype=np.float32
        )

        # Observation space focused on roll control and camera stability
        self.observation_space = spaces.Box(
            low=np.array([
                0.0,      # altitude (m)
                -200.0,   # vertical velocity (m/s)
                -np.pi,   # roll angle (rad)
                -np.deg2rad(self.config.max_roll_rate),  # roll rate (rad/s)
                -10.0,    # roll acceleration (rad/s²)
                0.0,      # dynamic pressure (Pa)
                0.0,      # time since launch (s)
                0.0,      # thrust fraction (0-1)
                -1.0,     # last action
                0.0,      # camera shake metric
            ], dtype=np.float32),
            high=np.array([
                self.config.max_altitude,
                200.0,    # vertical velocity (m/s)
                np.pi,    # roll angle (rad)
                np.deg2rad(self.config.max_roll_rate),   # roll rate (rad/s)
                10.0,     # roll acceleration (rad/s²)
                5000.0,   # dynamic pressure (Pa)
                20.0,     # time since launch (s)
                1.0,      # thrust fraction (0-1)
                1.0,      # last action
                100.0,    # camera shake metric
            ], dtype=np.float32),
            dtype=np.float32
        )

        # Initialize state variables
        self.reset()

    def reset(self, seed=None, options=None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)

        # Initial conditions
        self.altitude = 0.0
        self.vertical_velocity = 0.0
        self.vertical_acceleration = 0.0

        # Roll state (main focus)
        self.roll_angle = 0.0
        self.roll_rate = np.random.normal(0, np.deg2rad(30))  # Small initial disturbance
        self.roll_acceleration = 0.0

        # Time and propellant
        self.time = 0.0
        self.propellant_remaining = self.config.propellant_mass

        # Control state
        self.last_action = 0.0
        self.tab_deflection = 0.0

        # Camera metrics
        self.camera_shake_history = []
        self.total_rotation = 0.0  # Total roll for horizontal camera

        # Performance tracking
        self.max_altitude_reached = 0.0
        self.integrated_roll_error = 0.0

        return self._get_observation(), self._get_info()

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one time step of roll control"""

        # Process action - convert to tab deflection
        self.last_action = float(action[0])
        self.tab_deflection = self.last_action * np.deg2rad(self.config.max_tab_deflection)

        # Time update
        dt = self.config.dt
        self.time += dt

        # Update mass and thrust
        thrust, mass = self._update_propulsion()

        # Calculate aerodynamic conditions
        q = 0.5 * self._get_air_density() * self.vertical_velocity**2  # Dynamic pressure

        # Vertical dynamics (simplified - rocket assumed to be vertical)
        drag_coefficient = 0.35 if self.time < self.config.burn_time else 0.45  # Higher Cd after burnout
        frontal_area = np.pi * (self.config.diameter/2)**2
        drag_force = drag_coefficient * q * frontal_area

        self.vertical_acceleration = (thrust - drag_force - mass * 9.81) / mass
        self.vertical_velocity += self.vertical_acceleration * dt
        self.altitude += self.vertical_velocity * dt

        # Roll dynamics (main focus)
        roll_torque = self._calculate_roll_torque(q)
        I_roll = self._calculate_roll_inertia(mass)

        self.roll_acceleration = roll_torque / I_roll
        self.roll_rate += self.roll_acceleration * dt
        self.roll_angle += self.roll_rate * dt

        # Wrap roll angle to [-π, π]
        self.roll_angle = np.arctan2(np.sin(self.roll_angle), np.cos(self.roll_angle))

        # Update total rotation for camera metric
        self.total_rotation += abs(self.roll_rate) * dt

        # Calculate camera shake metric
        camera_shake = self._calculate_camera_shake()
        self.camera_shake_history.append(camera_shake)

        # Track maximum altitude
        self.max_altitude_reached = max(self.max_altitude_reached, self.altitude)

        # Calculate reward
        reward = self._calculate_reward(camera_shake)

        # Check termination conditions
        terminated = self._is_terminated()
        truncated = self.time > 20.0  # 20 second max flight time

        # Update integrated error for info
        self.integrated_roll_error += abs(self.roll_rate) * dt

        return self._get_observation(), reward, terminated, truncated, self._get_info()

    def _update_propulsion(self) -> Tuple[float, float]:
        """Update thrust and mass based on motor burn"""
        if self.time < self.config.burn_time:
            # Propellant consumption
            burn_fraction = self.config.dt / self.config.burn_time
            self.propellant_remaining -= self.config.propellant_mass * burn_fraction
            self.propellant_remaining = max(0, self.propellant_remaining)

            # Thrust curve (can be progressive, neutral, or regressive)
            if self.config.thrust_curve == "progressive":
                # Thrust increases over burn
                thrust_fraction = 0.7 + 0.6 * (self.time / self.config.burn_time)
            elif self.config.thrust_curve == "regressive":
                # Thrust decreases over burn
                thrust_fraction = 1.3 - 0.6 * (self.time / self.config.burn_time)
            else:  # neutral
                thrust_fraction = 1.0

            thrust = self.config.average_thrust * thrust_fraction
        else:
            thrust = 0.0

        mass = self.config.dry_mass + self.propellant_remaining
        return thrust, mass

    def _calculate_roll_torque(self, dynamic_pressure: float) -> float:
        """
        Calculate roll torque from fin tabs and aerodynamic damping.
        This is the key control mechanism for spin stabilization.
        """

        # Control torque from deflected tabs
        if dynamic_pressure > 1.0:  # Need some airspeed for control
            # Tab effectiveness
            tab_area = (self.config.tab_chord_fraction * self.config.fin_root_chord *
                       self.config.tab_span_fraction * self.config.fin_span)

            # Lift coefficient for small deflections (approximately linear)
            Cl_tab = 2 * np.pi * np.sin(self.tab_deflection)

            # Force per tab
            tab_force = 0.5 * Cl_tab * dynamic_pressure * tab_area

            # Moment arm (distance from rocket centerline to tab center of pressure)
            moment_arm = self.config.diameter/2 + 0.5 * self.config.fin_span

            # Total control torque (2 fins with tabs in differential deflection)
            control_torque = 2 * tab_force * moment_arm

            # Effectiveness factor (tabs less effective at very high or low speeds)
            speed_effectiveness = np.tanh(dynamic_pressure / 500.0)  # Saturates around 500 Pa
            control_torque *= speed_effectiveness

        else:
            control_torque = 0.0

        # Aerodynamic damping torque (opposes rotation)
        # This naturally stabilizes the rocket
        damping_coefficient = 0.001 * self.config.num_fins  # Empirical damping
        damping_torque = -damping_coefficient * self.roll_rate * dynamic_pressure

        # Add small disturbance torque (simulates asymmetries, turbulence)
        if dynamic_pressure > 10:
            disturbance = np.random.normal(0, 0.01 * np.sqrt(dynamic_pressure))
        else:
            disturbance = 0.0

        total_torque = control_torque + damping_torque + disturbance

        return total_torque

    def _calculate_roll_inertia(self, mass: float) -> float:
        """
        Calculate roll moment of inertia for composite rocket.
        Thin-walled tube approximation for fiberglass body.
        """
        # Body tube (thin-walled cylinder)
        radius = self.config.diameter / 2
        body_inertia = mass * radius**2  # Simplified for thin wall

        # Add fin contribution (small for composite fins)
        fin_mass_each = (self.config.fin_span * self.config.fin_root_chord *
                        self.config.fin_thickness * 1600)  # 1600 kg/m³ for G10
        fin_distance = radius + self.config.fin_span/2
        fins_inertia = self.config.num_fins * fin_mass_each * fin_distance**2

        return body_inertia + fins_inertia

    def _calculate_camera_shake(self) -> float:
        """
        Calculate camera shake metric based on roll rate and acceleration.
        Higher values = worse footage quality.
        """
        # Horizontal camera is mainly affected by roll rate
        horizontal_shake = abs(self.roll_rate) * 10.0

        # Add effect of roll acceleration (jerky motion)
        horizontal_shake += abs(self.roll_acceleration) * 1.0

        # Downward camera is less affected but still experiences some shake
        downward_shake = abs(self.roll_rate) * 2.0 + abs(self.roll_acceleration) * 0.5

        # Combined metric (weighted average favoring horizontal camera)
        camera_shake = 0.7 * horizontal_shake + 0.3 * downward_shake

        return camera_shake

    def _calculate_reward(self, camera_shake: float) -> float:
        """
        Reward function optimized for camera stability during flight.
        """
        reward = 0.0

        # 1. Primary objective: Minimize roll rate (most important for cameras)
        roll_rate_deg = np.rad2deg(abs(self.roll_rate))
        if roll_rate_deg < 10:  # Excellent stability
            reward += 10.0
        elif roll_rate_deg < 30:  # Good stability
            reward += 5.0
        elif roll_rate_deg < 60:  # Acceptable
            reward += 2.0
        else:  # Poor stability
            reward -= roll_rate_deg * 0.1

        # 2. Camera shake penalty (direct footage quality metric)
        reward -= camera_shake * 0.5

        # 3. Bonus for very stable footage (both cameras)
        if camera_shake < 1.0:
            reward += 5.0  # Excellence bonus

        # 4. Control effort penalty (reduce actuator wear and power consumption)
        reward -= abs(self.last_action) * 0.2

        # 5. Control smoothness bonus (avoid jerky movements)
        if hasattr(self, 'previous_action'):
            action_change = abs(self.last_action - self.previous_action)
            reward -= action_change * 1.0
        self.previous_action = self.last_action

        # 6. Altitude achievement bonus (still want successful flight)
        if self.altitude > 0:
            altitude_progress = min(self.altitude / 200.0, 1.0)  # Normalize to 200m
            reward += altitude_progress * 2.0

        # 7. Phase-specific bonuses
        if self.time < self.config.burn_time:
            # During boost: extra importance on stability
            if roll_rate_deg < 20:
                reward += 3.0
        else:
            # During coast: even more critical for camera footage
            if roll_rate_deg < 15:
                reward += 5.0

        return float(np.clip(reward, -50, 50))

    def _is_terminated(self) -> bool:
        """Check termination conditions"""
        # Crash
        if self.altitude < 0 and self.time > 0.1:
            return True

        # Excessive spin (unrecoverable for cameras)
        if abs(self.roll_rate) > np.deg2rad(self.config.max_roll_rate):
            return True

        # Reached apogee and descending significantly
        if self.altitude < self.max_altitude_reached - 50 and self.time > self.config.burn_time:
            return True

        return False

    def _get_air_density(self) -> float:
        """Simple atmospheric density model"""
        # Exponential atmosphere model
        sea_level_density = 1.225  # kg/m³
        scale_height = 8000  # m
        return sea_level_density * np.exp(-self.altitude / scale_height)

    def _get_observation(self) -> np.ndarray:
        """Get current observation focused on roll control"""
        # Calculate dynamic pressure for observation
        q = 0.5 * self._get_air_density() * self.vertical_velocity**2

        # Thrust fraction (0 during coast)
        thrust_fraction = min(1.0, max(0.0,
                              (self.config.burn_time - self.time) / self.config.burn_time))

        # Recent camera shake (average over last few samples)
        if len(self.camera_shake_history) > 0:
            recent_shake = np.mean(self.camera_shake_history[-10:])
        else:
            recent_shake = 0.0

        obs = np.array([
            self.altitude,
            self.vertical_velocity,
            self.roll_angle,
            self.roll_rate,
            self.roll_acceleration,
            q,
            self.time,
            thrust_fraction,
            self.last_action,
            recent_shake
        ], dtype=np.float32)

        return obs

    def _get_info(self) -> Dict[str, Any]:
        """Get additional info for debugging and analysis"""
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
            'average_roll_rate_deg_s': np.rad2deg(self.integrated_roll_error / max(self.time, 0.01)),
            'max_altitude_m': self.max_altitude_reached,
            'horizontal_camera_quality': self._assess_camera_quality('horizontal'),
            'downward_camera_quality': self._assess_camera_quality('downward'),
        }

    def _assess_camera_quality(self, camera_type: str) -> str:
        """Assess footage quality for each camera"""
        roll_rate_deg = np.rad2deg(abs(self.roll_rate))

        if camera_type == 'horizontal':
            if roll_rate_deg < 10:
                return "Excellent - Stable footage"
            elif roll_rate_deg < 30:
                return "Good - Minor motion blur"
            elif roll_rate_deg < 60:
                return "Fair - Noticeable blur"
            else:
                return "Poor - Severe blur/spinning"
        else:  # downward
            if roll_rate_deg < 20:
                return "Excellent - Clear ground view"
            elif roll_rate_deg < 45:
                return "Good - Slight rotation visible"
            elif roll_rate_deg < 90:
                return "Fair - Rotating ground view"
            else:
                return "Poor - Disorienting spin"

    def render(self, mode='human'):
        """Render environment state"""
        if mode == 'human':
            print(f"\n--- T: {self.time:.2f}s ---")
            print(f"Altitude: {self.altitude:.1f}m | Vertical Vel: {self.vertical_velocity:.1f}m/s")
            print(f"Roll Rate: {np.rad2deg(self.roll_rate):.1f}°/s | Total Rotations: {self.total_rotation/(2*np.pi):.2f}")
            print(f"Tab Deflection: {np.rad2deg(self.tab_deflection):.1f}° | Action: {self.last_action:.3f}")
            print(f"Phase: {'BOOST' if self.time < self.config.burn_time else 'COAST'}")
            print(f"Camera Quality - H: {self._assess_camera_quality('horizontal')}")
            print(f"Camera Quality - V: {self._assess_camera_quality('downward')}")

        elif mode == 'camera_view':
            # Simulate camera views
            return self._render_camera_views()

    def _render_camera_views(self) -> Dict[str, np.ndarray]:
        """Simulate what each camera would see (for advanced rendering)"""
        views = {}

        # Horizontal camera - affected by roll
        horizontal_blur = min(abs(self.roll_rate) * 10, 1.0)
        views['horizontal'] = {
            'roll_angle': self.roll_angle,
            'blur_factor': horizontal_blur,
            'stability': 1.0 - horizontal_blur
        }

        # Downward camera - shows ground rotation
        views['downward'] = {
            'ground_rotation': self.total_rotation,
            'altitude_agl': self.altitude,
            'blur_factor': min(abs(self.roll_rate) * 3, 1.0)
        }

        return views


# Example usage and testing
if __name__ == "__main__":

    # Create environment with custom configuration
    config = CompositeRocketConfig(
        dry_mass=1.5,           # 1.5 kg rocket
        propellant_mass=0.2,    # 200g motor
        diameter=0.054,         # 54mm diameter
        average_thrust=100.0,   # 100N average thrust
        burn_time=2.0,          # 2 second burn
        max_tab_deflection=15.0 # ±15° tab deflection
    )

    env = SpinStabilizedCameraRocket(config)

    print("=== Spin-Stabilized Camera Rocket Environment ===")
    print(f"Rocket: {config.dry_mass}kg composite, {config.diameter*1000:.0f}mm diameter")
    print(f"Motor: {config.average_thrust}N for {config.burn_time}s")
    print(f"Control: ±{config.max_tab_deflection}° fin tabs on {config.num_controlled_fins} fins")
    print(f"Cameras: Horizontal ({config.horizontal_camera_fov}°) + Downward ({config.downward_camera_fov}°)\n")

    # Test episode with simple proportional controller
    obs, info = env.reset()
    print("Initial conditions:")
    print(f"  Roll rate: {info['roll_rate_deg_s']:.1f}°/s")

    total_reward = 0
    episode_length = 0

    # Simple P-controller for demonstration
    Kp = 0.5  # Proportional gain

    while True:
        # Proportional control on roll rate
        roll_rate = obs[3]  # rad/s from observation
        action = np.array([-Kp * roll_rate])  # Negative feedback
        action = np.clip(action, -1, 1)

        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        episode_length += 1

        # Render every 50 steps (0.5 seconds of flight)
        if episode_length % 50 == 0:
            env.render()

        if terminated or truncated:
            break

    # Final statistics
    print("\n=== Episode Summary ===")
    print(f"Flight duration: {info['time_s']:.2f}s")
    print(f"Max altitude: {info['max_altitude_m']:.1f}m")
    print(f"Total reward: {total_reward:.2f}")
    print(f"Average roll rate: {info['average_roll_rate_deg_s']:.1f}°/s")
    print(f"Total rotations: {info['roll_total_rotations']:.2f}")
    print(f"Final camera assessment:")
    print(f"  Horizontal: {info['horizontal_camera_quality']}")
    print(f"  Downward: {info['downward_camera_quality']}")

    env.close()
