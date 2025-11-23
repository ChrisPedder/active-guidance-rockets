import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from scipy import integrate, interpolate

from spin_stabilized_control_env import SpinStabilizedCameraRocket, CompositeRocketConfig
from thrustcurve_motor_data import MotorData

class RealisticMotorRocket(SpinStabilizedCameraRocket):
    """
    Extended rocket environment using real motor data from thrustcurve.org
    """

    def __init__(self, motor_data: MotorData, config: CompositeRocketConfig = None):
        """
        Initialize with real motor data

        Args:
            motor_data: Parsed motor data from thrustcurve.org
            config: Rocket configuration (will be updated with motor specs)
        """
        if config is None:
            config = CompositeRocketConfig()

        # Update config with motor specifications
        config.propellant_mass = motor_data.propellant_mass
        config.burn_time = motor_data.burn_time
        config.average_thrust = motor_data.average_thrust

        # Adjust rocket diameter if motor is larger
        if motor_data.diameter > config.diameter:
            print(f"Adjusting rocket diameter from {config.diameter*1000:.1f}mm "
                  f"to {motor_data.diameter*1000:.1f}mm to fit motor")
            config.diameter = motor_data.diameter * 1.2  # Add some margin

        self.motor = motor_data

        # Override mass calculation to include motor case
        self.motor_case_mass = motor_data.case_mass

        # Track propellant consumption more accurately
        self.propellant_consumed = 0.0

        print(f"Loaded motor: {motor_data.manufacturer} {motor_data.designation}")
        print(f"Total impulse: {motor_data.total_impulse:.1f} N·s "
              f"({self._get_motor_class()})")
        print(f"Burn time: {motor_data.burn_time:.2f}s, "
              f"Max thrust: {motor_data.max_thrust:.1f}N")

        super().__init__(config)

    def _get_motor_class(self) -> str:
        """Determine motor classification (A, B, C, etc.)"""
        impulse = self.motor.total_impulse

        # Motor classifications (total impulse ranges in N·s)
        classifications = [
            (0, 1.25, "1/4A"),
            (1.25, 2.5, "1/2A"),
            (2.5, 5, "A"),
            (5, 10, "B"),
            (10, 20, "C"),
            (20, 40, "D"),
            (40, 80, "E"),
            (80, 160, "F"),
            (160, 320, "G"),
            (320, 640, "H"),
            (640, 1280, "I"),
            (1280, 2560, "J"),
            (2560, 5120, "K"),
            (5120, 10240, "L"),
            (10240, 20480, "M"),
        ]

        for min_impulse, max_impulse, letter in classifications:
            if min_impulse <= impulse < max_impulse:
                return letter

        return ">" + classifications[-1][2]

    def _update_propulsion(self) -> Tuple[float, float]:
        """
        Update thrust and mass based on real motor curve
        Overrides parent method to use actual thrust curve data
        """
        if self.time < self.motor.burn_time:
            # Get thrust from real curve
            thrust = self.motor.get_thrust(self.time)

            # Get current motor mass
            motor_mass = self.motor.get_mass(self.time)

            # Track propellant consumption
            self.propellant_consumed = self.motor.propellant_mass - \
                                      (motor_mass - self.motor.case_mass)

            # Update remaining propellant for parent class
            self.propellant_remaining = self.motor.propellant_mass - self.propellant_consumed
        else:
            thrust = 0.0
            motor_mass = self.motor.case_mass
            self.propellant_remaining = 0.0
            self.propellant_consumed = self.motor.propellant_mass

        # Total mass includes rocket dry mass and current motor mass
        total_mass = self.config.dry_mass + motor_mass

        return thrust, total_mass

    def get_current_cg(self) -> float:
        """
        Calculate CG including motor CG shift as propellant burns
        """
        # Base rocket CG (without motor)
        rocket_cg = 0.4  # 40cm from nose for typical small rocket

        # Motor CG position (at base of rocket)
        motor_position = self.config.length - self.motor.length / 2

        # Motor CG shift as propellant burns
        motor_cg_shift = self.motor.get_cg_shift(self.time)
        motor_cg = motor_position - motor_cg_shift

        # Combined CG (weighted average)
        motor_mass = self.motor.get_mass(self.time)
        total_mass = self.config.dry_mass + motor_mass

        combined_cg = (self.config.dry_mass * rocket_cg + motor_mass * motor_cg) / total_mass

        return combined_cg

    def _get_info(self) -> Dict:
        """Extended info including motor data"""
        info = super()._get_info()

        # Add motor-specific information
        info.update({
            'motor': f"{self.motor.manufacturer} {self.motor.designation}",
            'motor_class': self._get_motor_class(),
            'current_thrust_N': self.motor.get_thrust(self.time) if self.time < self.motor.burn_time else 0,
            'propellant_consumed_g': self.propellant_consumed * 1000,
            'propellant_remaining_g': self.propellant_remaining * 1000,
            'motor_mass_g': self.motor.get_mass(self.time) * 1000,
            'impulse_delivered_Ns': integrate.simpson(
                [self.motor.get_thrust(t) for t in np.linspace(0, min(self.time, self.motor.burn_time), 100)],
                np.linspace(0, min(self.time, self.motor.burn_time), 100)
            ) if self.time > 0 else 0
        })

        return info


# Utility class for common motor configurations
class CommonMotors:
    """Pre-defined configurations for common motors"""

    @staticmethod
    def estes_c6() -> MotorData:
        """Estes C6-5 motor data (common small rocket motor)"""
        return MotorData(
            manufacturer="Estes",
            designation="C6",
            diameter=18.0,  # mm
            length=70.0,    # mm
            total_mass=24.0,  # g
            propellant_mass=12.3,  # g
            case_mass=11.7,  # g
            total_impulse=10.0,  # N·s
            burn_time=1.85,  # s
            average_thrust=5.4,  # N
            max_thrust=14.0,  # N
            time_points=np.array([0.0, 0.04, 0.13, 0.5, 1.0, 1.5, 1.85]),
            thrust_points=np.array([0.0, 14.0, 12.0, 6.0, 5.0, 4.5, 0.0]),
            delays=[3, 5, 7]
        )

    @staticmethod
    def aerotech_f40() -> MotorData:
        """Aerotech F40-10W motor data (mid-power rocket motor)"""
        return MotorData(
            manufacturer="Aerotech",
            designation="F40W",
            diameter=29.0,  # mm
            length=124.0,   # mm
            total_mass=90.0,   # g
            propellant_mass=39.0,  # g
            case_mass=51.0,   # g
            total_impulse=80.0,  # N·s
            burn_time=2.0,   # s
            average_thrust=40.0,  # N
            max_thrust=65.0,  # N
            time_points=np.array([0.0, 0.05, 0.1, 0.5, 1.0, 1.5, 1.9, 2.0]),
            thrust_points=np.array([0.0, 65.0, 55.0, 45.0, 40.0, 35.0, 20.0, 0.0]),
            delays=[4, 7, 10]
        )

    @staticmethod
    def cesaroni_g79() -> MotorData:
        """Cesaroni G79-SS motor data (small high-power motor)"""
        return MotorData(
            manufacturer="Cesaroni",
            designation="G79-SS",
            diameter=29.0,  # mm
            length=152.0,   # mm
            total_mass=149.0,  # g
            propellant_mass=62.5,  # g
            case_mass=86.5,   # g
            total_impulse=130.0,  # N·s
            burn_time=1.6,   # s
            average_thrust=79.0,  # N
            max_thrust=110.0,  # N
            time_points=np.array([0.0, 0.02, 0.1, 0.3, 0.6, 0.9, 1.2, 1.5, 1.6]),
            thrust_points=np.array([0.0, 110.0, 95.0, 85.0, 80.0, 75.0, 70.0, 45.0, 0.0]),
            delays=[6, 9, 12],
            sparky=True  # This motor produces sparks
        )


# Example usage with real motor data
if __name__ == "__main__":

    # Method 1: Use pre-defined common motor
    motor = CommonMotors.cesaroni_g79()

    # Method 2: Parse from downloaded .eng file
    # parser = ThrustCurveParser()
    # motor = parser.parse_eng_file("motors/Estes_C6.eng")

    # Method 3: Download directly from thrustcurve.org
    # motor = ThrustCurveParser.download_from_thrustcurve("Estes_C6", "eng")

    # Create rocket configuration for this motor
    rocket_config = CompositeRocketConfig(
        dry_mass=0.150,  # 150g rocket (typical for C motor)
        diameter=0.024,   # 24mm body tube
        length=0.45,      # 45cm length
        num_fins=4,
        fin_span=0.04,
        max_tab_deflection=15.0
    )

    # Create environment with real motor
    env = RealisticMotorRocket(motor, rocket_config)

    # Visualize the motor thrust curve
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 10))

    # Thrust curve
    ax1.plot(motor.time_points, motor.thrust_points, 'b-', linewidth=2)
    ax1.fill_between(motor.time_points, 0, motor.thrust_points, alpha=0.3)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Thrust (N)')
    ax1.set_title(f'{motor.manufacturer} {motor.designation} Thrust Curve')
    ax1.grid(True, alpha=0.3)
    ax1.axhline(y=motor.average_thrust, color='r', linestyle='--',
                label=f'Average: {motor.average_thrust:.1f}N')
    ax1.legend()

    # Mass curve
    times = np.linspace(0, motor.burn_time * 1.1, 100)
    masses = [motor.get_mass(t) * 1000 for t in times]  # Convert to grams
    ax2.plot(times, masses, 'g-', linewidth=2)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Motor Mass (g)')
    ax2.set_title('Motor Mass During Burn')
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=motor.case_mass * 1000, color='r', linestyle='--',
                label=f'Empty: {motor.case_mass*1000:.1f}g')
    ax2.legend()

    # Impulse delivered over time
    impulses = []
    for t in times:
        if t > 0:
            t_range = np.linspace(0, t, 50)
            impulse = integrate.simpson(
                [motor.get_thrust(ti) for ti in t_range],
                t_range
            )
            impulses.append(impulse)
        else:
            impulses.append(0)

    ax3.plot(times, impulses, 'r-', linewidth=2)
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Impulse Delivered (N·s)')
    ax3.set_title('Cumulative Impulse')
    ax3.grid(True, alpha=0.3)
    ax3.axhline(y=motor.total_impulse, color='b', linestyle='--',
                label=f'Total: {motor.total_impulse:.1f}N·s')
    ax3.legend()

    plt.tight_layout()
    plt.savefig('motor_analysis.png', dpi=150)
    plt.show()

    # Run a test episode
    print("\n=== Running Simulation with Real Motor ===")
    obs, info = env.reset()

    total_reward = 0
    max_altitude = 0
    thrust_history = []
    altitude_history = []
    roll_rate_history = []
    time_history = []

    # Simple controller
    Kp = 0.3  # Proportional gain for roll control

    while True:
        # Control action based on roll rate
        roll_rate = obs[3]
        action = np.array([-Kp * roll_rate])
        action = np.clip(action, -1, 1)

        obs, reward, terminated, truncated, info = env.step(action)

        # Record data
        thrust_history.append(info['current_thrust_N'])
        altitude_history.append(info['altitude_m'])
        roll_rate_history.append(info['roll_rate_deg_s'])
        time_history.append(info['time_s'])

        total_reward += reward
        max_altitude = max(max_altitude, info['altitude_m'])

        # Print status at key moments
        if len(time_history) == 1:
            print(f"Ignition: {info['current_thrust_N']:.1f}N thrust")
        elif info['phase'] == 'coast' and len(time_history) > 1 and thrust_history[-2] > 0:
            print(f"Burnout at T+{info['time_s']:.2f}s, altitude {info['altitude_m']:.1f}m")
            print(f"  Delivered impulse: {info['impulse_delivered_Ns']:.1f}N·s")

        if terminated or truncated:
            break

    print(f"\n=== Flight Summary ===")
    print(f"Motor: {motor.manufacturer} {motor.designation} ({env._get_motor_class()})")
    print(f"Max altitude: {max_altitude:.1f}m")
    print(f"Flight duration: {info['time_s']:.2f}s")
    print(f"Average roll rate: {np.mean(np.abs(roll_rate_history)):.1f}°/s")
    print(f"Camera quality - Horizontal: {info['horizontal_camera_quality']}")
    print(f"Camera quality - Downward: {info['downward_camera_quality']}")
    print(f"Total reward: {total_reward:.1f}")

    # Plot flight profile
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    # Altitude and thrust
    ax1 = axes[0]
    ax1_twin = ax1.twinx()
    ax1.plot(time_history, altitude_history, 'b-', label='Altitude')
    ax1_twin.plot(time_history, thrust_history, 'r-', alpha=0.7, label='Thrust')
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Altitude (m)', color='b')
    ax1_twin.set_ylabel('Thrust (N)', color='r')
    ax1.set_title('Flight Profile')
    ax1.grid(True, alpha=0.3)

    # Roll rate
    axes[1].plot(time_history, roll_rate_history, 'g-')
    axes[1].set_xlabel('Time (s)')
    axes[1].set_ylabel('Roll Rate (°/s)')
    axes[1].set_title('Roll Stability')
    axes[1].grid(True, alpha=0.3)
    axes[1].axhline(y=0, color='k', linestyle='-', alpha=0.3)
    axes[1].axhline(y=30, color='r', linestyle='--', alpha=0.5, label='Good footage limit')
    axes[1].axhline(y=-30, color='r', linestyle='--', alpha=0.5)
    axes[1].legend()

    # Phase indicator
    axes[2].fill_between(
        [0, motor.burn_time], [0, 0], [1, 1],
        color='orange', alpha=0.3, label='Boost'
    )
    axes[2].fill_between(
        [motor.burn_time, time_history[-1]], [0, 0], [1, 1],
        color='blue', alpha=0.3, label='Coast'
    )
    axes[2].set_xlabel('Time (s)')
    axes[2].set_ylabel('Flight Phase')
    axes[2].set_title('Motor Burn Phases')
    axes[2].set_ylim(0, 1)
    axes[2].legend()

    plt.tight_layout()
    plt.savefig('flight_profile.png', dpi=150)
    plt.show()
