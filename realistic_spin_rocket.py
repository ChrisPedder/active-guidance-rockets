"""
Realistic Motor Rocket

This extends the spin-stabilized environment to use real motor data
from ThrustCurve.org with accurate thrust curves.

REQUIRES: A RocketAirframe instance must be provided for physics calculations.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import integrate

from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig
from airframe import RocketAirframe

# Try to import motor_loader for config-based motors
try:
    from motor_loader import Motor as ConfigMotor

    MOTOR_LOADER_AVAILABLE = True
except ImportError:
    MOTOR_LOADER_AVAILABLE = False
    ConfigMotor = None

# Try to import motor data classes
try:
    from thrustcurve_motor_data import MotorData

    MOTOR_DATA_AVAILABLE = True
except ImportError:
    MOTOR_DATA_AVAILABLE = False
    # Define a simple MotorData class if not available
    from dataclasses import dataclass
    from scipy import interpolate

    @dataclass
    class MotorData:
        manufacturer: str
        designation: str
        diameter: float  # mm
        length: float  # mm
        total_mass: float  # g
        propellant_mass: float  # g
        case_mass: float  # g (computed as total - propellant)
        total_impulse: float  # N·s
        burn_time: float  # s
        average_thrust: float  # N
        max_thrust: float  # N
        time_points: np.ndarray
        thrust_points: np.ndarray
        delays: list = None
        sparky: bool = False

        def __post_init__(self):
            # Convert to SI units
            self.diameter = self.diameter / 1000  # mm to m
            self.length = self.length / 1000  # mm to m
            self.total_mass = self.total_mass / 1000  # g to kg
            self.propellant_mass = self.propellant_mass / 1000
            self.case_mass = self.case_mass / 1000

            # Create interpolation function
            self._thrust_interp = interpolate.interp1d(
                self.time_points,
                self.thrust_points,
                kind="linear",
                bounds_error=False,
                fill_value=0.0,
            )

        def get_thrust(self, time: float) -> float:
            return float(self._thrust_interp(time))

        def get_mass(self, time: float) -> float:
            if time >= self.burn_time:
                return self.case_mass
            burn_fraction = time / self.burn_time
            remaining_prop = self.propellant_mass * (1 - burn_fraction)
            return self.case_mass + remaining_prop


class RealisticMotorRocket(SpinStabilizedCameraRocket):
    """
    Rocket environment using real motor thrust curves.

    Requires:
    - A RocketAirframe defining the rocket geometry
    - Motor data (either MotorData object or config dict from YAML)

    Example:
        >>> from airframe import RocketAirframe
        >>> airframe = RocketAirframe.load('my_rocket.ork')
        >>> env = RealisticMotorRocket(
        ...     airframe=airframe,
        ...     motor_config=motor_cfg
        ... )
    """

    def __init__(
        self,
        airframe: RocketAirframe,
        motor_data: Optional[MotorData] = None,
        motor_config: Optional[Dict[str, Any]] = None,
        config: Optional[RocketConfig] = None,
    ):
        """
        Initialize with airframe and motor data.

        Args:
            airframe: RocketAirframe defining rocket geometry (REQUIRED)
            motor_data: Motor specifications (MotorData object)
            motor_config: Motor config dict from YAML file
            config: RocketConfig with simulation parameters

        Must provide either motor_data OR motor_config.

        Example:
            >>> airframe = RocketAirframe.load('my_rocket.ork')
            >>> env = RealisticMotorRocket(
            ...     airframe=airframe,
            ...     motor_config=motor_cfg
            ... )
        """
        if airframe is None:
            raise ValueError(
                "RocketAirframe is required. Create one with:\n"
                "  airframe = RocketAirframe.load('my_rocket.ork')  # From OpenRocket\n"
                "  airframe = RocketAirframe.estes_alpha()          # Factory method\n"
                "  airframe = RocketAirframe.load('airframe.yaml')  # From YAML"
            )

        if config is None:
            config = RocketConfig()

        # Load motor from config if provided
        if motor_config is not None:
            if not MOTOR_LOADER_AVAILABLE:
                raise ImportError(
                    "motor_loader.py is required to use motor_config!\n"
                    "Please ensure motor_loader.py is in your project directory."
                )

            # Load motor using ConfigMotor from motor_loader.py
            self.motor = ConfigMotor(motor_config)
            self.motor_case_mass = self.motor.case_mass

            # Update config with motor specifications
            config.propellant_mass = self.motor.propellant_mass
            config.burn_time = self.motor.burn_time
            config.average_thrust = self.motor.average_thrust

            print(f"✓ Loaded motor: {self.motor.manufacturer} {self.motor.designation}")

        # Use provided motor_data
        elif motor_data is not None:
            self.motor = motor_data
            self.motor_case_mass = motor_data.case_mass

            # Update config with motor specifications
            config.propellant_mass = motor_data.propellant_mass
            config.burn_time = motor_data.burn_time
            config.average_thrust = motor_data.average_thrust

            print(f"✓ Using motor: {motor_data.manufacturer} {motor_data.designation}")

        else:
            raise ValueError(
                "Must provide either 'motor_data' or 'motor_config'!\n"
                "  - motor_data: Pass a MotorData object\n"
                "  - motor_config: Pass config dict from YAML file"
            )

        # Display motor info
        print(f"  Total impulse: {self.motor.total_impulse:.1f} N·s")
        print(f"  Burn time: {self.motor.burn_time:.2f}s")
        print(
            f"  Avg/Max thrust: {self.motor.average_thrust:.1f}N / {self.motor.max_thrust:.1f}N"
        )

        # Display airframe info
        print(f"✓ Using airframe: {airframe.name}")
        print(f"    Dry mass: {airframe.dry_mass*1000:.1f}g")
        print(f"    Diameter: {airframe.body_diameter*1000:.1f}mm")

        # Calculate and display TWR
        total_mass = airframe.dry_mass + self.motor.propellant_mass
        twr = self.motor.average_thrust / (total_mass * 9.81)
        print(f"  Rocket mass: {total_mass*1000:.1f}g")
        print(f"  TWR: {twr:.2f}")

        if twr < 1.0:
            print(f"  ⚠️  WARNING: TWR < 1.0 - rocket cannot lift off!")
        elif twr < 2.0:
            print(f"  ⚠️  WARNING: TWR < 2.0 - marginal performance")

        super().__init__(airframe=airframe, config=config)

    def _update_propulsion(self) -> Tuple[float, float]:
        """
        Get thrust from real motor curve.
        Works with both MotorData and ConfigMotor.
        """
        if self.time < self.motor.burn_time:
            thrust = self.motor.get_thrust(self.time)
            motor_mass = self.motor.get_mass(self.time)
        else:
            thrust = 0.0
            motor_mass = self.motor_case_mass

        # Total mass = airframe dry mass + current motor mass
        total_mass = self.airframe.dry_mass + motor_mass

        # Update propellant remaining for parent class
        self.propellant_remaining = max(0, motor_mass - self.motor_case_mass)

        return thrust, total_mass

    def _get_info(self) -> Dict[str, Any]:
        """Extended info with motor data."""
        info = super()._get_info()

        # Add motor-specific info
        current_thrust = (
            self.motor.get_thrust(self.time) if self.time < self.motor.burn_time else 0
        )

        info.update(
            {
                "motor": f"{self.motor.manufacturer} {self.motor.designation}",
                "current_thrust_N": current_thrust,
                "propellant_remaining_g": self.propellant_remaining * 1000,
            }
        )

        return info


def create_environment_from_config(
    config_path: str, rank: int = 0
) -> RealisticMotorRocket:
    """
    Create environment from config YAML file.

    The config file MUST specify an airframe via the 'airframe_file' key
    in the physics section.

    Args:
        config_path: Path to config YAML file
        rank: Environment rank for seeding

    Returns:
        Configured RealisticMotorRocket environment

    Config file format:
        physics:
          airframe_file: "path/to/rocket.ork"  # REQUIRED
          tab_chord_fraction: 0.25
          tab_span_fraction: 0.5
          max_tab_deflection: 15.0
          disturbance_scale: 0.0001
          damping_scale: 1.0

        motor:
          name: "estes_c6"
          # ... motor config

    Example:
        >>> env = create_environment_from_config('configs/my_rocket.yaml')
        ✓ Loaded airframe: My Custom Rocket
        ✓ Loaded motor: Estes C6
    """
    import yaml
    import os

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # Extract configs
    motor_config = config.get("motor", {})
    physics = config.get("physics", {})
    env_cfg = config.get("environment", {})

    # Load airframe (REQUIRED)
    airframe_file = physics.get("airframe_file")
    if not airframe_file:
        raise ValueError(
            f"Config file {config_path} must specify 'physics.airframe_file'.\n"
            "Example:\n"
            "  physics:\n"
            "    airframe_file: 'configs/airframes/my_rocket.ork'\n"
            "\n"
            "Create an airframe file from OpenRocket (.ork) or define in YAML."
        )

    # Resolve relative paths relative to config file
    if not os.path.isabs(airframe_file):
        config_dir = os.path.dirname(os.path.abspath(config_path))
        airframe_file = os.path.join(config_dir, airframe_file)

    airframe = RocketAirframe.load(airframe_file)

    # Create RocketConfig with physics tuning parameters
    rocket_config = RocketConfig(
        max_tab_deflection=physics.get("max_tab_deflection", 15.0),
        tab_chord_fraction=physics.get("tab_chord_fraction", 0.25),
        tab_span_fraction=physics.get("tab_span_fraction", 0.5),
        num_controlled_fins=physics.get("num_controlled_fins", 2),
        disturbance_scale=physics.get("disturbance_scale", 0.0001),
        damping_scale=physics.get("damping_scale", 1.0),
        initial_spin_std=physics.get("initial_spin_std", 15.0),
        max_roll_rate=physics.get("max_roll_rate", 360.0),
        max_episode_time=physics.get("max_episode_time", 15.0),
        dt=env_cfg.get("dt", 0.01),
    )

    # Create environment
    env = RealisticMotorRocket(
        airframe=airframe,
        motor_config=motor_config,
        config=rocket_config,
    )

    # Seed the environment
    if rank > 0:
        env.reset(seed=rank)

    return env


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Realistic Motor Rocket")
    print("=" * 60)

    # Create airframe (required)
    airframe = RocketAirframe.estes_alpha()

    # Create a simple motor config for testing
    motor_config = {
        "name": "estes_c6",
        "manufacturer": "Estes",
        "designation": "C6",
        "total_impulse_Ns": 10.0,
        "avg_thrust_N": 5.4,
        "max_thrust_N": 14.0,
        "burn_time_s": 1.85,
        "propellant_mass_g": 12.3,
        "case_mass_g": 12.7,
        "thrust_curve": {
            "time_s": [0.0, 0.1, 0.5, 1.0, 1.5, 1.85],
            "thrust_N": [0.0, 14.0, 6.0, 5.0, 4.0, 0.0],
        },
    }

    # Create config
    config = RocketConfig(
        max_tab_deflection=15.0,
        initial_spin_std=15.0,
        disturbance_scale=0.0001,
    )

    print(f"\nCreating environment with:")
    print(f"  Airframe: {airframe.name}")
    print(f"  Motor: {motor_config['designation']}")

    # Create environment
    env = RealisticMotorRocket(
        airframe=airframe,
        motor_config=motor_config,
        config=config,
    )

    print("\n--- Passive flight test (no control) ---")
    obs, info = env.reset()
    print(f"Initial: roll_rate={info['roll_rate_deg_s']:.1f}°/s")

    max_alt = 0
    for step in range(300):
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        max_alt = max(max_alt, info["altitude_m"])

        if step % 30 == 0:
            print(
                f"T={info['time_s']:.2f}s: alt={info['altitude_m']:.1f}m, "
                f"roll={info['roll_rate_deg_s']:.1f}°/s, "
                f"thrust={info['current_thrust_N']:.1f}N"
            )

        if terminated or truncated:
            break

    print(f"\nMax altitude: {max_alt:.1f}m")
    print(f"Duration: {info['time_s']:.2f}s")
    print(f"Final spin: {info['roll_rate_deg_s']:.1f}°/s")

    print("\n" + "=" * 60)
    print("Test complete!")
    print("=" * 60)
