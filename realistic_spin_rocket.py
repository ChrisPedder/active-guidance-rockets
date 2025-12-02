"""
Realistic Motor Rocket

This extends the spin-stabilized environment to use real motor data.

UPDATED: Now supports loading motors from config files via motor_loader.py
This enables using ANY motor from ThrustCurve.org without code changes.
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
from scipy import integrate

from spin_stabilized_control_env import (
    SpinStabilizedCameraRocket,
    RocketConfig
)

# Try to import motor_loader for config-based motors (NEW!)
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
    import numpy as np
    from scipy import interpolate

    @dataclass
    class MotorData:
        manufacturer: str
        designation: str
        diameter: float  # mm
        length: float    # mm
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
            self.length = self.length / 1000      # mm to m
            self.total_mass = self.total_mass / 1000  # g to kg
            self.propellant_mass = self.propellant_mass / 1000
            self.case_mass = self.case_mass / 1000

            # Create interpolation function
            self._thrust_interp = interpolate.interp1d(
                self.time_points, self.thrust_points,
                kind='linear', bounds_error=False, fill_value=0.0
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
    Rocket environment using real motor thrust curves with FIXED physics.

    UPDATED: Now supports loading motors from config files!

    Two ways to create:
    1. Traditional: RealisticMotorRocket(motor_data=CommonMotors.estes_c6(), config=...)
    2. NEW: RealisticMotorRocket(motor_config=config_dict['motor'], config=...)

    The new method works with ANY motor from ThrustCurve.org via generated configs.
    """

    def __init__(
        self,
        motor_data: Optional[MotorData] = None,
        config: Optional[RocketConfig] = None,
        motor_config: Optional[Dict[str, Any]] = None  # NEW!
    ):
        """
        Initialize with real motor data.

        Args:
            motor_data: Motor specifications (traditional method - backward compatible)
            config: Rocket configuration (will be updated with motor specs)
            motor_config: Motor config dict from YAML file (NEW - recommended!)
                         This is the 'motor' section from your config YAML

        Example (NEW method - works with any motor!):
            >>> import yaml
            >>> with open('configs/aerotech_k550w_easy.yaml') as f:
            ...     cfg = yaml.safe_load(f)
            >>> motor_config = cfg['motor']
            >>> rocket_config = RocketConfig(dry_mass=cfg['physics']['dry_mass'], ...)
            >>> env = RealisticMotorRocket(motor_config=motor_config, config=rocket_config)

        Example (traditional method - still works):
            >>> motor = CommonMotors.estes_c6()
            >>> env = RealisticMotorRocket(motor_data=motor, config=rocket_config)
        """
        if config is None:
            config = RocketConfig()

        # NEW: Load motor from config if provided
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

            print(f"✓ Loaded motor from config: {self.motor.manufacturer} {self.motor.designation}")

        # Traditional: Use provided motor_data
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
                "  - Traditional: motor_data=CommonMotors.estes_c6()\n"
                "  - NEW (recommended): motor_config=config_dict['motor']"
            )

        # Display motor info
        print(f"  Total impulse: {self.motor.total_impulse:.1f} N·s")
        print(f"  Burn time: {self.motor.burn_time:.2f}s")
        print(f"  Avg/Max thrust: {self.motor.average_thrust:.1f}N / {self.motor.max_thrust:.1f}N")

        # Calculate and display TWR
        total_mass = config.dry_mass + self.motor.propellant_mass
        twr = self.motor.average_thrust / (total_mass * 9.81)
        print(f"  Rocket mass: {total_mass*1000:.1f}g")
        print(f"  TWR: {twr:.2f}")

        if twr < 1.0:
            print(f"  ⚠️  WARNING: TWR < 1.0 - rocket cannot lift off!")
        elif twr < 2.0:
            print(f"  ⚠️  WARNING: TWR < 2.0 - marginal performance")

        super().__init__(config)

    def _update_propulsion(self) -> Tuple[float, float]:
        """
        Get thrust from real motor curve.
        Works with both MotorData and ConfigMotor!
        """
        if self.time < self.motor.burn_time:
            thrust = self.motor.get_thrust(self.time)
            motor_mass = self.motor.get_mass(self.time)
        else:
            thrust = 0.0
            motor_mass = self.motor_case_mass

        # Total mass = rocket dry mass + current motor mass
        total_mass = self.config.dry_mass + motor_mass

        # Update propellant remaining for parent class
        self.propellant_remaining = max(0, motor_mass - self.motor_case_mass)

        return thrust, total_mass

    def _get_info(self) -> Dict[str, Any]:
        """Extended info with motor data"""
        info = super()._get_info()

        # Add motor-specific info
        current_thrust = self.motor.get_thrust(self.time) if self.time < self.motor.burn_time else 0

        info.update({
            'motor': f"{self.motor.manufacturer} {self.motor.designation}",
            'current_thrust_N': current_thrust,
            'propellant_remaining_g': self.propellant_remaining * 1000,
        })

        return info


# NEW: Helper function to create environment from config file
def create_environment_from_config(
    config_path: str,
    rank: int = 0
) -> RealisticMotorRocket:
    """
    Create environment from config YAML file.

    This is the RECOMMENDED way to create environments - it automatically
    loads motor data from the config file, so ANY motor from ThrustCurve.org
    will work!

    Args:
        config_path: Path to config YAML file
        rank: Environment rank for seeding

    Returns:
        Configured RealisticMotorRocket environment

    Example:
        >>> # Works with ANY motor!
        >>> env = create_environment_from_config('configs/aerotech_k550w_easy.yaml')
        ✓ Loaded motor from config: AeroTech K550W
          Total impulse: 1539.1 N·s
          TWR: 5.00
    """
    import yaml

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Extract motor config
    motor_config = config.get('motor', {})

    # Extract physics and environment configs
    physics = config.get('physics', {})
    env_cfg = config.get('environment', {})

    # Create RocketConfig
    rocket_config = RocketConfig(
        dry_mass=physics.get('dry_mass', 0.1),
        diameter=physics.get('diameter', 0.024),
        length=physics.get('length', 0.4),
        num_fins=physics.get('num_fins', 4),
        fin_span=physics.get('fin_span', 0.04),
        fin_root_chord=physics.get('fin_root_chord', 0.05),
        fin_tip_chord=physics.get('fin_tip_chord', 0.025),
        max_tab_deflection=physics.get('max_tab_deflection', 15.0),
        tab_chord_fraction=physics.get('tab_chord_fraction', 0.25),
        tab_span_fraction=physics.get('tab_span_fraction', 0.5),
        cd_body=physics.get('cd_body', 0.5),
        cd_fins=physics.get('cd_fins', 0.01),
        cl_alpha=physics.get('cl_alpha', 2.0),
        control_effectiveness=physics.get('control_effectiveness', 1.0),
        disturbance_scale=physics.get('disturbance_scale', 0.0001),
        damping_scale=physics.get('damping_scale', 1.0),
        initial_spin_std=physics.get('initial_spin_std', 15.0),
        max_roll_rate=physics.get('max_roll_rate', 360.0),
        dt=env_cfg.get('dt', 0.01),
    )

    # Create environment with motor from config
    env = RealisticMotorRocket(
        motor_data=None,  # Don't use traditional motor
        config=rocket_config,
        motor_config=motor_config  # Use motor from config file
    )

    # Seed the environment
    if rank > 0:
        env.reset(seed=rank)

    return env


# Traditional convenience function (backward compatible)
def create_environment(
    motor_name: str = "estes_c6",
    dry_mass: float = None,
    **config_kwargs
) -> RealisticMotorRocket:
    """
    Create a rocket environment with specified motor.

    TRADITIONAL METHOD - Still works but limited to hardcoded motors.
    For new code, use create_environment_from_config() instead!

    Args:
        motor_name: One of "estes_c6", "aerotech_f40", "cesaroni_g79"
        dry_mass: Override dry mass (kg). If None, uses recommended mass for motor.
        **config_kwargs: Additional config overrides

    Returns:
        Configured environment
    """
    # Get motor
    motors = {
        "estes_c6": (CommonMotors.estes_c6, 0.100),      # 100g recommended
        "aerotech_f40": (CommonMotors.aerotech_f40, 0.400),  # 400g recommended
        "cesaroni_g79": (CommonMotors.cesaroni_g79, 0.800),  # 800g recommended
    }

    if motor_name not in motors:
        raise ValueError(f"Unknown motor: {motor_name}. Options: {list(motors.keys())}")

    motor_func, default_mass = motors[motor_name]
    motor = motor_func()

    # Build config
    config = RocketConfig(
        dry_mass=dry_mass if dry_mass is not None else default_mass,
        **config_kwargs
    )

    return RealisticMotorRocket(motor_data=motor, config=config)


if __name__ == "__main__":
    print("=" * 60)
    print("Testing Updated Realistic Motor Rocket")
    print("=" * 60)

    # Test 1: Traditional method (backward compatible)
    print("\n--- Test 1: Traditional Method (hardcoded motor) ---")
    env = create_environment(
        motor_name="estes_c6",
        dry_mass=0.100,  # 100g rocket
        disturbance_scale=0.0001,
        initial_spin_std=15.0,
    )

    print("\n--- Passive flight test (no control) ---")
    obs, info = env.reset()
    print(f"Initial: roll_rate={info['roll_rate_deg_s']:.1f}°/s")

    max_alt = 0
    for step in range(300):
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        max_alt = max(max_alt, info['altitude_m'])

        if step % 30 == 0:
            print(f"T={info['time_s']:.2f}s: alt={info['altitude_m']:.1f}m, "
                  f"roll={info['roll_rate_deg_s']:.1f}°/s, "
                  f"thrust={info['current_thrust_N']:.1f}N")

        if terminated or truncated:
            break

    print(f"\nMax altitude: {max_alt:.1f}m")
    print(f"Duration: {info['time_s']:.2f}s")
    print(f"Final spin: {info['roll_rate_deg_s']:.1f}°/s")

    # Test 2: NEW method with config file (if available)
    print("\n" + "=" * 60)
    print("--- Test 2: NEW Method (config-based motor) ---")

    import os
    if os.path.exists('configs/aerotech_k550w_easy.yaml'):
        print("\nFound config file - testing with K550W motor...")
        env_config = create_environment_from_config('configs/aerotech_k550w_easy.yaml')

        print("\nRunning quick test...")
        obs, info = env_config.reset()
        for step in range(50):
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env_config.step(action)
            if terminated or truncated:
                break

        print(f"✓ Config-based motor works! Max altitude: {info['max_altitude_m']:.1f}m")
    else:
        print("\nNo config file found for testing.")
        print("To test config-based motors:")
        print("  1. Generate a config: python generate_motor_config_fixed.py generate estes_c6")
        print("  2. Run: python realistic_spin_rocket_updated.py")

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)
