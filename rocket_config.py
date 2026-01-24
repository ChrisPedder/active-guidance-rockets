#!/usr/bin/env python3
"""
Rocket Training Configuration System

This module provides a clean way to parametrize all training settings,
avoiding magic numbers and enabling systematic experimentation.

Usage:
    from rocket_config import RocketTrainingConfig, load_config

    # Load from YAML file
    config = load_config("configs/experiment_01.yaml")

    # Or create programmatically
    config = RocketTrainingConfig.for_estes_c6()
"""

import yaml
from dataclasses import dataclass, field, asdict
from typing import Dict, Any, Optional, List
from pathlib import Path


@dataclass
class RocketPhysicsConfig:
    """
    Physics configuration for rocket training.

    Rocket geometry is defined separately in a RocketAirframe file.
    This config contains only physics tuning parameters, control surface
    configuration, and simulation settings.

    REQUIRED: airframe_file must specify path to .ork or .yaml airframe file.
    """

    # === Airframe Reference (REQUIRED for new configs) ===
    # Path to airframe file (.ork from OpenRocket or .yaml)
    airframe_file: str = None

    # Control surfaces - applied to airframe fins
    max_tab_deflection: float = 15.0  # degrees
    tab_chord_fraction: float = 0.25  # fraction of fin chord
    tab_span_fraction: float = 0.5  # fraction of fin span
    num_controlled_fins: int = 2  # number of fins with active tabs

    # Aerodynamics tuning
    cd_body: float = 0.5  # Body drag coefficient
    cd_fins: float = 0.01  # Fin drag coefficient
    cl_alpha: float = 2.0  # Lift curve slope (per radian)
    control_effectiveness: float = 1.0  # Multiplier for control authority

    # === Physics Tuning ===
    disturbance_scale: float = 0.0001  # Random disturbance magnitude
    damping_scale: float = 1.0  # Multiplier for aerodynamic damping
    initial_spin_std: float = 15.0  # Initial spin disturbance (deg/s std)
    max_roll_rate: float = 720.0  # deg/s - termination threshold
    max_episode_time: float = 15.0  # seconds - max episode duration

    # === Legacy fields (for backward compatibility with old configs) ===
    # These are populated when loading old-style configs that specify
    # rocket geometry directly instead of using airframe_file
    dry_mass: Optional[float] = None
    propellant_mass: Optional[float] = None
    diameter: Optional[float] = None
    length: Optional[float] = None
    num_fins: Optional[int] = None
    fin_span: Optional[float] = None
    fin_root_chord: Optional[float] = None
    fin_tip_chord: Optional[float] = None

    def resolve_airframe(self) -> "RocketAirframe":
        """
        Load and return the RocketAirframe from the specified file.

        Returns:
            RocketAirframe instance

        Raises:
            ValueError: If airframe_file is not specified
        """
        from airframe import RocketAirframe

        if not self.airframe_file:
            raise ValueError(
                "airframe_file is REQUIRED in RocketPhysicsConfig.\n"
                "Specify a path to an OpenRocket (.ork) or YAML (.yaml) file:\n"
                "  physics:\n"
                "    airframe_file: 'configs/airframes/my_rocket.ork'\n"
                "\n"
                "Or use a factory method that includes an airframe:\n"
                "  config = RocketTrainingConfig.for_estes_alpha()"
            )

        return RocketAirframe.load(self.airframe_file)


@dataclass
class MotorConfig:
    """Motor selection and configuration"""

    name: str = "estes_c6"  # Motor identifier
    thrust_multiplier: float = 1.0  # Scale thrust (for testing)

    # Extended motor specifications (optional, for auto-generated configs)
    manufacturer: Optional[str] = None
    designation: Optional[str] = None
    diameter_mm: Optional[float] = None
    length_mm: Optional[float] = None
    total_mass_g: Optional[float] = None
    propellant_mass_g: Optional[float] = None
    case_mass_g: Optional[float] = None
    impulse_class: Optional[str] = None
    total_impulse_Ns: Optional[float] = None
    avg_thrust_N: Optional[float] = None
    max_thrust_N: Optional[float] = None
    burn_time_s: Optional[float] = None
    thrust_curve: Optional[Dict[str, List[float]]] = None

    def to_motor(self) -> "Motor":
        """
        Convert this config to a Motor object from motor_loader.

        Returns:
            Motor object that can provide thrust/mass as functions of time
        """
        from motor_loader import Motor

        # Convert dataclass to dict format expected by Motor class
        motor_dict = {
            "name": self.name,
            "manufacturer": self.manufacturer,
            "designation": self.designation,
            "diameter_mm": self.diameter_mm,
            "length_mm": self.length_mm,
            "total_mass_g": self.total_mass_g,
            "propellant_mass_g": self.propellant_mass_g,
            "case_mass_g": self.case_mass_g,
            "impulse_class": self.impulse_class,
            "total_impulse_Ns": self.total_impulse_Ns,
            "avg_thrust_N": self.avg_thrust_N,
            "max_thrust_N": self.max_thrust_N,
            "burn_time_s": self.burn_time_s,
            "thrust_curve": self.thrust_curve,
            "thrust_multiplier": self.thrust_multiplier,
        }

        # Remove None values
        motor_dict = {k: v for k, v in motor_dict.items() if v is not None}

        return Motor(motor_dict)

    def get_specs_dict(self) -> Dict[str, float]:
        """
        Get motor specifications as a dictionary.
        Uses data from the config if available.

        Returns:
            Dictionary with motor specs (for backward compatibility with validation)
        """
        # Use actual config data if available
        if self.avg_thrust_N is not None and self.propellant_mass_g is not None:
            return {
                "average_thrust": self.avg_thrust_N,
                "max_thrust": self.max_thrust_N or self.avg_thrust_N * 1.5,
                "total_impulse": self.total_impulse_Ns or 0,
                "burn_time": self.burn_time_s or 0,
                "propellant_mass": self.propellant_mass_g / 1000,  # Convert to kg
                "case_mass": self.case_mass_g / 1000 if self.case_mass_g else 0,
                "recommended_mass_range": self._estimate_mass_range(),
            }
        else:
            # Fallback: return minimal default values
            return {
                "average_thrust": 10.0,
                "max_thrust": 15.0,
                "total_impulse": 20.0,
                "burn_time": 2.0,
                "propellant_mass": 0.02,
                "case_mass": 0.02,
                "recommended_mass_range": (0.05, 0.5),
            }

    def _estimate_mass_range(self) -> tuple:
        """Estimate recommended rocket mass range based on motor thrust"""
        if self.avg_thrust_N is None:
            return (0.05, 0.5)

        # Rule of thumb: TWR between 3-10 at liftoff
        # Recommended mass = thrust / (TWR * g)
        g = 9.81
        min_mass = self.avg_thrust_N / (10 * g)  # TWR = 10
        max_mass = self.avg_thrust_N / (3 * g)  # TWR = 3

        return (min_mass, max_mass)


@dataclass
class EnvironmentConfig:
    """Environment simulation parameters"""

    # Simulation
    dt: float = 0.02  # Time step (seconds) - 50Hz
    max_episode_steps: int = 500  # Max steps per episode

    # Initial conditions
    initial_spin_rate_range: tuple = (-30.0, 30.0)  # deg/s
    initial_tilt_range: tuple = (-5.0, 5.0)  # degrees

    # Wind
    enable_wind: bool = True
    max_wind_speed: float = 5.0  # m/s
    max_gust_speed: float = 2.0  # m/s
    wind_variability: float = 0.5  # How much wind changes

    # Termination conditions
    max_tilt_angle: float = 45.0  # degrees - terminate if exceeded
    min_altitude: float = -1.0  # meters - ground level with tolerance
    max_altitude: float = 500.0  # meters - success threshold

    # Observation normalization
    normalize_observations: bool = True
    obs_clip_value: float = 10.0  # Clip normalized obs to this range


@dataclass
class RewardConfig:
    """Reward function weights and parameters"""

    # Primary objectives
    altitude_reward_scale: float = 0.01  # Reward per meter altitude
    spin_penalty_scale: float = -0.1  # Penalty per deg/s of spin

    # Stability bonuses
    low_spin_bonus: float = 1.0  # Bonus when spin < threshold
    low_spin_threshold: float = 10.0  # deg/s

    # Control penalties
    control_effort_penalty: float = -0.01  # Penalty for large actions
    control_smoothness_penalty: float = -0.05  # Penalty for action changes

    # Terminal rewards
    success_bonus: float = 100.0  # Bonus for reaching target altitude
    crash_penalty: float = -50.0  # Penalty for crash

    # Reward shaping
    use_potential_shaping: bool = True  # Use potential-based shaping
    gamma: float = 0.99  # Discount for shaping


@dataclass
class PPOConfig:
    """PPO algorithm hyperparameters"""

    # Core hyperparameters
    learning_rate: float = 3e-4
    n_steps: int = 2048  # Steps per rollout
    batch_size: int = 64
    n_epochs: int = 10
    gamma: float = 0.99  # Discount factor
    gae_lambda: float = 0.95  # GAE lambda

    # Clipping
    clip_range: float = 0.2
    clip_range_vf: Optional[float] = None  # Value function clip (None = same as policy)

    # Regularization
    ent_coef: float = 0.01  # Entropy coefficient
    vf_coef: float = 0.5  # Value function coefficient
    max_grad_norm: float = 0.5  # Gradient clipping

    # Advantage normalization
    normalize_advantage: bool = True

    # Network architecture
    policy_net_arch: List[int] = field(default_factory=lambda: [256, 256])
    value_net_arch: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "tanh"  # "relu", "tanh", "elu"

    # Training
    total_timesteps: int = 500_000
    n_envs: int = 8  # Parallel environments
    device: str = "auto"  # "auto", "cpu", "cuda"


@dataclass
class CurriculumConfig:
    """Curriculum learning settings"""

    enabled: bool = True

    # Stage definitions (progress through these as training improves)
    stages: List[Dict[str, Any]] = field(
        default_factory=lambda: [
            {
                "name": "basic",
                "initial_spin_range": (-10, 10),
                "wind_enabled": False,
                "target_reward": 50,
            },
            {
                "name": "moderate_spin",
                "initial_spin_range": (-30, 30),
                "wind_enabled": False,
                "target_reward": 100,
            },
            {
                "name": "with_wind",
                "initial_spin_range": (-30, 30),
                "wind_enabled": True,
                "max_wind_speed": 3.0,
                "target_reward": 150,
            },
            {
                "name": "full",
                "initial_spin_range": (-60, 60),
                "wind_enabled": True,
                "max_wind_speed": 5.0,
                "target_reward": 200,
            },
        ]
    )

    # Advancement criteria
    episodes_to_evaluate: int = 100
    advancement_threshold: float = 0.8  # Fraction of episodes meeting target


@dataclass
class LoggingConfig:
    """Logging and checkpointing settings"""

    log_dir: str = "logs"
    save_dir: str = "models"

    # Tensorboard
    tensorboard_log: bool = True

    # Checkpointing
    save_freq: int = 10_000  # Steps between saves
    keep_checkpoints: int = 5  # Number to keep

    # Evaluation
    eval_freq: int = 5_000
    n_eval_episodes: int = 20

    # Metrics logging
    log_episode_freq: int = 10  # Log every N episodes

    # Experiment tracking
    experiment_name: str = "rocket_spin_control"
    tags: List[str] = field(default_factory=list)


@dataclass
class RocketTrainingConfig:
    """Complete training configuration"""

    physics: RocketPhysicsConfig = field(default_factory=RocketPhysicsConfig)
    motor: MotorConfig = field(default_factory=MotorConfig)
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    ppo: PPOConfig = field(default_factory=PPOConfig)
    curriculum: CurriculumConfig = field(default_factory=CurriculumConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

    def save(self, path: str):
        """Save configuration to YAML file"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)

    @classmethod
    def load(cls, path: str) -> "RocketTrainingConfig":
        """Load configuration from YAML file"""
        with open(path, "r") as f:
            data = yaml.safe_load(f)

        # Handle backward compatibility with old-style physics configs
        physics_data = data.get("physics", {})
        physics_config = cls._load_physics_config(physics_data)

        return cls(
            physics=physics_config,
            motor=MotorConfig(**data.get("motor", {})),
            environment=EnvironmentConfig(**data.get("environment", {})),
            reward=RewardConfig(**data.get("reward", {})),
            ppo=PPOConfig(**data.get("ppo", {})),
            curriculum=CurriculumConfig(**data.get("curriculum", {})),
            logging=LoggingConfig(**data.get("logging", {})),
        )

    @classmethod
    def _load_physics_config(cls, physics_data: Dict[str, Any]) -> RocketPhysicsConfig:
        """
        Load physics config with backward compatibility for old-style configs.

        Old configs specified rocket geometry directly (dry_mass, diameter, etc.).
        New configs use airframe_file to reference a RocketAirframe definition.
        """
        # Valid fields for RocketPhysicsConfig
        valid_fields = {
            "airframe_file",
            "max_tab_deflection",
            "tab_chord_fraction",
            "tab_span_fraction",
            "num_controlled_fins",
            "cd_body",
            "cd_fins",
            "cl_alpha",
            "control_effectiveness",
            "disturbance_scale",
            "damping_scale",
            "initial_spin_std",
            "max_roll_rate",
            "max_episode_time",
        }

        # Legacy fields that were in old configs (now in airframe)
        legacy_geometry_fields = {
            "dry_mass",
            "propellant_mass",
            "diameter",
            "length",
            "num_fins",
            "fin_span",
            "fin_root_chord",
            "fin_tip_chord",
        }

        # Check if this is an old-style config
        has_legacy_fields = any(f in physics_data for f in legacy_geometry_fields)
        has_airframe = "airframe_file" in physics_data

        if has_legacy_fields and not has_airframe:
            # Create a temporary airframe file from legacy config
            airframe_file = cls._create_legacy_airframe(physics_data)
            physics_data = dict(physics_data)  # Copy to avoid mutation
            physics_data["airframe_file"] = airframe_file

        # Include both valid fields and legacy fields for backward compatibility
        all_valid_fields = valid_fields | legacy_geometry_fields
        filtered_data = {k: v for k, v in physics_data.items() if k in all_valid_fields}

        return RocketPhysicsConfig(**filtered_data)

    @classmethod
    def _create_legacy_airframe(cls, physics_data: Dict[str, Any]) -> str:
        """
        Create a temporary airframe YAML file from legacy physics config.

        Returns path to the created file.
        """
        import tempfile
        import os
        from pathlib import Path

        # Extract legacy parameters with defaults
        dry_mass = physics_data.get("dry_mass", 0.1)
        diameter = physics_data.get("diameter", 0.024)
        length = physics_data.get("length", 0.4)
        num_fins = physics_data.get("num_fins", 4)
        fin_span = physics_data.get("fin_span", 0.04)
        fin_root_chord = physics_data.get("fin_root_chord", 0.05)
        fin_tip_chord = physics_data.get("fin_tip_chord", 0.025)

        # Create airframe dict
        # Distribute mass: 20% nose, 60% body, 20% fins
        nose_length = 0.07
        body_length = length - nose_length

        airframe_dict = {
            "name": "Legacy Config Airframe",
            "description": "Auto-generated from legacy physics config",
            "components": [
                {
                    "type": "NoseCone",
                    "name": "Nose Cone",
                    "position": 0.0,
                    "length": nose_length,
                    "base_diameter": diameter,
                    "shape": "ogive",
                    "thickness": 0.002,
                    "material": "ABS Plastic",
                    "mass_override": dry_mass * 0.15,
                },
                {
                    "type": "BodyTube",
                    "name": "Body Tube",
                    "position": nose_length,
                    "length": body_length,
                    "outer_diameter": diameter,
                    "inner_diameter": diameter - 0.002,
                    "material": "Cardboard",
                    "mass_override": dry_mass * 0.65,
                },
                {
                    "type": "TrapezoidFinSet",
                    "name": "Fins",
                    "position": nose_length + body_length - fin_root_chord,
                    "num_fins": num_fins,
                    "root_chord": fin_root_chord,
                    "tip_chord": fin_tip_chord,
                    "span": fin_span,
                    "sweep_length": 0.0,
                    "thickness": 0.002,
                    "material": "Balsa",
                    "mass_override": dry_mass * 0.20,
                },
            ],
        }

        # Write to a temp file that persists for the session
        # Use a consistent location so repeated loads don't create many files
        cache_dir = Path(tempfile.gettempdir()) / "rocket_airframes"
        cache_dir.mkdir(exist_ok=True)

        # Create a hash-based filename for consistency
        import hashlib

        config_str = f"{dry_mass}_{diameter}_{length}_{num_fins}_{fin_span}_{fin_root_chord}_{fin_tip_chord}"
        config_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        airframe_path = cache_dir / f"legacy_airframe_{config_hash}.yaml"

        with open(airframe_path, "w") as f:
            yaml.dump(airframe_dict, f, default_flow_style=False, sort_keys=False)

        return str(airframe_path)

    @classmethod
    def for_estes_alpha(
        cls, airframe_path: str = "configs/airframes/estes_alpha.yaml"
    ) -> "RocketTrainingConfig":
        """
        Pre-configured settings for Estes Alpha III with C6 motor.

        Args:
            airframe_path: Path to airframe file (default: Estes Alpha YAML)
        """
        return cls(
            physics=RocketPhysicsConfig(
                airframe_file=airframe_path,
                max_tab_deflection=15.0,
                disturbance_scale=0.0001,
            ),
            motor=MotorConfig(name="estes_c6"),
            ppo=PPOConfig(
                learning_rate=1e-4,  # Lower for stability
                n_steps=1024,
                batch_size=32,
                n_epochs=20,
                clip_range=0.1,  # Smaller for fine control
            ),
        )

    @classmethod
    def for_high_power(cls, airframe_path: str) -> "RocketTrainingConfig":
        """
        Pre-configured settings for high power rockets.

        Args:
            airframe_path: Path to airframe file (REQUIRED)
        """
        return cls(
            physics=RocketPhysicsConfig(
                airframe_file=airframe_path,
                max_tab_deflection=20.0,
                disturbance_scale=0.0001,
            ),
            motor=MotorConfig(name="aerotech_f40"),
            ppo=PPOConfig(
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
            ),
        )

    def validate(self) -> List[str]:
        """Validate configuration and return list of warnings/errors"""
        issues = []

        # Check airframe is specified
        if not self.physics.airframe_file:
            issues.append("CRITICAL: physics.airframe_file is REQUIRED")
            return issues

        # Try to load airframe and validate
        try:
            airframe = self.physics.resolve_airframe()
            dry_mass = airframe.dry_mass
        except Exception as e:
            issues.append(f"CRITICAL: Failed to load airframe: {e}")
            return issues

        # Check thrust-to-weight ratio
        motor_specs = self.motor.get_specs_dict()
        total_mass = dry_mass + motor_specs["propellant_mass"]
        twr = motor_specs["average_thrust"] / (total_mass * 9.81)

        if twr < 1.0:
            issues.append(f"CRITICAL: TWR={twr:.2f} < 1.0 - rocket cannot fly!")
        elif twr < 2.0:
            issues.append(f"WARNING: TWR={twr:.2f} is marginal (recommend > 2.0)")

        # Check mass is in recommended range
        rec_range = motor_specs["recommended_mass_range"]
        if not (rec_range[0] <= dry_mass <= rec_range[1]):
            issues.append(
                f"WARNING: dry_mass={dry_mass}kg outside recommended "
                f"range [{rec_range[0]:.2f}, {rec_range[1]:.2f}] for {self.motor.name}"
            )

        # Check PPO parameters
        if self.ppo.batch_size > self.ppo.n_steps * self.ppo.n_envs:
            issues.append(
                f"WARNING: batch_size ({self.ppo.batch_size}) > rollout size "
                f"({self.ppo.n_steps * self.ppo.n_envs})"
            )

        return issues


def load_config(path: str) -> RocketTrainingConfig:
    """Convenience function to load configuration"""
    return RocketTrainingConfig.load(path)


def create_default_configs():
    """Create default configuration files for different scenarios"""

    configs_dir = Path("configs")
    configs_dir.mkdir(exist_ok=True)

    # Create Estes Alpha configuration
    RocketTrainingConfig.for_estes_alpha().save(configs_dir / "estes_alpha.yaml")

    # Create a debug/test configuration
    debug_config = RocketTrainingConfig.for_estes_alpha()
    debug_config.ppo.total_timesteps = 10_000
    debug_config.logging.eval_freq = 1_000
    debug_config.curriculum.enabled = False
    debug_config.save(configs_dir / "debug.yaml")

    print(f"Created configuration files in {configs_dir}/")


if __name__ == "__main__":
    create_default_configs()

    # Test loading and validation
    config = RocketTrainingConfig.for_estes_alpha()
    issues = config.validate()

    print("\nConfiguration validation:")
    if issues:
        for issue in issues:
            print(f"  {issue}")
    else:
        print("  All checks passed")

    # Load airframe for display
    try:
        airframe = config.physics.resolve_airframe()
        motor_specs = config.motor.get_specs_dict()
        total_mass = airframe.dry_mass + motor_specs["propellant_mass"]
        twr = motor_specs["average_thrust"] / (total_mass * 9.81)

        print(f"\nTWR calculation:")
        print(f"  Airframe: {airframe.name}")
        print(f"  Motor: {config.motor.name}")
        print(f"  Dry mass: {airframe.dry_mass*1000:.1f}g")
        print(f"  Total mass: {total_mass*1000:.1f}g")
        print(f"  Average thrust: {motor_specs['average_thrust']:.1f}N")
        print(f"  TWR: {twr:.2f}")
    except Exception as e:
        print(f"\nCould not load airframe: {e}")
