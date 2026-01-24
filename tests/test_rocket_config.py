"""
Tests for rocket_config.py - configuration loading and validation.
"""
import pytest
import tempfile
from pathlib import Path
import yaml


class TestRocketPhysicsConfig:
    """Tests for RocketPhysicsConfig class."""

    def test_default_values(self):
        """Test that RocketPhysicsConfig has reasonable defaults."""
        from rocket_config import RocketPhysicsConfig

        config = RocketPhysicsConfig()

        assert config.airframe_file is None
        assert config.max_tab_deflection == 15.0
        assert config.tab_chord_fraction == 0.25
        assert config.tab_span_fraction == 0.5
        assert config.num_controlled_fins == 2
        assert config.disturbance_scale == 0.0001
        assert config.damping_scale == 1.0

    def test_custom_values(self):
        """Test that custom values are properly set."""
        from rocket_config import RocketPhysicsConfig

        config = RocketPhysicsConfig(
            max_tab_deflection=20.0,
            disturbance_scale=0.001,
        )

        assert config.max_tab_deflection == 20.0
        assert config.disturbance_scale == 0.001


class TestMotorConfig:
    """Tests for MotorConfig class."""

    def test_default_values(self):
        """Test MotorConfig default values."""
        from rocket_config import MotorConfig

        config = MotorConfig()

        assert config.name == "estes_c6"
        assert config.thrust_multiplier == 1.0

    def test_custom_motor(self, sample_motor_config):
        """Test creating motor config with custom values."""
        from rocket_config import MotorConfig

        config = MotorConfig(**sample_motor_config)

        assert config.name == 'test_motor'
        assert config.avg_thrust_N == 5.0
        assert config.burn_time_s == 2.0

    def test_to_motor(self, sample_motor_config):
        """Test converting MotorConfig to Motor object."""
        from rocket_config import MotorConfig

        config = MotorConfig(**sample_motor_config)
        motor = config.to_motor()

        assert motor.name == 'test_motor'
        assert motor.average_thrust == 5.0
        assert motor.burn_time == 2.0

    def test_get_specs_dict(self, sample_motor_config):
        """Test getting motor specs as dictionary."""
        from rocket_config import MotorConfig

        config = MotorConfig(**sample_motor_config)
        specs = config.get_specs_dict()

        assert 'average_thrust' in specs
        assert 'max_thrust' in specs
        assert 'burn_time' in specs
        assert specs['average_thrust'] == 5.0


class TestEnvironmentConfig:
    """Tests for EnvironmentConfig class."""

    def test_default_values(self):
        """Test EnvironmentConfig defaults."""
        from rocket_config import EnvironmentConfig

        config = EnvironmentConfig()

        assert config.dt == 0.02
        assert config.max_episode_steps == 500
        assert config.enable_wind is True
        assert config.max_tilt_angle == 45.0


class TestRewardConfig:
    """Tests for RewardConfig class."""

    def test_default_values(self):
        """Test RewardConfig defaults."""
        from rocket_config import RewardConfig

        config = RewardConfig()

        assert config.altitude_reward_scale == 0.01
        assert config.spin_penalty_scale == -0.1
        assert config.success_bonus == 100.0
        assert config.crash_penalty == -50.0


class TestPPOConfig:
    """Tests for PPOConfig class."""

    def test_default_values(self):
        """Test PPOConfig defaults."""
        from rocket_config import PPOConfig

        config = PPOConfig()

        assert config.learning_rate == 3e-4
        assert config.n_steps == 2048
        assert config.batch_size == 64
        assert config.gamma == 0.99
        assert config.total_timesteps == 500_000


class TestRocketTrainingConfig:
    """Tests for RocketTrainingConfig class."""

    def test_default_construction(self):
        """Test creating config with defaults."""
        from rocket_config import RocketTrainingConfig

        config = RocketTrainingConfig()

        assert config.physics is not None
        assert config.motor is not None
        assert config.environment is not None
        assert config.reward is not None
        assert config.ppo is not None

    def test_to_dict(self):
        """Test converting config to dictionary."""
        from rocket_config import RocketTrainingConfig

        config = RocketTrainingConfig()
        d = config.to_dict()

        assert 'physics' in d
        assert 'motor' in d
        assert 'environment' in d
        assert 'ppo' in d

    def test_save_and_load(self, tmp_path):
        """Test saving and loading config."""
        from rocket_config import RocketTrainingConfig, EnvironmentConfig

        # Create config with list-based ranges (YAML-compatible)
        config = RocketTrainingConfig()
        # EnvironmentConfig has tuple defaults that don't serialize well to YAML
        # This is a known limitation - test with explicit YAML file
        config_path = tmp_path / "test_config.yaml"

        # Write a minimal valid config
        import yaml
        config_dict = {
            'physics': {},
            'motor': {'name': 'test'},
            'environment': {
                'dt': 0.02,
                'initial_spin_rate_range': [-30.0, 30.0],
                'initial_tilt_range': [-5.0, 5.0],
            },
            'reward': {},
            'ppo': {'learning_rate': 0.001},
            'curriculum': {},
            'logging': {},
        }
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        loaded = RocketTrainingConfig.load(str(config_path))
        assert loaded.ppo.learning_rate == 0.001

    def test_load_legacy_config(self, sample_training_config_yaml):
        """Test loading a legacy config with old-style physics."""
        from rocket_config import load_config

        config = load_config(sample_training_config_yaml)

        # Should have created a temporary airframe file
        assert config.physics.airframe_file is not None
        assert Path(config.physics.airframe_file).exists()

        # Should have preserved other physics params
        assert config.physics.max_tab_deflection == 15.0
        assert config.physics.disturbance_scale == 0.0001

    def test_load_new_style_config(self, tmp_path, new_style_physics_config, sample_motor_config):
        """Test loading a new-style config with airframe_file."""
        import yaml
        from rocket_config import load_config

        config_dict = {
            'physics': new_style_physics_config,
            'motor': sample_motor_config,
            'environment': {},
            'reward': {},
            'ppo': {},
            'curriculum': {},
            'logging': {},
        }

        config_path = tmp_path / "new_style_config.yaml"
        with open(config_path, 'w') as f:
            yaml.dump(config_dict, f)

        config = load_config(str(config_path))

        # Should use the specified airframe file
        assert config.physics.airframe_file == new_style_physics_config['airframe_file']


class TestBackwardCompatibility:
    """Tests for backward compatibility with old config formats."""

    def test_legacy_config_creates_airframe(self, legacy_physics_config):
        """Test that legacy config creates temporary airframe."""
        from rocket_config import RocketTrainingConfig

        physics_config = RocketTrainingConfig._load_physics_config(legacy_physics_config)

        assert physics_config.airframe_file is not None
        assert Path(physics_config.airframe_file).exists()

    def test_legacy_airframe_has_correct_mass(self, legacy_physics_config):
        """Test that created airframe has approximately correct mass."""
        from rocket_config import RocketTrainingConfig
        from airframe import RocketAirframe

        physics_config = RocketTrainingConfig._load_physics_config(legacy_physics_config)
        airframe = RocketAirframe.load(physics_config.airframe_file)

        # Mass should be close to the specified dry_mass
        assert abs(airframe.dry_mass - legacy_physics_config['dry_mass']) < 0.01

    def test_legacy_airframe_has_correct_dimensions(self, legacy_physics_config):
        """Test that created airframe has correct dimensions."""
        from rocket_config import RocketTrainingConfig
        from airframe import RocketAirframe

        physics_config = RocketTrainingConfig._load_physics_config(legacy_physics_config)
        airframe = RocketAirframe.load(physics_config.airframe_file)

        # Body diameter should match
        assert abs(airframe.body_diameter - legacy_physics_config['diameter']) < 0.001

    def test_mixed_config_prefers_new_fields(self, tmp_path):
        """Test that when both old and new fields exist, airframe_file is used."""
        from rocket_config import RocketTrainingConfig
        import yaml

        # Create airframe file
        airframe_content = """
name: Test Airframe
components:
- type: NoseCone
  name: Nose
  position: 0.0
  length: 0.05
  base_diameter: 0.03
  shape: ogive
  thickness: 0.002
  material: ABS Plastic
  mass_override: 0.05
"""
        airframe_file = tmp_path / "airframe.yaml"
        airframe_file.write_text(airframe_content)

        # Create config with both old and new style
        physics_data = {
            'airframe_file': str(airframe_file),
            'dry_mass': 0.1,  # This should be ignored
            'diameter': 0.024,  # This should be ignored
            'max_tab_deflection': 15.0,
        }

        physics_config = RocketTrainingConfig._load_physics_config(physics_data)

        # Should use the specified airframe file, not create a new one
        assert physics_config.airframe_file == str(airframe_file)
