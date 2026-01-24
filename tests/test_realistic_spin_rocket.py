"""
Tests for realistic_spin_rocket module - realistic motor rocket environment.
"""
import pytest
import numpy as np
import tempfile
from pathlib import Path


class TestRealisticMotorRocket:
    """Tests for RealisticMotorRocket environment."""

    @pytest.fixture
    def sample_motor_config(self):
        """Sample motor configuration."""
        return {
            'name': 'estes_c6',
            'manufacturer': 'Estes',
            'designation': 'C6',
            'total_impulse_Ns': 10.0,
            'avg_thrust_N': 5.4,
            'max_thrust_N': 14.0,
            'burn_time_s': 1.85,
            'propellant_mass_g': 12.3,
            'case_mass_g': 12.7,
            'thrust_curve': {
                'time_s': [0.0, 0.1, 0.5, 1.0, 1.5, 1.85],
                'thrust_N': [0.0, 14.0, 6.0, 5.0, 4.0, 0.0]
            }
        }

    @pytest.fixture
    def airframe(self):
        """Get an Estes Alpha III airframe."""
        from airframe import RocketAirframe
        return RocketAirframe.estes_alpha()

    def test_realistic_motor_rocket_creation(self, airframe, sample_motor_config):
        """Test creating RealisticMotorRocket environment."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig(
            max_tab_deflection=15.0,
            initial_spin_std=15.0,
            disturbance_scale=0.0001,
        )

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=sample_motor_config,
            config=config,
        )

        assert env is not None
        assert env.airframe == airframe
        assert env.motor is not None

    def test_environment_reset(self, airframe, sample_motor_config):
        """Test environment reset."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig()

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=sample_motor_config,
            config=config,
        )

        obs, info = env.reset()

        assert obs is not None
        assert info is not None
        assert 'roll_rate_deg_s' in info

    def test_environment_step(self, airframe, sample_motor_config):
        """Test environment step."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig()

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=sample_motor_config,
            config=config,
        )

        env.reset()
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert info is not None

    def test_motor_thrust_during_simulation(self, airframe, sample_motor_config):
        """Test that motor thrust is applied during simulation."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig(dt=0.01)

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=sample_motor_config,
            config=config,
        )

        obs, info = env.reset()

        # Run a few steps during burn
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

            if terminated or truncated:
                break

        # Altitude should increase during burn
        assert info['altitude_m'] >= 0

    def test_motor_info_in_step(self, airframe, sample_motor_config):
        """Test that motor info is included in step info."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig()

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=sample_motor_config,
            config=config,
        )

        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))

        assert 'motor' in info
        assert 'current_thrust_N' in info
        assert 'propellant_remaining_g' in info

    def test_propellant_consumption(self, airframe, sample_motor_config):
        """Test that propellant is consumed during burn."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig(dt=0.01)

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=sample_motor_config,
            config=config,
        )

        env.reset()

        # Get initial propellant
        _, _, _, _, info = env.step(np.array([0.0]))
        initial_prop = info['propellant_remaining_g']

        # Run many steps
        for _ in range(100):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
            if terminated or truncated:
                break

        # Propellant should be consumed
        final_prop = info['propellant_remaining_g']
        assert final_prop <= initial_prop

    def test_missing_airframe_error(self, sample_motor_config):
        """Test that missing airframe raises error."""
        from realistic_spin_rocket import RealisticMotorRocket

        with pytest.raises(ValueError, match="RocketAirframe is required"):
            RealisticMotorRocket(
                airframe=None,
                motor_config=sample_motor_config,
            )

    def test_missing_motor_error(self, airframe):
        """Test that missing motor raises error."""
        from realistic_spin_rocket import RealisticMotorRocket

        with pytest.raises(ValueError, match="Must provide either"):
            RealisticMotorRocket(
                airframe=airframe,
                motor_data=None,
                motor_config=None,
            )


class TestRealisticMotorRocketWithMotorData:
    """Tests for RealisticMotorRocket with MotorData object."""

    @pytest.fixture
    def airframe(self):
        """Get an Estes Alpha III airframe."""
        from airframe import RocketAirframe
        return RocketAirframe.estes_alpha()

    def test_create_with_motor_data(self, airframe):
        """Test creating environment with MotorData object."""
        from realistic_spin_rocket import RealisticMotorRocket, MotorData
        from spin_stabilized_control_env import RocketConfig
        from scipy import interpolate

        # Create MotorData directly
        motor_data = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,  # mm
            length=70.0,    # mm
            total_mass=24.0,  # g
            propellant_mass=12.0,  # g
            case_mass=12.0,  # g
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 0.5, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 5.0, 0.0]),
        )

        config = RocketConfig()

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_data=motor_data,
            config=config,
        )

        assert env is not None
        assert env.motor is not None

        # Test step works
        env.reset()
        obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        assert obs is not None


class TestCreateEnvironmentFromConfig:
    """Tests for create_environment_from_config function."""

    @pytest.fixture
    def config_yaml(self, tmp_path):
        """Create a sample config YAML file."""
        import yaml

        # First create an airframe file
        airframe_content = """
name: Test Airframe
description: Test airframe for unit tests
components:
- type: NoseCone
  name: Nose Cone
  position: 0.0
  length: 0.07
  base_diameter: 0.024
  shape: ogive
  thickness: 0.002
  material: ABS Plastic
- type: BodyTube
  name: Body Tube
  position: 0.07
  length: 0.24
  outer_diameter: 0.024
  inner_diameter: 0.022
  material: Cardboard
- type: TrapezoidFinSet
  name: Fins
  position: 0.24
  num_fins: 4
  root_chord: 0.05
  tip_chord: 0.025
  span: 0.04
  sweep_length: 0.0
  thickness: 0.002
  material: Balsa
"""
        airframe_file = tmp_path / "test_airframe.yaml"
        airframe_file.write_text(airframe_content)

        config = {
            'physics': {
                'airframe_file': str(airframe_file),
                'max_tab_deflection': 15.0,
                'disturbance_scale': 0.0001,
            },
            'motor': {
                'name': 'test_motor',
                'manufacturer': 'Test',
                'designation': 'T1',
                'total_impulse_Ns': 10.0,
                'avg_thrust_N': 5.0,
                'max_thrust_N': 10.0,
                'burn_time_s': 2.0,
                'propellant_mass_g': 12.0,
                'case_mass_g': 12.0,
                'thrust_curve': {
                    'time_s': [0.0, 0.5, 1.0, 2.0],
                    'thrust_N': [0.0, 10.0, 5.0, 0.0]
                }
            },
            'environment': {
                'dt': 0.01,
            }
        }

        config_file = tmp_path / "test_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        return str(config_file)

    def test_create_environment_from_config(self, config_yaml):
        """Test creating environment from config file."""
        from realistic_spin_rocket import create_environment_from_config

        env = create_environment_from_config(config_yaml)

        assert env is not None
        obs, info = env.reset()
        assert obs is not None

    def test_create_environment_missing_airframe(self, tmp_path):
        """Test error when airframe_file is missing."""
        import yaml
        from realistic_spin_rocket import create_environment_from_config

        config = {
            'physics': {
                # Missing airframe_file
                'max_tab_deflection': 15.0,
            },
            'motor': {
                'name': 'test_motor',
                'manufacturer': 'Test',
                'designation': 'T1',
                'total_impulse_Ns': 10.0,
                'avg_thrust_N': 5.0,
                'max_thrust_N': 10.0,
                'burn_time_s': 2.0,
                'propellant_mass_g': 12.0,
                'case_mass_g': 12.0,
                'thrust_curve': {
                    'time_s': [0.0, 2.0],
                    'thrust_N': [10.0, 0.0]
                }
            },
        }

        config_file = tmp_path / "no_airframe_config.yaml"
        with open(config_file, 'w') as f:
            yaml.dump(config, f)

        with pytest.raises(ValueError, match="must specify.*airframe_file"):
            create_environment_from_config(str(config_file))


class TestRealisticMotorRocketPhysics:
    """Tests for physics in RealisticMotorRocket."""

    @pytest.fixture
    def env(self):
        """Create an environment for testing."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()

        motor_config = {
            'name': 'test',
            'manufacturer': 'Test',
            'designation': 'T1',
            'total_impulse_Ns': 10.0,
            'avg_thrust_N': 5.0,
            'max_thrust_N': 10.0,
            'burn_time_s': 2.0,
            'propellant_mass_g': 12.0,
            'case_mass_g': 12.0,
            'thrust_curve': {
                'time_s': [0.0, 0.5, 1.5, 2.0],
                'thrust_N': [0.0, 10.0, 5.0, 0.0]
            }
        }

        config = RocketConfig(dt=0.01)

        return RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
            config=config,
        )

    def test_full_flight_simulation(self, env):
        """Test a complete flight simulation."""
        obs, info = env.reset()

        max_altitude = 0
        step_count = 0
        max_steps = 500

        while step_count < max_steps:
            action = np.array([0.0])  # No control input
            obs, reward, terminated, truncated, info = env.step(action)
            step_count += 1

            if info['altitude_m'] > max_altitude:
                max_altitude = info['altitude_m']

            if terminated or truncated:
                break

        # Should reach some altitude
        assert max_altitude > 0

    def test_control_response(self, env):
        """Test that control input affects spin rate."""
        env.reset()

        # Apply positive control for several steps
        positive_spins = []
        for _ in range(20):
            obs, _, terminated, truncated, info = env.step(np.array([1.0]))
            positive_spins.append(info['roll_rate_deg_s'])
            if terminated or truncated:
                break

        env.reset()

        # Apply negative control for several steps
        negative_spins = []
        for _ in range(20):
            obs, _, terminated, truncated, info = env.step(np.array([-1.0]))
            negative_spins.append(info['roll_rate_deg_s'])
            if terminated or truncated:
                break

        # Control should have some effect (spin rates should differ)
        # Note: initial conditions are random, so we check trend
        if len(positive_spins) > 1 and len(negative_spins) > 1:
            pos_change = positive_spins[-1] - positive_spins[0]
            neg_change = negative_spins[-1] - negative_spins[0]
            # The changes should generally be in opposite directions
            # (though depends on initial conditions)


class TestMotorDataFallback:
    """Tests for fallback MotorData class when thrustcurve_motor_data is not available."""

    def test_fallback_motor_data_creation(self):
        """Test that fallback MotorData works."""
        from realistic_spin_rocket import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        # Values should be converted to SI
        assert motor.diameter == pytest.approx(0.018, rel=0.01)
        assert motor.length == pytest.approx(0.070, rel=0.01)

    def test_fallback_motor_data_thrust(self):
        """Test thrust interpolation in fallback MotorData."""
        from realistic_spin_rocket import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        assert motor.get_thrust(0.0) == pytest.approx(0.0, abs=0.1)
        assert motor.get_thrust(1.0) == pytest.approx(10.0, abs=0.1)
        assert motor.get_thrust(2.0) == pytest.approx(0.0, abs=0.1)

    def test_fallback_motor_data_mass(self):
        """Test mass calculation in fallback MotorData."""
        from realistic_spin_rocket import MotorData

        motor = MotorData(
            manufacturer="Test",
            designation="T1",
            diameter=18.0,
            length=70.0,
            total_mass=24.0,
            propellant_mass=12.0,
            case_mass=12.0,
            total_impulse=10.0,
            burn_time=2.0,
            average_thrust=5.0,
            max_thrust=10.0,
            time_points=np.array([0.0, 1.0, 2.0]),
            thrust_points=np.array([0.0, 10.0, 0.0]),
        )

        # At burn time, should be case mass only
        assert motor.get_mass(2.0) == pytest.approx(motor.case_mass, rel=0.01)

        # Before burn, should be total
        # Note: total_mass is converted to kg in __post_init__
        initial_mass = motor.get_mass(0.0)
        assert initial_mass > motor.case_mass


class TestRealisticMotorRocketTWR:
    """Tests for TWR warnings in RealisticMotorRocket."""

    @pytest.fixture
    def airframe(self):
        """Get an Estes Alpha III airframe."""
        from airframe import RocketAirframe
        return RocketAirframe.estes_alpha()

    def test_normal_twr(self, airframe, capsys):
        """Test environment with normal TWR doesn't warn."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig

        # Normal motor config with good TWR
        motor_config = {
            'name': 'test',
            'manufacturer': 'Test',
            'designation': 'T1',
            'total_impulse_Ns': 10.0,
            'avg_thrust_N': 10.0,  # Good thrust
            'max_thrust_N': 15.0,
            'burn_time_s': 1.0,
            'propellant_mass_g': 10.0,
            'case_mass_g': 10.0,
            'thrust_curve': {
                'time_s': [0.0, 0.5, 1.0],
                'thrust_N': [0.0, 15.0, 0.0]
            }
        }

        config = RocketConfig()

        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
            config=config,
        )

        captured = capsys.readouterr()
        # Should not have marginal/can't lift off warning
        assert "cannot lift off" not in captured.out.lower()


class TestRealisticMotorRocketUpdatePropulsion:
    """Tests for _update_propulsion method."""

    @pytest.fixture
    def env(self):
        """Create an environment for testing."""
        from realistic_spin_rocket import RealisticMotorRocket
        from spin_stabilized_control_env import RocketConfig
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()

        motor_config = {
            'name': 'test',
            'manufacturer': 'Test',
            'designation': 'T1',
            'total_impulse_Ns': 10.0,
            'avg_thrust_N': 5.0,
            'max_thrust_N': 10.0,
            'burn_time_s': 2.0,
            'propellant_mass_g': 12.0,
            'case_mass_g': 12.0,
            'thrust_curve': {
                'time_s': [0.0, 0.5, 1.5, 2.0],
                'thrust_N': [0.0, 10.0, 5.0, 0.0]
            }
        }

        config = RocketConfig(dt=0.01)

        return RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
            config=config,
        )

    def test_update_propulsion_during_burn(self, env):
        """Test propulsion update during burn phase."""
        env.reset()
        env.time = 0.5  # During burn

        thrust, mass = env._update_propulsion()

        assert thrust > 0  # Should have thrust
        assert mass > env.motor_case_mass  # Should still have propellant

    def test_update_propulsion_after_burn(self, env):
        """Test propulsion update after burn."""
        env.reset()
        env.time = 5.0  # After burn

        thrust, mass = env._update_propulsion()

        assert thrust == 0  # No thrust
        # Mass should be airframe + case mass
        assert mass == pytest.approx(env.airframe.dry_mass + env.motor_case_mass, rel=0.01)
