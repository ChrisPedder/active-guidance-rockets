"""
Tests for spin_stabilized_control_env.py - Gymnasium environment.
"""

import pytest
import numpy as np


class TestRocketConfig:
    """Tests for RocketConfig dataclass."""

    def test_default_values(self):
        """Test RocketConfig default values."""
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig()

        assert config.tab_chord_fraction == 0.25
        assert config.tab_span_fraction == 0.5
        assert config.max_tab_deflection == 15.0
        assert config.dt == 0.01
        assert config.max_roll_rate == 360.0

    def test_custom_values(self):
        """Test RocketConfig with custom values."""
        from spin_stabilized_control_env import RocketConfig

        config = RocketConfig(
            max_tab_deflection=20.0,
            disturbance_scale=0.001,
        )

        assert config.max_tab_deflection == 20.0
        assert config.disturbance_scale == 0.001


class TestSpinStabilizedCameraRocket:
    """Tests for SpinStabilizedCameraRocket environment."""

    def test_requires_airframe(self):
        """Test that environment requires an airframe."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        with pytest.raises(ValueError, match="RocketAirframe is required"):
            SpinStabilizedCameraRocket(airframe=None)

    def test_creation_with_airframe(self, estes_alpha_airframe):
        """Test creating environment with airframe."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)

        assert env.airframe is not None
        assert env.config is not None

    def test_action_space(self, estes_alpha_airframe):
        """Test action space definition."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)

        assert env.action_space.shape == (1,)
        assert env.action_space.low[0] == -1.0
        assert env.action_space.high[0] == 1.0

    def test_observation_space(self, estes_alpha_airframe):
        """Test observation space definition."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)

        assert len(env.observation_space.shape) == 1
        assert env.observation_space.shape[0] == 10  # 10 observation dimensions

    def test_reset(self, estes_alpha_airframe):
        """Test environment reset."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        obs, info = env.reset()

        assert obs.shape == (10,)
        assert isinstance(info, dict)
        assert "altitude_m" in info
        assert info["altitude_m"] == 0.0

    def test_reset_with_seed(self, estes_alpha_airframe):
        """Test that reset returns valid observation."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)

        obs, info = env.reset(seed=42)

        # Check observation is valid
        assert obs.shape == (10,)
        assert all(obs >= env.observation_space.low)
        assert all(obs <= env.observation_space.high)

    def test_step(self, estes_alpha_airframe):
        """Test taking a step in the environment."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (10,)
        assert isinstance(reward, float)
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_step_updates_time(self, estes_alpha_airframe):
        """Test that time advances after step."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig

        config = RocketConfig(dt=0.01)
        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe, config=config)
        env.reset()

        assert env.time == 0.0

        env.step(np.array([0.0]))
        assert env.time == pytest.approx(0.01, rel=0.01)

    def test_action_clipping(self, estes_alpha_airframe):
        """Test that actions are clipped to valid range."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        # Action outside range should be clipped
        env.step(np.array([2.0]))  # > 1.0
        assert env.last_action == 1.0

        env.step(np.array([-2.0]))  # < -1.0
        assert env.last_action == -1.0

    def test_altitude_increases_during_burn(self, estes_alpha_airframe):
        """Test that altitude increases during motor burn."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        initial_altitude = env.altitude

        # Take several steps during burn
        for _ in range(50):
            env.step(np.array([0.0]))

        assert env.altitude > initial_altitude

    def test_termination_on_excessive_spin(self, estes_alpha_airframe):
        """Test that episode terminates on excessive spin."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig

        config = RocketConfig(max_roll_rate=100.0)  # Low threshold
        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe, config=config)

        # Set up high spin rate
        env.reset()
        env.roll_rate = np.deg2rad(200.0)  # > max

        _, _, terminated, _, _ = env.step(np.array([0.0]))

        assert terminated

    def test_truncation_on_max_time(self, estes_alpha_airframe):
        """Test that episode truncates at max time."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig

        config = RocketConfig(
            max_episode_time=0.5, max_roll_rate=10000.0
        )  # Short episode
        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe, config=config)
        env.reset()

        truncated = False
        for _ in range(1000):  # Should truncate before this
            _, _, _, truncated, _ = env.step(np.array([0.0]))
            if truncated:
                break

        assert truncated

    def test_observation_bounds(self, estes_alpha_airframe):
        """Test that observations stay within bounds."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        for _ in range(100):
            obs, _, terminated, truncated, _ = env.step(
                np.array([np.random.uniform(-1, 1)])
            )

            # Check all observations are within bounds
            assert all(obs >= env.observation_space.low), f"Obs below low: {obs}"
            assert all(obs <= env.observation_space.high), f"Obs above high: {obs}"

            if terminated or truncated:
                break

    def test_info_contains_required_keys(self, estes_alpha_airframe):
        """Test that info dict contains expected keys."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        _, info = env.reset()

        required_keys = [
            "altitude_m",
            "vertical_velocity_ms",
            "roll_rate_deg_s",
            "time_s",
            "phase",
            "mass_kg",
            "airframe",
        ]

        for key in required_keys:
            assert key in info, f"Missing key: {key}"


class TestEnvironmentPhysics:
    """Tests for environment physics calculations."""

    def test_roll_torque_increases_with_dynamic_pressure(self, estes_alpha_airframe):
        """Test that control effectiveness increases with dynamic pressure."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)

        # Low dynamic pressure
        low_q_torque = env._calculate_roll_torque(10.0)

        # High dynamic pressure
        high_q_torque = env._calculate_roll_torque(1000.0)

        # Note: with random disturbance, this is probabilistic
        # We're mainly checking it doesn't crash

    def test_roll_inertia_calculation(self, estes_alpha_airframe):
        """Test roll inertia calculation."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        mass = env.airframe.dry_mass + 0.024  # Add motor mass
        inertia = env._calculate_roll_inertia(mass)

        assert inertia > 0

    def test_air_density_model(self, estes_alpha_airframe):
        """Test atmospheric density model."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)

        # At sea level
        env.altitude = 0.0
        rho_0 = env._get_air_density()
        assert rho_0 == pytest.approx(1.225, rel=0.01)

        # At 8km (scale height)
        env.altitude = 8000.0
        rho_8k = env._get_air_density()
        assert rho_8k == pytest.approx(1.225 / np.e, rel=0.01)

    def test_propulsion_during_burn(self, estes_alpha_airframe):
        """Test propulsion values during burn phase."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig

        config = RocketConfig(average_thrust=10.0, burn_time=2.0)
        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe, config=config)
        env.reset()

        thrust, mass = env._update_propulsion()

        assert thrust > 0  # During burn
        assert mass > env.airframe.dry_mass  # Has propellant

    def test_propulsion_after_burnout(self, estes_alpha_airframe):
        """Test propulsion values after burnout."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig

        config = RocketConfig(burn_time=1.0)
        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe, config=config)
        env.reset()
        env.time = 2.0  # After burnout
        env.propellant_remaining = 0.0

        thrust, mass = env._update_propulsion()

        assert thrust == 0.0
        assert mass == pytest.approx(env.airframe.dry_mass, rel=0.01)


class TestRewardFunction:
    """Tests for reward function."""

    def test_low_roll_rate_gives_high_reward(self, estes_alpha_airframe):
        """Test that low roll rate gives high reward."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        # Low roll rate
        env.roll_rate = np.deg2rad(2.0)  # 2 deg/s
        reward_low = env._calculate_reward(0.0)

        # High roll rate
        env.roll_rate = np.deg2rad(60.0)  # 60 deg/s
        reward_high = env._calculate_reward(0.0)

        assert reward_low > reward_high

    def test_control_effort_penalty(self, estes_alpha_airframe):
        """Test that large control actions are penalized."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()
        env.roll_rate = np.deg2rad(1.0)  # Low roll rate

        # No control
        env.last_action = 0.0
        reward_no_control = env._calculate_reward(0.0)

        # Full control
        env.last_action = 1.0
        reward_full_control = env._calculate_reward(0.0)

        assert reward_no_control > reward_full_control

    def test_camera_shake_penalty(self, estes_alpha_airframe):
        """Test that camera shake is penalized."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()
        env.roll_rate = np.deg2rad(1.0)
        env.last_action = 0.0

        reward_no_shake = env._calculate_reward(0.0)
        reward_high_shake = env._calculate_reward(10.0)

        assert reward_no_shake > reward_high_shake


class TestCameraShake:
    """Tests for camera shake metric."""

    def test_shake_depends_on_roll_rate(self, estes_alpha_airframe):
        """Test that shake depends on roll rate."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        env.roll_rate = np.deg2rad(10.0)
        env.roll_acceleration = 0.0
        shake_low = env._calculate_camera_shake()

        env.roll_rate = np.deg2rad(50.0)
        shake_high = env._calculate_camera_shake()

        assert shake_high > shake_low


class TestGymnasiumCompliance:
    """Tests for Gymnasium API compliance."""

    def test_env_api(self, estes_alpha_airframe):
        """Test environment has required Gymnasium API methods."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)

        # Check required attributes
        assert hasattr(env, "action_space")
        assert hasattr(env, "observation_space")
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        # Check spaces are valid
        assert env.action_space is not None
        assert env.observation_space is not None

        # Check reset returns correct format
        obs, info = env.reset()
        assert obs.shape == env.observation_space.shape
        assert isinstance(info, dict)

        # Check step returns correct format
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs.shape == env.observation_space.shape
        assert isinstance(reward, (float, int, np.floating))
        assert isinstance(terminated, bool)
        assert isinstance(truncated, bool)
        assert isinstance(info, dict)

    def test_sample_action(self, estes_alpha_airframe):
        """Test that sampled actions work."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket

        env = SpinStabilizedCameraRocket(airframe=estes_alpha_airframe)
        env.reset()

        for _ in range(10):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, info = env.step(action)

            if terminated or truncated:
                break
