"""
Tests for train_improved module - training script utilities and wrappers.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class TestImprovedRewardWrapper:
    """Tests for ImprovedRewardWrapper class."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment for testing."""

        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )
                self.step_count = 0
                self.altitude = 0.0

            def reset(self, seed=None, options=None):
                self.step_count = 0
                self.altitude = 0.0
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                self.step_count += 1
                self.altitude += 1.0

                obs = np.zeros(10, dtype=np.float32)
                reward = 1.0
                terminated = self.step_count >= 100
                truncated = False
                info = {
                    "altitude_m": self.altitude,
                    "roll_rate_deg_s": 10.0,
                    "phase": "boost",
                }
                return obs, reward, terminated, truncated, info

        return MockEnv()

    def test_wrapper_creation(self, mock_env):
        """Test creating ImprovedRewardWrapper."""
        from train_improved import ImprovedRewardWrapper

        reward_config = {
            "altitude_reward_scale": 0.01,
            "spin_penalty_scale": -0.1,
            "low_spin_threshold": 10.0,
            "low_spin_bonus": 1.0,
            "control_effort_penalty": -0.01,
            "control_smoothness_penalty": -0.05,
        }

        wrapped = ImprovedRewardWrapper(mock_env, reward_config)

        assert wrapped is not None
        assert wrapped.observation_space == mock_env.observation_space
        assert wrapped.action_space == mock_env.action_space

    def test_wrapper_reset(self, mock_env):
        """Test wrapper reset."""
        from train_improved import ImprovedRewardWrapper

        reward_config = {"altitude_reward_scale": 0.01}
        wrapped = ImprovedRewardWrapper(mock_env, reward_config)

        obs, info = wrapped.reset()

        assert obs is not None
        assert wrapped.prev_action is None
        assert wrapped.prev_altitude == 0.0

    def test_wrapper_step(self, mock_env):
        """Test wrapper step modifies reward."""
        from train_improved import ImprovedRewardWrapper

        reward_config = {
            "altitude_reward_scale": 0.01,
            "spin_penalty_scale": -0.1,
            "low_spin_threshold": 20.0,
            "low_spin_bonus": 1.0,
        }

        wrapped = ImprovedRewardWrapper(mock_env, reward_config)
        wrapped.reset()

        action = np.array([0.5])
        obs, reward, terminated, truncated, info = wrapped.step(action)

        # Reward should be modified from original
        assert isinstance(reward, float)

    def test_wrapper_tracks_previous_action(self, mock_env):
        """Test that wrapper tracks previous action."""
        from train_improved import ImprovedRewardWrapper

        reward_config = {"control_smoothness_penalty": -0.05}
        wrapped = ImprovedRewardWrapper(mock_env, reward_config)
        wrapped.reset()

        action1 = np.array([0.5])
        wrapped.step(action1)

        assert wrapped.prev_action is not None
        assert np.allclose(wrapped.prev_action, action1)

        action2 = np.array([-0.5])
        wrapped.step(action2)

        assert np.allclose(wrapped.prev_action, action2)

    def test_terminal_success_bonus(self, mock_env):
        """Test success bonus on terminal state."""
        from train_improved import ImprovedRewardWrapper

        class TerminatingEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    True,  # Always terminate
                    False,
                    {"altitude_m": 150.0, "roll_rate_deg_s": 5.0},
                )

        reward_config = {
            "success_altitude": 100.0,
            "success_bonus": 50.0,
        }

        env = TerminatingEnv()
        wrapped = ImprovedRewardWrapper(env, reward_config)
        wrapped.reset()

        _, reward, terminated, _, _ = wrapped.step(np.array([0.0]))

        # Should include success bonus since altitude > success_altitude
        assert terminated
        assert reward >= 50.0  # At least the bonus


class TestNormalizedActionWrapper:
    """Tests for NormalizedActionWrapper class."""

    @pytest.fixture
    def mock_env(self):
        """Create a mock environment with non-normalized actions."""

        class MockEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                # Non-normalized action space
                self.action_space = spaces.Box(
                    low=np.array([-15.0]), high=np.array([15.0]), dtype=np.float32
                )
                self.last_action = None

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                self.last_action = action
                return (np.zeros(10, dtype=np.float32), 0.0, False, False, {})

        return MockEnv()

    def test_wrapper_normalizes_action_space(self, mock_env):
        """Test that wrapper normalizes action space to [-1, 1]."""
        from train_improved import NormalizedActionWrapper

        wrapped = NormalizedActionWrapper(mock_env)

        assert np.allclose(wrapped.action_space.low, -1.0)
        assert np.allclose(wrapped.action_space.high, 1.0)

    def test_wrapper_scales_actions(self, mock_env):
        """Test that wrapper correctly scales actions."""
        from train_improved import NormalizedActionWrapper

        wrapped = NormalizedActionWrapper(mock_env)
        wrapped.reset()

        # Action of 0 should map to middle of original range
        wrapped.step(np.array([0.0]))
        assert np.allclose(mock_env.last_action, 0.0)

        # Action of 1 should map to high
        wrapped.step(np.array([1.0]))
        assert np.allclose(mock_env.last_action, 15.0)

        # Action of -1 should map to low
        wrapped.step(np.array([-1.0]))
        assert np.allclose(mock_env.last_action, -15.0)

        # Action of 0.5 should map to 7.5
        wrapped.step(np.array([0.5]))
        assert np.allclose(mock_env.last_action, 7.5)


class TestTrainingMetricsCallback:
    """Tests for TrainingMetricsCallback class."""

    @pytest.fixture
    def mock_config(self):
        """Create a mock config for callback."""
        from rocket_config import RocketTrainingConfig

        return RocketTrainingConfig()

    def test_callback_creation(self, mock_config):
        """Test creating TrainingMetricsCallback."""
        from train_improved import TrainingMetricsCallback

        callback = TrainingMetricsCallback(mock_config, verbose=0)

        assert callback is not None
        assert callback.episode_count == 0
        assert len(callback.episode_rewards) == 0

    def test_callback_tracks_metrics(self, mock_config):
        """Test that callback tracks episode metrics."""
        from train_improved import TrainingMetricsCallback

        callback = TrainingMetricsCallback(mock_config, verbose=0)

        # Simulate a completed episode
        callback.locals = {
            "dones": [True],
            "infos": [
                {
                    "episode": {"r": 100.0, "l": 50},
                    "altitude_m": 75.0,
                    "roll_rate_deg_s": 15.0,
                    "horizontal_camera_quality": "Good (under 5°/s)",
                }
            ],
        }

        result = callback._on_step()

        assert result is True
        assert callback.episode_count == 1
        assert len(callback.episode_altitudes) == 1
        assert callback.episode_altitudes[0] == 75.0

    def test_callback_camera_quality_scoring(self, mock_config):
        """Test camera quality scoring."""
        from train_improved import TrainingMetricsCallback

        callback = TrainingMetricsCallback(mock_config, verbose=0)

        quality_tests = [
            ("Excellent (under 1°/s)", 4),
            ("Good (under 5°/s)", 3),
            ("Fair (under 10°/s)", 2),
            ("Poor (over 10°/s)", 1),
        ]

        for quality, expected_score in quality_tests:
            callback.episode_camera_scores = []
            callback.locals = {
                "dones": [True],
                "infos": [
                    {
                        "altitude_m": 50.0,
                        "roll_rate_deg_s": 5.0,
                        "horizontal_camera_quality": quality,
                    }
                ],
            }
            callback._on_step()

            assert (
                callback.episode_camera_scores[-1] == expected_score
            ), f"Quality '{quality}' should score {expected_score}"


class TestWrapperIntegration:
    """Integration tests for wrappers with real environment."""

    @pytest.fixture
    def real_env(self):
        """Create a real rocket environment."""
        from spin_stabilized_control_env import SpinStabilizedCameraRocket, RocketConfig
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()
        config = RocketConfig(dt=0.02)

        return SpinStabilizedCameraRocket(airframe=airframe, config=config)

    def test_normalized_action_with_real_env(self, real_env):
        """Test NormalizedActionWrapper with real environment."""
        from train_improved import NormalizedActionWrapper

        wrapped = NormalizedActionWrapper(real_env)

        obs, info = wrapped.reset()
        assert obs is not None

        # Normalized action
        action = np.array([0.5])
        obs, reward, terminated, truncated, info = wrapped.step(action)

        assert obs is not None
        assert isinstance(reward, (int, float))

    def test_reward_wrapper_with_real_env(self, real_env):
        """Test ImprovedRewardWrapper with real environment."""
        from train_improved import ImprovedRewardWrapper

        reward_config = {
            "altitude_reward_scale": 0.01,
            "spin_penalty_scale": -0.1,
            "low_spin_threshold": 30.0,
            "low_spin_bonus": 1.0,
            "control_effort_penalty": -0.01,
            "control_smoothness_penalty": -0.02,
        }

        wrapped = ImprovedRewardWrapper(real_env, reward_config)

        obs, info = wrapped.reset()
        action = np.array([0.0])

        for _ in range(10):
            obs, reward, terminated, truncated, info = wrapped.step(action)
            if terminated or truncated:
                break

        assert obs is not None

    def test_stacked_wrappers(self, real_env):
        """Test stacking multiple wrappers."""
        from train_improved import ImprovedRewardWrapper, NormalizedActionWrapper

        reward_config = {
            "altitude_reward_scale": 0.01,
            "spin_penalty_scale": -0.1,
        }

        # Stack wrappers
        wrapped = NormalizedActionWrapper(real_env)
        wrapped = ImprovedRewardWrapper(wrapped, reward_config)

        obs, info = wrapped.reset()

        # Normalized action
        action = np.array([0.5])
        obs, reward, terminated, truncated, info = wrapped.step(action)

        assert obs is not None


class TestRewardCalculation:
    """Tests for reward calculation details."""

    def test_altitude_reward_component(self):
        """Test altitude reward component."""
        from train_improved import ImprovedRewardWrapper

        class SimpleEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    False,
                    False,
                    {"altitude_m": 100.0, "roll_rate_deg_s": 0.0},
                )

        env = SimpleEnv()
        reward_config = {
            "altitude_reward_scale": 0.1,
            "spin_penalty_scale": 0.0,  # No spin penalty for this test
        }

        wrapped = ImprovedRewardWrapper(env, reward_config)
        wrapped.reset()

        _, reward, _, _, _ = wrapped.step(np.array([0.0]))

        # With altitude=100 and scale=0.1, altitude component should be 10
        assert reward >= 10.0

    def test_spin_penalty_component(self):
        """Test spin penalty component."""
        from train_improved import ImprovedRewardWrapper

        class SpinnyEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    False,
                    False,
                    {"altitude_m": 0.0, "roll_rate_deg_s": 100.0},  # High spin
                )

        env = SpinnyEnv()
        reward_config = {
            "altitude_reward_scale": 0.0,
            "spin_penalty_scale": -0.1,  # -0.1 per deg/s
            "low_spin_threshold": 10.0,
        }

        wrapped = ImprovedRewardWrapper(env, reward_config)
        wrapped.reset()

        _, reward, _, _, _ = wrapped.step(np.array([0.0]))

        # With spin=100 deg/s and scale=-0.1, penalty should be -10
        # No low spin bonus since 100 > threshold
        assert reward < 0  # Should be negative due to penalty

    def test_low_spin_bonus(self):
        """Test low spin bonus."""
        from train_improved import ImprovedRewardWrapper

        class LowSpinEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    False,
                    False,
                    {"altitude_m": 0.0, "roll_rate_deg_s": 5.0},  # Low spin
                )

        env = LowSpinEnv()
        reward_config = {
            "altitude_reward_scale": 0.0,
            "spin_penalty_scale": 0.0,
            "low_spin_threshold": 10.0,
            "low_spin_bonus": 5.0,
        }

        wrapped = ImprovedRewardWrapper(env, reward_config)
        wrapped.reset()

        _, reward, _, _, _ = wrapped.step(np.array([0.0]))

        # Should get the low spin bonus
        assert reward >= 5.0

    def test_crash_penalty(self):
        """Test crash penalty on termination with low altitude."""
        from train_improved import ImprovedRewardWrapper

        class CrashEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    True,  # Terminate
                    False,
                    {"altitude_m": 0.5, "roll_rate_deg_s": 0.0},  # Low alt = crash
                )

        env = CrashEnv()
        reward_config = {
            "altitude_reward_scale": 0.0,
            "spin_penalty_scale": 0.0,
            "crash_penalty": -100.0,
        }

        wrapped = ImprovedRewardWrapper(env, reward_config)
        wrapped.reset()

        _, reward, terminated, _, _ = wrapped.step(np.array([0.0]))

        assert terminated
        # Crash penalty is applied when terminated and altitude < 1.0
        assert reward < 0  # Should have crash penalty

    def test_control_effort_penalty(self):
        """Test control effort penalty."""
        from train_improved import ImprovedRewardWrapper

        class ControlEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    False,
                    False,
                    {"altitude_m": 0.0, "roll_rate_deg_s": 0.0},
                )

        env = ControlEnv()
        reward_config = {
            "altitude_reward_scale": 0.0,
            "spin_penalty_scale": 0.0,
            "control_effort_penalty": -1.0,  # Strong penalty
        }

        wrapped = ImprovedRewardWrapper(env, reward_config)
        wrapped.reset()

        # With full control effort (action=1.0), penalty = -1.0
        _, reward_full, _, _, _ = wrapped.step(np.array([1.0]))

        wrapped.reset()
        # With no control effort (action=0.0), penalty = 0.0
        _, reward_zero, _, _, _ = wrapped.step(np.array([0.0]))

        assert reward_zero > reward_full

    def test_control_smoothness_penalty(self):
        """Test control smoothness penalty for action changes."""
        from train_improved import ImprovedRewardWrapper

        class SmoothEnv(gym.Env):
            def __init__(self):
                self.observation_space = spaces.Box(
                    low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32
                )
                self.action_space = spaces.Box(
                    low=-1.0, high=1.0, shape=(1,), dtype=np.float32
                )

            def reset(self, seed=None, options=None):
                return np.zeros(10, dtype=np.float32), {}

            def step(self, action):
                return (
                    np.zeros(10, dtype=np.float32),
                    0.0,
                    False,
                    False,
                    {"altitude_m": 0.0, "roll_rate_deg_s": 0.0},
                )

        env = SmoothEnv()
        reward_config = {
            "altitude_reward_scale": 0.0,
            "spin_penalty_scale": 0.0,
            "control_effort_penalty": 0.0,
            "control_smoothness_penalty": -1.0,  # Strong penalty
        }

        wrapped = ImprovedRewardWrapper(env, reward_config)
        wrapped.reset()

        # First step - no previous action
        wrapped.step(np.array([0.5]))

        # Second step - same action, no penalty
        _, reward_same, _, _, _ = wrapped.step(np.array([0.5]))

        wrapped.reset()
        wrapped.step(np.array([0.5]))
        # Second step - opposite action, penalty
        _, reward_change, _, _, _ = wrapped.step(np.array([-0.5]))

        assert reward_same > reward_change
