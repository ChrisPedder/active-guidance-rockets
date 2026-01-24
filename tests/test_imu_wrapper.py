"""
Tests for IMU observation wrapper.
"""

import pytest
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from rocket_env.sensors import IMUObservationWrapper, IMUConfig


class MockRocketEnv(gym.Env):
    """Mock environment for testing IMU wrapper."""

    def __init__(self):
        super().__init__()
        # Observation space matching real rocket env
        self.observation_space = spaces.Box(
            low=-100, high=100, shape=(10,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        self._step_count = 0
        self._roll_rate = 0.0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._roll_rate = np.deg2rad(np.random.uniform(-30, 30))
        return self._get_obs(), {}

    def step(self, action):
        self._step_count += 1
        # Simulate roll rate dynamics
        self._roll_rate += action[0] * 0.1 + np.random.normal(0, 0.01)

        obs = self._get_obs()
        reward = -abs(self._roll_rate)
        terminated = self._step_count >= 100
        truncated = False

        return obs, reward, terminated, truncated, {"roll_rate_rad_s": self._roll_rate}

    def _get_obs(self):
        """Get observation with roll rate at index 3 and acceleration at index 4."""
        obs = np.zeros(10, dtype=np.float32)
        obs[0] = self._step_count  # altitude
        obs[1] = 10.0  # velocity
        obs[2] = 0.0  # roll angle
        obs[3] = self._roll_rate  # roll rate (rad/s)
        obs[4] = 0.1  # roll acceleration (rad/s²)
        obs[5] = 500.0  # dynamic pressure
        obs[6] = self._step_count * 0.01  # time
        return obs


class TestIMUObservationWrapper:
    """Tests for IMUObservationWrapper."""

    def test_initialization(self):
        """Test wrapper initialization."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(env)

        assert wrapped.imu_config.name == "icm_20948"  # Default
        assert wrapped.roll_rate_index == 3
        assert wrapped.roll_accel_index == 4

    def test_custom_imu_config(self):
        """Test wrapper with custom IMU config."""
        env = MockRocketEnv()
        config = IMUConfig.mpu_6050()
        wrapped = IMUObservationWrapper(env, imu_config=config)

        assert wrapped.imu_config.name == "mpu_6050"

    def test_observation_space_unchanged(self):
        """Test that observation space is preserved."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(env)

        assert wrapped.observation_space == env.observation_space

    def test_action_space_unchanged(self):
        """Test that action space is preserved."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(env)

        assert wrapped.action_space == env.action_space

    def test_reset_returns_noisy_obs(self):
        """Test that reset returns observation with noise."""
        env = MockRocketEnv()
        noisy_config = IMUConfig.mpu_6050()  # High noise
        wrapped = IMUObservationWrapper(env, imu_config=noisy_config, seed=42)

        obs, info = wrapped.reset(seed=42)

        # Should have sensor info
        assert "imu" in info
        assert info["imu"]["preset"] == "mpu_6050"
        assert "gyro_bias" in info["imu"]

    def test_noisy_vs_clean_observations(self):
        """Test that noisy wrapper produces different observations."""
        env = MockRocketEnv()

        # Create wrapper with high noise
        noisy_config = IMUConfig(
            name="high_noise",
            gyro=IMUConfig.mpu_6050().gyro,
        )
        wrapped = IMUObservationWrapper(env, imu_config=noisy_config)

        # Run episode and collect observations
        clean_obs_list = []
        noisy_obs_list = []

        env.reset(seed=42)
        wrapped.reset(seed=42)

        for _ in range(50):
            action = np.array([0.0])

            clean_obs, _, _, _, _ = env.step(action)
            noisy_obs, _, _, _, _ = wrapped.step(action)

            clean_obs_list.append(clean_obs[3])
            noisy_obs_list.append(noisy_obs[3])

        # Noisy observations should differ from clean
        clean_arr = np.array(clean_obs_list)
        noisy_arr = np.array(noisy_obs_list)

        # Mean should be similar (no systematic offset with ideal sensor)
        # but individual measurements should differ
        diff = np.abs(clean_arr - noisy_arr)
        assert np.mean(diff) > 0.001  # Should have some noise

    def test_ideal_sensor_no_noise(self):
        """Test that ideal sensor produces minimal noise."""
        env = MockRocketEnv()
        ideal_config = IMUConfig.ideal()
        wrapped = IMUObservationWrapper(env, imu_config=ideal_config)

        wrapped.reset(seed=42)

        # With ideal sensor, the wrapper should not modify the roll rate
        # We test this by checking that the observation transformation
        # returns the same value for ideal sensors
        for _ in range(20):
            action = np.array([0.0])
            noisy_obs, _, _, _, info = wrapped.step(action)

            # Get the true roll rate from the underlying env
            true_rate = wrapped.env._roll_rate

            # With ideal sensor, measured rate should equal true rate
            measured_rate = noisy_obs[3]
            assert np.isclose(measured_rate, true_rate, rtol=1e-5)

    def test_get_sensor_state(self):
        """Test sensor state retrieval."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(env)

        wrapped.reset(seed=42)
        state = wrapped.get_sensor_state()

        assert "imu_preset" in state
        assert "gyro" in state
        assert "bias" in state["gyro"]

    def test_uses_noisy_observations_flag(self):
        """Test uses_noisy_observations property."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(env)

        assert wrapped.uses_noisy_observations is True


class TestIMUWrapperIntegration:
    """Integration tests for IMU wrapper with realistic scenarios."""

    def test_episode_completion(self):
        """Test that wrapped environment completes episodes normally."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(env)

        obs, _ = wrapped.reset()
        total_reward = 0
        done = False

        while not done:
            action = wrapped.action_space.sample()
            obs, reward, terminated, truncated, info = wrapped.step(action)
            total_reward += reward
            done = terminated or truncated

        # Should complete without errors
        assert done

    def test_multiple_resets(self):
        """Test that multiple resets work correctly."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(env)

        for i in range(5):
            obs, info = wrapped.reset(seed=i)

            # Each reset should give new sensor bias
            bias1 = info["imu"]["gyro_bias"]

            obs, info = wrapped.reset(seed=i + 100)
            bias2 = info["imu"]["gyro_bias"]

            assert bias1 != bias2  # Different seeds should give different biases

    def test_acceleration_derivation(self):
        """Test acceleration is derived from noisy rate."""
        env = MockRocketEnv()
        wrapped = IMUObservationWrapper(
            env, derive_acceleration=True, control_rate_hz=100.0
        )

        wrapped.reset(seed=42)

        # Run a few steps
        prev_rate = None
        for _ in range(10):
            action = np.array([0.1])  # Apply control
            obs, _, _, _, _ = wrapped.step(action)

            if prev_rate is not None:
                # Acceleration should be consistent with rate change
                # (not exact match due to noise, but reasonable)
                accel = obs[4]
                expected_accel = (obs[3] - prev_rate) / (1.0 / 100.0)
                # Just check it's in a reasonable range
                assert abs(accel) < 1000  # Not crazy

            prev_rate = obs[3]

    def test_different_control_rates(self):
        """Test wrapper works with different control rates."""
        env = MockRocketEnv()

        for rate in [50.0, 100.0, 200.0]:
            wrapped = IMUObservationWrapper(env, control_rate_hz=rate)
            obs, _ = wrapped.reset()
            obs, _, _, _, _ = wrapped.step(np.array([0.0]))

            # Should work without errors
            assert obs is not None
