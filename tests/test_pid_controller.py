"""
Tests for pid_controller.py

Tests for PID controller for rocket roll stabilization.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from pid_controller import (
    PIDConfig,
    PIDController,
    EpisodeResult,
    create_env,
    run_episode,
    evaluate_pid,
    print_summary,
)


class TestPIDConfig:
    """Tests for PIDConfig dataclass."""

    def test_default_values(self):
        """Test default PID gains."""
        config = PIDConfig()
        assert config.Cprop == 0.01
        assert config.Cint == 0.001
        assert config.Cderiv == 0.1
        assert config.max_roll_rate == 100.0
        assert config.max_deflection == 30.0
        assert config.launch_accel_threshold == 20.0

    def test_custom_values(self):
        """Test custom PID gains."""
        config = PIDConfig(
            Cprop=0.02,
            Cint=0.005,
            Cderiv=0.05,
            max_roll_rate=50.0,
            max_deflection=15.0,
            launch_accel_threshold=10.0,
        )
        assert config.Cprop == 0.02
        assert config.Cint == 0.005
        assert config.Cderiv == 0.05
        assert config.max_roll_rate == 50.0
        assert config.max_deflection == 15.0
        assert config.launch_accel_threshold == 10.0


class TestPIDController:
    """Tests for PIDController class."""

    def test_init_default_config(self):
        """Test initialization with default config."""
        controller = PIDController()
        assert controller.config is not None
        assert controller.launch_detected is False
        assert controller.integ_error == 0.0

    def test_init_custom_config(self):
        """Test initialization with custom config."""
        config = PIDConfig(Cprop=0.05)
        controller = PIDController(config)
        assert controller.config.Cprop == 0.05

    def test_reset(self):
        """Test controller reset."""
        controller = PIDController()
        controller.launch_detected = True
        controller.integ_error = 10.0
        controller.launch_orient = 45.0

        controller.reset()

        assert controller.launch_detected is False
        assert controller.integ_error == 0.0
        assert controller.launch_orient == 0.0
        assert controller.target_orient == 0.0

    def test_step_before_launch(self):
        """Test step returns zero before launch detection."""
        controller = PIDController()
        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.1,
            "roll_rate_deg_s": 5.0,
            "vertical_acceleration_ms2": 5.0,  # Below threshold
        }

        action = controller.step(obs, info)

        assert controller.launch_detected is False
        assert action[0] == 0.0

    def test_step_launch_detection(self):
        """Test launch detection when acceleration exceeds threshold."""
        controller = PIDController()
        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.1,
            "roll_rate_deg_s": 5.0,
            "vertical_acceleration_ms2": 25.0,  # Above threshold
        }

        action = controller.step(obs, info)

        assert controller.launch_detected is True
        assert controller.launch_orient == np.degrees(0.1)

    def test_step_pid_control(self):
        """Test PID control after launch."""
        config = PIDConfig(Cprop=0.1, Cint=0.0, Cderiv=0.0)
        controller = PIDController(config)
        controller.launch_detected = True
        controller.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(10.0),  # 10 degrees error
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
        }

        action = controller.step(obs, info)

        # Proportional control: 10 deg error * 0.1 gain = 1.0 deg
        # Normalized: 1.0 / 30.0 = 0.033
        assert action[0] > 0  # Positive action for positive error
        assert -1.0 <= action[0] <= 1.0

    def test_step_derivative_control(self):
        """Test derivative (rate) control."""
        config = PIDConfig(Cprop=0.0, Cint=0.0, Cderiv=0.1)
        controller = PIDController(config)
        controller.launch_detected = True
        controller.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 50.0,  # High roll rate
            "vertical_acceleration_ms2": 50.0,
        }

        action = controller.step(obs, info)

        # D term: 50 deg/s * 0.1 = 5.0 deg deflection
        # Normalized: 5.0 / 30.0 = 0.167
        assert action[0] > 0
        assert -1.0 <= action[0] <= 1.0

    def test_step_integral_control(self):
        """Test integral control accumulation."""
        config = PIDConfig(Cprop=0.0, Cint=0.1, Cderiv=0.0)
        controller = PIDController(config)
        controller.launch_detected = True
        controller.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(10.0),
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
        }

        # First step
        action1 = controller.step(obs, info, dt=0.01)
        # Integral: 10 deg * 0.01 s = 0.1 deg*s
        assert controller.integ_error != 0.0

        # Second step accumulates more
        action2 = controller.step(obs, info, dt=0.01)
        assert abs(action2[0]) > abs(action1[0])

    def test_step_angle_normalization(self):
        """Test angle normalization to [-180, 180]."""
        config = PIDConfig(Cprop=0.1, Cint=0.0, Cderiv=0.0)
        controller = PIDController(config)
        controller.launch_detected = True
        controller.target_orient = 0.0

        obs = np.zeros(10)
        # Roll angle of 350 degrees should be treated as -10 degrees
        info = {
            "roll_angle_rad": np.radians(350.0),
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
        }

        action = controller.step(obs, info)

        # Error should be -10 degrees, not 350 degrees
        assert action[0] < 0

    def test_step_roll_rate_clamping(self):
        """Test roll rate clamping."""
        config = PIDConfig(Cprop=0.0, Cint=0.0, Cderiv=0.1, max_roll_rate=50.0)
        controller = PIDController(config)
        controller.launch_detected = True

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 200.0,  # Way above max
            "vertical_acceleration_ms2": 50.0,
        }

        action = controller.step(obs, info)

        # Should be clamped to 50 deg/s
        # D term: 50 * 0.1 = 5.0 deg
        expected = 5.0 / config.max_deflection
        assert abs(action[0] - expected) < 0.01

    def test_step_output_clamping(self):
        """Test output clamping to [-1, 1]."""
        config = PIDConfig(Cprop=1.0, Cint=0.0, Cderiv=0.0)  # High gain
        controller = PIDController(config)
        controller.launch_detected = True
        controller.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(180.0),  # Large error
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
        }

        action = controller.step(obs, info)

        assert action[0] == 1.0 or action[0] == -1.0

    def test_anti_windup(self):
        """Test integral anti-windup."""
        config = PIDConfig(Cprop=0.0, Cint=0.1, Cderiv=0.0, max_deflection=10.0)
        controller = PIDController(config)
        controller.launch_detected = True
        controller.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(90.0),  # Large error
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
        }

        # Run many steps to accumulate integral
        for _ in range(1000):
            controller.step(obs, info, dt=0.01)

        # Integral should be clamped
        max_integ = config.max_deflection / config.Cint
        assert abs(controller.integ_error) <= max_integ + 1e-6


class TestEpisodeResult:
    """Tests for EpisodeResult dataclass."""

    def test_creation(self):
        """Test creating episode result."""
        result = EpisodeResult(
            max_altitude=100.0,
            mean_spin_rate=15.0,
            final_spin_rate=5.0,
            total_reward=50.0,
            episode_length=200,
            times=np.array([0.0, 0.01, 0.02]),
            altitudes=np.array([0.0, 1.0, 2.0]),
            roll_rates=np.array([10.0, 5.0, 2.0]),
            actions=np.array([0.1, 0.05, 0.02]),
        )

        assert result.max_altitude == 100.0
        assert result.mean_spin_rate == 15.0
        assert result.final_spin_rate == 5.0
        assert result.total_reward == 50.0
        assert result.episode_length == 200
        assert len(result.times) == 3
        assert len(result.altitudes) == 3
        assert len(result.roll_rates) == 3
        assert len(result.actions) == 3


class TestCreateEnv:
    """Tests for create_env function."""

    def test_create_env_from_config(self, sample_training_config_yaml):
        """Test environment creation from config file."""
        from rocket_config import load_config

        config = load_config(sample_training_config_yaml)
        env = create_env(config)

        assert env is not None
        assert hasattr(env, "reset")
        assert hasattr(env, "step")

        env.close()


class TestRunEpisode:
    """Tests for run_episode function."""

    def test_run_episode(self, sample_training_config_yaml):
        """Test running a single episode."""
        from rocket_config import load_config

        config = load_config(sample_training_config_yaml)
        env = create_env(config)
        controller = PIDController()

        result = run_episode(env, controller, dt=0.02)

        assert isinstance(result, EpisodeResult)
        assert result.max_altitude >= 0
        assert result.episode_length > 0
        assert len(result.times) == result.episode_length
        assert len(result.altitudes) == result.episode_length
        assert len(result.roll_rates) == result.episode_length
        assert len(result.actions) == result.episode_length

        env.close()


class TestEvaluatePID:
    """Tests for evaluate_pid function."""

    def test_evaluate_pid(self, sample_training_config_yaml, capsys):
        """Test PID evaluation over multiple episodes."""
        pid_config = PIDConfig(Cprop=0.02, Cint=0.001, Cderiv=0.05)
        results = evaluate_pid(sample_training_config_yaml, pid_config, n_episodes=5)

        assert len(results) == 5
        assert all(isinstance(r, EpisodeResult) for r in results)


class TestPrintSummary:
    """Tests for print_summary function."""

    def test_print_summary(self, capsys):
        """Test summary printing."""
        results = [
            EpisodeResult(
                max_altitude=100.0,
                mean_spin_rate=20.0,
                final_spin_rate=5.0,
                total_reward=50.0,
                episode_length=200,
                times=np.array([0.0]),
                altitudes=np.array([100.0]),
                roll_rates=np.array([20.0]),
                actions=np.array([0.1]),
            ),
            EpisodeResult(
                max_altitude=110.0,
                mean_spin_rate=25.0,
                final_spin_rate=8.0,
                total_reward=60.0,
                episode_length=210,
                times=np.array([0.0]),
                altitudes=np.array([110.0]),
                roll_rates=np.array([25.0]),
                actions=np.array([0.15]),
            ),
        ]

        print_summary(results, "Test")

        captured = capsys.readouterr()
        assert "Test Controller Results" in captured.out
        assert "Max Altitude" in captured.out
        assert "Mean Spin Rate" in captured.out
        assert "Success" in captured.out


class TestPlotComparison:
    """Tests for plot_comparison function."""

    def test_plot_comparison_no_display(self):
        """Test plot generation without display."""
        import matplotlib

        matplotlib.use("Agg")  # Non-interactive backend

        from pid_controller import plot_comparison

        results = [
            EpisodeResult(
                max_altitude=100.0,
                mean_spin_rate=20.0,
                final_spin_rate=5.0,
                total_reward=50.0,
                episode_length=100,
                times=np.linspace(0, 1, 100),
                altitudes=np.linspace(0, 100, 100),
                roll_rates=np.sin(np.linspace(0, 10, 100)) * 20,
                actions=np.sin(np.linspace(0, 10, 100)) * 0.5,
            ),
        ]

        with patch("matplotlib.pyplot.show"):
            plot_comparison(results)

    def test_plot_comparison_with_rl_results(self, tmp_path):
        """Test plot with RL comparison."""
        import matplotlib

        matplotlib.use("Agg")

        from pid_controller import plot_comparison

        pid_results = [
            EpisodeResult(
                max_altitude=100.0,
                mean_spin_rate=20.0,
                final_spin_rate=5.0,
                total_reward=50.0,
                episode_length=100,
                times=np.linspace(0, 1, 100),
                altitudes=np.linspace(0, 100, 100),
                roll_rates=np.sin(np.linspace(0, 10, 100)) * 20,
                actions=np.sin(np.linspace(0, 10, 100)) * 0.5,
            ),
        ]

        rl_results = [
            EpisodeResult(
                max_altitude=110.0,
                mean_spin_rate=15.0,
                final_spin_rate=3.0,
                total_reward=70.0,
                episode_length=100,
                times=np.linspace(0, 1, 100),
                altitudes=np.linspace(0, 110, 100),
                roll_rates=np.sin(np.linspace(0, 10, 100)) * 15,
                actions=np.sin(np.linspace(0, 10, 100)) * 0.3,
            ),
        ]

        save_path = str(tmp_path / "comparison.png")
        with patch("matplotlib.pyplot.show"):
            plot_comparison(pid_results, rl_results, save_path)

        # Check file was created
        assert (tmp_path / "comparison.png").exists()


class TestMainFunction:
    """Tests for main function argument parsing."""

    def test_main_argparse(self):
        """Test argument parsing setup."""
        import argparse

        # Just verify the function exists and has proper structure
        from pid_controller import main

        assert callable(main)
