"""
Tests for LeadCompensatedGSPIDController.

Covers the uncovered lines 309-432 in pid_controller.py.
"""

import numpy as np
import pytest

from controllers.pid_controller import (
    PIDConfig,
    LeadCompensatedGSPIDController,
)


class TestLeadCompensatedGSPIDController:
    """Test LeadCompensatedGSPIDController."""

    @pytest.fixture
    def config(self):
        return PIDConfig(
            Cprop=0.02,
            Cint=0.001,
            Cderiv=0.015,
            q_ref=500.0,
            max_deflection=30.0,
        )

    def test_init_default(self):
        ctrl = LeadCompensatedGSPIDController()
        assert ctrl.config is not None
        assert not ctrl.launch_detected
        assert ctrl._lead_b0 != 0
        assert ctrl._lead_b1 != 0

    def test_init_custom(self, config):
        ctrl = LeadCompensatedGSPIDController(
            config=config,
            use_observations=False,
            lead_zero=5.0,
            lead_pole=50.0,
        )
        assert ctrl.config.Cprop == 0.02
        assert ctrl._ref_effectiveness > 0

    def test_reset(self, config):
        ctrl = LeadCompensatedGSPIDController(config=config)
        ctrl.launch_detected = True
        ctrl.integ_error = 10.0
        ctrl._lead_x_prev = 1.0
        ctrl._lead_y_prev = 1.0

        ctrl.reset()
        assert not ctrl.launch_detected
        assert ctrl.integ_error == 0.0
        assert ctrl._lead_x_prev == 0.0
        assert ctrl._lead_y_prev == 0.0

    def test_gain_scale_nominal(self, config):
        ctrl = LeadCompensatedGSPIDController(config=config)
        # At q_ref, scale should be ~1.0
        scale = ctrl._gain_scale(config.q_ref)
        assert 0.5 <= scale <= 2.0

    def test_gain_scale_low_q(self, config):
        ctrl = LeadCompensatedGSPIDController(config=config)
        scale = ctrl._gain_scale(0.001)
        assert scale == 5.0  # Capped at max

    def test_gain_scale_high_q(self, config):
        ctrl = LeadCompensatedGSPIDController(config=config)
        scale = ctrl._gain_scale(100000.0)
        assert scale >= 0.5  # Capped at min

    def test_lead_filter(self, config):
        ctrl = LeadCompensatedGSPIDController(config=config)
        # Zero input should give zero output
        y = ctrl._lead_filter(0.0)
        assert y == 0.0

        # Step input should produce non-zero output
        y = ctrl._lead_filter(1.0)
        assert y != 0.0

    def test_lead_filter_dc_gain(self, config):
        """DC gain should be normalized to ~1.0."""
        ctrl = LeadCompensatedGSPIDController(config=config)
        # Apply constant input for many steps
        for _ in range(1000):
            y = ctrl._lead_filter(1.0)
        # Output should converge to ~1.0 (DC gain normalized)
        assert abs(y - 1.0) < 0.1

    def test_step_before_launch(self, config):
        ctrl = LeadCompensatedGSPIDController(config=config, use_observations=False)
        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 0.0,
            "dynamic_pressure_Pa": 0.0,
            "vertical_acceleration_ms2": 0.0,  # Not launched yet
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)
        assert action[0] == 0.0  # No action before launch

    def test_step_after_launch(self, config):
        ctrl = LeadCompensatedGSPIDController(config=config, use_observations=False)
        obs = np.zeros(10)

        # Trigger launch
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 5.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,  # Above threshold
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected

        # Next step with some spin rate
        info["roll_rate_deg_s"] = 10.0
        info["roll_angle_rad"] = 0.1
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_step_with_observations(self, config):
        """Test using observation-based mode."""
        ctrl = LeadCompensatedGSPIDController(config=config, use_observations=True)
        obs = np.array(
            [50.0, 30.0, 0.1, np.radians(10.0), 0.0, 500.0, 1.0, 0.5, 0.0, 0.0]
        )
        info = {
            "roll_rate_deg_s": 10.0,
            "dynamic_pressure_Pa": 500.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        assert ctrl.launch_detected  # Should auto-detect in obs mode
        assert action.shape == (1,)
        assert -1.0 <= action[0] <= 1.0

    def test_integral_windup_clamping(self, config):
        """Integral error should be clamped."""
        ctrl = LeadCompensatedGSPIDController(config=config, use_observations=False)

        # Launch
        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 0.0,
            "dynamic_pressure_Pa": 500.0,
            "vertical_acceleration_ms2": 50.0,
        }
        ctrl.step(obs, info, dt=0.01)

        # Apply large error for many steps to hit integral limit
        for _ in range(1000):
            info["roll_angle_rad"] = np.radians(90)
            info["roll_rate_deg_s"] = 0.0
            ctrl.step(obs, info, dt=0.01)

        max_integ = config.max_deflection / (config.Cint + 1e-6)
        assert abs(ctrl.integ_error) <= max_integ + 1.0

    def test_full_episode_integration(self):
        """Run a full episode with LeadCompensatedGSPIDController."""
        from compare_controllers import create_env, run_controller_episode
        from rocket_config import load_config

        config = load_config("configs/estes_c6_sac_wind.yaml")
        env = create_env(config, wind_speed=1.0)
        pid_config = PIDConfig(Cprop=0.0203, Cint=0.0002, Cderiv=0.0118, q_ref=500.0)
        ctrl = LeadCompensatedGSPIDController(pid_config)
        metrics = run_controller_episode(env, ctrl, dt=0.01)
        assert metrics.mean_spin_rate >= 0
        assert metrics.episode_length > 0
        env.close()
