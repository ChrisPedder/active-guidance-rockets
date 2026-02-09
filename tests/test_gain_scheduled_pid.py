"""
Tests for GainScheduledPIDController.

Verifies that gain scheduling correctly scales Kp and Kd with dynamic
pressure while leaving Ki unscaled, and that the controller produces
physically correct behavior across the flight envelope.
"""

import pytest
import numpy as np

from controllers.pid_controller import (
    GainScheduledPIDController,
    PIDController,
    PIDConfig,
)


class TestGainScaling:
    """Tests for the _gain_scale method."""

    def test_scale_at_reference_q_is_unity(self):
        """At q_ref, gain scale should be 1.0."""
        config = PIDConfig(q_ref=500.0)
        ctrl = GainScheduledPIDController(config)
        scale = ctrl._gain_scale(500.0)
        assert abs(scale - 1.0) < 0.01, f"Scale at q_ref should be ~1.0, got {scale}"

    def test_scale_increases_at_low_q(self):
        """At low dynamic pressure, gains should be scaled up."""
        config = PIDConfig(q_ref=500.0)
        ctrl = GainScheduledPIDController(config)
        scale_low = ctrl._gain_scale(100.0)
        assert scale_low > 1.0, (
            f"Scale at q=100 should be > 1.0 (got {scale_low}), "
            "controller needs higher gains when control effectiveness is low"
        )

    def test_scale_decreases_at_high_q(self):
        """At high dynamic pressure, gains should be scaled down."""
        config = PIDConfig(q_ref=500.0)
        ctrl = GainScheduledPIDController(config)
        scale_high = ctrl._gain_scale(1000.0)
        assert scale_high < 1.0, (
            f"Scale at q=1000 should be < 1.0 (got {scale_high}), "
            "controller needs lower gains when control effectiveness is high"
        )

    def test_scale_clamped_at_minimum(self):
        """Scale should not go below 0.5 even at very high q."""
        ctrl = GainScheduledPIDController(PIDConfig(q_ref=500.0))
        scale = ctrl._gain_scale(5000.0)
        assert scale >= 0.5, f"Scale should be clamped >= 0.5, got {scale}"

    def test_scale_clamped_at_maximum(self):
        """Scale should not exceed 5.0 even at very low q."""
        ctrl = GainScheduledPIDController(PIDConfig(q_ref=500.0))
        scale = ctrl._gain_scale(1.0)
        assert scale <= 5.0, f"Scale should be clamped <= 5.0, got {scale}"

    def test_scale_at_zero_q(self):
        """At q=0 (no airspeed), scale should be clamped to max."""
        ctrl = GainScheduledPIDController(PIDConfig(q_ref=500.0))
        scale = ctrl._gain_scale(0.0)
        assert scale == 5.0, f"Scale at q=0 should be 5.0 (max clamp), got {scale}"

    def test_scale_monotonically_decreasing(self):
        """Scale should decrease monotonically with increasing q (within clamp range)."""
        ctrl = GainScheduledPIDController(PIDConfig(q_ref=500.0))
        q_values = [50, 100, 200, 300, 500, 700, 1000]
        scales = [ctrl._gain_scale(q) for q in q_values]
        for i in range(len(scales) - 1):
            assert scales[i] >= scales[i + 1], (
                f"Scale should decrease: at q={q_values[i]} got {scales[i]}, "
                f"at q={q_values[i+1]} got {scales[i+1]}"
            )


class TestGainScheduledPIDInterface:
    """Tests that GainScheduledPIDController has the same interface as PIDController."""

    def test_has_reset_method(self):
        ctrl = GainScheduledPIDController()
        assert hasattr(ctrl, "reset") and callable(ctrl.reset)

    def test_has_step_method(self):
        ctrl = GainScheduledPIDController()
        assert hasattr(ctrl, "step") and callable(ctrl.step)

    def test_default_config_uses_optimized_gains(self):
        ctrl = GainScheduledPIDController()
        assert ctrl.config.Cprop == 0.005208
        assert ctrl.config.Cint == 0.000324
        assert ctrl.config.Cderiv == 0.016524

    def test_reset_clears_state(self):
        ctrl = GainScheduledPIDController()
        ctrl.launch_detected = True
        ctrl.integ_error = 10.0
        ctrl.launch_orient = 45.0

        ctrl.reset()

        assert ctrl.launch_detected is False
        assert ctrl.integ_error == 0.0
        assert ctrl.launch_orient == 0.0
        assert ctrl.target_orient == 0.0


class TestGainScheduledPIDControl:
    """Tests for gain-scheduled PID control behavior."""

    def test_zero_action_before_launch(self):
        """Should return zero before launch detection (ground-truth mode)."""
        ctrl = GainScheduledPIDController()
        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.1,
            "roll_rate_deg_s": 5.0,
            "vertical_acceleration_ms2": 5.0,  # Below threshold
            "dynamic_pressure_Pa": 0.0,
        }
        action = ctrl.step(obs, info)
        assert action[0] == 0.0
        assert ctrl.launch_detected is False

    def test_launch_detection(self):
        """Should detect launch when acceleration exceeds threshold."""
        ctrl = GainScheduledPIDController()
        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.1,
            "roll_rate_deg_s": 5.0,
            "vertical_acceleration_ms2": 25.0,
            "dynamic_pressure_Pa": 200.0,
        }
        ctrl.step(obs, info)
        assert ctrl.launch_detected is True
        assert ctrl.launch_orient == np.degrees(0.1)

    def test_produces_corrective_action(self):
        """Should produce action opposing the spin."""
        ctrl = GainScheduledPIDController()
        ctrl.launch_detected = True
        ctrl.target_orient = 0.0
        obs = np.zeros(10)
        obs[5] = 500.0  # dynamic pressure at reference
        info = {
            "roll_angle_rad": np.radians(5.0),
            "roll_rate_deg_s": 15.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 500.0,
        }
        action = ctrl.step(obs, info, dt=0.01)
        # PID negates output, so positive spin -> negative action
        assert action[0] < 0
        assert -1.0 <= action[0] <= 1.0

    def test_action_clamped_to_range(self):
        """Output should always be in [-1, 1]."""
        config = PIDConfig(Cprop=1.0, Cint=0.0, Cderiv=0.0)
        ctrl = GainScheduledPIDController(config)
        ctrl.launch_detected = True
        ctrl.target_orient = 0.0
        obs = np.zeros(10)
        obs[5] = 500.0
        info = {
            "roll_angle_rad": np.radians(180.0),
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 500.0,
        }
        action = ctrl.step(obs, info)
        assert action[0] == 1.0 or action[0] == -1.0

    def test_integral_not_scaled_by_q(self):
        """Ki term should NOT be affected by gain scheduling."""
        config = PIDConfig(Cprop=0.0, Cint=0.1, Cderiv=0.0)

        # Run at low q
        ctrl_low = GainScheduledPIDController(config)
        ctrl_low.launch_detected = True
        ctrl_low.target_orient = 0.0
        obs = np.zeros(10)
        info_low = {
            "roll_angle_rad": np.radians(10.0),
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 100.0,  # low q
        }
        for _ in range(10):
            ctrl_low.step(obs, info_low, dt=0.01)

        # Run at high q
        ctrl_high = GainScheduledPIDController(config)
        ctrl_high.launch_detected = True
        ctrl_high.target_orient = 0.0
        info_high = {
            "roll_angle_rad": np.radians(10.0),
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 1000.0,  # high q
        }
        for _ in range(10):
            ctrl_high.step(obs, info_high, dt=0.01)

        # Integral accumulation should be identical (same error, same dt)
        assert abs(ctrl_low.integ_error - ctrl_high.integ_error) < 1e-6, (
            f"Integral should not depend on q: "
            f"low_q={ctrl_low.integ_error}, high_q={ctrl_high.integ_error}"
        )

    def test_kd_scales_with_q(self):
        """Kd term should produce different output at different q values."""
        config = PIDConfig(Cprop=0.0, Cint=0.0, Cderiv=0.1)

        # At low q
        ctrl_low = GainScheduledPIDController(config)
        ctrl_low.launch_detected = True
        ctrl_low.target_orient = 0.0
        obs = np.zeros(10)
        info_low = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 30.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 100.0,
        }
        action_low = ctrl_low.step(obs, info_low)

        # At high q
        ctrl_high = GainScheduledPIDController(config)
        ctrl_high.launch_detected = True
        ctrl_high.target_orient = 0.0
        info_high = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 30.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 1000.0,
        }
        action_high = ctrl_high.step(obs, info_high)

        # Low q should produce LARGER action (gains scaled up)
        assert abs(action_low[0]) > abs(action_high[0]), (
            f"Action at low q ({abs(action_low[0]):.4f}) should be larger "
            f"than at high q ({abs(action_high[0]):.4f}) due to gain scheduling"
        )


class TestGainScheduledVsFixedPID:
    """Tests comparing gain-scheduled and fixed PID at reference q."""

    def test_matches_fixed_pid_at_reference_q(self):
        """At q_ref, gain-scheduled PID should produce same output as fixed PID."""
        config = PIDConfig(q_ref=500.0)
        gs_ctrl = GainScheduledPIDController(config)
        fixed_ctrl = PIDController(config)

        # Both start launched at same target
        gs_ctrl.launch_detected = True
        gs_ctrl.target_orient = 0.0
        fixed_ctrl.launch_detected = True
        fixed_ctrl.target_orient = 0.0

        obs = np.zeros(10)
        obs[5] = 500.0  # q at reference

        info = {
            "roll_angle_rad": np.radians(10.0),
            "roll_rate_deg_s": 20.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 500.0,
        }

        gs_action = gs_ctrl.step(obs, info, dt=0.01)
        fixed_action = fixed_ctrl.step(obs, info, dt=0.01)

        assert abs(gs_action[0] - fixed_action[0]) < 0.01, (
            f"At q_ref, GS-PID action ({gs_action[0]:.4f}) should match "
            f"fixed PID ({fixed_action[0]:.4f})"
        )

    def test_produces_more_action_at_low_q(self):
        """At low q, GS-PID should command more deflection than fixed PID."""
        config = PIDConfig(Cprop=0.0, Cint=0.0, Cderiv=0.1, q_ref=500.0)
        gs_ctrl = GainScheduledPIDController(config)
        fixed_ctrl = PIDController(config)

        gs_ctrl.launch_detected = True
        gs_ctrl.target_orient = 0.0
        fixed_ctrl.launch_detected = True
        fixed_ctrl.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 20.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 100.0,  # Well below q_ref
        }

        gs_action = gs_ctrl.step(obs, info)
        fixed_action = fixed_ctrl.step(obs, info)

        assert abs(gs_action[0]) > abs(fixed_action[0]), (
            f"At low q, GS-PID ({abs(gs_action[0]):.4f}) should command more "
            f"than fixed PID ({abs(fixed_action[0]):.4f})"
        )

    def test_produces_less_action_at_high_q(self):
        """At high q, GS-PID should command less deflection than fixed PID."""
        config = PIDConfig(Cprop=0.0, Cint=0.0, Cderiv=0.1, q_ref=500.0)
        gs_ctrl = GainScheduledPIDController(config)
        fixed_ctrl = PIDController(config)

        gs_ctrl.launch_detected = True
        gs_ctrl.target_orient = 0.0
        fixed_ctrl.launch_detected = True
        fixed_ctrl.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 20.0,
            "vertical_acceleration_ms2": 50.0,
            "dynamic_pressure_Pa": 1000.0,  # Well above q_ref
        }

        gs_action = gs_ctrl.step(obs, info)
        fixed_action = fixed_ctrl.step(obs, info)

        assert abs(gs_action[0]) < abs(fixed_action[0]), (
            f"At high q, GS-PID ({abs(gs_action[0]):.4f}) should command less "
            f"than fixed PID ({abs(fixed_action[0]):.4f})"
        )


class TestGainScheduledPIDObservationMode:
    """Tests for observation-based mode (reading from obs array instead of info)."""

    def test_reads_q_from_obs_index_5(self):
        """In observation mode, dynamic pressure should be read from obs[5]."""
        config = PIDConfig(Cprop=0.0, Cint=0.0, Cderiv=0.1, q_ref=500.0)

        # Obs mode at low q
        ctrl_low = GainScheduledPIDController(config, use_observations=True)
        obs_low = np.zeros(10, dtype=np.float32)
        obs_low[2] = 0.0  # roll angle
        obs_low[3] = 0.5  # roll rate rad/s (~28.6 deg/s)
        obs_low[5] = 100.0  # low q
        action_low = ctrl_low.step(obs_low, {})

        # Obs mode at high q
        ctrl_high = GainScheduledPIDController(config, use_observations=True)
        obs_high = np.zeros(10, dtype=np.float32)
        obs_high[2] = 0.0
        obs_high[3] = 0.5
        obs_high[5] = 1000.0  # high q
        action_high = ctrl_high.step(obs_high, {})

        # Low q should produce larger action (gains scaled up)
        assert abs(action_low[0]) > abs(action_high[0]), (
            f"Obs mode: action at low q ({abs(action_low[0]):.4f}) should be "
            f"larger than at high q ({abs(action_high[0]):.4f})"
        )

    def test_observation_mode_launches_immediately(self):
        """In observation mode, launch should be detected immediately."""
        ctrl = GainScheduledPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.1  # some roll angle
        obs[3] = 0.0
        obs[5] = 500.0
        ctrl.step(obs, {})
        assert ctrl.launch_detected is True
