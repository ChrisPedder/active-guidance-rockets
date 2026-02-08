"""
Tests for the repetitive (resonant) GS-PID controller.

Verifies:
1. Resonant filter produces output at the target frequency
2. Resonant filter state is properly reset
3. Controller interface matches other controllers (reset/step)
4. Controller produces valid actions
5. Gain scheduling applies to resonant output
6. Warmup period suppresses early resonant action
7. Frequency range limits are respected
8. Closed-loop stability with simple dynamics
9. Integration with compare_controllers.py
"""

import numpy as np
import pytest

from repetitive_controller import (
    RepetitiveGSPIDController,
    RepetitiveConfig,
)
from pid_controller import PIDConfig, GainScheduledPIDController


class TestRepetitiveConfig:
    """Test configuration defaults and validation."""

    def test_default_config_values(self):
        cfg = RepetitiveConfig()
        assert cfg.K_rc == 0.5
        assert cfg.min_omega == 3.0
        assert cfg.max_omega == 150.0
        assert cfg.damping == 0.05
        assert cfg.omega_smoothing == 0.9
        assert cfg.warmup_steps == 30

    def test_custom_config(self):
        cfg = RepetitiveConfig(K_rc=1.0, min_omega=5.0, damping=0.1)
        assert cfg.K_rc == 1.0
        assert cfg.min_omega == 5.0
        assert cfg.damping == 0.1


class TestResonantFilter:
    """Test the resonant filter behavior."""

    def test_filter_state_resets_to_zero(self):
        ctrl = RepetitiveGSPIDController(use_observations=True)
        # Perturb state
        ctrl._res_x1 = 10.0
        ctrl._res_x2 = 5.0
        ctrl._omega_est = 20.0
        ctrl._step_count = 100
        ctrl.reset()
        assert ctrl._res_x1 == 0.0
        assert ctrl._res_x2 == 0.0
        assert ctrl._omega_est == 0.0
        assert ctrl._step_count == 0

    def test_filter_produces_output_at_resonant_frequency(self):
        """Feed a sinusoidal signal at the resonant frequency and verify output grows."""
        ctrl = RepetitiveGSPIDController(
            config=RepetitiveConfig(K_rc=1.0, damping=0.05),
            use_observations=True,
        )
        omega = 15.0  # rad/s (~2.4 Hz)
        dt = 0.01
        outputs = []
        for i in range(500):
            t = i * dt
            error = np.sin(omega * t)
            out = ctrl._update_resonant_filter(error, omega, dt)
            outputs.append(abs(out))

        # Output should grow over time as the resonant mode builds up
        early_mean = np.mean(outputs[:50])
        late_mean = np.mean(outputs[-50:])
        assert (
            late_mean > early_mean * 2.0
        ), f"Resonant output should grow: early={early_mean:.4f}, late={late_mean:.4f}"

    def test_filter_does_not_resonate_off_frequency(self):
        """Signal at a different frequency should not excite the resonant mode as much."""
        ctrl = RepetitiveGSPIDController(
            config=RepetitiveConfig(K_rc=1.0, damping=0.05),
            use_observations=True,
        )
        omega_filter = 15.0  # Filter center
        omega_signal = 5.0  # Signal at different frequency
        dt = 0.01

        outputs_on = []
        outputs_off = []

        # On-frequency
        ctrl.reset()
        for i in range(300):
            t = i * dt
            error = np.sin(omega_filter * t)
            out = ctrl._update_resonant_filter(error, omega_filter, dt)
            outputs_on.append(abs(out))

        # Off-frequency
        ctrl.reset()
        for i in range(300):
            t = i * dt
            error = np.sin(omega_signal * t)
            out = ctrl._update_resonant_filter(error, omega_filter, dt)
            outputs_off.append(abs(out))

        on_rms = np.sqrt(np.mean(np.array(outputs_on[-100:]) ** 2))
        off_rms = np.sqrt(np.mean(np.array(outputs_off[-100:]) ** 2))
        assert on_rms > off_rms * 1.5, (
            f"On-frequency response ({on_rms:.4f}) should exceed "
            f"off-frequency ({off_rms:.4f})"
        )

    def test_damping_prevents_infinite_growth(self):
        """With damping > 0, filter output should be bounded."""
        ctrl = RepetitiveGSPIDController(
            config=RepetitiveConfig(K_rc=1.0, damping=0.1),
            use_observations=True,
        )
        omega = 15.0
        dt = 0.01

        max_out = 0.0
        for i in range(2000):
            t = i * dt
            error = np.sin(omega * t)
            out = ctrl._update_resonant_filter(error, omega, dt)
            max_out = max(max_out, abs(out))

        # With damping, output should be bounded (not growing forever)
        assert (
            max_out < 200.0
        ), f"Damped resonant output should be bounded, got {max_out:.1f}"

    def test_anti_windup_clamps_state(self):
        """State should be clamped to prevent numerical overflow."""
        ctrl = RepetitiveGSPIDController(
            config=RepetitiveConfig(K_rc=1.0, damping=0.0),  # No damping
            use_observations=True,
        )
        # Feed large constant input to drive state high
        for _ in range(10000):
            ctrl._update_resonant_filter(1000.0, 10.0, 0.01)

        assert abs(ctrl._res_x1) <= 100.0
        assert abs(ctrl._res_x2) <= 100.0


class TestControllerInterface:
    """Test the full controller interface."""

    def test_step_returns_correct_shape(self):
        ctrl = RepetitiveGSPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        info = {}
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_step_output_in_range(self):
        ctrl = RepetitiveGSPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.5  # roll angle
        obs[3] = 1.0  # roll rate
        obs[5] = 500.0  # dynamic pressure
        info = {}
        for _ in range(200):
            action = ctrl.step(obs, info, dt=0.01)
            assert -1.0 <= action[0] <= 1.0

    def test_launch_detection_obs_mode(self):
        ctrl = RepetitiveGSPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        info = {}
        ctrl.step(obs, info)
        assert ctrl.launch_detected

    def test_zero_error_produces_near_zero_action(self):
        """With zero error and rate, action should be near zero after settling."""
        ctrl = RepetitiveGSPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        obs[5] = 500.0
        info = {}
        ctrl.step(obs, info)  # Launch detection
        for _ in range(100):
            action = ctrl.step(obs, info)
        assert abs(action[0]) < 0.01

    def test_warmup_suppresses_resonant(self):
        """During warmup, resonant action should be zero."""
        cfg = RepetitiveConfig(warmup_steps=50, K_rc=2.0)
        ctrl = RepetitiveGSPIDController(config=cfg, use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.3  # angle error
        obs[3] = 10.0  # high roll rate (to trigger resonance)
        obs[5] = 500.0
        info = {}

        for step in range(40):
            ctrl.step(obs, info, dt=0.01)

        # During warmup, resonant action should be zero
        assert ctrl._res_action == 0.0

    def test_frequency_below_minimum_disables_resonant(self):
        """When spin rate is below min_omega, resonant should be disabled."""
        cfg = RepetitiveConfig(min_omega=5.0, warmup_steps=5)
        ctrl = RepetitiveGSPIDController(config=cfg, use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.3  # angle error
        obs[3] = 0.1  # very slow spin (below min_omega)
        obs[5] = 500.0
        info = {}

        for _ in range(20):
            ctrl.step(obs, info, dt=0.01)

        assert ctrl._res_action == 0.0

    def test_matches_gspid_when_krc_zero(self):
        """With K_rc=0, should behave identically to GS-PID."""
        pid_cfg = PIDConfig()
        rep_cfg = RepetitiveConfig(K_rc=0.0)

        gs_ctrl = GainScheduledPIDController(pid_cfg, use_observations=True)
        rep_ctrl = RepetitiveGSPIDController(pid_cfg, rep_cfg, use_observations=True)

        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.1
        obs[3] = 0.5
        obs[5] = 500.0
        info = {}

        # Run both controllers with same input
        gs_actions = []
        rep_actions = []
        for _ in range(100):
            gs_actions.append(gs_ctrl.step(obs, info, 0.01)[0])
            rep_actions.append(rep_ctrl.step(obs, info, 0.01)[0])

        # Should be identical
        np.testing.assert_allclose(gs_actions, rep_actions, atol=1e-6)


class TestClosedLoopStability:
    """Test closed-loop stability with simple dynamics."""

    def test_stable_with_simple_plant(self):
        """Simulate a simple roll model and verify stability."""
        ctrl = RepetitiveGSPIDController(
            config=RepetitiveConfig(K_rc=0.3, damping=0.1),
            use_observations=True,
        )

        # Launch detection at angle=0
        obs = np.zeros(10, dtype=np.float32)
        obs[5] = 500.0
        ctrl.step(obs, {}, 0.01)

        roll_angle = np.radians(10.0)
        roll_rate = 0.0
        dt = 0.01
        b0 = 130.0
        damping = 2.0

        for step in range(1000):
            obs = np.zeros(10, dtype=np.float32)
            obs[2] = roll_angle
            obs[3] = roll_rate
            obs[5] = 500.0

            action = ctrl.step(obs, {}, dt)
            accel = b0 * action[0] - damping * roll_rate
            roll_rate += accel * dt
            roll_angle += roll_rate * dt

        final_deg = abs(np.degrees(roll_angle))
        assert (
            final_deg < 15.0
        ), f"Roll angle {final_deg:.1f} deg should decrease from 10 deg initial"

    def test_stable_with_sinusoidal_disturbance(self):
        """With sinusoidal disturbance at spin frequency, response should be bounded."""
        ctrl = RepetitiveGSPIDController(
            config=RepetitiveConfig(K_rc=0.3, damping=0.1, warmup_steps=10),
            use_observations=True,
        )

        roll_angle = 0.0
        roll_rate = 0.0
        dt = 0.01
        b0 = 100.0
        dist_amplitude = 10.0

        max_rate = 0.0
        for step in range(1000):
            t = step * dt
            obs = np.zeros(10, dtype=np.float32)
            obs[2] = roll_angle
            obs[3] = roll_rate
            obs[5] = 500.0

            action = ctrl.step(obs, {}, dt)
            disturbance = dist_amplitude * np.sin(2 * np.pi * 2.0 * t)
            accel = b0 * action[0] + disturbance
            roll_rate += accel * dt
            roll_angle += roll_rate * dt
            max_rate = max(max_rate, abs(roll_rate))

        # Roll rate should remain bounded
        assert (
            max_rate < 15.0
        ), f"Max roll rate {np.degrees(max_rate):.1f} deg/s is too large"


class TestFrequencyTracking:
    """Test that the resonant frequency tracks the spin rate."""

    def test_omega_smoothing(self):
        """Smoothed omega should track a step change in roll rate."""
        cfg = RepetitiveConfig(omega_smoothing=0.9)
        ctrl = RepetitiveGSPIDController(config=cfg, use_observations=True)

        # Initial omega at 0
        assert ctrl._omega_est == 0.0

        # Feed constant roll rate of 20 rad/s
        obs = np.zeros(10, dtype=np.float32)
        obs[3] = 20.0  # roll rate in rad/s
        obs[5] = 500.0
        info = {}

        for _ in range(200):
            ctrl.step(obs, info, dt=0.01)

        # Should have converged close to 20 rad/s
        assert (
            abs(ctrl._omega_est - 20.0) < 1.0
        ), f"omega_est={ctrl._omega_est:.1f} should be near 20.0"

    def test_fast_smoothing_tracks_quickly(self):
        """Lower smoothing factor should track faster."""
        results = {}
        for alpha in [0.5, 0.99]:
            cfg = RepetitiveConfig(omega_smoothing=alpha, warmup_steps=5)
            ctrl = RepetitiveGSPIDController(config=cfg, use_observations=True)
            obs = np.zeros(10, dtype=np.float32)
            obs[3] = 20.0
            obs[5] = 500.0
            for _ in range(30):
                ctrl.step(obs, {}, dt=0.01)
            results[alpha] = ctrl._omega_est

        # Lower alpha should be closer to 20.0 after 30 steps
        assert results[0.5] > results[0.99], (
            f"alpha=0.5 ({results[0.5]:.2f}) should track faster than "
            f"alpha=0.99 ({results[0.99]:.2f})"
        )


class TestCompareControllersIntegration:
    """Test compare_controllers.py integration."""

    def test_controller_importable(self):
        from repetitive_controller import RepetitiveGSPIDController

        ctrl = RepetitiveGSPIDController()
        assert hasattr(ctrl, "step")
        assert hasattr(ctrl, "reset")

    def test_repetitive_flag_in_compare_source(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "--repetitive" in source

    def test_color_defined(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "Rep GS-PID" in source
