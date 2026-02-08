"""
Tests for the lead-compensated gain-scheduled PID controller.

Verifies:
1. Lead filter discretization (Tustin transform) is correct
2. Lead filter adds phase lead at the target frequency band
3. Lead filter DC gain normalization preserves steady-state behavior
4. Controller interface matches other controllers (reset/step)
5. Calm-condition performance is not degraded
6. Controller is stable with typical flight observations
"""

import numpy as np
import pytest

from pid_controller import (
    LeadCompensatedGSPIDController,
    GainScheduledPIDController,
    PIDConfig,
)


class TestLeadFilterCoefficients:
    """Verify the discrete lead filter coefficients are correct."""

    def test_default_coefficients_finite(self):
        ctrl = LeadCompensatedGSPIDController()
        assert np.isfinite(ctrl._lead_b0)
        assert np.isfinite(ctrl._lead_b1)
        assert np.isfinite(ctrl._lead_a1)

    def test_dc_gain_is_z_over_p(self):
        """DC gain of (s+z)/(s+p) is z/p."""
        z, p = 5.0, 50.0
        ctrl = LeadCompensatedGSPIDController(lead_zero=z, lead_pole=p)
        # DC gain of discrete filter: (b0 + b1) / (1 + a1)
        dc_gain = (ctrl._lead_b0 + ctrl._lead_b1) / (1.0 + ctrl._lead_a1)
        assert abs(dc_gain - z / p) < 1e-10

    def test_dc_normalization_makes_unit_dc(self):
        """After DC normalization, the effective DC gain should be 1.0."""
        ctrl = LeadCompensatedGSPIDController(lead_zero=5.0, lead_pole=50.0)
        # Feed constant signal through the filter; output should converge to input
        ctrl.reset()
        x_const = 10.0
        for _ in range(500):
            y = ctrl._lead_filter(x_const)
        assert abs(y - x_const) < 0.01, f"DC output {y} should converge to {x_const}"

    def test_stability_pole_inside_unit_circle(self):
        """The discrete pole should be inside the unit circle for stability."""
        ctrl = LeadCompensatedGSPIDController(lead_zero=5.0, lead_pole=50.0)
        # Pole of H(z) = (b0 + b1*z^-1) / (1 + a1*z^-1) is at z = -a1
        assert abs(ctrl._lead_a1) < 1.0, f"|a1| = {abs(ctrl._lead_a1)} should be < 1"

    def test_different_zero_pole_values(self):
        """Filter should work with various zero/pole placements."""
        for z, p in [(1.0, 10.0), (3.0, 30.0), (10.0, 100.0)]:
            ctrl = LeadCompensatedGSPIDController(lead_zero=z, lead_pole=p)
            assert abs(ctrl._lead_a1) < 1.0
            dc = (ctrl._lead_b0 + ctrl._lead_b1) / (1.0 + ctrl._lead_a1)
            assert abs(dc - z / p) < 1e-10


class TestLeadFilterFrequencyResponse:
    """Verify the lead filter adds phase lead at the target frequency."""

    @staticmethod
    def _measure_phase_at_freq(ctrl, freq_hz, n_cycles=20):
        """Measure the phase shift of the lead filter at a given frequency.

        Returns phase in degrees (positive = lead).
        """
        dt = 0.01
        omega = 2 * np.pi * freq_hz
        n_steps = int(n_cycles / freq_hz / dt)

        ctrl.reset()
        inputs = []
        outputs = []
        for i in range(n_steps):
            t = i * dt
            x = np.sin(omega * t)
            y = ctrl._lead_filter(x)
            inputs.append(x)
            outputs.append(y)

        # Use cross-correlation to find phase shift (last few cycles)
        n_last = int(3 / freq_hz / dt)  # last 3 cycles
        inp = np.array(inputs[-n_last:])
        out = np.array(outputs[-n_last:])

        # Find phase via cross-correlation peak
        corr = np.correlate(out, inp, mode="full")
        lags = np.arange(-len(inp) + 1, len(inp))
        peak_lag = lags[np.argmax(corr)]
        phase_deg = -peak_lag * dt * freq_hz * 360.0

        return phase_deg

    def test_phase_lead_at_target_frequency(self):
        """Should add positive phase lead around 2-3 Hz (12-19 rad/s)."""
        ctrl = LeadCompensatedGSPIDController(lead_zero=5.0, lead_pole=50.0)
        phase = self._measure_phase_at_freq(ctrl, 2.5)
        # Expect roughly 30-50 degrees of phase lead at 2.5 Hz
        assert phase > 10.0, f"Expected positive phase lead, got {phase:.1f} deg"

    def test_no_phase_lead_at_dc(self):
        """At very low frequency, phase lead should be negligible."""
        ctrl = LeadCompensatedGSPIDController(lead_zero=5.0, lead_pole=50.0)
        phase = self._measure_phase_at_freq(ctrl, 0.1, n_cycles=5)
        assert abs(phase) < 15.0, f"Expected near-zero phase at DC, got {phase:.1f} deg"

    def test_gain_boost_at_high_frequency(self):
        """Lead filter should boost gain at frequencies above the zero."""
        ctrl = LeadCompensatedGSPIDController(lead_zero=5.0, lead_pole=50.0)
        dt = 0.01

        # Measure amplitude response at 0.1 Hz (low) and 3 Hz (in lead band)
        def measure_amplitude(freq_hz):
            omega = 2 * np.pi * freq_hz
            ctrl.reset()
            outputs = []
            for i in range(int(10 / freq_hz / dt)):
                t = i * dt
                y = ctrl._lead_filter(np.sin(omega * t))
                outputs.append(y)
            # Peak amplitude of last few cycles
            n_last = int(2 / freq_hz / dt)
            return np.max(np.abs(outputs[-n_last:]))

        amp_low = measure_amplitude(0.1)
        amp_mid = measure_amplitude(3.0)
        # After DC normalization, low-freq amp ≈ 1.0, high-freq should be > 1
        assert (
            amp_mid > amp_low * 1.2
        ), f"Expected gain boost: amp@3Hz={amp_mid:.3f} > amp@0.1Hz={amp_low:.3f}"


class TestLeadCompensatedController:
    """Test the full controller interface and behavior."""

    def test_reset_clears_state(self):
        ctrl = LeadCompensatedGSPIDController()
        ctrl.launch_detected = True
        ctrl.integ_error = 999.0
        ctrl._lead_x_prev = 1.0
        ctrl._lead_y_prev = 1.0
        ctrl.reset()
        assert not ctrl.launch_detected
        assert ctrl.integ_error == 0.0
        assert ctrl._lead_x_prev == 0.0
        assert ctrl._lead_y_prev == 0.0

    def test_step_returns_correct_shape(self):
        ctrl = LeadCompensatedGSPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        info = {}
        action = ctrl.step(obs, info, dt=0.01)
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_step_output_in_range(self):
        ctrl = LeadCompensatedGSPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.5  # roll angle
        obs[3] = 1.0  # roll rate
        obs[5] = 500.0  # dynamic pressure
        info = {}
        for _ in range(100):
            action = ctrl.step(obs, info, dt=0.01)
            assert -1.0 <= action[0] <= 1.0

    def test_launch_detection_info_mode(self):
        """Ground-truth mode should detect launch via acceleration."""
        ctrl = LeadCompensatedGSPIDController(use_observations=False)
        obs = np.zeros(10)
        info = {
            "roll_angle_rad": 0.0,
            "roll_rate_deg_s": 0.0,
            "vertical_acceleration_ms2": 5.0,
            "dynamic_pressure_Pa": 0.0,
        }
        action = ctrl.step(obs, info)
        assert not ctrl.launch_detected
        assert action[0] == 0.0

        info["vertical_acceleration_ms2"] = 30.0
        action = ctrl.step(obs, info)
        assert ctrl.launch_detected

    def test_zero_spin_produces_zero_action(self):
        """With zero error and zero rate, action should be near zero."""
        ctrl = LeadCompensatedGSPIDController(use_observations=True)
        obs = np.zeros(10, dtype=np.float32)
        obs[5] = 500.0
        info = {}
        # First step sets launch orientation
        ctrl.step(obs, info)
        # Subsequent steps with same state
        for _ in range(50):
            action = ctrl.step(obs, info)
        assert abs(action[0]) < 0.01

    def test_matches_gs_pid_at_dc(self):
        """At zero spin (DC), lead GS-PID should behave similarly to GS-PID."""
        config = PIDConfig()
        gs_ctrl = GainScheduledPIDController(config, use_observations=True)
        lead_ctrl = LeadCompensatedGSPIDController(config, use_observations=True)

        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.1  # Small angle offset
        obs[5] = 500.0
        info = {}

        # After many steps with constant input, lead filter settles to DC
        gs_ctrl.step(obs, info)
        lead_ctrl.step(obs, info)
        for _ in range(200):
            gs_action = gs_ctrl.step(obs, info)
            lead_action = lead_ctrl.step(obs, info)

        # Should be similar (not identical due to filter transient effect on integrator)
        assert (
            abs(gs_action[0] - lead_action[0]) < 0.05
        ), f"GS-PID={gs_action[0]:.4f}, Lead={lead_action[0]:.4f}"


class TestClosedLoopStability:
    """Test that the lead compensator maintains stability in simulation."""

    def test_stable_with_simple_dynamics(self):
        """Simulate a simple 2nd-order roll model and verify stability.

        Uses a realistic b0 matching the Estes Alpha airframe (b0 ~ 130 rad/s^2
        per normalized action at q=500 Pa). The PID gains were tuned for this
        plant, so convergence is expected.
        """
        ctrl = LeadCompensatedGSPIDController(use_observations=True)

        # First step: launch detection at angle=0 (sets target_orient=0)
        obs = np.zeros(10, dtype=np.float32)
        obs[5] = 500.0
        ctrl.step(obs, {}, 0.01)

        # Now offset the roll angle — controller should drive it back to 0
        roll_angle = np.radians(10.0)
        roll_rate = 0.0
        dt = 0.01
        b0 = 130.0  # rad/s^2 per normalized action (realistic for Estes Alpha)
        damping = 2.0  # Aerodynamic damping coefficient

        for step in range(1000):  # 10 seconds
            obs = np.zeros(10, dtype=np.float32)
            obs[2] = roll_angle
            obs[3] = roll_rate
            obs[5] = 500.0
            info = {}

            action = ctrl.step(obs, info, dt)
            accel = b0 * action[0] - damping * roll_rate
            roll_rate += accel * dt
            roll_angle += roll_rate * dt

        # Should have reduced from 10 deg initial offset (stability check)
        final_angle_deg = abs(np.degrees(roll_angle))
        assert (
            final_angle_deg < 10.0
        ), f"Roll angle {final_angle_deg:.1f} deg should decrease from 10 deg initial"
        assert (
            abs(np.degrees(roll_rate)) < 60.0
        ), f"Roll rate {np.degrees(roll_rate):.1f} deg/s should be bounded"

    def test_stable_with_sinusoidal_disturbance(self):
        """Simulate with a sinusoidal disturbance and verify bounded response."""
        ctrl = LeadCompensatedGSPIDController(use_observations=True)

        roll_angle = 0.0
        roll_rate = 0.0
        dt = 0.01
        b0 = 100.0
        dist_amplitude = 10.0  # rad/s^2

        max_rate = 0.0
        for step in range(1000):  # 10 seconds
            t = step * dt
            obs = np.zeros(10, dtype=np.float32)
            obs[2] = roll_angle
            obs[3] = roll_rate
            obs[5] = 500.0
            info = {}

            action = ctrl.step(obs, info, dt)
            # Sinusoidal disturbance at ~2 Hz (typical spin frequency)
            disturbance = dist_amplitude * np.sin(2 * np.pi * 2.0 * t)
            accel = b0 * action[0] + disturbance
            roll_rate += accel * dt
            roll_angle += roll_rate * dt
            max_rate = max(max_rate, abs(roll_rate))

        # Roll rate should remain bounded
        assert (
            max_rate < 10.0
        ), f"Max roll rate {np.degrees(max_rate):.1f} deg/s is too large"


class TestCompareControllersIntegration:
    """Test that compare_controllers.py integration is correct."""

    def test_lead_controller_importable(self):
        from pid_controller import LeadCompensatedGSPIDController

        ctrl = LeadCompensatedGSPIDController()
        assert hasattr(ctrl, "step")
        assert hasattr(ctrl, "reset")

    def test_lead_flag_in_compare_source(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "--lead" in source

    def test_lead_color_defined(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "Lead GS-PID" in source
