"""
Tests for IMU noise robustness across all controller types.

Verifies that:
1. IMU noise degrades performance gracefully (within 20% of ground-truth)
2. Controllers remain stable even with elevated noise levels

Note: Basic observation mode tests (launch detection, reading from obs array)
are in the individual controller test files (test_pid_controller.py,
test_gain_scheduled_pid.py, test_adrc_controller.py).
"""

import pytest
import numpy as np

from controllers.pid_controller import (
    PIDController,
    GainScheduledPIDController,
    PIDConfig,
)
from controllers.adrc_controller import ADRCController, ADRCConfig

# --- Helpers ---


def make_obs(roll_angle=0.0, roll_rate=0.0, q=500.0):
    """Create a standard observation array."""
    obs = np.zeros(10, dtype=np.float32)
    obs[2] = roll_angle
    obs[3] = roll_rate
    obs[5] = q
    return obs


def make_info(roll_angle_rad=0.0, roll_rate_deg_s=0.0, accel=50.0, q=500.0):
    """Create a standard info dict."""
    return {
        "roll_angle_rad": roll_angle_rad,
        "roll_rate_deg_s": roll_rate_deg_s,
        "vertical_acceleration_ms2": accel,
        "dynamic_pressure_Pa": q,
    }


class TestIMUNoiseRobustness:
    """Test that controllers are robust to realistic IMU noise levels.

    Runs a closed-loop simulation comparing ground-truth vs noisy observations
    for each controller type, verifying < 20% degradation.
    """

    def _run_simulation(self, ctrl, b0, dt, n_steps, noise_std=0.0):
        """Run a simple closed-loop simulation.

        Returns list of absolute spin rates (deg/s) after initial transient.
        """
        roll_angle = 0.0
        roll_rate = np.radians(30.0)  # 30 deg/s initial spin
        spin_rates = []

        for step in range(n_steps):
            obs = make_obs(
                roll_angle=roll_angle,
                roll_rate=roll_rate,
                q=500.0,
            )
            if noise_std > 0:
                obs[3] += np.random.normal(0, noise_std)

            info = make_info(
                roll_angle_rad=roll_angle,
                roll_rate_deg_s=np.degrees(roll_rate),
                q=500.0,
            )
            action = ctrl.step(obs, info, dt)

            alpha = b0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

            # Collect after initial transient
            if step > 50:
                spin_rates.append(abs(np.degrees(roll_rate)))

        return spin_rates

    def test_pid_imu_within_20pct(self):
        """PID with noisy obs should be within 20% of ground-truth."""
        b0 = 100.0
        dt = 0.01
        np.random.seed(42)

        # Ground truth
        ctrl_gt = PIDController(use_observations=True)
        rates_gt = self._run_simulation(ctrl_gt, b0, dt, 300, noise_std=0.0)

        # With IMU noise (0.003 rad/s ≈ 0.17 deg/s RMS)
        ctrl_imu = PIDController(use_observations=True)
        rates_imu = self._run_simulation(ctrl_imu, b0, dt, 300, noise_std=0.003)

        mean_gt = np.mean(rates_gt)
        mean_imu = np.mean(rates_imu)

        # Both should settle; IMU version should be within 20% or close
        # (when both are near zero, use absolute tolerance)
        if mean_gt > 1.0:
            assert mean_imu < mean_gt * 1.2, (
                f"PID IMU ({mean_imu:.2f}) should be within 20% of "
                f"ground-truth ({mean_gt:.2f})"
            )
        else:
            assert (
                mean_imu < 2.0
            ), f"PID IMU ({mean_imu:.2f}) should remain low when GT is near zero"

    def test_gs_pid_imu_within_20pct(self):
        """GS-PID with noisy obs should be within 20% of ground-truth."""
        b0 = 100.0
        dt = 0.01
        np.random.seed(42)

        ctrl_gt = GainScheduledPIDController(use_observations=True)
        rates_gt = self._run_simulation(ctrl_gt, b0, dt, 300, noise_std=0.0)

        ctrl_imu = GainScheduledPIDController(use_observations=True)
        rates_imu = self._run_simulation(ctrl_imu, b0, dt, 300, noise_std=0.003)

        mean_gt = np.mean(rates_gt)
        mean_imu = np.mean(rates_imu)

        if mean_gt > 1.0:
            assert mean_imu < mean_gt * 1.2
        else:
            assert mean_imu < 2.0

    def test_adrc_imu_within_20pct(self):
        """ADRC with noisy obs should be within 20% of ground-truth."""
        b0 = 100.0
        dt = 0.01
        np.random.seed(42)

        config = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, use_observations=True)
        ctrl_gt = ADRCController(config)
        rates_gt = self._run_simulation(ctrl_gt, b0, dt, 300, noise_std=0.0)

        config2 = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=b0, use_observations=True)
        ctrl_imu = ADRCController(config2)
        rates_imu = self._run_simulation(ctrl_imu, b0, dt, 300, noise_std=0.003)

        mean_gt = np.mean(rates_gt)
        mean_imu = np.mean(rates_imu)

        if mean_gt > 1.0:
            assert mean_imu < mean_gt * 1.2
        else:
            assert mean_imu < 2.0


class TestIMUNoiseDoesNotCauseInstability:
    """Test that even with higher noise levels, controllers don't diverge."""

    @pytest.mark.parametrize("noise_std", [0.003, 0.01, 0.03])
    def test_pid_stable_under_noise(self, noise_std):
        """PID should remain stable even with elevated noise."""
        ctrl = PIDController(use_observations=True)
        np.random.seed(42)

        roll_angle = 0.0
        roll_rate = np.radians(30.0)
        b0 = 100.0
        dt = 0.01

        for step in range(500):
            obs = make_obs(roll_angle=roll_angle, roll_rate=roll_rate, q=500.0)
            obs[3] += np.random.normal(0, noise_std)
            action = ctrl.step(obs, {}, dt)
            alpha = b0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        final_rate = abs(np.degrees(roll_rate))
        assert final_rate < 50.0, (
            f"PID should not diverge with noise_std={noise_std}: "
            f"final rate = {final_rate:.1f} deg/s"
        )

    @pytest.mark.parametrize("noise_std", [0.003, 0.01, 0.03])
    def test_adrc_stable_under_noise(self, noise_std):
        """ADRC should remain stable even with elevated noise."""
        config = ADRCConfig(omega_c=15.0, omega_o=50.0, b0=100.0, use_observations=True)
        ctrl = ADRCController(config)
        np.random.seed(42)

        roll_angle = 0.0
        roll_rate = np.radians(30.0)
        dt = 0.01

        for step in range(500):
            obs = make_obs(roll_angle=roll_angle, roll_rate=roll_rate, q=500.0)
            obs[3] += np.random.normal(0, noise_std)
            action = ctrl.step(obs, {}, dt)
            alpha = 100.0 * action[0]
            roll_rate += alpha * dt
            roll_angle += roll_rate * dt

        final_rate = abs(np.degrees(roll_rate))
        assert final_rate < 50.0, (
            f"ADRC should not diverge with noise_std={noise_std}: "
            f"final rate = {final_rate:.1f} deg/s"
        )
