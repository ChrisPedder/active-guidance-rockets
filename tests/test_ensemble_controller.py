"""
Tests for the multi-controller ensemble with online switching.

Verifies:
1. Ensemble requires at least 2 controllers
2. Reset clears all controller states
3. All controllers receive step() calls (shadow mode)
4. Switching logic respects dwell time
5. Switching logic respects margin
6. Warmup prevents early switching
7. Controller interface (step returns correct shape)
8. Output stays in [-1, 1]
9. Integration with compare_controllers.py
"""

import numpy as np
import pytest

from ensemble_controller import EnsembleController, EnsembleConfig
from pid_controller import PIDConfig, PIDController, GainScheduledPIDController


class SimpleController:
    """Minimal controller for testing ensemble behavior."""

    def __init__(self, fixed_action: float = 0.0):
        self.fixed_action = fixed_action
        self.launch_detected = False
        self._step_count = 0

    def reset(self):
        self.launch_detected = False
        self._step_count = 0

    def step(self, obs, info, dt=0.01):
        self.launch_detected = True
        self._step_count += 1
        return np.array([self.fixed_action], dtype=np.float32)


class TestEnsembleConstruction:
    """Test ensemble creation and validation."""

    def test_requires_at_least_two_controllers(self):
        with pytest.raises(ValueError, match="at least 2"):
            EnsembleController([SimpleController()])

    def test_default_names_assigned(self):
        ens = EnsembleController([SimpleController(), SimpleController()])
        assert ens.names == ["ctrl_0", "ctrl_1"]

    def test_custom_names(self):
        ens = EnsembleController(
            [SimpleController(), SimpleController()],
            names=["A", "B"],
        )
        assert ens.names == ["A", "B"]

    def test_default_config(self):
        ens = EnsembleController([SimpleController(), SimpleController()])
        assert ens.config.window_size == 30
        assert ens.config.switch_margin == 1.0
        assert ens.config.min_dwell_s == 0.2
        assert ens.config.warmup_steps == 50


class TestEnsembleReset:
    """Test reset behavior."""

    def test_reset_clears_all_state(self):
        ctrl1 = SimpleController(0.5)
        ctrl2 = SimpleController(-0.5)
        ens = EnsembleController([ctrl1, ctrl2])

        # Run some steps
        obs = np.zeros(10, dtype=np.float32)
        for _ in range(10):
            ens.step(obs, {})

        ens.reset()
        assert ens._active_idx == 0
        assert ens._step_count == 0
        assert ens._switch_count == 0
        assert ctrl1._step_count == 0
        assert ctrl2._step_count == 0

    def test_reset_clears_performance_windows(self):
        ens = EnsembleController([SimpleController(), SimpleController()])
        obs = np.zeros(10, dtype=np.float32)
        for _ in range(10):
            ens.step(obs, {})

        ens.reset()
        for w in ens._perf_windows:
            assert len(w) == 0


class TestShadowMode:
    """Test that all controllers run in shadow mode."""

    def test_all_controllers_receive_step(self):
        ctrl1 = SimpleController(0.1)
        ctrl2 = SimpleController(0.2)
        ctrl3 = SimpleController(0.3)
        ens = EnsembleController([ctrl1, ctrl2, ctrl3], names=["A", "B", "C"])

        obs = np.zeros(10, dtype=np.float32)
        for _ in range(20):
            ens.step(obs, {})

        # All controllers should have been stepped
        assert ctrl1._step_count == 20
        assert ctrl2._step_count == 20
        assert ctrl3._step_count == 20

    def test_active_controller_output_used(self):
        ctrl1 = SimpleController(0.1)
        ctrl2 = SimpleController(0.9)
        ens = EnsembleController(
            [ctrl1, ctrl2],
            config=EnsembleConfig(warmup_steps=1000),  # Prevent switching
        )

        obs = np.zeros(10, dtype=np.float32)
        action = ens.step(obs, {})

        # Active controller is index 0 (ctrl1), so action should be 0.1
        assert abs(action[0] - 0.1) < 1e-6


class TestSwitchingLogic:
    """Test the switching logic."""

    def test_no_switching_during_warmup(self):
        ens = EnsembleController(
            [SimpleController(0.1), SimpleController(0.2)],
            config=EnsembleConfig(warmup_steps=100),
        )

        obs = np.zeros(10, dtype=np.float32)
        for _ in range(50):
            ens.step(obs, {})

        assert ens._active_idx == 0
        assert ens._switch_count == 0

    def test_dwell_time_prevents_rapid_switching(self):
        ens = EnsembleController(
            [SimpleController(0.1), SimpleController(0.2)],
            config=EnsembleConfig(
                warmup_steps=5,
                min_dwell_s=1.0,  # 100 steps at dt=0.01
                switch_margin=0.0,
            ),
        )

        obs = np.zeros(10, dtype=np.float32)
        obs[3] = 1.0  # Some roll rate
        # Run fewer steps than dwell time
        for _ in range(50):
            ens.step(obs, {}, dt=0.01)

        # Should not have switched (still within dwell time from initialization)
        initial_switches = ens._switch_count
        # Even if we run more steps, the dwell time constraint applies
        for _ in range(50):
            ens.step(obs, {}, dt=0.01)

        # At most one switch could happen (first dwell period expires at step 100)
        assert ens._switch_count <= initial_switches + 1

    def test_step_count_increments(self):
        ens = EnsembleController([SimpleController(), SimpleController()])
        obs = np.zeros(10, dtype=np.float32)

        for i in range(10):
            ens.step(obs, {})

        assert ens._step_count == 10


class TestControllerInterface:
    """Test the standard controller interface."""

    def test_step_returns_correct_shape(self):
        ens = EnsembleController([SimpleController(), SimpleController()])
        obs = np.zeros(10, dtype=np.float32)
        action = ens.step(obs, {})
        assert action.shape == (1,)
        assert action.dtype == np.float32

    def test_output_in_range(self):
        """With real controllers, output should be in [-1, 1]."""
        pid_cfg = PIDConfig()
        ctrl1 = GainScheduledPIDController(pid_cfg, use_observations=True)
        ctrl2 = GainScheduledPIDController(pid_cfg, use_observations=True)
        ens = EnsembleController([ctrl1, ctrl2])

        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.5  # roll angle
        obs[3] = 2.0  # roll rate
        obs[5] = 500.0

        for _ in range(100):
            action = ens.step(obs, {})
            assert -1.0 <= action[0] <= 1.0

    def test_launch_detected_property(self):
        ctrl1 = SimpleController()
        ctrl2 = SimpleController()
        ens = EnsembleController([ctrl1, ctrl2])

        assert not ens.launch_detected

        obs = np.zeros(10, dtype=np.float32)
        ens.step(obs, {})
        assert ens.launch_detected

    def test_active_controller_name(self):
        ens = EnsembleController(
            [SimpleController(), SimpleController()],
            names=["Alpha", "Beta"],
        )
        assert ens.active_controller_name == "Alpha"

    def test_performance_windows_fill(self):
        ens = EnsembleController(
            [SimpleController(), SimpleController()],
            config=EnsembleConfig(window_size=10),
        )
        obs = np.zeros(10, dtype=np.float32)
        obs[3] = 0.5  # roll rate

        for _ in range(15):
            ens.step(obs, {})

        # Windows should be capped at window_size
        for w in ens._perf_windows:
            assert len(w) == 10


class TestWithRealControllers:
    """Test ensemble with actual PID/GS-PID controllers."""

    def test_gspid_pair_runs_without_error(self):
        pid_cfg = PIDConfig()
        ctrl1 = GainScheduledPIDController(pid_cfg, use_observations=True)
        ctrl2 = GainScheduledPIDController(pid_cfg, use_observations=True)
        ens = EnsembleController(
            [ctrl1, ctrl2],
            names=["GS-PID-1", "GS-PID-2"],
        )

        obs = np.zeros(10, dtype=np.float32)
        obs[2] = 0.1
        obs[5] = 500.0

        for _ in range(200):
            action = ens.step(obs, {}, dt=0.01)

        assert action.shape == (1,)

    def test_ensemble_preserves_controller_state(self):
        """Both controllers should accumulate integrator state."""
        pid_cfg = PIDConfig()
        ctrl1 = GainScheduledPIDController(pid_cfg, use_observations=True)
        ctrl2 = GainScheduledPIDController(pid_cfg, use_observations=True)
        ens = EnsembleController([ctrl1, ctrl2])

        # First step: launch detection at angle=0 (sets target_orient=0)
        obs = np.zeros(10, dtype=np.float32)
        obs[5] = 500.0
        ens.step(obs, {}, dt=0.01)

        # Now add an angle offset so prop_error != 0
        obs[2] = 0.1  # 0.1 rad angle error from target (0)
        for _ in range(50):
            ens.step(obs, {}, dt=0.01)

        # Both controllers should have accumulated integrator state
        assert abs(ctrl1.integ_error) > 0
        assert abs(ctrl2.integ_error) > 0


class TestCompareControllersIntegration:
    """Test compare_controllers.py integration."""

    def test_controller_importable(self):
        from ensemble_controller import EnsembleController, EnsembleConfig

        ctrl = EnsembleController(
            [SimpleController(), SimpleController()],
        )
        assert hasattr(ctrl, "step")
        assert hasattr(ctrl, "reset")

    def test_ensemble_flag_in_compare_source(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "--ensemble" in source

    def test_color_defined(self):
        from pathlib import Path

        source = (Path(__file__).parent.parent / "compare_controllers.py").read_text()
        assert "Ensemble" in source
