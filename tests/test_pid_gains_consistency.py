"""
Tests for PID gain consistency across the codebase.

Ensures the optimized PID gains (Kp=0.005208, Ki=0.000324, Kd=0.016524) are used
consistently in all default values, argparse defaults, and getattr fallbacks.

Background: A 5x performance regression was caused by compare_controllers.py
using argparse defaults (Kp=0.05, Ki=0.01, Kd=0.08) instead of the optimized
gains. This test prevents that from happening again.
"""

import ast
import re
from pathlib import Path

import pytest

# The single source of truth: optimized PID gains
OPTIMIZED_KP = 0.005208
OPTIMIZED_KI = 0.000324
OPTIMIZED_KD = 0.016524


class TestPIDConfigDefaults:
    """Test that PIDConfig dataclass defaults match optimized gains."""

    def test_pid_config_defaults(self):
        """PIDConfig defaults must be the optimized gains."""
        from pid_controller import PIDConfig

        config = PIDConfig()
        assert config.Cprop == OPTIMIZED_KP, (
            f"PIDConfig.Cprop default is {config.Cprop}, "
            f"expected optimized value {OPTIMIZED_KP}"
        )
        assert config.Cint == OPTIMIZED_KI, (
            f"PIDConfig.Cint default is {config.Cint}, "
            f"expected optimized value {OPTIMIZED_KI}"
        )
        assert config.Cderiv == OPTIMIZED_KD, (
            f"PIDConfig.Cderiv default is {config.Cderiv}, "
            f"expected optimized value {OPTIMIZED_KD}"
        )


class TestRocketPhysicsConfigDefaults:
    """Test that RocketPhysicsConfig PID defaults match optimized gains."""

    def test_rocket_physics_pid_defaults(self):
        """RocketPhysicsConfig pid_Kp/Ki/Kd defaults must be optimized gains."""
        from rocket_config import RocketPhysicsConfig

        config = RocketPhysicsConfig()
        assert config.pid_Kp == OPTIMIZED_KP, (
            f"RocketPhysicsConfig.pid_Kp default is {config.pid_Kp}, "
            f"expected {OPTIMIZED_KP}"
        )
        assert config.pid_Ki == OPTIMIZED_KI, (
            f"RocketPhysicsConfig.pid_Ki default is {config.pid_Ki}, "
            f"expected {OPTIMIZED_KI}"
        )
        assert config.pid_Kd == OPTIMIZED_KD, (
            f"RocketPhysicsConfig.pid_Kd default is {config.pid_Kd}, "
            f"expected {OPTIMIZED_KD}"
        )


class TestArgparseDefaults:
    """Test that argparse defaults in CLI scripts match optimized gains."""

    @staticmethod
    def _extract_argparse_defaults(filepath: Path, arg_names: list) -> dict:
        """Extract default values from argparse add_argument calls.

        Parses the source file to find add_argument calls and extract
        their default= keyword values.
        """
        source = filepath.read_text()
        defaults = {}

        for arg_name in arg_names:
            # Match patterns like: add_argument("--pid-Kp", type=float, default=0.05, ...)
            # or: add_argument("--Cprop", type=float, default=0.05, ...)
            pattern = (
                rf'add_argument\(["\']--{re.escape(arg_name)}["\']'
                rf".*?default\s*=\s*([0-9.]+)"
            )
            match = re.search(pattern, source)
            if match:
                defaults[arg_name] = float(match.group(1))

        return defaults

    def test_compare_controllers_argparse_defaults(self):
        """compare_controllers.py argparse defaults must use optimized gains."""
        filepath = Path("compare_controllers.py")
        if not filepath.exists():
            pytest.skip("compare_controllers.py not found")

        defaults = self._extract_argparse_defaults(
            filepath, ["pid-Kp", "pid-Ki", "pid-Kd"]
        )

        assert defaults.get("pid-Kp") == OPTIMIZED_KP, (
            f"compare_controllers.py --pid-Kp default is {defaults.get('pid-Kp')}, "
            f"expected {OPTIMIZED_KP}"
        )
        assert defaults.get("pid-Ki") == OPTIMIZED_KI, (
            f"compare_controllers.py --pid-Ki default is {defaults.get('pid-Ki')}, "
            f"expected {OPTIMIZED_KI}"
        )
        assert defaults.get("pid-Kd") == OPTIMIZED_KD, (
            f"compare_controllers.py --pid-Kd default is {defaults.get('pid-Kd')}, "
            f"expected {OPTIMIZED_KD}"
        )

    def test_pid_controller_argparse_defaults(self):
        """pid_controller.py argparse defaults must use optimized gains."""
        filepath = Path("pid_controller.py")
        if not filepath.exists():
            pytest.skip("pid_controller.py not found")

        defaults = self._extract_argparse_defaults(
            filepath, ["Cprop", "Cint", "Cderiv"]
        )

        assert defaults.get("Cprop") == OPTIMIZED_KP, (
            f"pid_controller.py --Cprop default is {defaults.get('Cprop')}, "
            f"expected {OPTIMIZED_KP}"
        )
        assert defaults.get("Cint") == OPTIMIZED_KI, (
            f"pid_controller.py --Cint default is {defaults.get('Cint')}, "
            f"expected {OPTIMIZED_KI}"
        )
        assert defaults.get("Cderiv") == OPTIMIZED_KD, (
            f"pid_controller.py --Cderiv default is {defaults.get('Cderiv')}, "
            f"expected {OPTIMIZED_KD}"
        )


class TestGetAttrFallbacks:
    """Test that getattr fallback values for PID gains match optimized gains.

    Several files use patterns like:
        getattr(config.physics, "pid_Kp", 0.05)
    The fallback value (0.05 in this example) must match the optimized gains.
    """

    # Files and their expected getattr patterns for PID gains
    FILES_WITH_FALLBACKS = [
        "visualizations/visualize_spin_agent.py",
        "optimize_pid.py",
        "train_residual_sac.py",
        "train_improved.py",
    ]

    @staticmethod
    def _extract_getattr_pid_fallbacks(filepath: Path) -> dict:
        """Extract PID gain fallback values from getattr calls."""
        source = filepath.read_text()
        fallbacks = {}

        for gain_name in ["pid_Kp", "pid_Ki", "pid_Kd"]:
            # Match: getattr(..., "pid_Kp", 0.05)
            pattern = (
                rf'getattr\([^)]*["\']{re.escape(gain_name)}["\']'
                rf"\s*,\s*([0-9.]+)\s*\)"
            )
            matches = re.findall(pattern, source)
            if matches:
                # Take the last match (in case there are multiple)
                fallbacks[gain_name] = float(matches[-1])

        return fallbacks

    @pytest.mark.parametrize("filepath", FILES_WITH_FALLBACKS)
    def test_getattr_fallbacks(self, filepath):
        """getattr fallback values for PID gains must match optimized gains."""
        path = Path(filepath)
        if not path.exists():
            pytest.skip(f"{filepath} not found")

        fallbacks = self._extract_getattr_pid_fallbacks(path)

        if not fallbacks:
            # File might not have any PID getattr calls (e.g., if refactored)
            return

        if "pid_Kp" in fallbacks:
            assert fallbacks["pid_Kp"] == OPTIMIZED_KP, (
                f"{filepath} getattr fallback for pid_Kp is {fallbacks['pid_Kp']}, "
                f"expected {OPTIMIZED_KP}"
            )
        if "pid_Ki" in fallbacks:
            assert fallbacks["pid_Ki"] == OPTIMIZED_KI, (
                f"{filepath} getattr fallback for pid_Ki is {fallbacks['pid_Ki']}, "
                f"expected {OPTIMIZED_KI}"
            )
        if "pid_Kd" in fallbacks:
            assert fallbacks["pid_Kd"] == OPTIMIZED_KD, (
                f"{filepath} getattr fallback for pid_Kd is {fallbacks['pid_Kd']}, "
                f"expected {OPTIMIZED_KD}"
            )


class TestPIDControllerWithOptimizedGains:
    """Functional test: PID with optimized gains produces reasonable output."""

    def test_optimized_gains_produce_corrective_action(self):
        """PID with default (optimized) gains should produce corrective action
        for a typical initial spin rate."""
        from pid_controller import PIDController, PIDConfig
        import numpy as np

        controller = PIDController()  # Uses optimized defaults
        controller.launch_detected = True
        controller.target_orient = 0.0

        obs = np.zeros(10)
        info = {
            "roll_angle_rad": np.radians(5.0),
            "roll_rate_deg_s": 15.0,  # Typical initial spin
            "vertical_acceleration_ms2": 50.0,
        }

        action = controller.step(obs, info, dt=0.01)

        # With Kd=0.016524 and 15 deg/s roll rate, the D term alone gives:
        # 15 * 0.016524 = 0.248 deg deflection, normalized to 0.248/30 = 0.00826
        # PID negates output (line 150: action = -servo_cmd / max_deflection)
        # so positive spin -> negative action (opposing torque)
        assert (
            abs(action[0]) > 0.001
        ), "Optimized PID gains should produce meaningful corrective action"
        assert action[0] < 0, (
            "Action should oppose positive roll rate "
            "(PID negates servo_cmd, so positive spin -> negative action)"
        )
