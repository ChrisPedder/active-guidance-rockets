"""
Tests for Phase 1 Hardware Parameter Studies (Research Plan Feb 7th v2).

Validates that:
1. Config files load correctly and have the expected parameter overrides
2. Environments can be created from each config
3. Physics responds correctly to parameter changes (control authority scales
   with num_controlled_fins, tab_deflection, tab_area)
4. Higher loop rates produce consistent episode durations
"""

import pytest
import numpy as np
from pathlib import Path

import yaml

from rocket_config import load_config


# ─── Config file paths ───────────────────────────────────────────────────────

CONFIGS_DIR = Path("configs")
BASELINE_CONFIG = CONFIGS_DIR / "estes_c6_sac_wind.yaml"

PHASE1_CONFIGS = {
    "4fin": CONFIGS_DIR / "estes_c6_4fin.yaml",
    "200hz": CONFIGS_DIR / "estes_c6_200hz.yaml",
    "500hz": CONFIGS_DIR / "estes_c6_500hz.yaml",
    "tab10": CONFIGS_DIR / "estes_c6_tab10.yaml",
    "tab15": CONFIGS_DIR / "estes_c6_tab15.yaml",
    "tab25": CONFIGS_DIR / "estes_c6_tab25.yaml",
    "tab30": CONFIGS_DIR / "estes_c6_tab30.yaml",
    "bigtab": CONFIGS_DIR / "estes_c6_bigtab.yaml",
}


# ─── Helpers ──────────────────────────────────────────────────────────────────


def _load_yaml_raw(path: Path) -> dict:
    """Load raw YAML without going through the config system."""
    with open(path) as f:
        return yaml.safe_load(f)


# ─── Test: Config files exist ─────────────────────────────────────────────────


class TestConfigFilesExist:
    """All Phase 1 config files must exist."""

    def test_baseline_config_exists(self):
        assert BASELINE_CONFIG.exists(), f"Baseline config missing: {BASELINE_CONFIG}"

    @pytest.mark.parametrize("name,path", list(PHASE1_CONFIGS.items()))
    def test_phase1_config_exists(self, name, path):
        assert path.exists(), f"Phase 1 config missing: {path} ({name})"


# ─── Test: Config parameter overrides ─────────────────────────────────────────


class TestConfigParameterOverrides:
    """Verify each config has the correct parameter changes vs baseline."""

    def test_4fin_has_4_controlled_fins(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["4fin"])
        assert raw["physics"]["num_controlled_fins"] == 4

    def test_4fin_unchanged_tab_params(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["4fin"])
        baseline = _load_yaml_raw(BASELINE_CONFIG)
        assert (
            raw["physics"]["max_tab_deflection"]
            == baseline["physics"]["max_tab_deflection"]
        )
        assert (
            raw["physics"]["tab_chord_fraction"]
            == baseline["physics"]["tab_chord_fraction"]
        )
        assert (
            raw["physics"]["tab_span_fraction"]
            == baseline["physics"]["tab_span_fraction"]
        )

    def test_200hz_has_correct_dt(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["200hz"])
        assert raw["environment"]["dt"] == 0.005

    def test_200hz_has_correct_sensor_rate(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["200hz"])
        assert raw["sensors"]["control_rate_hz"] == 200.0

    def test_500hz_has_correct_dt(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["500hz"])
        assert raw["environment"]["dt"] == 0.002

    def test_500hz_has_correct_sensor_rate(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["500hz"])
        assert raw["sensors"]["control_rate_hz"] == 500.0

    @pytest.mark.parametrize(
        "name,expected_deflection",
        [
            ("tab10", 10.0),
            ("tab15", 15.0),
            ("tab25", 25.0),
            ("tab30", 30.0),
        ],
    )
    def test_tab_deflection_configs(self, name, expected_deflection):
        raw = _load_yaml_raw(PHASE1_CONFIGS[name])
        assert raw["physics"]["max_tab_deflection"] == expected_deflection

    def test_bigtab_has_correct_area(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["bigtab"])
        assert raw["physics"]["tab_chord_fraction"] == 0.50
        assert raw["physics"]["tab_span_fraction"] == 1.0

    def test_bigtab_unchanged_deflection(self):
        raw = _load_yaml_raw(PHASE1_CONFIGS["bigtab"])
        baseline = _load_yaml_raw(BASELINE_CONFIG)
        assert (
            raw["physics"]["max_tab_deflection"]
            == baseline["physics"]["max_tab_deflection"]
        )


# ─── Test: Configs load through the config system ─────────────────────────────


class TestConfigsLoadCorrectly:
    """All Phase 1 configs must load without error through the config system."""

    @pytest.mark.parametrize("name,path", list(PHASE1_CONFIGS.items()))
    def test_config_loads(self, name, path):
        config = load_config(str(path))
        assert config is not None, f"Config {name} loaded as None"
        assert config.physics is not None
        assert config.motor is not None
        assert config.environment is not None

    def test_4fin_config_value_propagates(self):
        config = load_config(str(PHASE1_CONFIGS["4fin"]))
        assert getattr(config.physics, "num_controlled_fins", 2) == 4

    def test_200hz_dt_propagates(self):
        config = load_config(str(PHASE1_CONFIGS["200hz"]))
        assert config.environment.dt == 0.005

    def test_500hz_dt_propagates(self):
        config = load_config(str(PHASE1_CONFIGS["500hz"]))
        assert config.environment.dt == 0.002

    def test_tab_deflection_propagates(self):
        config = load_config(str(PHASE1_CONFIGS["tab25"]))
        assert config.physics.max_tab_deflection == 25.0

    def test_bigtab_area_propagates(self):
        config = load_config(str(PHASE1_CONFIGS["bigtab"]))
        assert config.physics.tab_chord_fraction == 0.50
        assert config.physics.tab_span_fraction == 1.0


# ─── Test: Environments can be created ────────────────────────────────────────


class TestEnvironmentCreation:
    """Environments can be created from each Phase 1 config."""

    @pytest.mark.parametrize("name,path", list(PHASE1_CONFIGS.items()))
    def test_env_creation(self, name, path):
        from compare_controllers import create_env

        config = load_config(str(path))
        env = create_env(config, wind_speed=0.0)
        assert env is not None
        obs, info = env.reset(seed=42)
        assert obs is not None
        assert len(obs) > 0

    @pytest.mark.parametrize("name,path", list(PHASE1_CONFIGS.items()))
    def test_env_step(self, name, path):
        from compare_controllers import create_env

        config = load_config(str(path))
        env = create_env(config, wind_speed=1.0)
        obs, info = env.reset(seed=42)
        action = np.array([0.0])
        obs2, reward, terminated, truncated, info2 = env.step(action)
        assert obs2 is not None


# ─── Test: Control authority scales with hardware parameters ──────────────────


class TestControlAuthorityScaling:
    """Physics model must respond to hardware parameter changes."""

    def _get_effectiveness(self, config_path: str) -> float:
        """Get control effectiveness (N*m/rad) at a reference dynamic pressure.

        Directly queries the airframe's control effectiveness calculation,
        bypassing the simulation loop for a deterministic comparison.
        """
        config = load_config(config_path)
        airframe = config.physics.resolve_airframe()
        q_ref = 500.0  # Reference dynamic pressure (mid-boost)
        return airframe.get_control_effectiveness(
            q_ref,
            tab_chord_fraction=config.physics.tab_chord_fraction,
            tab_span_fraction=config.physics.tab_span_fraction,
            num_controlled_fins=getattr(config.physics, "num_controlled_fins", 2),
        )

    def _get_max_control_torque(self, config_path: str) -> float:
        """Get maximum control torque (N*m) at reference q.

        This is effectiveness * max_deflection_rad, representing the
        maximum torque the hardware configuration can produce.
        """
        config = load_config(config_path)
        eff = self._get_effectiveness(config_path)
        max_defl_rad = np.radians(config.physics.max_tab_deflection)
        return eff * max_defl_rad

    def test_4fin_doubles_effectiveness(self):
        """4 active fins should exactly double control effectiveness vs 2 fins."""
        eff_2fin = self._get_effectiveness(str(BASELINE_CONFIG))
        eff_4fin = self._get_effectiveness(str(PHASE1_CONFIGS["4fin"]))

        ratio = eff_4fin / eff_2fin
        assert 1.9 < ratio < 2.1, (
            f"4-fin effectiveness should be 2x baseline, got ratio={ratio:.2f} "
            f"(2fin={eff_2fin:.6f}, 4fin={eff_4fin:.6f})"
        )

    def test_larger_deflection_increases_max_torque(self):
        """Larger tab deflection should increase max control torque proportionally."""
        torque_baseline = self._get_max_control_torque(str(BASELINE_CONFIG))
        torque_tab25 = self._get_max_control_torque(str(PHASE1_CONFIGS["tab25"]))

        # 25/30 = ~0.833x deflection range (tab25 is less than baseline 30 degrees)
        expected_ratio = 25.0 / 30.0
        actual_ratio = torque_tab25 / torque_baseline
        assert actual_ratio > expected_ratio * 0.9, (
            f"25-degree tabs should produce ~{expected_ratio:.2f}x max torque vs baseline, "
            f"got {actual_ratio:.2f}x (baseline={torque_baseline:.6f}, tab25={torque_tab25:.6f})"
        )

    def test_larger_area_increases_effectiveness(self):
        """Larger tab area should increase control effectiveness."""
        eff_baseline = self._get_effectiveness(str(BASELINE_CONFIG))
        eff_bigtab = self._get_effectiveness(str(PHASE1_CONFIGS["bigtab"]))

        # bigtab: chord 0.50 vs 0.25 (2x), span 1.0 vs 0.5 (2x) → 4x area
        ratio = eff_bigtab / eff_baseline
        assert 3.5 < ratio < 4.5, (
            f"Doubled chord+span should give ~4x effectiveness, got ratio={ratio:.2f} "
            f"(baseline={eff_baseline:.6f}, bigtab={eff_bigtab:.6f})"
        )

    def test_deflection_torque_monotonically_increases(self):
        """Max control torque should increase monotonically with tab deflection."""
        configs_ordered = [
            ("tab10", str(PHASE1_CONFIGS["tab10"])),
            ("tab15", str(PHASE1_CONFIGS["tab15"])),
            ("tab25", str(PHASE1_CONFIGS["tab25"])),
            ("tab30", str(PHASE1_CONFIGS["tab30"])),
        ]
        torques = [
            (name, self._get_max_control_torque(path)) for name, path in configs_ordered
        ]

        for i in range(len(torques) - 1):
            name_a, t_a = torques[i]
            name_b, t_b = torques[i + 1]
            assert (
                t_b > t_a
            ), f"Max torque should increase: {name_a}={t_a:.6f} vs {name_b}={t_b:.6f}"

    def test_effectiveness_independent_of_deflection(self):
        """Effectiveness (N*m/rad) should be the same regardless of max deflection,
        since it depends only on tab geometry and num_fins."""
        eff_baseline = self._get_effectiveness(str(BASELINE_CONFIG))
        eff_tab25 = self._get_effectiveness(str(PHASE1_CONFIGS["tab25"]))

        assert abs(eff_baseline - eff_tab25) < 1e-10, (
            f"Effectiveness should not depend on max_tab_deflection: "
            f"baseline={eff_baseline:.6f}, tab25={eff_tab25:.6f}"
        )


# ─── Test: Loop rate affects timestep correctly ───────────────────────────────


class TestLoopRatePhysics:
    """Higher loop rates should produce more steps per unit time."""

    def test_200hz_dt_in_env(self):
        from compare_controllers import create_env

        config = load_config(str(PHASE1_CONFIGS["200hz"]))
        env = create_env(config, wind_speed=0.0)
        assert env.config.dt == 0.005

    def test_500hz_dt_in_env(self):
        from compare_controllers import create_env

        config = load_config(str(PHASE1_CONFIGS["500hz"]))
        env = create_env(config, wind_speed=0.0)
        assert env.config.dt == 0.002

    def test_higher_rate_more_steps_per_second(self):
        """200 Hz config should take ~2x as many steps as 100 Hz for the same
        simulation time."""
        from compare_controllers import create_env

        # Run baseline (100 Hz) for a fixed number of steps
        config_100 = load_config(str(BASELINE_CONFIG))
        env_100 = create_env(config_100, wind_speed=0.0)
        env_100.reset(seed=42)
        steps_100 = 0
        for _ in range(100):
            _, _, terminated, truncated, info = env_100.step(np.array([0.0]))
            steps_100 += 1
            if terminated or truncated:
                break
        time_100 = steps_100 * 0.01

        # Run 200 Hz for same simulated time
        config_200 = load_config(str(PHASE1_CONFIGS["200hz"]))
        env_200 = create_env(config_200, wind_speed=0.0)
        env_200.reset(seed=42)
        steps_200 = 0
        for _ in range(200):
            _, _, terminated, truncated, info = env_200.step(np.array([0.0]))
            steps_200 += 1
            if terminated or truncated:
                break
        time_200 = steps_200 * 0.005

        # Both should simulate the same duration (1 second)
        assert (
            abs(time_100 - time_200) < 0.02
        ), f"Both should simulate ~1s: 100Hz={time_100:.3f}s, 200Hz={time_200:.3f}s"
        assert (
            steps_200 == steps_100 * 2
        ), f"200 Hz should use 2x as many steps: got {steps_200} vs {steps_100}"


# ─── Test: Only the intended parameter changes ───────────────────────────────


class TestNoUnintendedChanges:
    """Configs should only differ from baseline in the intended parameters."""

    def test_4fin_only_changes_num_controlled_fins(self):
        baseline = _load_yaml_raw(BASELINE_CONFIG)
        variant = _load_yaml_raw(PHASE1_CONFIGS["4fin"])

        # Physics should be identical except num_controlled_fins
        for key in baseline["physics"]:
            if key == "num_controlled_fins":
                continue
            if key in variant["physics"]:
                assert variant["physics"][key] == baseline["physics"][key], (
                    f"4fin config changed physics.{key}: "
                    f"{baseline['physics'][key]} -> {variant['physics'][key]}"
                )

    def test_200hz_only_changes_dt_and_sensor_rate(self):
        baseline = _load_yaml_raw(BASELINE_CONFIG)
        variant = _load_yaml_raw(PHASE1_CONFIGS["200hz"])

        # Physics should be identical
        for key in baseline["physics"]:
            if key in variant["physics"]:
                assert (
                    variant["physics"][key] == baseline["physics"][key]
                ), f"200hz config changed physics.{key}"

        # Environment should only differ in dt and max_episode_steps
        for key in baseline["environment"]:
            if key in ("dt", "max_episode_steps"):
                continue
            if key in variant["environment"]:
                assert (
                    variant["environment"][key] == baseline["environment"][key]
                ), f"200hz config changed environment.{key}"

    def test_tab_configs_only_change_deflection(self):
        baseline = _load_yaml_raw(BASELINE_CONFIG)

        for name in ["tab10", "tab15", "tab25", "tab30"]:
            variant = _load_yaml_raw(PHASE1_CONFIGS[name])
            for key in baseline["physics"]:
                if key == "max_tab_deflection":
                    continue
                if key in variant["physics"]:
                    assert variant["physics"][key] == baseline["physics"][key], (
                        f"{name} config changed physics.{key}: "
                        f"{baseline['physics'][key]} -> {variant['physics'][key]}"
                    )

    def test_bigtab_only_changes_tab_area(self):
        baseline = _load_yaml_raw(BASELINE_CONFIG)
        variant = _load_yaml_raw(PHASE1_CONFIGS["bigtab"])

        for key in baseline["physics"]:
            if key in ("tab_chord_fraction", "tab_span_fraction"):
                continue
            if key in variant["physics"]:
                assert (
                    variant["physics"][key] == baseline["physics"][key]
                ), f"bigtab config changed physics.{key}"
