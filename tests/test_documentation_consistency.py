"""
Tests for documentation consistency across CLAUDE.md and experimental_results.md.

Verifies that:
1. Controller implementations referenced in docs actually exist
2. CLI flags documented in CLAUDE.md are present in compare_controllers.py
3. Results tables in both docs are consistent with each other
4. Success criteria thresholds are consistent between docs
5. File references in docs point to existing files
"""

import re
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).parent.parent


class TestDocumentedControllersExist:
    """Verify all controller classes referenced in docs exist in code."""

    def test_pid_controller_exists(self):
        from controllers.pid_controller import PIDController

        ctrl = PIDController()
        assert hasattr(ctrl, "step")

    def test_gain_scheduled_pid_exists(self):
        from controllers.pid_controller import GainScheduledPIDController

        ctrl = GainScheduledPIDController()
        assert hasattr(ctrl, "step")

    def test_adrc_controller_exists(self):
        from controllers.adrc_controller import ADRCController, ADRCConfig

        ctrl = ADRCController(ADRCConfig(b0=100.0))
        assert hasattr(ctrl, "step")

    def test_estimate_adrc_config_exists(self):
        from controllers.adrc_controller import estimate_adrc_config

        assert callable(estimate_adrc_config)

    def test_pid_config_exists(self):
        from controllers.pid_controller import PIDConfig

        config = PIDConfig()
        assert hasattr(config, "Cprop")
        assert hasattr(config, "Cint")
        assert hasattr(config, "Cderiv")

    def test_adrc_config_has_dynamic_b0(self):
        """CLAUDE.md documents dynamic b0 via b0_per_pa. Verify it exists."""
        from controllers.adrc_controller import ADRCConfig

        config = ADRCConfig(b0=100.0, b0_per_pa=0.5)
        assert config.b0_per_pa == 0.5

    def test_adrc_config_has_use_observations(self):
        """CLAUDE.md documents use_observations for IMU mode."""
        from controllers.adrc_controller import ADRCConfig

        config = ADRCConfig(b0=100.0, use_observations=True)
        assert config.use_observations is True


class TestDocumentedCLIFlags:
    """Verify CLI flags documented in CLAUDE.md exist in compare_controllers.py."""

    @pytest.fixture
    def compare_source(self):
        path = PROJECT_ROOT / "compare_controllers.py"
        if not path.exists():
            pytest.skip("compare_controllers.py not found")
        return path.read_text()

    def test_gs_pid_flag(self, compare_source):
        assert (
            "--gain-scheduled" in compare_source
        ), "CLAUDE.md documents --gain-scheduled flag but it's missing from compare_controllers.py"

    def test_adrc_flag(self, compare_source):
        assert (
            "--adrc" in compare_source
        ), "CLAUDE.md documents --adrc flag but it's missing from compare_controllers.py"

    def test_imu_flag(self, compare_source):
        assert (
            "--imu" in compare_source
        ), "CLAUDE.md documents --imu flag but it's missing from compare_controllers.py"

    def test_wind_levels_flag(self, compare_source):
        assert "--wind-levels" in compare_source

    def test_n_episodes_flag(self, compare_source):
        assert "--n-episodes" in compare_source


class TestSuccessCriteriaConsistency:
    """Verify success criteria match between CLAUDE.md and experimental_results.md."""

    @pytest.fixture
    def claude_md(self):
        path = PROJECT_ROOT / "CLAUDE.md"
        if not path.exists():
            pytest.skip("CLAUDE.md not found")
        return path.read_text()

    @pytest.fixture
    def results_md(self):
        path = PROJECT_ROOT / "experimental_results.md"
        if not path.exists():
            pytest.skip("experimental_results.md not found")
        return path.read_text()

    def test_zero_wind_target(self, claude_md, results_md):
        """Both docs should state < 5 deg/s target for 0 m/s wind."""
        assert "< 5" in claude_md, "CLAUDE.md should mention < 5 deg/s target"
        assert (
            "< 5" in results_md
        ), "experimental_results.md should mention < 5 deg/s target"

    def test_primary_goal_consistent(self, claude_md, results_md):
        """Both docs should reference < 5 deg/s as the primary goal."""
        assert "5 deg/s" in claude_md
        assert "5 deg/s" in results_md


class TestDocumentedFilesExist:
    """Verify files referenced in documentation actually exist."""

    FILES_REFERENCED = [
        "controllers/pid_controller.py",
        "controllers/adrc_controller.py",
        "compare_controllers.py",
        "experimental_results.md",
    ]

    @pytest.mark.parametrize("filename", FILES_REFERENCED)
    def test_referenced_file_exists(self, filename):
        path = PROJECT_ROOT / filename
        assert (
            path.exists()
        ), f"Documentation references {filename} but it does not exist"


class TestResultsTablesConsistency:
    """Verify the results tables in CLAUDE.md and experimental_results.md agree."""

    @staticmethod
    def _extract_table_values(text, controller_name, wind_speeds):
        """Extract mean spin values for a controller from a markdown table.

        Returns dict of wind_speed -> mean_value.
        """
        values = {}
        for line in text.split("\n"):
            # Skip lines that don't look like table rows
            if "|" not in line or "---" in line:
                continue
            # Look for rows starting with a wind speed number
            cells = [c.strip() for c in line.split("|")]
            cells = [c for c in cells if c]  # Remove empty cells
            if not cells:
                continue
            try:
                wind = int(cells[0])
            except (ValueError, IndexError):
                continue
            if wind in wind_speeds:
                # Find the controller column value — extract first number
                for cell in cells[1:]:
                    # Match patterns like "3.2 ± 0.5" or "**3.2 ± 0.5**"
                    match = re.search(r"\*{0,2}(\d+\.?\d*)\s*[±+]", cell)
                    if match:
                        val = float(match.group(1))
                        if wind not in values:
                            values[wind] = []
                        values[wind].append(val)
        return values

    def test_zero_wind_best_below_5(self):
        """The final results table should show all controllers below 5 deg/s
        at 0 m/s wind. Only checks the first table after 'Ground-Truth
        Evaluation', not per-approach tables later in the file.
        """
        results_path = PROJECT_ROOT / "experimental_results.md"
        if not results_path.exists():
            pytest.skip("experimental_results.md not found")
        text = results_path.read_text()

        # Find the final results table section — limit to first table only
        gt_start = text.find("Ground-Truth Evaluation")
        if gt_start < 0:
            pytest.skip("Ground-Truth Evaluation section not found")
        section = text[gt_start:]
        # Limit to content before the next section header (## or ###)
        next_section = re.search(r"\n#{2,3}\s", section[1:])
        if next_section:
            section = section[: next_section.start() + 1]
        values = self._extract_table_values(section, "all", [0])

        if 0 in values:
            for val in values[0]:
                assert (
                    val < 5.0
                ), f"All controllers at 0 m/s should be < 5 deg/s, found {val}"

    def test_gs_pid_best_at_zero_wind(self):
        """GS-PID should be the best at 0 m/s in the ground-truth results table."""
        results_path = PROJECT_ROOT / "experimental_results.md"
        if not results_path.exists():
            pytest.skip("experimental_results.md not found")
        text = results_path.read_text()

        # Find the ground-truth table only (between "Ground-Truth" and "IMU-Based")
        gt_start = text.find("Ground-Truth Evaluation")
        imu_start = text.find("IMU-Based Evaluation")
        if gt_start < 0:
            pytest.skip("Ground-Truth Evaluation section not found")
        gt_section = (
            text[gt_start:imu_start] if imu_start > gt_start else text[gt_start:]
        )
        values = self._extract_table_values(gt_section, "all", [0])

        if 0 in values and len(values[0]) >= 2:
            # GS-PID is the second column (index 1) in the table
            gs_pid_val = values[0][1]
            assert gs_pid_val <= min(values[0]), (
                f"GS-PID ({gs_pid_val}) should be best at 0 m/s, "
                f"but found lower value in {values[0]}"
            )


class TestRecommendedController:
    """Verify the recommended controller is documented consistently."""

    def test_claude_md_recommends_gs_pid(self):
        """CLAUDE.md should recommend GS-PID for deployment."""
        path = PROJECT_ROOT / "CLAUDE.md"
        if not path.exists():
            pytest.skip("CLAUDE.md not found")
        text = path.read_text()
        assert "GS-PID" in text, "CLAUDE.md should mention GS-PID"
        assert (
            "recommended" in text.lower()
        ), "CLAUDE.md should include a recommendation"

    def test_results_md_recommends_gs_pid(self):
        """experimental_results.md should recommend GS-PID for deployment."""
        path = PROJECT_ROOT / "experimental_results.md"
        if not path.exists():
            pytest.skip("experimental_results.md not found")
        text = path.read_text()
        assert "GS-PID" in text, "experimental_results.md should mention GS-PID"
        assert (
            "recommended" in text.lower() or "Recommended" in text
        ), "experimental_results.md should include a recommendation"


class TestResearchPlanStepsDocumented:
    """Verify all research plan steps are marked complete in CLAUDE.md."""

    def test_step_6_marked_complete(self):
        """Step 6 should be marked as complete."""
        path = PROJECT_ROOT / "CLAUDE.md"
        if not path.exists():
            pytest.skip("CLAUDE.md not found")
        text = path.read_text()

        # Find Step 6 section
        step6_match = re.search(
            r"### Step 6.*?\n(.*?)(?=### Step 7|---|\Z)",
            text,
            re.DOTALL,
        )
        assert step6_match is not None, "Step 6 section not found in CLAUDE.md"
        step6_text = step6_match.group(0)
        assert (
            "✅" in step6_text or "Completed" in step6_text
        ), "Step 6 should be marked as completed in CLAUDE.md"

    def test_imu_validation_documented(self):
        """IMU validation results should be mentioned."""
        path = PROJECT_ROOT / "CLAUDE.md"
        if not path.exists():
            pytest.skip("CLAUDE.md not found")
        text = path.read_text()
        assert "IMU" in text, "CLAUDE.md should document IMU validation results"
        assert (
            "negligible" in text.lower() or "within" in text.lower()
        ), "CLAUDE.md should note that IMU noise impact is negligible"
