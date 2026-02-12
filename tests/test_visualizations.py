#!/usr/bin/env python3
"""
Tests for visualization scripts.

Verifies that each visualization can run in save mode without crashing,
that output files are created, and that they are non-empty. Uses minimal
simulation parameters (2 runs, short episodes) to keep tests fast.
"""

import os
import sys
import shutil
import tempfile
from unittest.mock import MagicMock

import numpy as np
import pytest

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


@pytest.fixture
def output_dir():
    """Create a temporary output directory and clean up after tests."""
    tmp = tempfile.mkdtemp(prefix="viz_test_")
    yield tmp
    shutil.rmtree(tmp, ignore_errors=True)


class TestRollRateMonteCarlo:
    """Tests for roll_rate_montecarlo.py."""

    def test_import(self):
        """Verify the module can be imported."""
        from visualizations.roll_rate_montecarlo import (
            collect_data,
            create_animation,
            get_config_and_gains,
            make_controller,
            run_episode,
            run_episode_rl,
        )

    def test_save_creates_file(self, output_dir):
        """Run in save mode and verify output file is created and non-empty."""
        import matplotlib

        matplotlib.use("Agg")

        from visualizations.roll_rate_montecarlo import (
            collect_data,
            compute_y_max,
            create_animation,
            get_config_and_gains,
        )
        from rocket_config import load_config

        config_path, pid_config = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        # Use 1 run and a single wind level for speed
        wind_levels = [1.0]
        n_runs = 1
        data = collect_data(config, wind_levels, n_runs, "pid", pid_config)
        y_max = compute_y_max(data)

        save_path = os.path.join(output_dir, "roll_rate_test.gif")
        create_animation(
            data,
            wind_levels,
            y_max,
            "estes_alpha",
            "pid",
            n_runs,
            save_path,
        )

        assert os.path.exists(save_path), f"Output file not created: {save_path}"
        assert os.path.getsize(save_path) > 0, "Output file is empty"

    def test_data_collection(self):
        """Verify data collection returns expected structure."""
        from visualizations.roll_rate_montecarlo import (
            collect_data,
            get_config_and_gains,
        )
        from rocket_config import load_config

        config_path, pid_config = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        data = collect_data(config, [1.0], 2, "pid", pid_config)

        assert 1.0 in data
        assert len(data[1.0]) == 2
        # Each trace is (times, roll_rates)
        t, rr = data[1.0][0]
        assert len(t) > 0
        assert len(rr) > 0
        assert len(t) == len(rr)


class TestWindFieldVisualization:
    """Tests for wind_field_visualization.py."""

    def test_import(self):
        """Verify the module can be imported."""
        from visualizations.wind_field_visualization import (
            create_animation,
            get_flight_params,
            sample_timeseries,
            sample_wind_field,
        )

    def test_save_creates_file(self, output_dir):
        """Run in save mode and verify output file is created and non-empty."""
        import matplotlib

        matplotlib.use("Agg")

        from visualizations.wind_field_visualization import (
            create_animation,
            get_flight_params,
        )
        from wind_model import WindModel, WindConfig

        max_alt, flight_dur, rocket_vel = get_flight_params("estes_alpha")

        # Use a short flight duration for speed
        flight_dur = 0.5
        fixed_alt = 50.0

        wind_config = WindConfig(
            enable=True,
            base_speed=2.0,
            max_gust_speed=1.0,
            variability=0.3,
        )
        wind_model = WindModel(wind_config)
        wind_model.reset(seed=42)

        save_path = os.path.join(output_dir, "wind_field_test.gif")
        create_animation(
            wind_model,
            "estes_alpha",
            2.0,
            max_alt,
            flight_dur,
            rocket_vel,
            fixed_alt,
            save_path,
        )

        assert os.path.exists(save_path), f"Output file not created: {save_path}"
        assert os.path.getsize(save_path) > 0, "Output file is empty"

    def test_flight_params(self):
        """Verify flight parameters are reasonable."""
        from visualizations.wind_field_visualization import get_flight_params

        max_alt, flight_dur, vel = get_flight_params("estes_alpha")
        assert max_alt > 0
        assert flight_dur > 0
        assert vel > 0

        max_alt_j, flight_dur_j, vel_j = get_flight_params("j800")
        assert max_alt_j > max_alt  # J800 flies higher
        assert vel_j > vel  # J800 is faster


class TestTrajectoryMonteCarlo:
    """Tests for trajectory_montecarlo.py."""

    def test_import(self):
        """Verify the module can be imported."""
        from visualizations.trajectory_montecarlo import (
            collect_data,
            compute_axis_ranges,
            create_2d_animation,
            create_3d_animation,
            get_config_and_gains,
            run_episode_trajectory,
            run_episode_trajectory_rl,
        )

    def test_save_2d_creates_file(self, output_dir):
        """Run 2D save mode and verify output is created and non-empty."""
        import matplotlib

        matplotlib.use("Agg")

        from visualizations.trajectory_montecarlo import (
            collect_data,
            compute_axis_ranges,
            create_2d_animation,
            get_config_and_gains,
        )
        from rocket_config import load_config

        config_path, pid_config = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        wind_levels = [1.0]
        n_runs = 1
        data = collect_data(config, wind_levels, n_runs, "pid", pid_config)
        ranges = compute_axis_ranges(data)

        save_path = os.path.join(output_dir, "trajectory_2d_test.gif")
        create_2d_animation(
            data,
            wind_levels,
            ranges,
            "estes_alpha",
            "pid",
            n_runs,
            save_path,
        )

        assert os.path.exists(save_path), f"Output file not created: {save_path}"
        assert os.path.getsize(save_path) > 0, "Output file is empty"

    def test_save_3d_creates_file(self, output_dir):
        """Run 3D save mode and verify output is created and non-empty."""
        import matplotlib

        matplotlib.use("Agg")

        from visualizations.trajectory_montecarlo import (
            collect_data,
            compute_axis_ranges,
            create_3d_animation,
            get_config_and_gains,
        )
        from rocket_config import load_config

        config_path, pid_config = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        wind_levels = [1.0]
        n_runs = 1
        data = collect_data(config, wind_levels, n_runs, "pid", pid_config)
        ranges = compute_axis_ranges(data)

        save_path = os.path.join(output_dir, "trajectory_3d_test.gif")
        create_3d_animation(
            data,
            wind_levels,
            ranges,
            "estes_alpha",
            "pid",
            n_runs,
            save_path,
        )

        assert os.path.exists(save_path), f"Output file not created: {save_path}"
        assert os.path.getsize(save_path) > 0, "Output file is empty"

    def test_trajectory_data_structure(self):
        """Verify trajectory data has expected fields."""
        from visualizations.trajectory_montecarlo import (
            get_config_and_gains,
            run_episode_trajectory,
        )
        from rocket_config import load_config

        config_path, pid_config = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        traj = run_episode_trajectory(config, 1.0, "pid", pid_config, seed=42)

        assert "time" in traj
        assert "altitude" in traj
        assert "x" in traj
        assert "y" in traj
        assert len(traj["time"]) == len(traj["altitude"])
        assert len(traj["x"]) == len(traj["y"])
        assert traj["altitude"].max() > 0  # Rocket should fly


class TestRollRateRL:
    """Tests for RL model support in roll_rate_montecarlo.py."""

    def test_run_episode_rl(self):
        """Verify run_episode_rl returns valid (times, roll_rates) tuple."""
        from visualizations.roll_rate_montecarlo import (
            get_config_and_gains,
            run_episode_rl,
        )
        from rocket_config import load_config

        config_path, _ = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        t, rr = run_episode_rl(config, 1.0, mock_model, vec_normalize=None, seed=42)

        assert len(t) > 0
        assert len(rr) > 0
        assert len(t) == len(rr)
        assert all(r >= 0 for r in rr)

    def test_run_episode_rl_with_vec_normalize(self):
        """Verify run_episode_rl works with a VecNormalize mock."""
        from visualizations.roll_rate_montecarlo import (
            get_config_and_gains,
            run_episode_rl,
        )
        from rocket_config import load_config

        config_path, _ = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)
        mock_vec_normalize = MagicMock()
        mock_vec_normalize.normalize_obs.side_effect = lambda x: x

        t, rr = run_episode_rl(
            config, 1.0, mock_model, vec_normalize=mock_vec_normalize, seed=42
        )

        assert len(t) > 0
        assert mock_vec_normalize.normalize_obs.call_count > 0

    def test_collect_data_with_model(self):
        """Verify collect_data dispatches to RL when model is provided."""
        from visualizations.roll_rate_montecarlo import (
            collect_data,
            get_config_and_gains,
        )
        from rocket_config import load_config

        config_path, pid_config = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        data = collect_data(config, [1.0], 2, "sac", pid_config, model=mock_model)

        assert 1.0 in data
        assert len(data[1.0]) == 2
        t, rr = data[1.0][0]
        assert len(t) > 0
        assert len(t) == len(rr)


class TestTrajectoryRL:
    """Tests for RL model support in trajectory_montecarlo.py."""

    def test_run_episode_trajectory_rl(self):
        """Verify run_episode_trajectory_rl returns valid trajectory dict."""
        from visualizations.trajectory_montecarlo import (
            get_config_and_gains,
            run_episode_trajectory_rl,
        )
        from rocket_config import load_config

        config_path, _ = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        traj = run_episode_trajectory_rl(
            config, 1.0, mock_model, vec_normalize=None, seed=42
        )

        assert "time" in traj
        assert "altitude" in traj
        assert "x" in traj
        assert "y" in traj
        assert "velocity" in traj
        assert len(traj["time"]) == len(traj["altitude"])
        assert len(traj["x"]) == len(traj["y"])
        assert traj["altitude"].max() > 0

    def test_run_episode_trajectory_rl_with_vec_normalize(self):
        """Verify run_episode_trajectory_rl works with a VecNormalize mock."""
        from visualizations.trajectory_montecarlo import (
            get_config_and_gains,
            run_episode_trajectory_rl,
        )
        from rocket_config import load_config

        config_path, _ = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)
        mock_vec_normalize = MagicMock()
        mock_vec_normalize.normalize_obs.side_effect = lambda x: x

        traj = run_episode_trajectory_rl(
            config, 1.0, mock_model, vec_normalize=mock_vec_normalize, seed=42
        )

        assert len(traj["time"]) > 0
        assert mock_vec_normalize.normalize_obs.call_count > 0

    def test_collect_data_with_model(self):
        """Verify collect_data dispatches to RL when model is provided."""
        from visualizations.trajectory_montecarlo import (
            collect_data,
            get_config_and_gains,
        )
        from rocket_config import load_config

        config_path, pid_config = get_config_and_gains("estes_alpha")
        config = load_config(config_path)

        mock_model = MagicMock()
        mock_model.predict.return_value = (np.array([0.0]), None)

        data = collect_data(config, [1.0], 2, "sac", pid_config, model=mock_model)

        assert 1.0 in data
        assert len(data[1.0]) == 2
        traj = data[1.0][0]
        assert "time" in traj
        assert "altitude" in traj
        assert len(traj["time"]) > 0
