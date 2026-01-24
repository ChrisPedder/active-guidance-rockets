"""
Pytest fixtures for rocket control tests.
"""
import pytest
import tempfile
import os
from pathlib import Path

import numpy as np


@pytest.fixture
def sample_motor_config():
    """Sample motor configuration dictionary."""
    return {
        'name': 'test_motor',
        'manufacturer': 'Test',
        'designation': 'T100',
        'diameter_mm': 18.0,
        'length_mm': 70.0,
        'total_mass_g': 24.0,
        'propellant_mass_g': 12.0,
        'case_mass_g': 12.0,
        'impulse_class': 'C',
        'total_impulse_Ns': 10.0,
        'avg_thrust_N': 5.0,
        'max_thrust_N': 14.0,
        'burn_time_s': 2.0,
        'thrust_multiplier': 1.0,
        'thrust_curve': {
            'time_s': [0.0, 0.1, 1.0, 2.0],
            'thrust_N': [0.0, 14.0, 5.0, 0.0]
        }
    }


@pytest.fixture
def legacy_physics_config():
    """Legacy physics configuration with old-style parameters."""
    return {
        'dry_mass': 0.1,
        'propellant_mass': 0.012,
        'diameter': 0.024,
        'length': 0.4,
        'num_fins': 4,
        'fin_span': 0.04,
        'fin_root_chord': 0.05,
        'fin_tip_chord': 0.025,
        'max_tab_deflection': 15.0,
        'disturbance_scale': 0.0001,
    }


@pytest.fixture
def new_style_physics_config(tmp_path):
    """New-style physics configuration with airframe_file."""
    # Create a temporary airframe file
    airframe_content = """
name: Test Airframe
description: Test airframe for unit tests
components:
- type: NoseCone
  name: Nose Cone
  position: 0.0
  length: 0.07
  base_diameter: 0.024
  shape: ogive
  thickness: 0.002
  material: ABS Plastic
- type: BodyTube
  name: Body Tube
  position: 0.07
  length: 0.24
  outer_diameter: 0.024
  inner_diameter: 0.022
  material: Cardboard
- type: TrapezoidFinSet
  name: Fins
  position: 0.24
  num_fins: 4
  root_chord: 0.05
  tip_chord: 0.025
  span: 0.04
  sweep_length: 0.0
  thickness: 0.002
  material: Balsa
"""
    airframe_file = tmp_path / "test_airframe.yaml"
    airframe_file.write_text(airframe_content)

    return {
        'airframe_file': str(airframe_file),
        'max_tab_deflection': 15.0,
        'disturbance_scale': 0.0001,
    }


@pytest.fixture
def sample_training_config_yaml(tmp_path, legacy_physics_config, sample_motor_config):
    """Create a complete training config YAML file."""
    import yaml

    config = {
        'physics': legacy_physics_config,
        'motor': sample_motor_config,
        'environment': {
            'dt': 0.02,
            'max_episode_steps': 500,
            'initial_spin_rate_range': [-30.0, 30.0],
            'initial_tilt_range': [-5.0, 5.0],
            'enable_wind': False,
        },
        'reward': {
            'altitude_reward_scale': 0.01,
            'spin_penalty_scale': -0.1,
        },
        'ppo': {
            'learning_rate': 0.0003,
            'n_steps': 2048,
            'batch_size': 64,
            'total_timesteps': 10000,
        },
        'curriculum': {
            'enabled': False,
        },
        'logging': {
            'log_dir': 'logs',
            'save_dir': 'models',
            'experiment_name': 'test_experiment',
        }
    }

    config_file = tmp_path / "test_config.yaml"
    with open(config_file, 'w') as f:
        yaml.dump(config, f)

    return str(config_file)


@pytest.fixture
def estes_alpha_airframe():
    """Get an Estes Alpha III airframe."""
    from airframe import RocketAirframe
    return RocketAirframe.estes_alpha()


@pytest.fixture
def temp_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
