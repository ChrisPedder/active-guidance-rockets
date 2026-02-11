"""
Tests for realistic_spin_rocket.py module.

Covers: RealisticMotorRocket creation, stepping, _update_propulsion, _get_info.
"""

import numpy as np
import pytest

from realistic_spin_rocket import RealisticMotorRocket
from airframe import RocketAirframe
from spin_stabilized_control_env import RocketConfig


@pytest.fixture
def airframe():
    return RocketAirframe.estes_alpha()


@pytest.fixture
def motor_config():
    """Minimal motor config dict for testing (matches motor_loader.py format)."""
    return {
        "name": "estes_c6",
        "manufacturer": "Estes",
        "designation": "C6",
        "total_impulse_Ns": 8.8,
        "burn_time_s": 1.85,
        "avg_thrust_N": 4.7,
        "max_thrust_N": 14.1,
        "propellant_mass_g": 12.6,
        "case_mass_g": 11.9,
        "thrust_curve": {
            "time_s": [0.0, 0.031, 0.15, 1.7, 1.85],
            "thrust_N": [0.0, 14.1, 8.0, 4.0, 0.0],
        },
    }


class TestRealisticMotorRocketCreation:
    """Test RealisticMotorRocket initialization."""

    def test_create_with_motor_config(self, airframe, motor_config):
        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
        )
        assert env is not None
        assert hasattr(env, "motor")
        env.close()

    def test_create_with_custom_rocket_config(self, airframe, motor_config):
        config = RocketConfig()
        config.max_tab_deflection = 25.0
        env = RealisticMotorRocket(
            airframe=airframe,
            motor_config=motor_config,
            config=config,
        )
        assert env is not None
        env.close()

    def test_create_without_airframe_raises(self, motor_config):
        with pytest.raises(ValueError, match="RocketAirframe is required"):
            RealisticMotorRocket(airframe=None, motor_config=motor_config)

    def test_create_without_motor_raises(self, airframe):
        with pytest.raises(ValueError, match="Must provide either"):
            RealisticMotorRocket(airframe=airframe)


class TestRealisticMotorRocketStepping:
    """Test environment stepping."""

    def test_reset(self, airframe, motor_config):
        env = RealisticMotorRocket(airframe=airframe, motor_config=motor_config)
        obs, info = env.reset()
        assert obs is not None
        assert len(obs) == 10
        assert "roll_rate_deg_s" in info
        env.close()

    def test_step(self, airframe, motor_config):
        env = RealisticMotorRocket(airframe=airframe, motor_config=motor_config)
        obs, info = env.reset()
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert obs is not None
        assert isinstance(reward, float)
        env.close()

    def test_full_episode(self, airframe, motor_config):
        env = RealisticMotorRocket(airframe=airframe, motor_config=motor_config)
        obs, info = env.reset()
        steps = 0
        while True:
            action = np.array([0.0])
            obs, reward, terminated, truncated, info = env.step(action)
            steps += 1
            if terminated or truncated:
                break
        assert steps > 10
        assert info.get("max_altitude_m", 0) > 0
        env.close()

    def test_info_has_motor_fields(self, airframe, motor_config):
        env = RealisticMotorRocket(airframe=airframe, motor_config=motor_config)
        obs, info = env.reset()
        action = np.array([0.0])
        obs, reward, terminated, truncated, info = env.step(action)
        assert "motor" in info
        assert "current_thrust_N" in info
        assert "propellant_remaining_g" in info
        env.close()

    def test_propulsion_during_burn(self, airframe, motor_config):
        env = RealisticMotorRocket(airframe=airframe, motor_config=motor_config)
        obs, info = env.reset()
        # Step a few times during burn phase
        for _ in range(10):
            obs, reward, terminated, truncated, info = env.step(np.array([0.0]))
        # During burn, thrust should be positive
        assert info["current_thrust_N"] >= 0
        env.close()
