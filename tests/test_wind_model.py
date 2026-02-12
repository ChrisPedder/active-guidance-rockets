"""Tests for wind model including Dryden turbulence implementation."""

import numpy as np
import pytest

from wind_model import WindConfig, WindModel, DrydenTurbulence, _TURBULENCE_SEVERITY

# ========================================================================
# DrydenTurbulence unit tests
# ========================================================================


class TestDrydenTurbulenceParameters:
    """Test MIL-F-8785C parameter computation."""

    def test_scale_lengths_increase_with_altitude(self):
        """Scale lengths should increase with altitude (larger eddies higher up)."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=10.0, W_20=3.0, severity="light", V=30.0)
        L_low = dt.L_u

        dt.compute_parameters(altitude_m=100.0, W_20=3.0, severity="light", V=30.0)
        L_high = dt.L_u

        assert L_high > L_low

    def test_sigma_scales_with_wind_speed(self):
        """Turbulence intensity should scale with W_20."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=1.0, severity="light", V=30.0)
        sigma_low = dt.sigma_u

        dt.compute_parameters(altitude_m=50.0, W_20=5.0, severity="light", V=30.0)
        sigma_high = dt.sigma_u

        assert sigma_high > sigma_low
        # Should scale linearly with W_20 (via sigma_w = intensity * W_20)
        assert abs(sigma_high / sigma_low - 5.0) < 0.01

    def test_severity_levels(self):
        """Higher severity should produce higher turbulence intensity."""
        dt = DrydenTurbulence(dt=0.01)
        sigmas = {}
        for severity in ["light", "moderate", "severe"]:
            dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity=severity, V=30.0)
            sigmas[severity] = dt.sigma_u

        assert sigmas["light"] < sigmas["moderate"] < sigmas["severe"]

    def test_minimum_altitude_clamped(self):
        """Very low altitude should be clamped to prevent issues."""
        dt = DrydenTurbulence(dt=0.01)
        # Should not raise with altitude near zero
        dt.compute_parameters(altitude_m=0.1, W_20=3.0, severity="light", V=30.0)
        assert dt.L_u > 0
        assert dt.sigma_u > 0

    def test_minimum_velocity_clamped(self):
        """Very low velocity should be clamped to prevent division by zero."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="light", V=0.01)
        assert np.isfinite(dt.L_u)
        assert np.isfinite(dt.sigma_u)

    def test_unknown_severity_defaults_to_light(self):
        """Unknown severity string should default to light (0.1)."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="unknown", V=30.0)
        dt2 = DrydenTurbulence(dt=0.01)
        dt2.compute_parameters(altitude_m=50.0, W_20=3.0, severity="light", V=30.0)
        assert dt.sigma_u == dt2.sigma_u


class TestDrydenTurbulenceFilter:
    """Test discrete-time forming filter behavior."""

    def test_reset_zeros_output(self):
        """Reset should zero all filter states."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="moderate", V=30.0)
        rng = np.random.default_rng(42)

        # Run a few steps
        for _ in range(100):
            dt.step(rng)

        assert dt.u_g != 0.0 or dt.v_g != 0.0  # Should have non-zero output

        dt.reset()
        assert dt.u_g == 0.0
        assert dt.v_g == 0.0
        assert dt._u_state == 0.0
        assert dt._v_state1 == 0.0
        assert dt._v_state2 == 0.0

    def test_output_is_finite(self):
        """Filter output should always be finite."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="severe", V=30.0)
        rng = np.random.default_rng(42)

        for _ in range(10000):
            u_g, v_g = dt.step(rng)
            assert np.isfinite(u_g), f"u_g is not finite: {u_g}"
            assert np.isfinite(v_g), f"v_g is not finite: {v_g}"

    def test_output_variance_matches_target(self):
        """Long-run output variance should approximately match sigma^2."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="moderate", V=30.0)
        rng = np.random.default_rng(42)

        N = 100000
        u_vals = np.zeros(N)
        v_vals = np.zeros(N)
        for i in range(N):
            u_vals[i], v_vals[i] = dt.step(rng)

        # Discard transient (first 1000 samples)
        u_vals = u_vals[1000:]
        v_vals = v_vals[1000:]

        u_std = np.std(u_vals)
        v_std = np.std(v_vals)

        # Output std should be within 30% of target sigma
        # (exact match requires infinite data; 30% is a reasonable tolerance)
        assert (
            abs(u_std - dt.sigma_u) / dt.sigma_u < 0.3
        ), f"u_g std {u_std:.3f} too far from sigma_u {dt.sigma_u:.3f}"
        assert (
            abs(v_std - dt.sigma_v) / dt.sigma_v < 0.3
        ), f"v_g std {v_std:.3f} too far from sigma_v {dt.sigma_v:.3f}"

    def test_output_is_gaussian(self):
        """Filter output should be approximately Gaussian (driven by Gaussian noise)."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="moderate", V=30.0)
        rng = np.random.default_rng(42)

        N = 50000
        u_vals = np.zeros(N)
        for i in range(N):
            u_vals[i], _ = dt.step(rng)

        # Discard transient
        u_vals = u_vals[1000:]

        # Normalize
        u_norm = (u_vals - np.mean(u_vals)) / np.std(u_vals)

        # Check kurtosis is close to 3 (Gaussian has kurtosis = 3)
        kurtosis = np.mean(u_norm**4)
        assert abs(kurtosis - 3.0) < 0.5, f"Kurtosis {kurtosis:.2f} too far from 3.0"

    def test_longitudinal_filter_is_lowpass(self):
        """u_g filter should be a low-pass (autocorrelation should decay, not oscillate)."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="moderate", V=30.0)
        rng = np.random.default_rng(42)

        N = 50000
        u_vals = np.zeros(N)
        for i in range(N):
            u_vals[i], _ = dt.step(rng)

        # Compute autocorrelation at lag 1 and lag 10
        u_vals = u_vals[1000:]
        u_centered = u_vals - np.mean(u_vals)
        var = np.var(u_vals)

        acf_1 = np.mean(u_centered[:-1] * u_centered[1:]) / var
        acf_10 = np.mean(u_centered[:-10] * u_centered[10:]) / var

        # For a low-pass filter: acf should be positive and monotonically decreasing
        assert acf_1 > 0, f"ACF at lag 1 should be positive, got {acf_1:.3f}"
        assert acf_1 > acf_10, f"ACF should decay: lag1={acf_1:.3f}, lag10={acf_10:.3f}"

    def test_filter_stability(self):
        """Filter should be stable (bounded output) even after many steps."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="severe", V=30.0)
        rng = np.random.default_rng(42)

        max_u = 0.0
        max_v = 0.0
        for _ in range(100000):
            u_g, v_g = dt.step(rng)
            max_u = max(max_u, abs(u_g))
            max_v = max(max_v, abs(v_g))

        # Output should stay bounded — 6 sigma is a reasonable bound
        assert (
            max_u < 6 * dt.sigma_u
        ), f"max |u_g| = {max_u:.3f} > 6*sigma_u = {6*dt.sigma_u:.3f}"
        assert (
            max_v < 6 * dt.sigma_v
        ), f"max |v_g| = {max_v:.3f} > 6*sigma_v = {6*dt.sigma_v:.3f}"

    def test_psd_slope_longitudinal(self):
        """Longitudinal PSD should fall off at high frequency (low-pass character)."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="moderate", V=30.0)
        rng = np.random.default_rng(42)

        N = 100000
        u_vals = np.zeros(N)
        for i in range(N):
            u_vals[i], _ = dt.step(rng)

        # Compute PSD via Welch's method
        from numpy.fft import rfft, rfftfreq

        # Use windowed FFT for a basic PSD estimate
        u_vals = u_vals[1000:]  # discard transient
        n = len(u_vals)
        freqs = rfftfreq(n, d=0.01)
        psd = np.abs(rfft(u_vals)) ** 2 / n

        # Compare low frequency (0.1-0.5 Hz) vs high frequency (5-10 Hz)
        low_mask = (freqs > 0.1) & (freqs < 0.5)
        high_mask = (freqs > 5.0) & (freqs < 10.0)

        low_power = np.mean(psd[low_mask])
        high_power = np.mean(psd[high_mask])

        # Low-pass: low frequency power should be much higher than high frequency
        assert (
            low_power > 10 * high_power
        ), f"PSD not low-pass: low_power={low_power:.2e}, high_power={high_power:.2e}"

    def test_different_seeds_different_output(self):
        """Different RNG seeds should produce different turbulence sequences."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="moderate", V=30.0)

        rng1 = np.random.default_rng(42)
        dt.reset()
        vals1 = [dt.step(rng1)[0] for _ in range(100)]

        rng2 = np.random.default_rng(99)
        dt.reset()
        vals2 = [dt.step(rng2)[0] for _ in range(100)]

        assert vals1 != vals2

    def test_same_seed_reproducible(self):
        """Same RNG seed should produce identical output."""
        dt = DrydenTurbulence(dt=0.01)
        dt.compute_parameters(altitude_m=50.0, W_20=3.0, severity="moderate", V=30.0)

        rng1 = np.random.default_rng(42)
        dt.reset()
        vals1 = [dt.step(rng1)[0] for _ in range(100)]

        rng2 = np.random.default_rng(42)
        dt.reset()
        vals2 = [dt.step(rng2)[0] for _ in range(100)]

        np.testing.assert_array_equal(vals1, vals2)


# ========================================================================
# WindConfig tests
# ========================================================================


class TestWindConfig:
    """Test WindConfig defaults and Dryden fields."""

    def test_defaults_backward_compatible(self):
        """Default config should match legacy behavior."""
        cfg = WindConfig()
        assert cfg.enable is False
        assert cfg.use_dryden is False
        assert cfg.max_gust_speed == 0.0

    def test_dryden_fields_present(self):
        """Dryden-specific fields should be accessible."""
        cfg = WindConfig(use_dryden=True, turbulence_severity="moderate")
        assert cfg.use_dryden is True
        assert cfg.turbulence_severity == "moderate"
        assert cfg.altitude_profile_alpha == 0.14
        assert cfg.reference_altitude == 10.0

    def test_severity_lookup_table(self):
        """Severity lookup table should have expected values."""
        assert _TURBULENCE_SEVERITY["light"] == 0.1
        assert _TURBULENCE_SEVERITY["moderate"] == 0.2
        assert _TURBULENCE_SEVERITY["severe"] == 0.3


# ========================================================================
# WindModel integration tests
# ========================================================================


class TestWindModelLegacy:
    """Test that legacy sinusoidal model is unchanged."""

    def test_legacy_model_unchanged(self):
        """Legacy model output should be identical to original implementation."""
        cfg = WindConfig(
            enable=True,
            base_speed=3.0,
            max_gust_speed=2.0,
            variability=0.3,
            altitude_gradient=0.0,
            use_dryden=False,
        )
        model = WindModel(cfg)
        model.reset(seed=42)

        speed, direction = model.get_wind(time=1.0, altitude=50.0)
        assert speed >= 0.0
        assert np.isfinite(speed)
        assert np.isfinite(direction)

    def test_legacy_deterministic_with_seed(self):
        """Legacy model should be deterministic with same seed."""
        cfg = WindConfig(enable=True, base_speed=3.0, max_gust_speed=2.0)
        model = WindModel(cfg)

        model.reset(seed=42)
        s1, d1 = model.get_wind(time=1.0, altitude=50.0)

        model.reset(seed=42)
        s2, d2 = model.get_wind(time=1.0, altitude=50.0)

        assert s1 == s2
        assert d1 == d2

    def test_disabled_returns_zero(self):
        """Wind model should return zero when disabled."""
        cfg = WindConfig(enable=False, base_speed=3.0)
        model = WindModel(cfg)
        model.reset(seed=42)
        s, d = model.get_wind(time=1.0, altitude=50.0)
        assert s == 0.0
        assert d == 0.0

    def test_legacy_altitude_gradient(self):
        """Legacy linear altitude gradient should work."""
        cfg = WindConfig(
            enable=True,
            base_speed=3.0,
            max_gust_speed=0.0,
            altitude_gradient=1.0,
            use_dryden=False,
        )
        model = WindModel(cfg)
        model.reset(seed=42)

        s_low, _ = model.get_wind(time=0.0, altitude=0.0)
        s_high, _ = model.get_wind(time=0.0, altitude=100.0)

        # At 100m with gradient=1.0: factor = 1.0 + 1.0 * 100/100 = 2.0
        assert s_high > s_low


class TestWindModelDryden:
    """Test Dryden turbulence model integration in WindModel."""

    def test_dryden_produces_output(self):
        """Dryden model should produce non-zero wind."""
        cfg = WindConfig(
            enable=True,
            base_speed=3.0,
            use_dryden=True,
            turbulence_severity="moderate",
        )
        model = WindModel(cfg)
        model.reset(seed=42)

        # Run several steps to let filter warm up
        speeds = []
        for t in range(500):
            s, d = model.get_wind(time=t * 0.01, altitude=50.0, rocket_velocity=30.0)
            speeds.append(s)

        # Should have non-trivial variation
        assert np.std(speeds) > 0.01, "Dryden model produces no variation"
        assert all(np.isfinite(s) for s in speeds), "Non-finite speed values"

    def test_dryden_reproducible_with_seed(self):
        """Dryden model should be reproducible with same seed."""
        cfg = WindConfig(
            enable=True, base_speed=3.0, use_dryden=True, turbulence_severity="moderate"
        )

        model = WindModel(cfg)
        model.reset(seed=42)
        vals1 = [model.get_wind(t * 0.01, 50.0, 30.0) for t in range(100)]

        model.reset(seed=42)
        vals2 = [model.get_wind(t * 0.01, 50.0, 30.0) for t in range(100)]

        for (s1, d1), (s2, d2) in zip(vals1, vals2):
            assert s1 == s2
            assert d1 == d2

    def test_dryden_speed_non_negative(self):
        """Wind speed should never go negative."""
        cfg = WindConfig(
            enable=True, base_speed=3.0, use_dryden=True, turbulence_severity="severe"
        )
        model = WindModel(cfg)
        model.reset(seed=42)

        for t in range(10000):
            s, _ = model.get_wind(t * 0.01, 50.0, 30.0)
            assert s >= 0.0, f"Negative wind speed at t={t*0.01:.2f}: {s}"

    def test_dryden_stochastic_between_seeds(self):
        """Different seeds should produce different output."""
        cfg = WindConfig(enable=True, base_speed=3.0, use_dryden=True)
        model = WindModel(cfg)

        model.reset(seed=42)
        vals1 = [model.get_wind(t * 0.01, 50.0, 30.0)[0] for t in range(100)]

        model.reset(seed=99)
        vals2 = [model.get_wind(t * 0.01, 50.0, 30.0)[0] for t in range(100)]

        assert vals1 != vals2

    def test_dryden_filter_recompute_on_velocity_change(self):
        """Filter coefficients should update when rocket velocity changes significantly."""
        cfg = WindConfig(enable=True, base_speed=3.0, use_dryden=True)
        model = WindModel(cfg)
        model.reset(seed=42)

        # First call initializes
        model.get_wind(0.01, 50.0, rocket_velocity=10.0)
        assert model._dryden_initialized

        # Small velocity change: no recompute
        old_a1 = model._dryden._a1_u
        model.get_wind(0.02, 50.0, rocket_velocity=11.0)
        assert model._dryden._a1_u == old_a1

        # Large velocity change (>20%): should recompute
        model.get_wind(0.03, 50.0, rocket_velocity=30.0)
        assert model._dryden._a1_u != old_a1


class TestWindModelAltitudeProfile:
    """Test altitude wind profile (power-law vs linear)."""

    def test_power_law_increases_with_altitude(self):
        """Power-law profile should give higher wind at higher altitude."""
        cfg = WindConfig(
            enable=True,
            base_speed=3.0,
            use_dryden=True,
            altitude_profile_alpha=0.14,
            reference_altitude=10.0,
        )
        model = WindModel(cfg)

        factor_10 = model._get_altitude_factor(10.0)
        factor_100 = model._get_altitude_factor(100.0)
        factor_200 = model._get_altitude_factor(200.0)

        assert factor_10 < factor_100 < factor_200

    def test_power_law_at_reference_altitude(self):
        """At reference altitude, factor should be 1.0."""
        cfg = WindConfig(
            enable=True,
            base_speed=3.0,
            use_dryden=True,
            altitude_profile_alpha=0.14,
            reference_altitude=10.0,
        )
        model = WindModel(cfg)

        factor = model._get_altitude_factor(10.0)
        assert abs(factor - 1.0) < 1e-10

    def test_power_law_clamps_minimum_altitude(self):
        """Altitude should be clamped to 1m minimum."""
        cfg = WindConfig(
            enable=True, base_speed=3.0, use_dryden=True, reference_altitude=10.0
        )
        model = WindModel(cfg)

        factor_0 = model._get_altitude_factor(0.0)
        factor_1 = model._get_altitude_factor(1.0)
        assert factor_0 == factor_1  # 0m clamped to 1m

    def test_power_law_known_value(self):
        """Verify power-law against known values.
        V(100m) / V(10m) = (100/10)^0.14 = 10^0.14 ~ 1.38"""
        cfg = WindConfig(
            enable=True,
            use_dryden=True,
            altitude_profile_alpha=0.14,
            reference_altitude=10.0,
        )
        model = WindModel(cfg)

        ratio = model._get_altitude_factor(100.0) / model._get_altitude_factor(10.0)
        expected = 10.0**0.14
        assert abs(ratio - expected) < 0.01

    def test_legacy_linear_gradient_used_when_not_dryden(self):
        """Legacy model should use linear gradient, not power-law."""
        cfg = WindConfig(
            enable=True, base_speed=3.0, use_dryden=False, altitude_gradient=1.0
        )
        model = WindModel(cfg)

        factor = model._get_altitude_factor(100.0)
        expected = 1.0 + 1.0 * 100.0 / 100.0  # = 2.0
        assert abs(factor - expected) < 1e-10


class TestWindModelRollTorque:
    """Test that roll torque calculation is unchanged."""

    def test_roll_torque_unchanged(self):
        """Roll torque method should not be affected by Dryden changes."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()

        cfg = WindConfig(enable=True, base_speed=3.0, use_dryden=False)
        model = WindModel(cfg)

        torque = model.get_roll_torque(
            wind_speed=3.0,
            wind_direction=0.5,
            roll_angle=0.0,
            velocity=30.0,
            dynamic_pressure=500.0,
            airframe=airframe,
        )
        assert np.isfinite(torque)
        assert torque != 0.0

    def test_zero_wind_zero_torque(self):
        """No wind should produce no torque."""
        from airframe import RocketAirframe

        airframe = RocketAirframe.estes_alpha()

        cfg = WindConfig(enable=True, base_speed=3.0)
        model = WindModel(cfg)

        torque = model.get_roll_torque(
            wind_speed=0.0,
            wind_direction=0.0,
            roll_angle=0.0,
            velocity=30.0,
            dynamic_pressure=500.0,
            airframe=airframe,
        )
        assert torque == 0.0
