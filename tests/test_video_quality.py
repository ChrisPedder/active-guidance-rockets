"""Tests for Gyroflow video quality metric."""

import numpy as np
import pytest

from controllers.video_quality_metric import (
    CameraPreset,
    EpisodeVideoQuality,
    FrameQuality,
    GyroflowVideoMetric,
    QualityThresholds,
    print_video_quality_table,
)

# =========================================================================
# Camera presets
# =========================================================================


class TestCameraPresets:
    def test_runcam_1080p60_defaults(self):
        cam = CameraPreset.runcam_1080p60()
        assert cam.fps == 60.0
        assert cam.exposure_time_s == pytest.approx(1 / 120.0)
        assert cam.fov_diagonal_deg == 120.0
        assert cam.rolling_shutter_ms == 12.0
        assert cam.name == "RunCam 1080p60"

    def test_runcam_4k30_defaults(self):
        cam = CameraPreset.runcam_4k30()
        assert cam.fps == 30.0
        assert cam.exposure_time_s == pytest.approx(1 / 60.0)
        assert cam.rolling_shutter_ms == 16.0

    def test_runcam_1080p120_defaults(self):
        cam = CameraPreset.runcam_1080p120()
        assert cam.fps == 120.0
        assert cam.exposure_time_s == pytest.approx(1 / 240.0)
        assert cam.rolling_shutter_ms == 8.0

    def test_all_presets_returns_three(self):
        presets = CameraPreset.all_presets()
        assert len(presets) == 3
        names = {p.name for p in presets}
        assert "RunCam 1080p60" in names
        assert "RunCam 4K30" in names
        assert "RunCam 1080p120" in names


# =========================================================================
# Motion blur
# =========================================================================


class TestMotionBlur:
    def setup_method(self):
        self.cam = CameraPreset.runcam_1080p60()
        self.metric = GyroflowVideoMetric(self.cam)

    def test_zero_spin_zero_blur(self):
        q = self.metric.compute_frame_quality(0.0)
        assert q.motion_blur_deg == 0.0

    def test_formula_20_deg_s(self):
        # 1/120 s * 20 deg/s = 0.16667 deg
        q = self.metric.compute_frame_quality(20.0)
        assert q.motion_blur_deg == pytest.approx(20.0 / 120.0, abs=1e-6)

    def test_proportional_to_rate(self):
        q10 = self.metric.compute_frame_quality(10.0)
        q20 = self.metric.compute_frame_quality(20.0)
        assert q20.motion_blur_deg == pytest.approx(2.0 * q10.motion_blur_deg, rel=1e-6)

    def test_proportional_to_exposure(self):
        cam_fast = CameraPreset(
            name="fast",
            fps=60,
            exposure_time_s=1 / 240.0,
            fov_diagonal_deg=120.0,
            rolling_shutter_ms=12.0,
        )
        m_fast = GyroflowVideoMetric(cam_fast)
        q_normal = self.metric.compute_frame_quality(30.0)
        q_fast = m_fast.compute_frame_quality(30.0)
        # Half exposure → half blur
        assert q_fast.motion_blur_deg == pytest.approx(
            q_normal.motion_blur_deg / 2.0, rel=1e-6
        )

    def test_negative_spin_uses_absolute(self):
        q_pos = self.metric.compute_frame_quality(15.0)
        q_neg = self.metric.compute_frame_quality(-15.0)
        assert q_neg.motion_blur_deg == pytest.approx(q_pos.motion_blur_deg)


# =========================================================================
# Rolling shutter
# =========================================================================


class TestRollingShutter:
    def setup_method(self):
        self.cam = CameraPreset.runcam_1080p60()
        self.metric = GyroflowVideoMetric(self.cam)

    def test_zero_spin_zero_residual(self):
        q = self.metric.compute_frame_quality(0.0)
        assert q.rolling_shutter_residual_deg == 0.0

    def test_formula_verification(self):
        # readout=12ms, rate=20 deg/s, residual_fraction=0.05
        # → 0.012 * 20 * 0.05 = 0.012 deg
        q = self.metric.compute_frame_quality(20.0)
        expected = 0.012 * 20.0 * 0.05
        assert q.rolling_shutter_residual_deg == pytest.approx(expected, abs=1e-8)

    def test_configurable_residual_fraction(self):
        m_high = GyroflowVideoMetric(self.cam, rolling_shutter_residual_fraction=0.10)
        q_default = self.metric.compute_frame_quality(20.0)
        q_high = m_high.compute_frame_quality(20.0)
        assert q_high.rolling_shutter_residual_deg == pytest.approx(
            2.0 * q_default.rolling_shutter_residual_deg, rel=1e-6
        )


# =========================================================================
# FoV crop
# =========================================================================


class TestFoVCrop:
    def setup_method(self):
        self.cam = CameraPreset.runcam_1080p60()
        self.metric = GyroflowVideoMetric(self.cam)

    def test_zero_spin_zero_crop(self):
        q = self.metric.compute_frame_quality(0.0)
        assert q.fov_crop_pct == 0.0

    def test_formula_verification(self):
        # rate=20, fps=60, fov=120 → (20/60)/120 * 100 = 0.2778 %
        q = self.metric.compute_frame_quality(20.0)
        expected = (20.0 / 60.0) / 120.0 * 100.0
        assert q.fov_crop_pct == pytest.approx(expected, abs=1e-6)

    def test_higher_fps_less_crop(self):
        cam120 = CameraPreset.runcam_1080p120()
        m120 = GyroflowVideoMetric(cam120)
        q60 = self.metric.compute_frame_quality(30.0)
        q120 = m120.compute_frame_quality(30.0)
        assert q120.fov_crop_pct < q60.fov_crop_pct


# =========================================================================
# Composite score
# =========================================================================


class TestCompositeScore:
    def setup_method(self):
        self.cam = CameraPreset.runcam_1080p60()
        self.metric = GyroflowVideoMetric(self.cam)

    def test_zero_spin_perfect_score(self):
        q = self.metric.compute_frame_quality(0.0)
        assert q.composite_score == pytest.approx(1.0)

    def test_bounded_0_1(self):
        for rate in [0, 5, 10, 20, 50, 100, 500, 1000]:
            q = self.metric.compute_frame_quality(rate)
            assert 0.0 <= q.composite_score <= 1.0

    def test_decreases_with_spin(self):
        q5 = self.metric.compute_frame_quality(5.0)
        q50 = self.metric.compute_frame_quality(50.0)
        assert q50.composite_score < q5.composite_score

    def test_weights_sum_to_one(self):
        # weights: 0.5 + 0.3 + 0.2 = 1.0
        # At zero spin all sub-scores are 1.0, so composite = 1.0
        q = self.metric.compute_frame_quality(0.0)
        assert q.composite_score == pytest.approx(1.0)

    def test_extreme_spin_floors_to_zero(self):
        # Very high rate should drive all sub-scores to 0
        q = self.metric.compute_frame_quality(100000.0)
        assert q.composite_score == pytest.approx(0.0, abs=1e-6)


# =========================================================================
# Episode evaluation
# =========================================================================


class TestEpisodeEvaluation:
    def setup_method(self):
        self.cam = CameraPreset.runcam_1080p60()
        self.metric = GyroflowVideoMetric(self.cam)

    def test_constant_rate(self):
        # 5 seconds at 10 deg/s, dt=0.01 → 500 samples
        rates = np.full(500, 10.0)
        eq = self.metric.evaluate_episode(rates, dt=0.01)
        # All frames should have identical blur
        assert eq.mean_blur_deg == pytest.approx(eq.max_blur_deg, rel=1e-3)
        assert eq.verdict in ("Excellent", "Good", "Acceptable", "Poor")

    def test_varying_rate(self):
        # Ramp from 0 to 60 deg/s over 5 s
        rates = np.linspace(0, 60, 500)
        eq = self.metric.evaluate_episode(rates, dt=0.01)
        assert eq.max_blur_deg > eq.mean_blur_deg
        assert eq.min_composite < eq.mean_composite

    def test_resampling_100hz_to_60fps(self):
        # 10 s episode at 100 Hz → 1000 samples → should produce ~600 frames
        rates = np.full(1000, 5.0)
        eq = self.metric.evaluate_episode(rates, dt=0.01)
        # Check that evaluation doesn't crash and gives sensible results
        assert (
            eq.pct_excellent + eq.pct_good + eq.pct_acceptable + eq.pct_poor
            == pytest.approx(100.0, abs=0.1)
        )

    def test_short_episode(self):
        # Only 5 samples at 0.01s → 0.05s → ~3 frames at 60fps
        rates = np.full(5, 2.0)
        eq = self.metric.evaluate_episode(rates, dt=0.01)
        assert eq.verdict in ("Excellent", "Good", "Acceptable", "Poor")

    def test_empty_episode(self):
        rates = np.array([])
        eq = self.metric.evaluate_episode(rates, dt=0.01)
        assert eq.verdict == "Excellent"
        assert eq.mean_composite == 1.0

    def test_percentages_sum_to_100(self):
        rates = np.random.uniform(0, 50, 500)
        eq = self.metric.evaluate_episode(rates, dt=0.01)
        total = eq.pct_excellent + eq.pct_good + eq.pct_acceptable + eq.pct_poor
        assert total == pytest.approx(100.0, abs=0.1)


# =========================================================================
# Verdict thresholds
# =========================================================================


class TestVerdict:
    def setup_method(self):
        self.cam = CameraPreset.runcam_1080p60()
        self.metric = GyroflowVideoMetric(self.cam)

    def test_5_deg_s_excellent(self):
        q = self.metric.compute_frame_quality(5.0)
        verdict = self.metric._verdict_from_composite(q.composite_score)
        assert verdict == "Excellent"

    def test_20_deg_s_excellent(self):
        # Key hypothesis: 20 deg/s should still be excellent at 1080p60
        q = self.metric.compute_frame_quality(20.0)
        verdict = self.metric._verdict_from_composite(q.composite_score)
        assert verdict == "Excellent"

    def test_very_high_spin_poor(self):
        q = self.metric.compute_frame_quality(500.0)
        verdict = self.metric._verdict_from_composite(q.composite_score)
        assert verdict == "Poor"


# =========================================================================
# Expected values (key hypothesis tests)
# =========================================================================


class TestExpectedValues:
    def test_20_deg_s_1080p60_watchable(self):
        """The key hypothesis: 20 deg/s at 1080p60 produces watchable video."""
        cam = CameraPreset.runcam_1080p60()
        m = GyroflowVideoMetric(cam)
        q = m.compute_frame_quality(20.0)
        # Should be Excellent or Good
        assert (
            q.composite_score >= 0.65
        ), f"20 deg/s at 1080p60 should be at least Good, got composite={q.composite_score:.3f}"

    def test_30_deg_s_4k30_acceptable(self):
        """30 deg/s at 4K30 should be at least acceptable."""
        cam = CameraPreset.runcam_4k30()
        m = GyroflowVideoMetric(cam)
        q = m.compute_frame_quality(30.0)
        assert (
            q.composite_score >= 0.40
        ), f"30 deg/s at 4K30 should be at least Acceptable, got composite={q.composite_score:.3f}"

    def test_blur_at_20_deg_s_small(self):
        """At 20 deg/s 1080p60, motion blur should be < 0.2 deg."""
        cam = CameraPreset.runcam_1080p60()
        m = GyroflowVideoMetric(cam)
        q = m.compute_frame_quality(20.0)
        assert q.motion_blur_deg < 0.2

    def test_crop_at_20_deg_s_small(self):
        """At 20 deg/s 1080p60, FoV crop should be < 1%."""
        cam = CameraPreset.runcam_1080p60()
        m = GyroflowVideoMetric(cam)
        q = m.compute_frame_quality(20.0)
        assert q.fov_crop_pct < 1.0

    def test_rs_at_20_deg_s_negligible(self):
        """At 20 deg/s 1080p60, rolling shutter residual should be < 0.1 deg."""
        cam = CameraPreset.runcam_1080p60()
        m = GyroflowVideoMetric(cam)
        q = m.compute_frame_quality(20.0)
        assert q.rolling_shutter_residual_deg < 0.1


# =========================================================================
# Custom thresholds
# =========================================================================


class TestQualityThresholds:
    def test_custom_thresholds_change_verdict(self):
        cam = CameraPreset.runcam_1080p60()
        # Very strict thresholds
        strict = QualityThresholds(
            excellent_min=0.99,
            good_min=0.95,
            acceptable_min=0.90,
        )
        m = GyroflowVideoMetric(cam, thresholds=strict)
        q = m.compute_frame_quality(20.0)
        # With strict thresholds, 20 deg/s might not be Excellent
        verdict = m._verdict_from_composite(q.composite_score)
        # Composite ~0.96 for 20 deg/s, so with strict thresholds should be Good
        assert verdict in ("Good", "Acceptable", "Poor")

    def test_default_thresholds(self):
        th = QualityThresholds()
        assert th.excellent_min == 0.85
        assert th.good_min == 0.65
        assert th.acceptable_min == 0.40
        assert th.blur_ceiling_deg == 2.0
        assert th.rs_ceiling_deg == 1.0
        assert th.crop_ceiling_pct == 10.0

    def test_custom_blur_ceiling(self):
        cam = CameraPreset.runcam_1080p60()
        # Low ceiling → worse scores at same rate
        tight = QualityThresholds(blur_ceiling_deg=0.5)
        loose = QualityThresholds(blur_ceiling_deg=5.0)
        m_tight = GyroflowVideoMetric(cam, thresholds=tight)
        m_loose = GyroflowVideoMetric(cam, thresholds=loose)
        q_tight = m_tight.compute_frame_quality(20.0)
        q_loose = m_loose.compute_frame_quality(20.0)
        assert q_tight.composite_score < q_loose.composite_score


# =========================================================================
# Integration / print tests
# =========================================================================


class TestIntegration:
    def test_print_summary_no_crash(self, capsys):
        cam = CameraPreset.runcam_1080p60()
        m = GyroflowVideoMetric(cam)
        rates = np.full(500, 10.0)
        eq = m.evaluate_episode(rates, dt=0.01)
        m.print_summary([eq], "TestPID", 2.0)
        out = capsys.readouterr().out
        assert "TestPID" in out

    def test_batch_evaluation(self):
        cam = CameraPreset.runcam_1080p60()
        m = GyroflowVideoMetric(cam)
        series = [np.full(500, r) for r in [5, 10, 20]]
        results = m.evaluate_episodes(series, dt=0.01)
        assert len(results) == 3
        # Lower rate → better verdict
        assert results[0].mean_composite >= results[2].mean_composite

    def test_print_video_quality_table_no_crash(self, capsys):
        """Ensure the table printer doesn't crash with mock data."""
        from dataclasses import dataclass, field
        from typing import List

        @dataclass
        class MockEpisode:
            mean_spin_rate: float = 10.0
            spin_rate_series: np.ndarray = None

        @dataclass
        class MockResult:
            controller_name: str = "PID"
            wind_speed: float = 0.0
            episodes: list = None

        ep1 = MockEpisode(spin_rate_series=np.full(500, 10.0))
        ep2 = MockEpisode(spin_rate_series=np.full(500, 20.0))
        r0 = MockResult(wind_speed=0.0, episodes=[ep1, ep2])
        r1 = MockResult(wind_speed=3.0, episodes=[ep1, ep2])

        all_results = {"PID (IMU)": [r0, r1]}
        print_video_quality_table(
            all_results, dt=0.01, camera_presets=[CameraPreset.runcam_1080p60()]
        )
        out = capsys.readouterr().out
        assert "VIDEO QUALITY" in out
        assert "PID (IMU)" in out

    def test_print_table_missing_series(self, capsys):
        """Table should handle episodes without spin_rate_series gracefully."""
        from dataclasses import dataclass

        @dataclass
        class MockEpisode:
            mean_spin_rate: float = 10.0

        @dataclass
        class MockResult:
            controller_name: str = "PID"
            wind_speed: float = 0.0
            episodes: list = None

        ep = MockEpisode()
        r0 = MockResult(wind_speed=0.0, episodes=[ep])
        all_results = {"PID": [r0]}
        print_video_quality_table(
            all_results, dt=0.01, camera_presets=[CameraPreset.runcam_1080p60()]
        )
        out = capsys.readouterr().out
        assert "N/A" in out
