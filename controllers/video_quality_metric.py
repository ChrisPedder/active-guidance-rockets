#!/usr/bin/env python3
"""
Gyroflow Video Quality Metric

Models post-processing video quality after gyroscope-based stabilization
(e.g., Gyroflow). Given a spin rate time series from the simulation,
computes per-frame quality metrics:

- Motion blur: exposure_time * |spin_rate| (degrees of blur per frame)
- Rolling shutter residual: readout_time * |spin_rate| * residual_fraction
- FoV crop: percentage of frame cropped for stabilization

These are combined into a composite score and a human-readable verdict.

Usage:
    # Standalone: evaluate specific spin rates
    uv run python video_quality_metric.py --spin-rate 5 10 20 30 50

    # Integrated with compare_controllers.py via --video-quality flag
"""

import argparse
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class CameraPreset:
    """Camera configuration for video quality analysis."""

    name: str
    fps: float  # frames per second
    exposure_time_s: float  # shutter speed (seconds)
    fov_diagonal_deg: float  # diagonal field of view (degrees)
    rolling_shutter_ms: float  # sensor readout time (milliseconds)

    @classmethod
    def runcam_1080p60(cls) -> "CameraPreset":
        return cls(
            name="RunCam 1080p60",
            fps=60.0,
            exposure_time_s=1.0 / 120.0,  # typical 180-degree shutter
            fov_diagonal_deg=120.0,
            rolling_shutter_ms=12.0,
        )

    @classmethod
    def runcam_4k30(cls) -> "CameraPreset":
        return cls(
            name="RunCam 4K30",
            fps=30.0,
            exposure_time_s=1.0 / 60.0,
            fov_diagonal_deg=120.0,
            rolling_shutter_ms=16.0,
        )

    @classmethod
    def runcam_1080p120(cls) -> "CameraPreset":
        return cls(
            name="RunCam 1080p120",
            fps=120.0,
            exposure_time_s=1.0 / 240.0,
            fov_diagonal_deg=120.0,
            rolling_shutter_ms=8.0,
        )

    @classmethod
    def all_presets(cls) -> List["CameraPreset"]:
        return [cls.runcam_1080p60(), cls.runcam_4k30(), cls.runcam_1080p120()]


@dataclass
class QualityThresholds:
    """Thresholds for mapping composite score to verdict strings."""

    excellent_min: float = 0.85  # composite >= this → "Excellent"
    good_min: float = 0.65  # composite >= this → "Good"
    acceptable_min: float = 0.40  # composite >= this → "Acceptable"
    # below acceptable_min → "Poor"

    # Per-metric ceilings used to normalize scores to [0, 1]
    blur_ceiling_deg: float = 2.0  # motion blur above this → score 0
    rs_ceiling_deg: float = 1.0  # RS residual above this → score 0
    crop_ceiling_pct: float = 10.0  # FoV crop above this → score 0


@dataclass
class FrameQuality:
    """Quality metrics for a single video frame."""

    motion_blur_deg: float
    rolling_shutter_residual_deg: float
    fov_crop_pct: float
    composite_score: float  # 0 (worst) to 1 (best)


@dataclass
class EpisodeVideoQuality:
    """Aggregated video quality for one simulation episode."""

    mean_blur_deg: float
    max_blur_deg: float
    mean_rs_residual_deg: float
    max_rs_residual_deg: float
    mean_crop_pct: float
    max_crop_pct: float
    mean_composite: float
    min_composite: float
    pct_excellent: float  # % of frames rated Excellent
    pct_good: float  # % of frames rated Good
    pct_acceptable: float  # % of frames rated Acceptable
    pct_poor: float  # % of frames rated Poor
    verdict: str  # overall verdict (based on mean composite)


# ---------------------------------------------------------------------------
# Main metric class
# ---------------------------------------------------------------------------


class GyroflowVideoMetric:
    """Compute post-stabilization video quality from spin rate data."""

    def __init__(
        self,
        camera: CameraPreset,
        thresholds: Optional[QualityThresholds] = None,
        rolling_shutter_residual_fraction: float = 0.05,
    ):
        self.camera = camera
        self.thresholds = thresholds or QualityThresholds()
        self.rs_residual_fraction = rolling_shutter_residual_fraction

    # -- per-frame ---------------------------------------------------------

    def compute_frame_quality(self, spin_rate_deg_s: float) -> FrameQuality:
        """Compute quality metrics for a single frame at a given spin rate."""
        rate = abs(spin_rate_deg_s)

        # Motion blur: degrees of rotation during exposure
        motion_blur = self.camera.exposure_time_s * rate

        # Rolling shutter residual: readout time * rate * residual fraction
        readout_s = self.camera.rolling_shutter_ms / 1000.0
        rs_residual = readout_s * rate * self.rs_residual_fraction

        # FoV crop: rotation per frame / FoV * 100
        rotation_per_frame = rate / self.camera.fps
        fov_crop = (rotation_per_frame / self.camera.fov_diagonal_deg) * 100.0

        # Normalize each to [0, 1] (1 = perfect)
        blur_score = max(0.0, 1.0 - motion_blur / self.thresholds.blur_ceiling_deg)
        rs_score = max(0.0, 1.0 - rs_residual / self.thresholds.rs_ceiling_deg)
        crop_score = max(0.0, 1.0 - fov_crop / self.thresholds.crop_ceiling_pct)

        composite = 0.5 * blur_score + 0.3 * rs_score + 0.2 * crop_score

        return FrameQuality(
            motion_blur_deg=motion_blur,
            rolling_shutter_residual_deg=rs_residual,
            fov_crop_pct=fov_crop,
            composite_score=composite,
        )

    # -- per-episode -------------------------------------------------------

    def evaluate_episode(
        self,
        spin_rates_deg_s: np.ndarray,
        dt: float,
    ) -> EpisodeVideoQuality:
        """Evaluate video quality for one episode.

        Resamples from simulation rate (1/dt Hz) to camera fps, then
        computes per-frame metrics and aggregates.
        """
        n_sim = len(spin_rates_deg_s)
        if n_sim == 0:
            return self._empty_quality()

        sim_duration = n_sim * dt
        n_frames = max(1, int(np.floor(sim_duration * self.camera.fps)))

        # Time arrays
        sim_times = np.arange(n_sim) * dt
        frame_times = np.arange(n_frames) / self.camera.fps

        # Resample (clamp frame_times to sim range)
        frame_times = np.clip(frame_times, sim_times[0], sim_times[-1])
        frame_rates = np.interp(frame_times, sim_times, spin_rates_deg_s)

        # Per-frame quality
        qualities = [self.compute_frame_quality(r) for r in frame_rates]

        blurs = np.array([q.motion_blur_deg for q in qualities])
        rs = np.array([q.rolling_shutter_residual_deg for q in qualities])
        crops = np.array([q.fov_crop_pct for q in qualities])
        composites = np.array([q.composite_score for q in qualities])

        # Verdict counts
        th = self.thresholds
        n = len(composites)
        pct_excellent = np.sum(composites >= th.excellent_min) / n * 100.0
        pct_good = (
            np.sum((composites >= th.good_min) & (composites < th.excellent_min))
            / n
            * 100.0
        )
        pct_acceptable = (
            np.sum((composites >= th.acceptable_min) & (composites < th.good_min))
            / n
            * 100.0
        )
        pct_poor = np.sum(composites < th.acceptable_min) / n * 100.0

        mean_composite = float(np.mean(composites))
        verdict = self._verdict_from_composite(mean_composite)

        return EpisodeVideoQuality(
            mean_blur_deg=float(np.mean(blurs)),
            max_blur_deg=float(np.max(blurs)),
            mean_rs_residual_deg=float(np.mean(rs)),
            max_rs_residual_deg=float(np.max(rs)),
            mean_crop_pct=float(np.mean(crops)),
            max_crop_pct=float(np.max(crops)),
            mean_composite=mean_composite,
            min_composite=float(np.min(composites)),
            pct_excellent=pct_excellent,
            pct_good=pct_good,
            pct_acceptable=pct_acceptable,
            pct_poor=pct_poor,
            verdict=verdict,
        )

    def evaluate_episodes(
        self,
        spin_rate_series_list: List[np.ndarray],
        dt: float,
    ) -> List[EpisodeVideoQuality]:
        """Evaluate video quality for a batch of episodes."""
        return [self.evaluate_episode(s, dt) for s in spin_rate_series_list]

    # -- printing ----------------------------------------------------------

    def print_summary(
        self,
        qualities: List[EpisodeVideoQuality],
        controller_name: str,
        wind_speed: float,
    ):
        """Print a formatted summary for one controller at one wind level."""
        if not qualities:
            return
        mean_blur = np.mean([q.mean_blur_deg for q in qualities])
        mean_rs = np.mean([q.mean_rs_residual_deg for q in qualities])
        mean_crop = np.mean([q.mean_crop_pct for q in qualities])
        mean_comp = np.mean([q.mean_composite for q in qualities])
        verdict = self._verdict_from_composite(mean_comp)
        pct_exc = np.mean([q.pct_excellent for q in qualities])

        print(
            f"  {controller_name} @ {wind_speed:.0f} m/s: "
            f"verdict={verdict}, composite={mean_comp:.3f}, "
            f"blur={mean_blur:.3f} deg, RS={mean_rs:.4f} deg, "
            f"crop={mean_crop:.3f}%, excellent_frames={pct_exc:.0f}%"
        )

    # -- helpers -----------------------------------------------------------

    def _verdict_from_composite(self, composite: float) -> str:
        th = self.thresholds
        if composite >= th.excellent_min:
            return "Excellent"
        elif composite >= th.good_min:
            return "Good"
        elif composite >= th.acceptable_min:
            return "Acceptable"
        else:
            return "Poor"

    def _empty_quality(self) -> EpisodeVideoQuality:
        return EpisodeVideoQuality(
            mean_blur_deg=0.0,
            max_blur_deg=0.0,
            mean_rs_residual_deg=0.0,
            max_rs_residual_deg=0.0,
            mean_crop_pct=0.0,
            max_crop_pct=0.0,
            mean_composite=1.0,
            min_composite=1.0,
            pct_excellent=100.0,
            pct_good=0.0,
            pct_acceptable=0.0,
            pct_poor=0.0,
            verdict="Excellent",
        )


# ---------------------------------------------------------------------------
# Video quality table printer (used by compare_controllers.py)
# ---------------------------------------------------------------------------


def print_video_quality_table(
    all_results: dict,
    dt: float,
    camera_presets: Optional[List[CameraPreset]] = None,
):
    """Print video quality analysis table for all controllers and wind levels.

    Args:
        all_results: dict mapping controller_name -> list of ControllerResult,
            where each ControllerResult has .episodes with .spin_rate_series.
        dt: simulation timestep (seconds).
        camera_presets: list of CameraPreset to evaluate. Defaults to all.
    """
    if camera_presets is None:
        camera_presets = CameraPreset.all_presets()

    controllers = list(all_results.keys())
    if not controllers:
        return

    wind_levels = [r.wind_speed for r in list(all_results.values())[0]]

    for camera in camera_presets:
        metric = GyroflowVideoMetric(camera)

        print(f"\n{'='*80}")
        print(f"VIDEO QUALITY ANALYSIS ({camera.name})")
        print(f"{'='*80}")

        # Build per-controller, per-wind verdicts and metrics
        verdicts = {}  # controller -> [verdict_per_wind]
        blur_vals = {}  # controller -> [mean_blur_per_wind]
        crop_vals = {}  # controller -> [mean_crop_per_wind]
        comp_vals = {}  # controller -> [mean_composite_per_wind]

        for cname in controllers:
            verdicts[cname] = []
            blur_vals[cname] = []
            crop_vals[cname] = []
            comp_vals[cname] = []

            for result in all_results[cname]:
                # Collect spin rate series from episodes
                series_list = []
                for ep in result.episodes:
                    if (
                        hasattr(ep, "spin_rate_series")
                        and ep.spin_rate_series is not None
                    ):
                        series_list.append(ep.spin_rate_series)

                if not series_list:
                    verdicts[cname].append("N/A")
                    blur_vals[cname].append(float("nan"))
                    crop_vals[cname].append(float("nan"))
                    comp_vals[cname].append(float("nan"))
                    continue

                qualities = metric.evaluate_episodes(series_list, dt)
                mean_comp = np.mean([q.mean_composite for q in qualities])
                mean_blur = np.mean([q.mean_blur_deg for q in qualities])
                mean_crop = np.mean([q.mean_crop_pct for q in qualities])
                verdict = metric._verdict_from_composite(mean_comp)

                verdicts[cname].append(verdict)
                blur_vals[cname].append(mean_blur)
                crop_vals[cname].append(mean_crop)
                comp_vals[cname].append(mean_comp)

        # Column width
        col_w = max(15, max(len(c) for c in controllers) + 2)

        # Verdict sub-table
        print("\nVerdict:")
        header = f"{'Wind (m/s)':>12}"
        for c in controllers:
            header += f" | {c:>{col_w}}"
        print(header)
        print("-" * len(header))
        for i, wind in enumerate(wind_levels):
            row = f"{wind:>12.0f}"
            for c in controllers:
                row += f" | {verdicts[c][i]:>{col_w}}"
            print(row)

        # Motion blur sub-table
        print("\nMotion Blur (deg/frame):")
        header = f"{'Wind (m/s)':>12}"
        for c in controllers:
            header += f" | {c:>{col_w}}"
        print(header)
        print("-" * len(header))
        for i, wind in enumerate(wind_levels):
            row = f"{wind:>12.0f}"
            for c in controllers:
                val = blur_vals[c][i]
                val_str = f"{val:.4f}" if np.isfinite(val) else "N/A"
                row += f" | {val_str:>{col_w}}"
            print(row)

        # FoV Crop sub-table
        print("\nFoV Crop (%):")
        header = f"{'Wind (m/s)':>12}"
        for c in controllers:
            header += f" | {c:>{col_w}}"
        print(header)
        print("-" * len(header))
        for i, wind in enumerate(wind_levels):
            row = f"{wind:>12.0f}"
            for c in controllers:
                val = crop_vals[c][i]
                val_str = f"{val:.4f}" if np.isfinite(val) else "N/A"
                row += f" | {val_str:>{col_w}}"
            print(row)

        # Composite score sub-table
        print("\nComposite Score (0-1):")
        header = f"{'Wind (m/s)':>12}"
        for c in controllers:
            header += f" | {c:>{col_w}}"
        print(header)
        print("-" * len(header))
        for i, wind in enumerate(wind_levels):
            row = f"{wind:>12.0f}"
            for c in controllers:
                val = comp_vals[c][i]
                val_str = f"{val:.4f}" if np.isfinite(val) else "N/A"
                row += f" | {val_str:>{col_w}}"
            print(row)

        print(f"{'='*80}")


# ---------------------------------------------------------------------------
# Standalone CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate video quality at given spin rates",
    )
    parser.add_argument(
        "--spin-rate",
        type=float,
        nargs="+",
        required=True,
        help="Spin rates to evaluate (deg/s)",
    )
    parser.add_argument(
        "--camera",
        type=str,
        default="all",
        choices=["1080p60", "4k30", "1080p120", "all"],
        help="Camera preset",
    )
    args = parser.parse_args()

    preset_map = {
        "1080p60": [CameraPreset.runcam_1080p60()],
        "4k30": [CameraPreset.runcam_4k30()],
        "1080p120": [CameraPreset.runcam_1080p120()],
        "all": CameraPreset.all_presets(),
    }
    cameras = preset_map[args.camera]

    for camera in cameras:
        metric = GyroflowVideoMetric(camera)
        print(f"\n{'='*70}")
        print(f"Camera: {camera.name}")
        print(
            f"  FPS: {camera.fps}, Exposure: {camera.exposure_time_s*1000:.1f} ms, "
            f"FoV: {camera.fov_diagonal_deg} deg, RS: {camera.rolling_shutter_ms} ms"
        )
        print(f"{'='*70}")
        print(
            f"{'Spin (deg/s)':>14} | {'Blur (deg)':>10} | {'RS Res (deg)':>12} | "
            f"{'Crop (%)':>9} | {'Composite':>10} | {'Verdict':<12}"
        )
        print("-" * 80)

        for rate in args.spin_rate:
            q = metric.compute_frame_quality(rate)
            verdict = metric._verdict_from_composite(q.composite_score)
            print(
                f"{rate:>14.1f} | {q.motion_blur_deg:>10.4f} | "
                f"{q.rolling_shutter_residual_deg:>12.5f} | "
                f"{q.fov_crop_pct:>9.4f} | {q.composite_score:>10.4f} | "
                f"{verdict:<12}"
            )

    # Summary
    print(f"\n{'='*70}")
    print("KEY FINDINGS:")
    cam60 = CameraPreset.runcam_1080p60()
    m = GyroflowVideoMetric(cam60)
    for rate in [5, 10, 20, 30, 50]:
        q = m.compute_frame_quality(rate)
        v = m._verdict_from_composite(q.composite_score)
        print(
            f"  {rate:>3} deg/s @ 1080p60: {v} (composite={q.composite_score:.3f}, "
            f"blur={q.motion_blur_deg:.3f} deg)"
        )
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
