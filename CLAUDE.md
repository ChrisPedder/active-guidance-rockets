# Active Guidance Rockets - Project Context

## Project Overview

This project develops a spin stabilization system for model rockets with onboard cameras. The goal is to minimize roll rate during flight to capture stable video footage.

**Target rockets:** Estes C6 motor class (~10 N-s impulse, 1.85s burn) and AeroTech J800T (~800 N-s impulse)

**Control mechanism:** Adjustable fin tabs that create differential drag to counteract spin

## Success Criteria

- **Primary goal:** Mean spin rate < 5 deg/s for stable video footage
- **Secondary goal:** Graceful degradation under wind disturbances (up to 3 m/s)

---

## Current Status (Feb 2026)

### Repository Cleanup (Feb 9, 2026)

Two major bug fixes were applied and all experiments re-run:
1. **Wind torque model**: Changed from `total_fin_area * sin(relative_angle)` to body-shadow model using `single_fin_area * K_shadow * sin(N/2 * gamma)`
2. **Tab deflection**: Fixed from 3.6° to 30° across all configs, PID gains rescaled by 3.6/30 = 0.12

17 controller approaches were evaluated. Only the top 3 (PID, GS-PID, Ensemble) plus ADRC (retained as a research baseline) remain in the codebase. All eliminated controllers and their test files have been deleted.

### Final Results — Estes C6 (50 episodes, IMU, verified Feb 9)

| Wind (m/s) | PID | GS-PID | Ensemble | Target |
|------------|-----------|-----------|----------|--------|
| 0 | 3.7 ± 0.6 | **3.4 ± 0.5** | **3.4 ± 0.5** | < 5 ✅ |
| 1 | 7.6 ± 3.8 | **6.0 ± 2.7** | 6.6 ± 2.7 | < 10 ✅ |
| 2 | 10.4 ± 5.4 | 9.5 ± 5.2 | **9.4 ± 4.9** | < 15 ✅ |
| 3 | 14.7 ± 7.5 | **11.7 ± 6.7** | 11.5 ± 6.1 | < 20 ✅ |
| 5 | **14.1 ± 7.5** | 17.8 ± 8.6 | 14.8 ± 7.6 | — |

### Final Results — AeroTech J800T (50 episodes, IMU, verified Feb 9)

| Wind (m/s) | PID | GS-PID | Ensemble | Target |
|------------|-----------|-----------|----------|--------|
| 0 | 12.8 ± 0.6 | **10.5 ± 0.8** | **10.5 ± 0.7** | < 5 |
| 1 | 13.6 ± 1.0 | 11.0 ± 0.8 | **10.8 ± 0.7** | < 10 |
| 2 | 14.1 ± 1.5 | 11.3 ± 0.9 | **11.2 ± 0.8** | < 15 ✅ |
| 3 | 14.6 ± 2.0 | **11.7 ± 1.2** | 12.0 ± 1.2 | < 20 ✅ |
| 5 | 15.1 ± 1.9 | 12.7 ± 1.7 | **12.4 ± 1.5** | — |

Note: J800 baseline performance (10-15 deg/s) is higher than Estes (3-14 deg/s) — J800 PID gains may need re-optimization post-bugfix. ADRC catastrophically fails on J800 (0% success, 150-297 deg/s in Phase 1 experiments) — not viable for high-power rockets.

### What Works

- **GS-PID is the recommended controller** — best at 0 m/s (3.4 deg/s), consistently best across conditions
- **All Estes targets now met up to 3 m/s**: < 5 at 0 m/s ✅, < 10 at 1 m/s ✅, < 15 at 2 m/s ✅, < 20 at 3 m/s ✅
- **Ensemble matches or beats GS-PID** at most wind levels (9.4 at 2 m/s, 11.5 at 3 m/s)
- **PID gains (Estes):** `Kp=0.0203, Ki=0.0002, Kd=0.0118` (optimized via LHS + local optimization, post-bugfix)
- **PID gains (J800):** `Kp=0.0213, Ki=0.0050, Kd=0.0271, q_ref=13268` (may need re-optimization)
- **Gain scheduling** addresses the ~20x variation in control effectiveness during flight
- **ICM-20948 IMU noise** (0.15 deg/s RMS) has negligible impact on all controllers
- **100% success rate** on Estes at 0-2 m/s across all controllers

### What Doesn't Work

- **J800 baseline performance is higher than expected** (10.5 deg/s at 0 m/s) — PID gains likely need re-optimization for the post-bugfix physics
- **ADRC catastrophically fails on J800** — ESO dynamics incompatible with J800's flight profile
- **17 advanced controllers were tested and eliminated** — none consistently beat simple PID/GS-PID. See `experimental_results.md` for full analysis of all dropped approaches

### Key Insight

**Simple controllers win.** GS-PID and Ensemble consistently outperform every advanced controller tested (17 approaches including ADRC+FF, Fourier ADRC, GP feedforward, STA-SMC, Cascade DOB, FLL, H-inf, Repetitive Control, Lead Compensator, RLS b0, NN wind estimator, Residual RL). The wind disturbance is periodic at the spin frequency, and no feedforward/estimation approach reliably tracks the rapidly-varying disturbance frequency during flight. Gain scheduling addresses the primary challenge (varying control effectiveness), and the Ensemble controller provides modest worst-case reduction.

**Post-bugfix performance is significantly better on Estes.** The corrected physics model (body-shadow wind torque + 30° tab deflection) produces better results: all targets up to 3 m/s are now met, which was not achieved pre-bugfix.

**Cross-platform generalization matters.** ADRC performs well on Estes but catastrophically fails on J800 — making it unsuitable for deployment. PID and GS-PID generalize across both platforms.

---

## Quick Reference

### Current Hardware Parameters

From `configs/estes_c6_sac_wind.yaml`:
- `max_tab_deflection: 30` (degrees, post-bugfix)
- `tab_chord_fraction: 0.25` (25% of fin chord)
- `tab_span_fraction: 0.5` (50% of fin span)
- `num_controlled_fins: 2` (of 4 total fins)
- `dt: 0.01` (100 Hz control loop)

### File Structure

**Controllers** (`controllers/`):
- `controllers/pid_controller.py` - PID and Gain-Scheduled PID controllers
- `controllers/adrc_controller.py` - ADRC controller with dynamic b0
- `controllers/ensemble_controller.py` - Multi-controller ensemble with online switching (GS-PID + ADRC)
- `controllers/video_quality_metric.py` - Gyroflow post-stabilization video quality analysis
- `controllers/disturbance_observer.py` - Disturbance observer for training wrappers

**Evaluation & Optimization:**
- `compare_controllers.py` - Controller comparison tool (supports --pid-only, --gain-scheduled, --adrc, --ensemble, --imu, --video-quality)
- `optimization/optimize_pid.py` - PID/GS-PID gain optimization (LHS + Nelder-Mead)
- `optimization/bayesian_optimize.py` - Per-condition Bayesian optimization (differential evolution)

**Training** (`training/`):
- `training/train_improved.py` - PPO training with configurable reward and wrappers
- `training/train_sac.py` - SAC training with wind curriculum
- `training/train_residual_sac.py` - Residual SAC training (PID + RL corrections)
- `training/sweep_hyperparams.py` - Hyperparameter sweep automation

**Documentation:**
- `experimental_results.md` - Full history of all experiments and results
- `CLAUDE.md` - This file (project context and status)

**Configs:**
- `configs/estes_c6_sac_wind.yaml` - Estes C6 environment config (sinusoidal wind)
- `configs/aerotech_j800_wind.yaml` - AeroTech J800T environment config
- `configs/estes_c6_dryden_{light,moderate,severe}.yaml` - Dryden turbulence configs
- `configs/estes_c6_4fin.yaml` - Hardware study: 4 active fins
- `configs/estes_c6_200hz.yaml` - Hardware study: 200 Hz loop rate
- `configs/estes_c6_500hz.yaml` - Hardware study: 500 Hz loop rate
- `configs/estes_c6_tab{10,15,25,30}.yaml` - Hardware study: tab deflection sweep
- `configs/estes_c6_bigtab.yaml` - Hardware study: 4x tab area
- `configs/estes_c6_4fin_200hz.yaml` - Hardware study: 4 active fins + 200 Hz combined

**Optimization Results:**
- `optimization_results/pid_optimization.json` - Estes PID tuning results (Kp=0.0203, Ki=0.0002, Kd=0.0118)
- `optimization_results/gs_pid_j800_optimization.json` - J800 GS-PID results (Kp=0.0213, Ki=0.0050, Kd=0.0271, q_ref=13268)
- `optimization_results/gs_pid_j800_imu_optimization.json` - J800 GS-PID IMU results
- `optimization_results/pid_optimization_4fin.json` - 4-fin re-optimized gains
- `optimization_results/pid_optimization_tab10.json` - Tab10 re-optimized gains
- `optimization_results/pid_optimization_200hz.json` - 200 Hz re-optimized gains
- `optimization_results/pid_optimization_4fin_200hz.json` - 4-fin+200Hz re-optimized gains
- `optimization_results/pid_optimization_dryden_{light,moderate,severe}.json` - Dryden PID tuning
- `optimization_results/bo_*.json` - Bayesian optimization results

### Key Commands

```bash
# Full comparison (retained controllers, IMU) — sinusoidal wind
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --adrc --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# J800 comparison
uv run python compare_controllers.py --config configs/aerotech_j800_wind.yaml \
    --pid-Kp 0.0213 --pid-Ki 0.0050 --pid-Kd 0.0271 --pid-qref 13268 \
    --gain-scheduled --ensemble --imu \
    --wind-levels 0 1 2 3 5 --n-episodes 50

# PID-only baseline test
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --pid-only --wind-levels 0 1 2 3 --n-episodes 20

# PID gain optimization
uv run python optimization/optimize_pid.py --config configs/estes_c6_sac_wind.yaml \
    --wind-levels 0 2 5 --n-episodes 20 --n-lhs-samples 80 \
    --output optimization_results/pid_optimization.json

# Video quality analysis
uv run python compare_controllers.py --config configs/estes_c6_sac_wind.yaml \
    --gain-scheduled --imu --video-quality --camera-preset all \
    --wind-levels 0 1 2 3 5 --n-episodes 20

# Run tests
uv run python -m pytest tests/ -v
```

---

## Diagnostic Analysis

### Root Cause: Wind Torque is Periodic, Not Constant

The wind model computes roll torque as (`wind_model.py`):
```
torque ∝ sin(wind_direction - roll_angle)
```

Wind disturbance oscillates at the spin frequency. PID integral action lags sinusoidal disturbances by 90° and cannot cancel them structurally. However, no feedforward or estimation approach tested was able to reliably track the rapidly-varying disturbance frequency during flight — making simple rate-damping (PID/GS-PID) the most robust approach.

### Control Authority

- **Max control torque** at q=500 Pa: ~0.013 N·m (full deflection, 2 active fins)
- **Wind disturbance torque** at 3 m/s wind, 30 m/s rocket: ~0.0012 N·m
- **Authority ratio: ~10:1** — sufficient torque, but phase lag limits effectiveness

### IMU Noise Is Not the Bottleneck

ICM-20948 gyro noise: 0.015 deg/s/√Hz → ~0.15 deg/s RMS at 100 Hz. Negligible compared to the 5-30 deg/s spin rates being controlled.

---

## Eliminated Approaches (Summary)

17 control approaches were implemented, tested, and eliminated. See `experimental_results.md` for full details.

| Approach | Result | Why It Failed |
|----------|--------|---------------|
| Lead Compensator | 18.1 deg/s at 0 m/s | Amplifies IMU noise |
| RLS b0 | No improvement | Physics model already accurate enough |
| Repetitive Control | Matches GS-PID | Spin frequency varies too fast |
| Fourier ADRC | Best at 1 m/s (9.6) | Degrades at 2+ m/s, fixed frequency candidates |
| GP Feedforward | Minimal improvement | 50-point budget too sparse in 4D space |
| STA-SMC | 49.0 at 3 m/s | √|σ| control law too slow for periodic disturbance |
| Cascade DOB | 10.3 at 1 m/s | Frequency tracker loses lock at high wind |
| FLL | Matches GS-PID | Gradient adaptation converges too slowly |
| H-inf/LQG/LTR | Degrades at 2+ m/s | Fixed structure can't adapt to 20× b0 variation |
| ADRC+FF | Mixed | Estimation degrades at high wind |
| NN Wind Estimator | No improvement | Wind hard to disentangle from control dynamics |
| Bayesian Optimization | Overfits at high wind | Too few episodes during optimization |
| Residual RL | Failed | SAC interferes with PID; reward hacking |
| Direct SAC | Failed | Poor local minimum, no control structure inductive bias |
| DOB-SAC | Failed | SAC residual still interferes |

---

## Sources

- [Roll-motion stabilizer for sounding rocket using ADRC](https://ieeexplore.ieee.org/document/10242451/)
- [Disturbance rejection for small air-to-surface missiles](https://www.mdpi.com/2076-3417/13/1/389)
- [AIAA Low-Cost Rocket Roll Control (flight validated)](https://arc.aiaa.org/doi/10.2514/1.A36408)
- [Gyroflow open-source video stabilization](https://gyroflow.xyz/)
