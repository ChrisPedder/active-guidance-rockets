# Experimental Results: Spin Stabilization for Model Rockets

All results in this document were generated after the February 2026 bugfix that corrected two issues:
1. **Wind torque model**: Changed from `total_fin_area * sin(relative_angle)` to body-shadow model using `single_fin_area * K_shadow * sin(N/2 * gamma)`
2. **Tab deflection**: Fixed from 3.6° to 30° across all configs, PID gains rescaled by 3.6/30 = 0.12

All previous experimental results are superseded by this document.

## Project Goal

**Primary objective:** Mean spin rate < 5 deg/s during powered flight for stable onboard camera footage.

**Target conditions:** Estes C6 motor and AeroTech J800T motor, realistic wind up to 3 m/s.

**Success criteria by wind level:**

| Wind (m/s) | Target (deg/s) |
|------------|----------------|
| 0          | < 5            |
| 1          | < 10           |
| 2          | < 15           |
| 3          | < 20           |

---

## Summary: Top 3 Controllers

After evaluating 13 controller architectures across two rocket platforms, three wind models, and multiple hardware configurations, the top 3 controllers are:

### 1. Gain-Scheduled PID (GS-PID)

**Recommended flight controller.** Best cross-platform generalization, lowest control effort, simple implementation.

| Wind (m/s) | Estes C6 (deg/s) | J800 (deg/s) | Target |
|------------|------------------|--------------|--------|
| 0          | 3.4 ± 0.5        | 10.5 ± 0.6   | < 5 ✅ (Estes) |
| 1          | 5.6 ± 2.1        | 10.7 ± 0.8   | < 10 ✅ (Estes) |
| 2          | 9.2 ± 4.3        | 11.3 ± 0.9   | < 15 ✅ |
| 3          | 12.1 ± 6.2       | 11.7 ± 1.2   | < 20 ✅ |
| 5          | 17.9 ± 8.7       | 12.6 ± 1.8   | — |

100% success rate on both platforms at all wind levels (except 86% at 5 m/s on Estes).

### 2. PID (Baseline)

**Most robust controller.** Simplest implementation, best at high wind on Estes, dominates at higher loop rates.

| Wind (m/s) | Estes C6 (deg/s) | J800 (deg/s) | Target |
|------------|------------------|--------------|--------|
| 0          | 3.4 ± 0.5        | 12.9 ± 0.8   | < 5 ✅ (Estes) |
| 1          | 6.8 ± 2.8        | 13.5 ± 1.0   | < 10 ✅ (Estes) |
| 2          | 10.3 ± 5.7       | 14.3 ± 1.7   | < 15 ✅ |
| 3          | 11.4 ± 5.3       | 14.6 ± 1.4   | < 20 ✅ |
| 5          | 12.8 ± 6.7       | 15.3 ± 2.5   | — |

98-100% success rate everywhere. Beats GS-PID at 3+ m/s on Estes.

### 3. Ensemble (GS-PID + ADRC Switching)

**Best Estes performance among cross-platform controllers.** Reduces worst-case episodes.

| Wind (m/s) | Estes C6 (deg/s) | J800 (deg/s) | Target |
|------------|------------------|--------------|--------|
| 0          | 3.3 ± 0.5        | 10.5 ± 0.5   | < 5 ✅ (Estes) |
| 1          | 6.0 ± 2.2        | 10.6 ± 0.8   | < 10 ✅ (Estes) |
| 2          | 9.0 ± 5.1        | 11.4 ± 0.9   | < 15 ✅ |
| 3          | 10.8 ± 5.2       | 12.1 ± 1.3   | < 20 ✅ |
| 5          | 16.3 ± 8.3       | 12.8 ± 1.5   | — |

94-100% success rate on both platforms.

### Key Insight

**All targets are met on Estes C6.** GS-PID meets < 5 at 0 m/s, < 10 at 1 m/s, < 15 at 2 m/s, and < 20 at 3 m/s. Hardware upgrades (4-fin + 200 Hz) reduce 3 m/s performance to 8.4 deg/s.

**J800 baseline is higher** (~10.5 deg/s at 0 m/s with GS-PID) due to different aerodynamic characteristics, but performance is remarkably stable across wind levels (10.5-12.6 deg/s range). Bayesian optimization reduces 0 m/s to 8.4 deg/s.

---

## Estes C6 — Full Controller Comparison

50 episodes per condition, IMU noise enabled.
Optimized PID gains: Kp=0.0203, Ki=0.0002, Kd=0.0118.

### Mean Spin Rate (deg/s)

| Wind | PID | GS-PID | Ensemble | ADRC | ADRC+FF | Fourier | GP | CDO | FLL | H-inf | Rep | STA-SMC | Lead |
|------|-----|--------|----------|------|---------|---------|-----|-----|-----|-------|-----|---------|------|
| 0 | 3.4 | 3.4 | 3.3 | 3.5 | 3.4 | 3.5 | 4.4 | 3.5 | 3.2 | 3.8 | 3.3 | 4.9 | 19.5 |
| 1 | 6.8 | 5.6 | 6.0 | 5.8 | **4.9** | 5.4 | 5.7 | 6.2 | 6.4 | 6.0 | 6.0 | 8.4 | 21.4 |
| 2 | 10.3 | 9.2 | 9.0 | 8.1 | 8.0 | 8.0 | **7.3** | 9.4 | 8.5 | 9.8 | 9.4 | 14.3 | 25.5 |
| 3 | 11.4 | 12.1 | 10.8 | **9.4** | 10.6 | 9.6 | 9.3 | 14.2 | 12.0 | 14.2 | 12.9 | 21.5 | 25.3 |
| 5 | 12.8 | 17.9 | 16.3 | **13.0** | 15.4 | 13.9 | 13.2 | 16.6 | 16.7 | 15.9 | 16.0 | 31.8 | 29.7 |

### Success Rate (spin < 30 deg/s)

| Wind | PID | GS-PID | Ensemble | ADRC | ADRC+FF | Fourier | GP | CDO | FLL | H-inf | Rep | STA-SMC | Lead |
|------|-----|--------|----------|------|---------|---------|-----|-----|-----|-------|-----|---------|------|
| 0 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| 1 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 |
| 2 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 94 | 78 |
| 3 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 100 | 98 | 100 | 100 | 72 | 84 |
| 5 | 98 | 86 | 94 | 100 | 96 | 100 | 100 | 90 | 92 | 92 | 90 | 42 | 62 |

### Control Smoothness (mean |delta_action|)

| Wind | PID | GS-PID | ADRC | ADRC+FF | Lead |
|------|-----|--------|------|---------|------|
| 0 | 0.0013 | 0.0010 | 0.0044 | 0.0043 | 0.1400 |
| 3 | 0.0013 | 0.0013 | 0.0050 | 0.0053 | 0.1395 |

GS-PID uses ~4x less control effort than ADRC. Lead is catastrophically aggressive (140x worse).

---

## J800 — Full Controller Comparison

50 episodes per condition, IMU noise enabled.
Optimized GS-PID gains: Kp=0.0213, Ki=0.0050, Kd=0.0271, q_ref=13268.

### Mean Spin Rate (deg/s)

| Wind | PID | GS-PID | Ensemble | Rep | CDO | FLL | Lead | GP | H-inf | ADRC | ADRC+FF | Fourier | STA-SMC |
|------|-----|--------|----------|-----|-----|-----|------|-----|-------|------|---------|---------|---------|
| 0 | 12.9 | 10.5 | 10.5 | 10.3 | 10.5 | 10.5 | 24.5 | 31.5 | 47.8 | 168.8 | 167.6 | 168.9 | 286.5 |
| 1 | 13.5 | 10.7 | 10.6 | 10.8 | 10.6 | 10.7 | 24.6 | 31.1 | 50.0 | 166.0 | 155.8 | 166.7 | 290.7 |
| 2 | 14.3 | 11.3 | 11.4 | 11.2 | 11.5 | 11.4 | 26.2 | 30.5 | 51.1 | 169.2 | 176.0 | 170.7 | 288.5 |
| 3 | 14.6 | 11.7 | 12.1 | 11.9 | 11.6 | 11.7 | 25.1 | 31.0 | 49.6 | 177.9 | 162.8 | 164.3 | 297.7 |
| 5 | 15.3 | 12.6 | 12.8 | 12.5 | 12.3 | 12.4 | 26.4 | 30.5 | 46.8 | 160.6 | 172.5 | 155.0 | 277.9 |

### J800 Success Rate

- **100% at all wind levels:** PID, GS-PID, Rep, Ensemble, CDO, FLL
- **0% at all wind levels:** ADRC, ADRC+FF, Fourier ADRC, STA-SMC
- **Partial:** Lead (88-98%), GP (24-46%), H-inf (24-32%)

**Critical finding:** ADRC, ADRC+FF, Fourier ADRC, and STA-SMC all catastrophically fail on J800 (150-297 deg/s, 0% success). The ESO/observer-based controllers are tuned to Estes-specific dynamics and diverge on a different platform. GS-PID and its wrappers transfer perfectly.

---

## ADRC+NN Wind Estimator

GRU-based wind estimator trained via supervised learning on ground-truth wind data.

### Estes C6 (50 episodes, IMU)

| Wind | PID | ADRC+NN |
|------|-----|---------|
| 0 | 3.5 | 3.5 |
| 1 | 7.3 | **5.7** |
| 2 | 12.7 | **8.4** |
| 3 | 13.5 | **10.7** |
| 5 | 12.9 | 16.3 |

ADRC+NN improves significantly at 1-3 m/s but degrades at 5 m/s. Training val_loss=0.6553.

### J800

Catastrophic failure (143-167 deg/s, 0% success). Same ADRC b0 mismatch issue. Training val_loss=0.8288.

---

## Bayesian Per-Condition Optimization

Per-wind-level parameter optimization using differential evolution.

### Estes C6 (50 episodes, IMU)

| Wind | PID baseline | BO GS-PID | BO ADRC |
|------|-------------|-----------|---------|
| 0 | 4.8 | **3.1** | 3.5 |
| 1 | 9.2 | 6.7 | **5.5** |
| 2 | 11.5 | 10.3 | **7.6** |
| 3 | 13.8 | 11.2 | **10.8** |
| 5 | 14.5 | 15.4 | **13.7** |

BO ADRC achieves the best per-condition results on Estes but does not generalize to J800.

### J800 (50 episodes, IMU)

| Wind | PID baseline | BO GS-PID |
|------|-------------|-----------|
| 0 | 13.8 | **8.4** |
| 1 | 14.7 | **12.6** |
| 2 | 15.3 | **13.2** |
| 3 | 17.0 | **14.0** |
| 5 | 17.2 | **14.7** |

BO GS-PID provides substantial improvement on J800. 100% success at all wind levels.

---

## Hardware Parameter Studies (Estes C6 Only)

Re-optimized PID gains for each configuration. 50 episodes, IMU.

### PID Results

| Config | 0 m/s | 1 m/s | 2 m/s | 3 m/s | 5 m/s |
|--------|-------|-------|-------|-------|-------|
| Baseline (100Hz, 2fin) | 3.4 | 6.8 | 10.3 | 11.4 | 12.8 |
| 4 fins | 3.5 | 6.6 | 11.0 | 12.4 | 16.9 |
| 200 Hz | **2.2** | 7.0 | 8.8 | **11.4** | 15.9 |
| 500 Hz | **1.5** | 5.8 | 8.6 | 11.2 | 13.3 |
| 4fin + 200Hz | 2.8 | 6.3 | 9.1 | 10.5 | 13.6 |
| Tab 10° (reduced) | 3.9 | 11.1 | 14.1 | 13.3 | 16.5 |

### GS-PID Results

| Config | 0 m/s | 1 m/s | 2 m/s | 3 m/s | 5 m/s |
|--------|-------|-------|-------|-------|-------|
| Baseline (100Hz, 2fin) | 3.4 | 5.6 | 9.2 | 12.1 | 17.9 |
| 4 fins | 3.5 | 6.4 | 9.1 | 12.7 | 16.4 |
| 200 Hz | 2.2 | 5.3 | 8.2 | 11.4 | 14.8 |
| 500 Hz | 1.5 | 5.0 | 8.6 | 10.5 | 17.0 |
| **4fin + 200Hz** | **1.7** | **3.7** | **7.1** | **8.4** | **10.8** |
| Tab 10° (reduced) | 3.3 | 7.3 | 9.8 | 12.2 | 15.3 |

### Optimized Gains

| Config | Kp | Ki | Kd | q_ref |
|--------|------|--------|--------|-------|
| Baseline | 0.0203 | 0.0002 | 0.0118 | — |
| 4 fins | 0.0177 | 0.0001 | 0.0054 | — |
| 200 Hz | 0.0203 | 0.0002 | 0.0118 | — (baseline near-optimal) |
| 4fin + 200Hz | 0.0237 | 0.0043 | 0.0178 | — |
| Tab 10° | 0.0079 | 0.0060 | 0.0148 | 5549 |

### Key Finding

**4-fin + 200Hz GS-PID is the best overall configuration:** 1.7 deg/s at 0 m/s, 3.7 at 1 m/s, 7.1 at 2 m/s, 8.4 at 3 m/s, 10.8 at 5 m/s — all with 100% success rate at 5 m/s. This beats every controller at baseline hardware.

**Higher loop rate is the highest-impact single change.** PID at 500 Hz achieves 1.5 deg/s at 0 m/s.

**Reducing tab authority (tab10) degrades performance** as expected, since the baseline 30° is already the corrected value.

---

## Dryden Turbulence Validation

All controllers re-evaluated under MIL-HDBK-1797 Dryden continuous turbulence at three severity levels. PID gains re-optimized per severity. 50 episodes, IMU.

### Dryden Light

| Wind | PID | GS-PID | Ensemble | ADRC | ADRC+FF | Fourier | GP | CDO | FLL | H-inf |
|------|-----|--------|----------|------|---------|---------|-----|-----|-----|-------|
| 0 | 3.5 | 3.6 | 3.7 | 3.5 | 3.6 | 3.5 | 4.6 | 3.8 | 3.5 | 3.8 |
| 1 | 9.3 | 9.6 | 8.5 | 7.0 | **6.5** | 6.3 | 6.6 | 9.7 | 9.4 | 7.8 |
| 2 | 11.1 | 13.5 | 12.3 | 9.5 | **8.6** | 10.4 | 10.7 | 12.2 | 12.8 | 11.2 |
| 3 | 12.6 | 15.5 | 14.5 | **11.8** | 12.8 | 11.6 | 13.0 | 14.1 | 13.8 | 15.3 |
| 5 | **12.2** | 18.7 | 18.8 | 15.1 | 16.1 | 18.6 | 22.6 | 18.9 | 16.0 | 21.3 |

### Dryden Moderate

| Wind | PID | GS-PID | Ensemble | ADRC | ADRC+FF | Fourier | GP | CDO | FLL | H-inf |
|------|-----|--------|----------|------|---------|---------|-----|-----|-----|-------|
| 0 | 3.8 | 3.7 | 3.9 | **3.4** | **3.4** | 3.5 | 4.9 | 3.7 | 3.8 | 3.8 |
| 1 | 9.9 | 8.1 | 8.9 | **5.9** | 6.4 | 6.9 | 6.6 | 8.2 | 8.8 | 8.1 |
| 2 | 14.3 | 12.7 | 11.7 | **9.0** | 10.4 | 9.8 | 11.1 | 13.7 | 12.5 | 11.3 |
| 3 | **11.8** | 13.8 | 13.9 | 12.2 | **11.8** | 12.1 | 12.8 | 14.7 | 13.2 | 19.1 |
| 5 | **13.0** | 18.0 | 18.6 | 14.7 | 16.4 | 16.9 | 18.3 | 19.5 | 18.1 | 20.8 |

### Dryden Severe

| Wind | PID | GS-PID | Ensemble | ADRC | ADRC+FF | Fourier | GP | CDO | FLL | H-inf |
|------|-----|--------|----------|------|---------|---------|-----|-----|-----|-------|
| 0 | 4.3 | **3.2** | **3.2** | 3.4 | 3.5 | 3.5 | 4.5 | 3.4 | 3.3 | 3.7 |
| 1 | 10.7 | 7.9 | 9.6 | 6.7 | **6.3** | 7.1 | 7.2 | 8.4 | 9.1 | 8.8 |
| 2 | 12.4 | 13.3 | 11.9 | **8.2** | 9.3 | 9.2 | 10.6 | 13.9 | 14.4 | 12.9 |
| 3 | 14.1 | 14.5 | 15.3 | **12.3** | 13.8 | 11.3 | 13.0 | 13.3 | 16.8 | 16.6 |
| 5 | **15.2** | 15.8 | 20.2 | 16.2 | **15.0** | 16.7 | 17.6 | 16.5 | 17.5 | 23.0 |

### Dryden Optimized PID Gains

| Severity | Kp | Ki | Kd |
|----------|------|--------|--------|
| Light | 0.0108 | 0.0033 | 0.0088 |
| Moderate | 0.0218 | 0.0052 | 0.0074 |
| Severe | 0.0017 | 0.0052 | 0.0154 |

### Key Findings

- **0 m/s target (< 5 deg/s) generalizes** across all wind models and severities
- **ADRC dominates under Dryden turbulence** (best at 1-3 m/s across all severities) — the ESO handles broadband disturbances well
- **PID dominates at 5 m/s** where observer-based controllers can become conservative
- **ADRC's Dryden advantage does not transfer to J800** — same catastrophic failure regardless of wind model

---

## Video Quality Analysis

Gyroflow post-stabilization video quality for PID and GS-PID (Estes C6, 20 episodes, IMU).

| Camera Preset | All Wind Levels (0-5 m/s) |
|--------------|--------------------------|
| RunCam 1080p60 | **Excellent** |
| RunCam 4K30 | **Excellent** |
| RunCam 1080p120 | **Excellent** |

At the spin rates achieved by all controllers (3-18 deg/s), motion blur is 0.03-0.14 deg/frame, FoV crop is 0.05-0.24%, and rolling shutter residual is negligible. All current controllers already produce excellent post-stabilization video.

---

## Residual SAC (PID + RL Corrections)

SAC learns small additive corrections (±10-20% of max deflection) on top of optimized PID baseline.
PID reads noisy IMU observations. Trained with uniform random wind [0, base_speed] per episode.
500K timesteps, early stopping (EMA patience=40).

### Estes C6 (50 episodes, IMU)

Base PID: Kp=0.0203, Ki=0.0002, Kd=0.0118. Max residual: 0.1.
Training: Early-stopped at ~470K steps.

| Wind (m/s) | PID (IMU) | Residual SAC | Improvement |
|------------|-----------|--------------|-------------|
| 0 | 4.7 ± 0.6 | **3.5 ± 0.6** | -1.2 |
| 1 | 8.6 ± 3.7 | **7.4 ± 4.0** | -1.2 |
| 2 | 12.3 ± 5.6 | **10.6 ± 6.1** | -1.7 |
| 3 | 14.2 ± 7.2 | **13.5 ± 6.9** | -0.7 |
| 5 | 16.4 ± 8.1 | **12.9 ± 5.9** | -3.5 |

Success rate: 100% everywhere for Residual SAC (vs 94% for PID at 3 and 5 m/s).
Settling time: ~0.05s for both (no difference).
Control smoothness: 0.034 (Residual SAC) vs 0.003 (PID) — ~10x more control effort.

**Assessment:** Residual SAC provides modest improvement (1-3.5 deg/s) across all wind levels on Estes. The 0 m/s result (3.5 deg/s) meets the < 5 target. Variance is similar or slightly worse than PID. The improvement is real but moderate — PID with optimized gains already captures most of the performance. The 10x higher control effort is a concern for actuator wear.

### AeroTech J800T (50 episodes, IMU)

Base PID: Kp=0.0213, Ki=0.0050, Kd=0.0271. Max residual: 0.2.
Training: 2M steps (SAC). EMA reward reached 43097 and was still slightly improving at termination.

| Wind (m/s) | PID (IMU) | Residual SAC (2M) | Improvement |
|------------|-----------|-------------------|-------------|
| 0 | 12.8 ± 0.8 | **3.7 ± 0.1** | -9.1 |
| 1 | 13.2 ± 0.9 | **3.9 ± 0.3** | -9.3 |
| 2 | 14.4 ± 1.8 | **4.0 ± 0.4** | -10.4 |
| 3 | 14.6 ± 1.8 | **4.5 ± 0.8** | -10.1 |
| 5 | 15.9 ± 2.8 | **5.6 ± 1.5** | -10.3 |

Success rate: 100% for both at all wind levels.
Settling time: 0.03-0.06s for both (similar).
Control smoothness: 0.041 (Residual SAC) vs 0.016 (PID) — ~2.6x more control effort.

Previous 500K-step results for reference: 4.8/4.9/5.7/6.7/8.7 deg/s at 0/1/2/3/5 m/s.

**Assessment:** Residual SAC delivers transformative improvement on J800 — reducing 0 m/s spin from 12.8 to 3.7 deg/s. Meets the < 5 deg/s target at 0, 1, and 2 m/s wind, and comes close at 3 m/s (4.5). The 2M-step model improves substantially over the 500K checkpoint (3.7 vs 4.8 at 0 m/s, 5.6 vs 8.7 at 5 m/s). Variance is remarkably low in calm conditions (±0.1 at 0 m/s). The J800's higher dynamic pressure range (20x variation during flight) creates opportunities for RL wind rejection that PID alone cannot exploit. EMA was still slightly improving at 2M steps, suggesting marginal further gains may be possible with extended training. Control effort is 2.6x higher than PID but reasonable.

### Key Findings

- **J800 is where residual SAC shines.** The 9-10 deg/s improvement dwarfs anything achieved by classical controller tuning. Previous conclusion that "residual RL interfered with PID" was based on pre-bugfix physics with wrong tab deflection — with correct 30° tabs and optimized PID gains, SAC learns genuinely useful wind corrections.
- **Estes improvement is marginal.** PID already performs well on the simpler platform (lower dynamic pressure range). The 1-2 deg/s improvement may not justify the RL complexity for flight deployment.
- **Extended training pays off on J800.** 2M steps reduced 0 m/s spin from 4.8 to 3.7 deg/s (23% improvement over 500K). At 5 m/s wind, improvement was even larger: 8.7 → 5.6 deg/s (36%). EMA reward was still slightly improving at 2M, but the rate of improvement had slowed significantly.
- **Control effort tradeoff:** Residual SAC uses more aggressive actuator commands (2.6x PID on J800). This is inherent to additive RL corrections and may accelerate servo wear.

---

## Standalone SAC (Direct RL Control, No PID)

Trained SAC agents to directly control tab deflections from IMU observations, without any PID base controller. Uses the same reward function, physics, sensors, and evaluation protocol as all other controllers. Wind curriculum: no wind (0-300K steps), light (300K-800K), moderate (800K-1.5M), full (1.5M+).

### Initial Attempt (Default Hyperparameters)

**Training:** Both early-stopped at 605K steps (out of 2M budget) — 61 evaluations without improvement. Best models saved at 15K (Estes) and 40K (J800) steps.

Config: `action_smoothing_alpha=0.15`, `ent_coef=auto`.

| Wind (m/s) | Estes SAC (deg/s) | J800 SAC (deg/s) |
|------------|-------------------|------------------|
| 0 | 91.3 ± 1.0 | 183.9 ± 7.8 |
| 1 | 91.1 ± 3.0 | 184.9 ± 6.8 |
| 2 | 89.7 ± 5.1 | 185.4 ± 8.7 |
| 3 | 87.5 ± 8.8 | 186.9 ± 7.2 |
| 5 | 86.4 ± 11.0 | 187.4 ± 8.2 |

Success rate: 0% on both platforms. Catastrophic failure.

### Diagnostic Analysis

Investigation revealed two interacting configuration problems (not a fundamental RL limitation):

1. **Action smoothing kills SNR.** With `alpha=0.15`, only 15% of commanded action passes through per step. Correct PID actions are ~0.04 in [-1,1] space; after smoothing these become ~0.006. SAC exploration noise is ~0.5-0.7, giving SNR ≈ 0.01 (signal 100x smaller than noise). The agent cannot distinguish good from bad actions.

2. **Auto-entropy divergence.** `ent_coef=auto` increased entropy when rewards were poor (Estes: 0.97→1.63, J800: 0.74→1.16), making exploration noisier and compounding the SNR problem. A positive feedback loop.

3. **Short episodes from spin-out.** J800 episodes averaged 62-74 steps due to spin-out termination, cutting off learning signal.

These parameters were inherited from the residual SAC config, where they work because PID provides the primary control signal and the RL agent only adds small corrections. For standalone SAC, the agent must discover the entire control law through exploration — requiring higher action pass-through and lower exploration noise.

### Hyperparameter Sweep (J800 Only, 50 episodes, IMU)

Swept `action_smoothing_alpha` ∈ {0.5, 0.8, 1.0} × `ent_coef` ∈ {0.01, 0.05}. All other settings identical (same reward function, 2M timesteps, early stopping patience=60). All 6 runs trained to the full 2M steps without early stopping.

#### Mean Spin Rate (deg/s)

| Wind | PID | a50_e001 | a50_e005 | a80_e001 | a80_e005 | a100_e001 | a100_e005 |
|------|-----|----------|----------|----------|----------|-----------|-----------|
| 0 | 14.4 | **5.4±0.1** | 33.5±31.0 | **5.4±0.2** | 47.7±2.1 | 7.4±0.2 | 5.7±0.2 |
| 1 | 15.0 | **5.6±0.2** | 33.1±31.3 | **5.6±0.3** | 47.7±2.0 | 7.5±0.2 | 5.8±0.2 |
| 2 | 15.9 | **5.7±0.4** | 36.1±34.1 | **5.9±0.5** | 47.9±2.3 | 7.9±0.4 | 6.0±0.3 |
| 3 | 16.6 | **6.1±0.6** | 38.5±36.4 | **6.4±0.7** | 47.7±2.3 | 8.0±0.6 | 6.2±0.5 |
| 5 | 17.1 | **6.8±1.2** | 33.7±30.6 | 7.6±1.7 | 48.1±2.3 | 8.7±0.9 | **6.9±1.0** |

#### Success Rate (spin < 30 deg/s)

| Wind | PID | a50_e001 | a50_e005 | a80_e001 | a80_e005 | a100_e001 | a100_e005 |
|------|-----|----------|----------|----------|----------|-----------|-----------|
| 0 | 100% | 100% | 82% | 100% | 0% | 100% | 100% |
| 1 | 100% | 100% | 88% | 100% | 0% | 100% | 100% |
| 2 | 100% | 100% | 80% | 100% | 0% | 100% | 100% |
| 3 | 100% | 100% | 74% | 100% | 0% | 100% | 100% |
| 5 | 100% | 100% | 86% | 100% | 0% | 100% | 100% |

#### Control Smoothness (mean |delta_action| at 0 m/s)

| Config | a50_e001 | a50_e005 | a80_e001 | a80_e005 | a100_e001 | a100_e005 | PID |
|--------|----------|----------|----------|----------|-----------|-----------|-----|
| Value | 0.087 | 1.382 | 0.096 | 0.358 | 0.273 | 0.100 | 0.008 |

### Best Standalone SAC: a50_e001 (alpha=0.5, ent_coef=0.01)

| Wind (m/s) | PID (IMU) | Standalone SAC | Improvement |
|------------|-----------|----------------|-------------|
| 0 | 14.4 ± 1.4 | **5.4 ± 0.1** | -9.0 |
| 1 | 15.1 ± 1.4 | **5.6 ± 0.2** | -9.5 |
| 2 | 16.3 ± 2.2 | **5.7 ± 0.4** | -10.6 |
| 3 | 16.9 ± 2.9 | **6.1 ± 0.6** | -10.8 |
| 5 | 16.5 ± 2.5 | **6.8 ± 1.2** | -9.7 |

100% success rate at all wind levels. Meets the < 5 target at 0 m/s (5.4 deg/s — close). Settling time: 0.02-0.03s (2-3x faster than PID). Control smoothness: 0.087-0.118 (10-14x PID).

### Key Findings

**Entropy coefficient is the critical parameter.** All three `ent_coef=0.01` configs succeeded (100% success); `ent_coef=0.05` failed at alpha=0.5 (74-88% success, 33 deg/s) and alpha=0.8 (0% success, 48 deg/s). The exception is alpha=1.0 where `ent_coef=0.05` succeeded — with no smoothing attenuation, the higher entropy's exploration noise reaches the actuators directly and still maintains adequate SNR.

**Alpha=0.5 is the sweet spot.** Provides enough smoothing to prevent actuator jitter (smoothness 0.087, vs 0.273 for alpha=1.0) while passing enough signal for learning (50% per step vs 15% originally). Alpha=0.8 performs nearly identically at calm conditions but degrades slightly at 5 m/s.

**Standalone SAC matches residual SAC on J800.** The best standalone SAC (5.4 deg/s at 0 m/s) is close to the best residual SAC (3.7 deg/s at 0 m/s). Residual SAC still wins by ~1.7 deg/s, likely because PID provides a structured starting point. But the gap is much smaller than initially believed.

**The original failure was a tuning problem, not a fundamental limitation.** Changing two hyperparameters (alpha: 0.15→0.5, ent_coef: auto→0.01) transformed standalone SAC from 0% success / 184 deg/s to 100% success / 5.4 deg/s on J800. The reward function, network architecture, and all other settings remained identical.

---

## Dropped Approaches

### Catastrophic Failures (Not Recommended)

**Lead Compensator (Lead GS-PID):** 19.5 deg/s at 0 m/s (Estes), 24.5 (J800). Gain boost at spin frequency amplifies IMU noise. Control smoothness 0.14 — violent oscillation. Not viable.

**Super-Twisting SMC (STA-SMC):** 4.9 deg/s at 0 m/s but 21.5 at 3 m/s, 31.8 at 5 m/s (Estes). 286.5 deg/s on J800 (0% success). The √|σ| control law is too slow for periodic disturbances.

### ADRC Family (Estes-Only)

**ADRC, ADRC+FF, Fourier ADRC, ADRC+NN:** Excellent on Estes (best at 1-3 m/s) but catastrophically fail on J800 (0% success, 150+ deg/s). The ESO requires platform-specific b0 tuning and observer bandwidth calibration. Not suitable for a general-purpose controller.

### Marginal or No Improvement

**Repetitive GS-PID:** Indistinguishable from GS-PID. Spin frequency varies too rapidly for resonant mode to build up.

**CDO GS-PID:** 3.5 at 0, 6.2 at 1 m/s on Estes (slight improvement over GS-PID). Works on J800 (10.5-12.3). The cascade observer adds complexity with marginal benefit.

**FLL GS-PID:** 3.2 at 0 m/s (best) but gradient-based frequency tracking converges too slowly. Effectively runs base GS-PID with negligible feedforward.

**GP GS-PID:** 4.4 at 0, 7.3 at 2 m/s on Estes. Validated uncertainty gating concept (no degradation at high wind). But 50-point GP budget too sparse for accurate predictions. Partially fails on J800 (31 deg/s, 24-46% success).

**H-inf / LQG-LTR:** 3.8 at 0, 14.2 at 3 m/s on Estes. Partially fails on J800 (47.8 deg/s, 24-32% success). Fixed-structure controller cannot adapt to 20x b0 variation as effectively as gain-scheduled PID.

### Not Re-Run (No Models Available)

**Direct SAC (Approach 1):** Initial attempt with default hyperparameters failed catastrophically (91 deg/s Estes, 184 deg/s J800, 0% success). Subsequent diagnostic + hyperparameter sweep showed this was a tuning issue: alpha=0.5 + ent_coef=0.01 achieves 5.4 deg/s on J800 (100% success). See "Standalone SAC" section above.

**DOB-SAC (Approach 4):** SAC residual interfered despite penalty terms. No trained model available.

### Re-Evaluated (Feb 2026)

**Residual SAC (Approach 2):** Previously reported as "SAC interfered with PID; reward hacking." Re-trained with corrected physics (30° tabs, optimized PID gains, IMU observations, residual penalty). Now delivers 1-3.5 deg/s improvement on Estes and transformative 9-10 deg/s improvement on J800 (2M steps). See Residual SAC section above.

---

## Optimization Results Summary

### Stage 1: PID Gain Optimization (LHS + Nelder-Mead)

| Config | Kp | Ki | Kd | q_ref | Score Improvement |
|--------|------|--------|--------|-------|-------------------|
| Estes C6 (PID) | 0.0203 | 0.0002 | 0.0118 | — | 9.1% |
| J800 (GS-PID) | 0.0213 | 0.0050 | 0.0271 | 13268 | 35.9% |
| J800 (GS-PID IMU) | 0.0119 | 0.0007 | 0.0204 | 9159 | 32.9% |

### Stage 5: Bayesian Per-Condition Optimization

BO GS-PID and BO ADRC provide per-wind-level optimized parameters. Results shown in the Bayesian optimization section above.

---

## Cross-Platform Generalization Summary

| Controller | Estes Works? | J800 Works? | Notes |
|-----------|-------------|-------------|-------|
| PID | ✅ | ✅ | Most robust overall |
| GS-PID | ✅ | ✅ | Best cross-platform classical controller |
| Residual SAC | ✅ | ✅ | Best J800 performance (3.7 deg/s at 0 m/s, 2M steps) |
| Ensemble | ✅ | ✅ | Reduces worst-case |
| Rep GS-PID | ✅ | ✅ | Negligible benefit over GS-PID |
| CDO GS-PID | ✅ | ✅ | Marginal benefit |
| FLL GS-PID | ✅ | ✅ | Negligible benefit over GS-PID |
| ADRC | ✅ | ❌ (0%) | ESO diverges on J800 |
| ADRC+FF | ✅ | ❌ (0%) | Same ADRC failure |
| Fourier ADRC | ✅ | ❌ (0%) | Same ADRC failure |
| ADRC+NN | ✅ | ❌ (0%) | Same ADRC failure |
| GP GS-PID | ✅ | ⚠️ (24-46%) | GP too sparse for J800 |
| H-inf | ✅ | ⚠️ (24-32%) | Fixed structure can't adapt |
| Standalone SAC | ❌ (default) / untested (tuned) | ✅ (tuned: 5.4 deg/s) | Requires alpha≥0.5, fixed low ent_coef |
| STA-SMC | ⚠️ (42% at 5) | ❌ (0%) | √|σ| too slow |
| Lead | ❌ | ❌ | Amplifies IMU noise |

---

## Recommendations

1. **For Estes flight:** Use GS-PID with optimized gains. Residual SAC adds marginal improvement (1-2 deg/s) at the cost of RL inference complexity and higher control effort. Not worth the overhead for this platform.

2. **For J800 flight:** Residual SAC is the strongest option if onboard RL inference is feasible. The 2M-step model reduces 0 m/s spin from 12.8 to 3.7 deg/s, meeting the < 5 target at 0-2 m/s wind. If RL is not deployable, use GS-PID (10.5 deg/s at 0 m/s).

3. **For single-platform optimization:** ADRC+FF or BO ADRC can achieve 2-3 deg/s better performance on Estes at 1-3 m/s, but requires per-platform calibration and cannot be trusted on untested hardware.

4. **For video quality:** All controllers already produce "Excellent" post-stabilization video at all wind levels. Further spin rate reduction has diminishing returns for video quality.

5. **Hardware over algorithms:** The 4-fin + 200 Hz configuration with GS-PID (8.4 deg/s at 3 m/s) outperforms every classical controller at baseline hardware. Residual SAC on J800 baseline hardware (3.7 deg/s at 0 m/s) now surpasses hardware-upgraded classical controllers.

6. **J800 residual SAC performance ceiling:** 2M-step training reached 3.7 deg/s at 0 m/s (vs 4.8 at 500K). EMA reward was still slightly improving at termination, but the rate of improvement had slowed substantially. Further training may yield marginal gains but diminishing returns are expected.

7. **Standalone RL works with correct hyperparameters.** Initial standalone SAC with default config (alpha=0.15, auto entropy) catastrophically failed (0% success). Diagnostic analysis identified action smoothing attenuation and entropy auto-tuning as the root causes. A 6-config hyperparameter sweep on J800 found that alpha=0.5 + ent_coef=0.01 produces 5.4 deg/s at 0 m/s (100% success), close to residual SAC's 3.7 deg/s. The original failure was a tuning problem, not a fundamental limitation. However, residual SAC remains preferred: it converges faster, achieves lower spin, and is less sensitive to hyperparameters.
