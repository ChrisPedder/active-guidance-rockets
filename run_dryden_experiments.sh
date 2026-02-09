#!/usr/bin/env zsh
#
# run_dryden_experiments.sh
#
# Re-runs all controller experiments using the Dryden turbulence wind model
# at three severity levels (light, moderate, severe).
#
# For each severity level:
#   1. Creates a Dryden config from the base YAML
#   2. Optimizes PID gains
#   3. Trains NN wind estimator
#   4. Runs Bayesian per-wind-level optimization
#   5. Runs full controller comparison (IMU)
#   6. Runs ADRC+NN comparison with trained estimator
#   7. Runs Bayesian-optimized parameters comparison
#
# All output is logged per-severity to logs/dryden_experiment_<severity>.log
# Final summary written to dryden_results_summary.txt
#
# Usage:
#   chmod +x run_dryden_experiments.sh
#   ./run_dryden_experiments.sh
#

set -e

# ─── Configuration ───────────────────────────────────────────────────────────

BASE_CONFIG="configs/estes_c6_sac_wind.yaml"
SEVERITIES=(light moderate severe)
WIND_LEVELS="0 1 2 3 5"
N_EVAL_EPISODES=50
N_OPT_EPISODES=20
N_LHS_SAMPLES=80
N_BO_TRIALS=80
NN_TRAIN_EPISODES=100
NN_TRAIN_EPOCHS=50
SUMMARY_FILE="dryden_results_summary.txt"

# ─── Directories ─────────────────────────────────────────────────────────────

mkdir -p configs logs models optimization_results

# ─── Step 1: Create Dryden config files ──────────────────────────────────────

echo "=== Creating Dryden config files ==="

for severity in "${SEVERITIES[@]}"; do
    config_out="configs/estes_c6_dryden_${severity}.yaml"
    echo "  Creating ${config_out} (severity=${severity})"

    # Copy base config and modify Dryden settings
    sed -e 's/^  use_dryden: false/  use_dryden: true/' \
        -e "s/^  turbulence_severity: light/  turbulence_severity: ${severity}/" \
        "${BASE_CONFIG}" > "${config_out}"
done

echo ""

# ─── Helper: extract PID gains from optimization JSON ────────────────────────

extract_pid_gains() {
    local json_file="$1"
    python3 -c "
import json, sys
with open('${json_file}') as f:
    data = json.load(f)
opt = data.get('optimized', data.get('baseline'))
print(f\"{opt['kp']} {opt['ki']} {opt['kd']}\")
"
}

# ─── Helper: log + run a command ─────────────────────────────────────────────

run_logged() {
    local logfile="$1"
    shift
    echo "  >> $*" | tee -a "${logfile}"
    "$@" 2>&1 | tee -a "${logfile}"
}

# ─── Step 2: Per-severity experiment loop ────────────────────────────────────

for severity in "${SEVERITIES[@]}"; do
    config="configs/estes_c6_dryden_${severity}.yaml"
    logfile="logs/dryden_experiment_${severity}.log"
    pid_json="optimization_results/pid_optimization_dryden_${severity}.json"
    bo_json="optimization_results/bo_gs_pid_dryden_${severity}.json"
    nn_model="models/wind_estimator_dryden_${severity}.pt"

    echo "============================================================"
    echo "=== Dryden severity: ${severity}"
    echo "=== Config: ${config}"
    echo "=== Log: ${logfile}"
    echo "============================================================"
    echo ""

    # Clear previous log
    : > "${logfile}"
    echo "=== Dryden experiment: ${severity} ===" >> "${logfile}"
    echo "Started: $(date)" >> "${logfile}"
    echo "" >> "${logfile}"

    # ── 2a: PID gain optimization ────────────────────────────────────────

    echo "--- [${severity}] Step 1/6: PID gain optimization ---"
    run_logged "${logfile}" \
        uv run python optimization/optimize_pid.py \
            --config "${config}" \
            --wind-levels ${=WIND_LEVELS} \
            --n-episodes "${N_OPT_EPISODES}" \
            --n-lhs-samples "${N_LHS_SAMPLES}" \
            --output "${pid_json}"

    # Extract optimized gains
    gains=($(extract_pid_gains "${pid_json}"))
    OPT_KP="${gains[1]}"
    OPT_KI="${gains[2]}"
    OPT_KD="${gains[3]}"
    echo "  Optimized gains: Kp=${OPT_KP}, Ki=${OPT_KI}, Kd=${OPT_KD}"
    echo "  Optimized gains: Kp=${OPT_KP}, Ki=${OPT_KI}, Kd=${OPT_KD}" >> "${logfile}"
    echo ""

    # ── 2b: NN wind estimator training ───────────────────────────────────

    echo "--- [${severity}] Step 2/6: NN wind estimator training ---"
    run_logged "${logfile}" \
        uv run python wind_estimator.py \
            --train \
            --config "${config}" \
            --wind-levels ${=WIND_LEVELS} \
            --episodes "${NN_TRAIN_EPISODES}" \
            --epochs "${NN_TRAIN_EPOCHS}" \
            --model "${nn_model}"

    echo ""

    # ── 2c: Bayesian optimization (per-wind-level) ───────────────────────

    echo "--- [${severity}] Step 3/6: Bayesian optimization (GS-PID) ---"
    run_logged "${logfile}" \
        uv run python optimization/bayesian_optimize.py \
            --config "${config}" \
            --controller gs-pid \
            --wind-level ${=WIND_LEVELS} \
            --n-episodes "${N_OPT_EPISODES}" \
            --n-trials "${N_BO_TRIALS}" \
            --output "${bo_json}"

    echo ""

    # ── 2d: Full controller comparison (IMU) ─────────────────────────────

    echo "--- [${severity}] Step 4/6: Full controller comparison (IMU) ---"
    run_logged "${logfile}" \
        uv run python compare_controllers.py \
            --config "${config}" \
            --pid-Kp "${OPT_KP}" --pid-Ki "${OPT_KI}" --pid-Kd "${OPT_KD}" \
            --gain-scheduled --adrc --adrc-ff --fourier-ff --gp-ff --ensemble \
            --imu \
            --wind-levels ${=WIND_LEVELS} \
            --n-episodes "${N_EVAL_EPISODES}" \
            --save-plot "dryden_comparison_${severity}.png"

    echo ""

    # ── 2e: ADRC+NN comparison ───────────────────────────────────────────

    echo "--- [${severity}] Step 5/6: ADRC+NN comparison ---"
    run_logged "${logfile}" \
        uv run python compare_controllers.py \
            --config "${config}" \
            --pid-Kp "${OPT_KP}" --pid-Ki "${OPT_KI}" --pid-Kd "${OPT_KD}" \
            --adrc-nn "${nn_model}" \
            --imu \
            --wind-levels ${=WIND_LEVELS} \
            --n-episodes "${N_EVAL_EPISODES}"

    echo ""

    # ── 2f: Bayesian-optimized parameters comparison ─────────────────────

    echo "--- [${severity}] Step 6/6: Bayesian-optimized parameters comparison ---"
    run_logged "${logfile}" \
        uv run python compare_controllers.py \
            --config "${config}" \
            --pid-Kp "${OPT_KP}" --pid-Ki "${OPT_KI}" --pid-Kd "${OPT_KD}" \
            --optimized-params "${bo_json}" \
            --imu \
            --wind-levels ${=WIND_LEVELS} \
            --n-episodes "${N_EVAL_EPISODES}"

    echo ""
    echo "=== Completed: ${severity} ($(date)) ==="
    echo "Completed: $(date)" >> "${logfile}"
    echo ""
done

# ─── Step 3: Collate summary ────────────────────────────────────────────────

echo "=== Collating results into ${SUMMARY_FILE} ==="

{
    echo "================================================================"
    echo "  DRYDEN TURBULENCE EXPERIMENT RESULTS"
    echo "  Generated: $(date)"
    echo "================================================================"
    echo ""

    for severity in "${SEVERITIES[@]}"; do
        logfile="logs/dryden_experiment_${severity}.log"
        pid_json="optimization_results/pid_optimization_dryden_${severity}.json"
        bo_json="optimization_results/bo_gs_pid_dryden_${severity}.json"
        nn_model="models/wind_estimator_dryden_${severity}.pt"

        echo "────────────────────────────────────────────────────────────"
        echo "  SEVERITY: ${severity}"
        echo "────────────────────────────────────────────────────────────"
        echo ""

        # Print optimized PID gains
        if [[ -f "${pid_json}" ]]; then
            echo "  Optimized PID gains:"
            python3 -c "
import json
with open('${pid_json}') as f:
    data = json.load(f)
opt = data.get('optimized', data.get('baseline'))
print(f\"    Kp = {opt['kp']:.4f}\")
print(f\"    Ki = {opt['ki']:.4f}\")
print(f\"    Kd = {opt['kd']:.4f}\")
print(f\"    Score = {opt['score']:.2f}\")
for wl, res in sorted(opt.get('by_wind', {}).items(), key=lambda x: float(x[0])):
    print(f\"    Wind {wl} m/s: mean={res['mean_spin']:.1f}, std={res['std_spin']:.1f}, success={res['success_rate']:.0%}\")
"
        else
            echo "  [WARNING] PID optimization output not found: ${pid_json}"
        fi
        echo ""

        # Print BO results
        if [[ -f "${bo_json}" ]]; then
            echo "  Bayesian-optimized GS-PID params:"
            python3 -c "
import json
with open('${bo_json}') as f:
    data = json.load(f)
for wl_key, res in sorted(data.get('results', {}).items(), key=lambda x: float(x[0])):
    params = res['params']
    param_str = ', '.join(f'{k}={v:.4f}' for k, v in sorted(params.items()))
    print(f\"    Wind {res['wind_level']} m/s: mean={res['mean_spin']:.1f}, std={res['std_spin']:.1f} | {param_str}\")
"
        else
            echo "  [WARNING] BO output not found: ${bo_json}"
        fi
        echo ""

        # Print NN model status
        if [[ -f "${nn_model}" ]]; then
            echo "  NN wind estimator: ${nn_model} ($(du -h "${nn_model}" | cut -f1) )"
        else
            echo "  [WARNING] NN model not found: ${nn_model}"
        fi
        echo ""

        # Extract comparison tables from log
        echo "  Controller comparison tables:"
        echo "  (see full log: ${logfile})"
        echo ""

        # Extract the comparison table sections from the full comparison run (Step 4/6)
        if [[ -f "${logfile}" ]]; then
            # Print lines between "Mean Spin Rate" header and "Success Rate" end
            awk '
                /Step 4\/6:/ { in_step4 = 1 }
                /Step 5\/6:/ { in_step4 = 0 }
                in_step4 && /^Wind \(m\/s\)/ { printing = 1 }
                in_step4 && printing { print "    " $0 }
                in_step4 && printing && /^$/ { blank_count++ }
                in_step4 && blank_count >= 2 { printing = 0; blank_count = 0 }
            ' "${logfile}"
        fi
        echo ""
    done

    echo "================================================================"
    echo "  END OF RESULTS"
    echo "================================================================"
} > "${SUMMARY_FILE}"

echo ""
echo "=== All experiments complete ==="
echo "  Summary: ${SUMMARY_FILE}"
echo "  Logs:    logs/dryden_experiment_{light,moderate,severe}.log"
echo "  Plots:   dryden_comparison_{light,moderate,severe}.png"
echo ""
