#!/bin/bash
# SAC Hyperparameter Sweep - J800 only
# 6 combinations: alpha x ent_coef
# Runs sequentially with early stopping (patience=60)

set -e
cd /Users/chrispedder/Documents/Projects/active-guidance-rockets

CONFIGS=(
    "configs/sac_sweep/j800_sac_a50_e001.yaml"
    "configs/sac_sweep/j800_sac_a50_e005.yaml"
    "configs/sac_sweep/j800_sac_a80_e001.yaml"
    "configs/sac_sweep/j800_sac_a80_e005.yaml"
    "configs/sac_sweep/j800_sac_a100_e001.yaml"
    "configs/sac_sweep/j800_sac_a100_e005.yaml"
)

echo "========================================"
echo "SAC HYPERPARAMETER SWEEP (6 configs)"
echo "========================================"
echo ""

for i in "${!CONFIGS[@]}"; do
    cfg="${CONFIGS[$i]}"
    name=$(basename "$cfg" .yaml)
    echo ""
    echo "========================================"
    echo "[$(( i + 1 ))/6] Training: $name"
    echo "Config: $cfg"
    echo "Started: $(date)"
    echo "========================================"
    echo ""

    uv run python training/train_sac.py \
        --config "$cfg" \
        --early-stopping 60 \
        2>&1

    echo ""
    echo "[$(( i + 1 ))/6] Completed: $name at $(date)"
    echo ""
done

echo ""
echo "========================================"
echo "SWEEP COMPLETE at $(date)"
echo "========================================"
