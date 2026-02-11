#!/bin/bash
# Evaluate all 6 SAC sweep models against PID baseline
set -e
cd /Users/chrispedder/Documents/Projects/active-guidance-rockets

MODELS=(
    "models/rocket_sac_j800_a50_e001_aerotech_j800t_20260210_192732"
    "models/rocket_sac_j800_a50_e005_aerotech_j800t_20260210_233722"
    "models/rocket_sac_j800_a80_e001_aerotech_j800t_20260211_003534"
    "models/rocket_sac_j800_a80_e005_aerotech_j800t_20260211_043141"
    "models/rocket_sac_j800_a100_e001_aerotech_j800t_20260211_053014"
    "models/rocket_sac_j800_a100_e005_aerotech_j800t_20260211_092539"
)

NAMES=(
    "a50_e001"
    "a50_e005"
    "a80_e001"
    "a80_e005"
    "a100_e001"
    "a100_e005"
)

echo "========================================"
echo "SAC SWEEP EVALUATION (6 models)"
echo "========================================"
echo ""

for i in "${!MODELS[@]}"; do
    model_dir="${MODELS[$i]}"
    name="${NAMES[$i]}"
    echo ""
    echo "========================================"
    echo "[$(( i + 1 ))/6] Evaluating: $name"
    echo "Model: $model_dir/best_model.zip"
    echo "========================================"
    echo ""

    uv run python compare_controllers.py \
        --config "$model_dir/config.yaml" \
        --sac "$model_dir/best_model.zip" \
        --imu \
        --wind-levels 0 1 2 3 5 \
        --n-episodes 50 \
        2>&1

    echo ""
    echo "[$(( i + 1 ))/6] Done: $name"
    echo ""
done

echo ""
echo "========================================"
echo "ALL EVALUATIONS COMPLETE"
echo "========================================"
