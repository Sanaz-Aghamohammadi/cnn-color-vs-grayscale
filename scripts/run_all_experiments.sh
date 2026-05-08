#!/usr/bin/env bash
# Run all 15 experiments: 5 conditions x 3 seeds.
#
# Run from project root:
#     bash scripts/run_all_experiments.sh
#
# Each run writes to results/experiments/<condition>/seed_<seed>/
# Total time depends on your GPU. Ballpark on an RTX 3060-class card:
#     ~10-15 min/run x 15 runs = ~2.5-4 hours total.

set -e  # Stop on first failure

CONDITIONS=("color" "luma" "cie_y" "cie_lstar" "lstar_g0")
SEEDS=(42 123 2024)

for condition in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo "  Training: condition=$condition seed=$seed"
        echo "=========================================================="
        python -m src.train --condition "$condition" --seed "$seed"
    done
done

echo ""
echo "=========================================================="
echo "  All runs complete. Aggregating results..."
echo "=========================================================="
python -m src.aggregate_results
