#!/usr/bin/env bash
# Run from-scratch experiments: 3 conditions x 3 seeds = 9 runs.
#
# Budget: ~55 min per run x 9 runs ~= 8 hours of training.
# Early stopping may cut some runs short.
#
# Run from project root:
#     bash scripts/run_scratch_experiments.sh
#
# Each run writes to results/experiments_scratch/<condition>/seed_<seed>/
# Results are saved after every epoch, so a crash won't lose everything.

set -e  # Stop on first failure

CONDITIONS=("color" "cie_lstar" "lstar_g0")
SEEDS=(42 123 2024)

# Log directory for per-run logs
LOG_DIR="results/experiments_scratch/logs"
mkdir -p "$LOG_DIR"

START_TIME=$(date +%s)

for condition in "${CONDITIONS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo ""
        echo "=========================================================="
        echo "  FROM SCRATCH: condition=$condition seed=$seed"
        echo "  Started at: $(date)"
        echo "=========================================================="
        LOG_FILE="$LOG_DIR/${condition}_seed${seed}.log"
        python -m src.train_scratch --condition "$condition" --seed "$seed" 2>&1 | tee "$LOG_FILE"
    done
done

END_TIME=$(date +%s)
ELAPSED_MIN=$(( (END_TIME - START_TIME) / 60 ))

echo ""
echo "=========================================================="
echo "  All from-scratch runs complete in $ELAPSED_MIN minutes."
echo "  Aggregating results..."
echo "=========================================================="
python -m src.aggregate_results --mode scratch
