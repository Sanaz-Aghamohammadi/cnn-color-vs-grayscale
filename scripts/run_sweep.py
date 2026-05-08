"""
Run the full 15-run training sweep (5 conditions x 3 seeds).

Skips any run whose results.json already has a completed test evaluation,
so re-running after an interruption is safe.

Usage:
    python scripts/run_sweep.py                   # GTSRB, 20 epochs
    python scripts/run_sweep.py --dataset fmd
    python scripts/run_sweep.py --dataset fmd --epochs 30
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import CONDITIONS, DATASETS, NUM_EPOCHS, SEEDS, get_dataset_config


def is_complete(results_dir: Path, condition: str, seed: int) -> bool:
    path = results_dir / condition / f"seed_{seed}" / "results.json"
    if not path.exists():
        return False
    with open(path) as f:
        data = json.load(f)
    return data.get("test") is not None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gtsrb", choices=DATASETS)
    parser.add_argument("--epochs",  type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    cfg         = get_dataset_config(args.dataset)
    results_dir = cfg["results_dir"]

    runs  = [(c, s) for c in CONDITIONS for s in SEEDS]
    total = len(runs)
    print(f"Dataset: {args.dataset} | Sweep: {total} runs x {args.epochs} epochs")

    completed = sum(1 for c, s in runs if is_complete(results_dir, c, s))
    if completed:
        print(f"{completed} already complete — will skip those.")

    for i, (condition, seed) in enumerate(runs, 1):
        if is_complete(results_dir, condition, seed):
            print(f"[{i}/{total}] SKIP  condition={condition}  seed={seed}  (already done)")
            continue

        print(f"\n[{i}/{total}] START  condition={condition}  seed={seed}")
        result = subprocess.run([
            sys.executable, "-m", "src.train",
            "--dataset",   args.dataset,
            "--condition", condition,
            "--seed",      str(seed),
            "--epochs",    str(args.epochs),
        ])

        if result.returncode != 0:
            print(f"\nSweep FAILED at run {i}/{total}: condition={condition} seed={seed}")
            print("Fix the error and re-run — completed runs will be skipped automatically.")
            sys.exit(1)

    print(f"\nAll {total} runs complete.")


if __name__ == "__main__":
    main()
