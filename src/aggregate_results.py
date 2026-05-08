"""
Aggregate results across all (condition, seed) runs.

Usage:
    python -m src.aggregate_results                  # defaults to gtsrb
    python -m src.aggregate_results --dataset fmd
"""

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, stdev

from src.config import CONDITIONS, DATASETS, PROJECT_ROOT, SEEDS, get_dataset_config


def load_all_results(results_dir: Path) -> dict:
    """Returns {condition: [result_dict, ...]} for all completed runs."""
    by_condition = {c: [] for c in CONDITIONS}

    for condition in CONDITIONS:
        for seed in SEEDS:
            path = results_dir / condition / f"seed_{seed}" / "results.json"
            if not path.exists():
                print(f"Missing:  {path}")
                continue
            with open(path) as f:
                data = json.load(f)
            if data.get("test") is None:
                print(f"Incomplete (no test eval yet): {path}")
                continue
            by_condition[condition].append(data)

    return by_condition


def summarize(by_condition: dict) -> list[dict]:
    rows = []
    for condition, runs in by_condition.items():
        if not runs:
            continue

        def _std(xs):
            return stdev(xs) if len(xs) > 1 else 0.0

        test_top1  = [r["test"]["top1"]     for r in runs]
        test_top5  = [r["test"]["top5"]     for r in runs]
        test_f1    = [r["test"]["macro_f1"] for r in runs]
        val_top1   = [r["best_val_top1"]    for r in runs]
        times_min  = [r["total_time_s"] / 60.0 for r in runs if "total_time_s" in r]
        epochs_run = [r["epochs_run"]       for r in runs]

        rows.append({
            "condition":        condition,
            "n_runs":           len(runs),
            "test_top1_mean":   mean(test_top1),
            "test_top1_std":    _std(test_top1),
            "test_top5_mean":   mean(test_top5),
            "test_top5_std":    _std(test_top5),
            "test_f1_mean":     mean(test_f1),
            "test_f1_std":      _std(test_f1),
            "val_top1_mean":    mean(val_top1),
            "val_top1_std":     _std(val_top1),
            "avg_train_min":    mean(times_min) if times_min else 0.0,
            "avg_epochs":       mean(epochs_run),
        })

    rows.sort(key=lambda r: r["test_top1_mean"], reverse=True)
    return rows


def print_table(rows: list[dict]) -> None:
    if not rows:
        print("No completed results found.")
        return

    header = (
        f"{'condition':<12} {'n':>2}  "
        f"{'test_top1':>15}  {'test_top5':>15}  "
        f"{'test_f1':>15}  {'val_top1':>15}  "
        f"{'train_min':>10}  {'epochs':>6}"
    )
    print("\n" + header)
    print("-" * len(header))

    for r in rows:
        print(
            f"{r['condition']:<12} {r['n_runs']:>2}  "
            f"{r['test_top1_mean']*100:>6.2f} ± {r['test_top1_std']*100:>4.2f}  "
            f"{r['test_top5_mean']*100:>6.2f} ± {r['test_top5_std']*100:>4.2f}  "
            f"{r['test_f1_mean']*100:>6.2f} ± {r['test_f1_std']*100:>4.2f}  "
            f"{r['val_top1_mean']*100:>6.2f} ± {r['val_top1_std']*100:>4.2f}  "
            f"{r['avg_train_min']:>10.1f}  "
            f"{r['avg_epochs']:>6.1f}"
        )


def write_csv(rows: list[dict], out_path: Path) -> None:
    if not rows:
        return
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote summary CSV -> {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="gtsrb", choices=DATASETS)
    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)
    results_dir = cfg["results_dir"]

    print(f"Dataset: {args.dataset} | Aggregating from: {results_dir}")
    by_condition = load_all_results(results_dir)
    rows = summarize(by_condition)
    print_table(rows)
    write_csv(rows, PROJECT_ROOT / "results" / f"summary_{args.dataset}.csv")


if __name__ == "__main__":
    main()
