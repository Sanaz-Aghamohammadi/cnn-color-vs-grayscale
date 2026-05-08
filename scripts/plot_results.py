"""
Generate a two-panel bar chart comparing test top-1 accuracy across
all five input conditions on GTSRB and FMD.

Usage:
    python scripts/plot_results.py
Output:
    report/figures/results.pdf  (embed in LaTeX)
    report/figures/results.png  (quick preview)
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = PROJECT_ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

CONDITION_LABELS = {
    "color":     "Color\n(RGB)",
    "luma":      "Luma\n(BT.601)",
    "cie_y":     "CIE Y",
    "cie_lstar": "CIE L*",
    "lstar_g0":  r"$L^*_{G_0}$",
}

# Consistent color palette across both panels
PALETTE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]

# Display order: color first, rest sorted by GTSRB ranking
DISPLAY_ORDER = ["color", "luma", "cie_lstar", "lstar_g0", "cie_y"]


def read_csv(path: Path) -> dict[str, dict]:
    with open(path) as f:
        return {row["condition"]: row for row in csv.DictReader(f)}


def draw_panel(ax, data: dict, title: str, chance_pct: float, ylim: tuple) -> None:
    means  = [float(data[c]["test_top1_mean"]) * 100 for c in DISPLAY_ORDER]
    stds   = [float(data[c]["test_top1_std"])  * 100 for c in DISPLAY_ORDER]
    labels = [CONDITION_LABELS[c] for c in DISPLAY_ORDER]
    x      = np.arange(len(DISPLAY_ORDER))

    ax.bar(
        x, means, yerr=stds, capsize=4,
        color=PALETTE, edgecolor="white", linewidth=0.5,
        error_kw=dict(elinewidth=1.2, ecolor="#333333"),
    )
    ax.axhline(
        chance_pct, color="#888888", linestyle="--", linewidth=1.0,
        label=f"Chance ({chance_pct:.1f}%)",
    )

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_ylabel("Top-1 Accuracy (%)", fontsize=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(*ylim)
    ax.legend(fontsize=8, framealpha=0.7)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle=":", linewidth=0.6, color="#cccccc", zorder=0)
    ax.set_axisbelow(True)

    for xi, (m, s) in enumerate(zip(means, stds)):
        ax.text(xi, m + s + (ylim[1] - ylim[0]) * 0.015,
                f"{m:.1f}", ha="center", va="bottom", fontsize=7.5, color="#222222")


def main() -> None:
    gtsrb = read_csv(PROJECT_ROOT / "results" / "summary.csv")
    fmd   = read_csv(PROJECT_ROOT / "results" / "summary_fmd.csv")

    fig, (ax_g, ax_f) = plt.subplots(1, 2, figsize=(8.5, 3.6))
    fig.subplots_adjust(wspace=0.32)

    draw_panel(ax_g, gtsrb, "GTSRB  (43 classes, 20 epochs)",
               chance_pct=100 / 43, ylim=(88, 100))
    draw_panel(ax_f, fmd,   "FMD  (10 classes, 30 epochs)",
               chance_pct=10.0,     ylim=(0, 42))

    for fmt in ("pdf", "png"):
        out = FIG_DIR / f"results.{fmt}"
        fig.savefig(out, bbox_inches="tight", dpi=150)
        print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
