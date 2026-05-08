"""
Central configuration for the color-vs-grayscale CNN experiments.

Per-dataset settings (paths, num_classes, img_size) live in DATASET_CONFIGS.
Hyperparameters that are shared across all datasets live at the top level.
"""

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]

# ----- Shared across all datasets -----
CONDITIONS = ["color", "luma", "cie_y", "cie_lstar", "lstar_g0"]
SEEDS      = [42, 123, 2024]

BATCH_SIZE  = 64
NUM_WORKERS = 4
NUM_EPOCHS  = 20

LR           = 0.01
MOMENTUM     = 0.9
WEIGHT_DECAY = 5e-4
NESTEROV     = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ----- Per-dataset configuration -----
DATASET_CONFIGS = {
    "gtsrb": {
        "num_classes": 43,
        "img_size":    64,
        "split_seed":  0,
        # results/experiments/ holds existing runs; rename to results/gtsrb/
        # with `mv results/experiments results/gtsrb` once the sweep is done.
        "results_dir": PROJECT_ROOT / "results" / "gtsrb",
        "data_roots": {
            "color":     PROJECT_ROOT / "data" / "raw"       / "GTSRB",
            "luma":      PROJECT_ROOT / "data" / "processed" / "luma",
            "cie_y":     PROJECT_ROOT / "data" / "processed" / "cie_y",
            "cie_lstar": PROJECT_ROOT / "data" / "processed" / "cie_lstar",
            "lstar_g0":  PROJECT_ROOT / "data" / "processed" / "lstar_g0",
        },
    },
    "fmd": {
        "num_classes":   10,
        "img_size":      128,
        "split_seed":    0,
        "test_fraction": 0.20,
        "results_dir":   PROJECT_ROOT / "results" / "fmd",
        "data_roots": {
            "color":     PROJECT_ROOT / "data" / "raw"       / "FMD" / "image",
            "luma":      PROJECT_ROOT / "data" / "processed" / "fmd" / "luma",
            "cie_y":     PROJECT_ROOT / "data" / "processed" / "fmd" / "cie_y",
            "cie_lstar": PROJECT_ROOT / "data" / "processed" / "fmd" / "cie_lstar",
            "lstar_g0":  PROJECT_ROOT / "data" / "processed" / "fmd" / "lstar_g0",
        },
    },
}

DATASETS = list(DATASET_CONFIGS.keys())


def get_dataset_config(dataset: str) -> dict:
    if dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset '{dataset}'. Choose from: {DATASETS}")
    return DATASET_CONFIGS[dataset]
