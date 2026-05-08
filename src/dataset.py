"""
Dataset utilities for the color-vs-grayscale experiments.

GTSRB: track-stratified train/val split + GTSRBTestDataset from Test.csv.
FMD:   stratified random 80/20 train+test split, then 10% val from train.
       All three splits are Subset views of one ImageFolder, so class_to_idx
       is automatically consistent across train, val, and test.
"""

import csv
import random
from collections import defaultdict
from pathlib import Path

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms

from src.config import (
    BATCH_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    NUM_WORKERS,
    get_dataset_config,
)


# ---------------------------------------------------------------------------
# Image loader
# ---------------------------------------------------------------------------

def _pil_loader_as_rgb(path: str) -> Image.Image:
    with open(path, "rb") as f:
        return Image.open(f).convert("RGB")


# ---------------------------------------------------------------------------
# Transforms
# ---------------------------------------------------------------------------

def build_transform(img_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# GTSRB: track-stratified train/val split
# ---------------------------------------------------------------------------

def _track_stratified_split(
    samples: list,
    val_fraction: float = 0.10,
    seed: int = 0,
) -> tuple[list[int], list[int]]:
    """
    Split by video track. GTSRB filenames: <class>_<track>_<frame>.png.
    All frames from a held-out track go to val — no frame leaks across splits.
    """
    by_track: dict = defaultdict(list)
    for i, (path, class_idx) in enumerate(samples):
        stem = Path(path).stem
        track_id = int(stem.split("_")[1])
        by_track[(class_idx, track_id)].append(i)

    tracks_by_class: dict = defaultdict(list)
    for class_idx, track_id in by_track:
        tracks_by_class[class_idx].append(track_id)

    train_idx: list[int] = []
    val_idx:   list[int] = []

    for class_idx in sorted(tracks_by_class):
        tracks = sorted(tracks_by_class[class_idx])
        n_val = max(1, round(len(tracks) * val_fraction))
        n_val = min(n_val, len(tracks) - 1)

        if n_val == 0:
            print(f"  Warning: class {class_idx} has only 1 track; all frames -> train")
            for t in tracks:
                train_idx.extend(by_track[(class_idx, t)])
            continue

        rng = random.Random(seed * 1000 + class_idx)
        val_tracks = set(rng.sample(tracks, n_val))
        for t in tracks:
            bucket = val_idx if t in val_tracks else train_idx
            bucket.extend(by_track[(class_idx, t)])

    return train_idx, val_idx


class GTSRBTestDataset(Dataset):
    """Test set from a flat Test/ folder using Test.csv labels."""

    def __init__(self, processed_root: Path, transform=None,
                 class_to_idx: dict | None = None):
        self.transform = transform
        self.samples: list[tuple[Path, int]] = []

        with open(processed_root / "Test.csv", newline="") as f:
            for row in csv.DictReader(f):
                img_path = processed_root / row["Path"]
                raw_id = str(int(row["ClassId"]))
                class_id = class_to_idx[raw_id] if class_to_idx else int(row["ClassId"])
                self.samples.append((img_path, class_id))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)
        return img, label


def _build_gtsrb_dataloaders(data_root: Path, cfg: dict,
                              transform) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dir = data_root / "Train"
    if not train_dir.exists():
        raise FileNotFoundError(f"Train directory not found: {train_dir}")

    full_train = datasets.ImageFolder(
        root=str(train_dir),
        transform=transform,
        loader=_pil_loader_as_rgb,
    )
    train_idx, val_idx = _track_stratified_split(
        full_train.samples, val_fraction=0.10, seed=cfg["split_seed"]
    )
    train_set = Subset(full_train, train_idx)
    val_set   = Subset(full_train, val_idx)

    test_set = GTSRBTestDataset(
        data_root, transform=transform,
        class_to_idx=full_train.class_to_idx,
    )

    print(f"  split: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test  (from {len(full_train)} train images)")
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# FMD: stratified random train / val / test split
# ---------------------------------------------------------------------------

def _stratified_split(
    samples: list,
    test_fraction: float,
    val_fraction: float,
    seed: int,
) -> tuple[list[int], list[int], list[int]]:
    """
    Stratified split for datasets with no predefined test set.

    For each class:
      1. Hold out ~test_fraction of images -> test
      2. From the remainder, hold out ~val_fraction -> val
      3. The rest -> train

    Returns train_idx, val_idx, test_idx.
    """
    by_class: dict = defaultdict(list)
    for i, (_, class_idx) in enumerate(samples):
        by_class[class_idx].append(i)

    train_idx: list[int] = []
    val_idx:   list[int] = []
    test_idx:  list[int] = []

    for class_idx in sorted(by_class):
        indices = by_class[class_idx]
        n = len(indices)
        n_test = max(1, round(n * test_fraction))
        n_val  = max(1, round((n - n_test) * val_fraction))

        rng = random.Random(seed * 10000 + class_idx)
        shuffled = indices.copy()
        rng.shuffle(shuffled)

        test_idx.extend(shuffled[:n_test])
        val_idx.extend(shuffled[n_test:n_test + n_val])
        train_idx.extend(shuffled[n_test + n_val:])

    return train_idx, val_idx, test_idx


def _build_fmd_dataloaders(data_root: Path, cfg: dict,
                           transform) -> tuple[DataLoader, DataLoader, DataLoader]:
    if not data_root.exists():
        raise FileNotFoundError(f"Data directory not found: {data_root}")

    full_ds = datasets.ImageFolder(
        root=str(data_root),
        transform=transform,
        loader=_pil_loader_as_rgb,
    )
    train_idx, val_idx, test_idx = _stratified_split(
        full_ds.samples,
        test_fraction=cfg["test_fraction"],
        val_fraction=0.10,
        seed=cfg["split_seed"],
    )
    train_set = Subset(full_ds, train_idx)
    val_set   = Subset(full_ds, val_idx)
    test_set  = Subset(full_ds, test_idx)

    print(f"  split: {len(train_set)} train / {len(val_set)} val / "
          f"{len(test_set)} test  (from {len(full_ds)} total images)")
    return train_set, val_set, test_set


# ---------------------------------------------------------------------------
# DataLoader builder (entry point)
# ---------------------------------------------------------------------------

def build_dataloaders(
    condition: str,
    dataset: str,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Build train, val, and test DataLoaders.

    Parameters
    ----------
    condition : str  e.g. "color", "luma"
    dataset   : str  "gtsrb" or "fmd"

    Returns
    -------
    train_loader, val_loader, test_loader
    """
    cfg       = get_dataset_config(dataset)
    data_root = Path(cfg["data_roots"][condition])
    transform = build_transform(cfg["img_size"])

    if dataset == "gtsrb":
        train_set, val_set, test_set = _build_gtsrb_dataloaders(
            data_root, cfg, transform)
    else:
        train_set, val_set, test_set = _build_fmd_dataloaders(
            data_root, cfg, transform)

    use_pin_memory = torch.cuda.is_available()
    persistent     = NUM_WORKERS > 0

    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=NUM_WORKERS, pin_memory=use_pin_memory,
                              drop_last=True, persistent_workers=persistent)
    val_loader   = DataLoader(val_set,   batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=use_pin_memory,
                              persistent_workers=persistent)
    test_loader  = DataLoader(test_set,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=NUM_WORKERS, pin_memory=use_pin_memory,
                              persistent_workers=persistent)
    return train_loader, val_loader, test_loader
