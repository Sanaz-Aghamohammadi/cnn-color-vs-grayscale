"""
Shared helpers for preprocessing GTSRB into per-method grayscale variants.

Reads RGB PNGs from data/raw/GTSRB and writes single-channel PNG outputs
under data/processed/<method>/, preserving the Train/Test layout and
copying Test.csv unchanged.
"""

from pathlib import Path
from typing import Callable
import shutil

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT = PROJECT_ROOT / "data" / "raw" / "GTSRB"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed"

ConvertFn = Callable[[np.ndarray], np.ndarray]


def _load_rgb(path: Path):
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        return None
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _save_png(path: Path, gray_uint8: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ok = cv2.imwrite(str(path), gray_uint8)
    if not ok:
        raise RuntimeError(f"Failed to write {path}")


def _process_train(method: str, convert_fn: ConvertFn) -> tuple[int, int]:
    train_in = RAW_ROOT / "Train"
    train_out = PROCESSED_ROOT / method / "Train"

    class_dirs = sorted(
        (p for p in train_in.iterdir() if p.is_dir()),
        key=lambda p: int(p.name),
    )
    print(f"Train: {len(class_dirs)} class folders")

    total_done = 0
    total_skipped = 0
    for cdir in class_dirs:
        files = sorted(p for p in cdir.iterdir() if p.suffix.lower() == ".png")
        out_class = train_out / cdir.name
        n_done = n_skip = 0
        for f in files:
            out_path = out_class / f.name
            if out_path.exists():
                n_skip += 1
                continue
            rgb = _load_rgb(f)
            if rgb is None:
                print(f"  unreadable: {f}")
                continue
            gray = convert_fn(rgb)
            _save_png(out_path, gray)
            n_done += 1
        print(f"  class {cdir.name:>2}: wrote {n_done:5d}, skipped {n_skip:5d} (of {len(files)})")
        total_done += n_done
        total_skipped += n_skip
    return total_done, total_skipped


def _process_test(method: str, convert_fn: ConvertFn) -> tuple[int, int]:
    test_in = RAW_ROOT / "Test"
    test_out = PROCESSED_ROOT / method / "Test"

    files = sorted(p for p in test_in.iterdir() if p.suffix.lower() == ".png")
    print(f"Test: {len(files)} files")

    n_done = n_skip = 0
    for f in files:
        out_path = test_out / f.name
        if out_path.exists():
            n_skip += 1
            continue
        rgb = _load_rgb(f)
        if rgb is None:
            print(f"  unreadable: {f}")
            continue
        gray = convert_fn(rgb)
        _save_png(out_path, gray)
        n_done += 1
        if n_done % 1000 == 0:
            print(f"  ...{n_done} written")
    print(f"Test: wrote {n_done}, skipped {n_skip} (of {len(files)})")
    return n_done, n_skip


def _copy_test_csv(method: str) -> None:
    src = RAW_ROOT / "Test.csv"
    dst = PROCESSED_ROOT / method / "Test.csv"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"Copied {src.name} -> {dst}")


def preprocess_dataset(method: str, convert_fn: ConvertFn) -> None:
    """
    Run the full GTSRB preprocess for one method.

    Parameters
    ----------
    method : str
        Name of the output subfolder under data/processed/, e.g. "luma".
    convert_fn : callable
        Takes an RGB uint8 image of shape (H, W, 3) and returns a uint8
        grayscale image of shape (H, W).
    """
    print(f"=== Preprocessing GTSRB -> {method} ===")
    print(f"raw:       {RAW_ROOT}")
    print(f"processed: {PROCESSED_ROOT / method}")
    if not RAW_ROOT.exists():
        raise SystemExit(f"Raw GTSRB not found at {RAW_ROOT}")

    train_done, train_skip = _process_train(method, convert_fn)
    test_done, test_skip = _process_test(method, convert_fn)
    _copy_test_csv(method)

    print(
        f"\nDone [{method}]: "
        f"train wrote {train_done} (skipped {train_skip}); "
        f"test wrote {test_done} (skipped {test_skip})"
    )
