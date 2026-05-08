"""
Shared helpers for preprocessing FMD into per-method grayscale variants.

Reads RGB JPEGs from data/raw/FMD/image/<class>/ and writes single-channel
PNG outputs under data/processed/fmd/<method>/<class>/, preserving filenames
(extension changes .jpg -> .png). No train/test split is created here —
that is handled at load time in src/dataset.py.
"""

from pathlib import Path
from typing import Callable

import cv2
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_ROOT     = PROJECT_ROOT / "data" / "raw" / "FMD" / "image"
PROCESSED_ROOT = PROJECT_ROOT / "data" / "processed" / "fmd"

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png"}
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


def preprocess_dataset(method: str, convert_fn: ConvertFn) -> None:
    """
    Convert all FMD images for one grayscale method.

    Parameters
    ----------
    method : str
        Output subfolder name under data/processed/fmd/, e.g. "luma".
    convert_fn : callable
        Takes RGB uint8 (H, W, 3) and returns grayscale uint8 (H, W).
    """
    print(f"=== Preprocessing FMD -> {method} ===")
    print(f"raw:       {RAW_ROOT}")
    print(f"processed: {PROCESSED_ROOT / method}")

    if not RAW_ROOT.exists():
        raise SystemExit(f"Raw FMD not found at {RAW_ROOT}")

    class_dirs = sorted(p for p in RAW_ROOT.iterdir() if p.is_dir())
    print(f"Found {len(class_dirs)} class folders")

    total_done = total_skipped = 0

    for cdir in class_dirs:
        files = sorted(
            p for p in cdir.iterdir()
            if p.suffix.lower() in VALID_EXTENSIONS
        )
        out_class = PROCESSED_ROOT / method / cdir.name
        n_done = n_skip = 0

        for f in files:
            out_path = out_class / (f.stem + ".png")
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

        print(f"  {cdir.name:<10}: wrote {n_done:4d}, skipped {n_skip:4d} (of {len(files)})")
        total_done    += n_done
        total_skipped += n_skip

    print(f"\nDone [{method}]: wrote {total_done} (skipped {total_skipped})")
