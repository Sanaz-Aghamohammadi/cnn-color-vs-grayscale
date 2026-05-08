"""Preprocess FMD into Rec. 601 luma grayscale."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.luma import luma_grayscale
from scripts._preprocess_fmd import preprocess_dataset

if __name__ == "__main__":
    preprocess_dataset("luma", luma_grayscale)
