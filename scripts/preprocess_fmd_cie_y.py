"""Preprocess FMD into linear CIE Y grayscale."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.cie_y import cie_y_grayscale
from scripts._preprocess_fmd import preprocess_dataset

if __name__ == "__main__":
    preprocess_dataset("cie_y", cie_y_grayscale)
