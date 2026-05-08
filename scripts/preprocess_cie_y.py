"""Preprocess GTSRB into linear CIE Y grayscale."""

from methods.cie_y import cie_y_grayscale
from scripts._preprocess_common import preprocess_dataset


if __name__ == "__main__":
    preprocess_dataset("cie_y", cie_y_grayscale)
