"""Preprocess GTSRB into perceptual CIE L* grayscale."""

from methods.cie_lstar import cie_lstar_grayscale
from scripts._preprocess_common import preprocess_dataset


if __name__ == "__main__":
    preprocess_dataset("cie_lstar", cie_lstar_grayscale)
