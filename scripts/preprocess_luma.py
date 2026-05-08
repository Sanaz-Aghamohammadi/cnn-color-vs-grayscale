"""Preprocess GTSRB into Rec. 601 luma grayscale."""

from methods.luma import luma_grayscale
from scripts._preprocess_common import preprocess_dataset


if __name__ == "__main__":
    preprocess_dataset("luma", luma_grayscale)
