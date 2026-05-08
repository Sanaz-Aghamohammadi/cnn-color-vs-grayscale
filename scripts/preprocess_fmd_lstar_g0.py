"""Preprocess FMD into L*G0 grayscale (chromaticity-conditioned max luminance)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from methods.g0_model import G0Model
from methods.lstar_g0 import lstar_g0_grayscale
from scripts._preprocess_fmd import PROJECT_ROOT, preprocess_dataset


def main() -> None:
    print("Loading G0 model...")
    ref = PROJECT_ROOT / "data" / "reference"
    g0_model = G0Model(
        cmf_path=str(ref / "CMF2deg.xlsx"),
        d65_path=str(ref / "D65.xlsx"),
        lut_path=str(ref / "LUT.xlsx"),
        white_y=1.0,
    )
    print("G0 model loaded.")
    preprocess_dataset("lstar_g0", lambda rgb: lstar_g0_grayscale(rgb, g0_model))


if __name__ == "__main__":
    main()
