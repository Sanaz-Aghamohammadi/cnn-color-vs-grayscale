import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt

from methods.luma import luma_grayscale
from methods.cie_y import cie_y_grayscale
from methods.cie_lstar import cie_lstar_grayscale
from methods.g0_model import G0Model
from methods.lstar_g0 import lstar_g0_grayscale


def main():
    input_path = Path("data/test_image.png")
    output_dir = Path("results/test_methods")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load image
    img_bgr = cv2.imread(str(input_path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {input_path}")

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    print("Loaded image:", input_path)
    print("Image shape:", img_rgb.shape)

    # Build G0 model once
    print("\nLoading G0 model...")
    g0_model = G0Model(
        cmf_path="data/reference/CMF2deg.xlsx",
        d65_path="data/reference/D65.xlsx",
        lut_path="data/reference/LUT.xlsx",
        white_y=1.0,
    )
    print("G0 model loaded.")

    # Apply methods
    print("\nApplying luma...")
    gray_luma = luma_grayscale(img_rgb.astype("float32"))

    print("Applying CIE Y...")
    gray_cie_y = cie_y_grayscale(img_rgb)

    print("Applying CIE L*...")
    gray_cie_lstar = cie_lstar_grayscale(img_rgb)

    print("Applying L*G0...")
    gray_lstar_g0 = lstar_g0_grayscale(img_rgb, g0_model)

    # Save outputs
    luma_path = output_dir / "luma.jpg"
    cie_y_path = output_dir / "cie_y.jpg"
    cie_lstar_path = output_dir / "cie_lstar.jpg"
    lstar_g0_path = output_dir / "lstar_g0.jpg"

    cv2.imwrite(str(luma_path), gray_luma)
    cv2.imwrite(str(cie_y_path), gray_cie_y)
    cv2.imwrite(str(cie_lstar_path), gray_cie_lstar)
    cv2.imwrite(str(lstar_g0_path), gray_lstar_g0)

    print("\nSaved outputs:")
    print(luma_path)
    print(cie_y_path)
    print(cie_lstar_path)
    print(lstar_g0_path)

    # Show comparison
    plt.figure(figsize=(14, 10))

    plt.subplot(2, 3, 1)
    plt.imshow(img_rgb)
    plt.title("Original RGB")
    plt.axis("off")

    plt.subplot(2, 3, 2)
    plt.imshow(gray_luma, cmap="gray")
    plt.title("Luma")
    plt.axis("off")

    plt.subplot(2, 3, 3)
    plt.imshow(gray_cie_y, cmap="gray")
    plt.title("CIE Y")
    plt.axis("off")

    plt.subplot(2, 3, 4)
    plt.imshow(gray_cie_lstar, cmap="gray")
    plt.title("CIE L*")
    plt.axis("off")

    plt.subplot(2, 3, 5)
    plt.imshow(gray_lstar_g0, cmap="gray")
    plt.title("L*G0")
    plt.axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()