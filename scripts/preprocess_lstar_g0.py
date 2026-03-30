import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

import cv2

from methods.g0_model import G0Model
from methods.lstar_g0 import lstar_g0_grayscale


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_to_lstar_g0_grayscale(input_path: Path, output_path: Path, g0_model: G0Model) -> None:
    """
    Read one image, convert it to L*G0 grayscale, and save it.
    """
    img = cv2.imread(str(input_path))

    if img is None:
        print(f"Skipping unreadable image: {input_path}")
        return

    # OpenCV loads BGR, convert to RGB
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Apply L*G0 method
    gray = lstar_g0_grayscale(img_rgb, g0_model)

    # Ensure output folder exists
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save grayscale image
    cv2.imwrite(str(output_path), gray)


def is_valid_image_file(path: Path) -> bool:
    """
    Return True if the file has a supported image extension.
    """
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def process_classified_split(split_name: str, g0_model: G0Model) -> None:
    """
    Process a split such as train or valid where images are inside class folders.
    """
    input_root = Path("data/raw/flowers") / split_name
    output_root = Path("data/processed/lstar_g0/flowers") / split_name

    if not input_root.exists():
        print(f"Input split not found: {input_root}")
        return

    class_folders = [p for p in input_root.iterdir() if p.is_dir()]

    print(f"\nProcessing classified split: {split_name}")
    print(f"Found {len(class_folders)} class folders.")

    total_images = 0

    for class_folder in sorted(class_folders):
        image_paths = [p for p in class_folder.iterdir() if is_valid_image_file(p)]
        print(f"  Class {class_folder.name}: {len(image_paths)} images")

        for image_path in image_paths:
            output_path = output_root / class_folder.name / image_path.name
            convert_to_lstar_g0_grayscale(image_path, output_path, g0_model)
            total_images += 1

    print(f"Finished {split_name}: {total_images} images processed.")


def process_flat_split(split_name: str, g0_model: G0Model) -> None:
    """
    Process a split such as test where images are directly inside the folder.
    """
    input_root = Path("data/raw/flowers") / split_name
    output_root = Path("data/processed/lstar_g0/flowers") / split_name

    if not input_root.exists():
        print(f"Input split not found: {input_root}")
        return

    image_paths = [p for p in input_root.iterdir() if is_valid_image_file(p)]

    print(f"\nProcessing flat split: {split_name}")
    print(f"Found {len(image_paths)} images.")

    for image_path in sorted(image_paths):
        output_path = output_root / image_path.name
        convert_to_lstar_g0_grayscale(image_path, output_path, g0_model)

    print(f"Finished {split_name}: {len(image_paths)} images processed.")


def main() -> None:
    print("Loading G0 model...")
    g0_model = G0Model(
        cmf_path="data/reference/CMF2deg.xlsx",
        d65_path="data/reference/D65.xlsx",
        lut_path="data/reference/LUT.xlsx",
        white_y=1.0,
    )
    print("G0 model loaded.")

    process_classified_split("train", g0_model)
    process_classified_split("valid", g0_model)
    process_flat_split("test", g0_model)

    print("\nAll L*G0 preprocessing complete.")


if __name__ == "__main__":
    main()