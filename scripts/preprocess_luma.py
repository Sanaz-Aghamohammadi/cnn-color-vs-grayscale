from pathlib import Path
import cv2
import numpy as np

from methods.luma import luma_grayscale


VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def convert_to_luma_grayscale(input_path: Path, output_path: Path) -> None:
    """
    Read one image, convert it to luma grayscale, and save it.
    """
    img = cv2.imread(str(input_path))

    if img is None:
        print(f"Skipping unreadable image: {input_path}")
        return

    # OpenCV reads images in BGR format, so convert to RGB first
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

    # Apply grayscale method from methods/luma.py
    gray = luma_grayscale(img_rgb)

    # Create output folder if needed
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save grayscale image
    cv2.imwrite(str(output_path), gray)


def is_valid_image_file(path: Path) -> bool:
    """
    Return True if the file has a valid image extension.
    """
    return path.is_file() and path.suffix.lower() in VALID_EXTENSIONS


def process_classified_split(split_name: str) -> None:
    """
    Process a split such as train or valid where images are grouped in class folders.
    """
    input_root = Path("data/raw/flowers") / split_name
    output_root = Path("data/processed/luma/flowers") / split_name

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
            convert_to_luma_grayscale(image_path, output_path)
            total_images += 1

    print(f"Finished {split_name}: {total_images} images processed.")


def process_flat_split(split_name: str) -> None:
    """
    Process a split such as test where images are directly inside the folder.
    """
    input_root = Path("data/raw/flowers") / split_name
    output_root = Path("data/processed/luma/flowers") / split_name

    if not input_root.exists():
        print(f"Input split not found: {input_root}")
        return

    image_paths = [p for p in input_root.iterdir() if is_valid_image_file(p)]

    print(f"\nProcessing flat split: {split_name}")
    print(f"Found {len(image_paths)} images.")

    for image_path in sorted(image_paths):
        output_path = output_root / image_path.name
        convert_to_luma_grayscale(image_path, output_path)

    print(f"Finished {split_name}: {len(image_paths)} images processed.")


def main() -> None:
    process_classified_split("train")
    process_classified_split("valid")
    process_flat_split("test")
    print("\nAll luma preprocessing complete.")


if __name__ == "__main__":
    main()