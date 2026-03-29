from pathlib import Path

dataset_root = Path("data/raw/flowers")

print("Dataset root exists:", dataset_root.exists())
print()

for item in sorted(dataset_root.iterdir()):
    print(item.name)

print("\nChecking one level deeper...\n")

for split in ["train", "valid", "test"]:
    split_path = dataset_root / split
    print(f"{split}: exists = {split_path.exists()}")

    if split_path.exists():
        subfolders = [p for p in split_path.iterdir() if p.is_dir()]
        print(f"  Number of class folders: {len(subfolders)}")

        if len(subfolders) > 0:
            print("  First few class folders:")
            for folder in sorted(subfolders)[:5]:
                print("   ", folder.name)
    print()