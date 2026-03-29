from pathlib import Path

test_path = Path("data/raw/flowers/test")

print("Test path exists:", test_path.exists())
print()

items = list(test_path.iterdir())
print("Number of items in test:", len(items))
print()

print("First 20 items:")
for item in sorted(items)[:20]:
    print(item.name, "| is_dir =", item.is_dir(), "| suffix =", item.suffix)