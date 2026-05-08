"""
Standalone evaluation for a single saved checkpoint.

Loads best.pt for a given (condition, seed) and reports validation metrics.
Useful for double-checking the numbers in results.json.

Example:
    python -m src.evaluate --condition luma --seed 42
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn

from src.config import CONDITIONS, DATA_ROOTS, RESULTS_DIR
from src.dataset import build_dataloaders
from src.train import build_model, evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()

    ckpt_path = RESULTS_DIR / args.condition / f"seed_{args.seed}" / "best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, val_loader, num_classes = build_dataloaders(DATA_ROOTS[args.condition])

    model = build_model(num_classes).to(device)
    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    metrics = evaluate(model, val_loader, nn.CrossEntropyLoss(), device)

    print(f"Condition: {args.condition} | Seed: {args.seed}")
    print(f"  val_loss: {metrics['val_loss']:.4f}")
    print(f"  top1:     {metrics['top1']*100:.2f}%")
    print(f"  top5:     {metrics['top5']*100:.2f}%")
    print(f"  macro_f1: {metrics['macro_f1']*100:.2f}%")


if __name__ == "__main__":
    main()
