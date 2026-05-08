"""
Single-run training script for the color-vs-grayscale comparison.

Trains ONE (dataset, condition, seed) triple from scratch. Writes a
results.json and best checkpoint to results/<dataset>/<condition>/seed_<seed>/.

Usage:
    python -m src.train --dataset gtsrb --condition color     --seed 42
    python -m src.train --dataset fmd   --condition color     --seed 42
    python -m src.train --dataset fmd   --condition luma      --seed 42
    (--dataset defaults to gtsrb for backward compatibility)
"""

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torchvision import models

from src.config import (
    CONDITIONS,
    DATASETS,
    LR,
    MOMENTUM,
    NESTEROV,
    NUM_EPOCHS,
    WEIGHT_DECAY,
    get_dataset_config,
)
from src.dataset import build_dataloaders


# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------

def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.mps.manual_seed(seed)
    except AttributeError:
        pass


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_model(num_classes: int) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


# ---------------------------------------------------------------------------
# Training / evaluation loops
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    running_loss, total = 0.0, 0
    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        loss = criterion(model(images), targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / total


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict:
    model.eval()
    running_loss, total = 0.0, 0
    top1_correct, top5_correct = 0, 0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        running_loss += criterion(logits, targets).item() * images.size(0)
        total += images.size(0)

        _, top5 = logits.topk(5, dim=1)
        top1_correct += (top5[:, 0] == targets).sum().item()
        top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        all_preds.append(top5[:, 0].cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    return {
        "loss":      running_loss / total,
        "top1":      top1_correct / total,
        "top5":      top5_correct / total,
        "macro_f1":  float(f1_score(targets, preds, average="macro", zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Results persistence
# ---------------------------------------------------------------------------

def _save_results(path: Path, payload: dict) -> None:
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset",   default="gtsrb", choices=DATASETS)
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed",      type=int, required=True)
    parser.add_argument("--epochs",    type=int, default=NUM_EPOCHS)
    args = parser.parse_args()

    cfg = get_dataset_config(args.dataset)

    set_seed(args.seed)
    device = pick_device()
    print(f"Dataset: {args.dataset} | Condition: {args.condition} | "
          f"Seed: {args.seed} | Device: {device}")

    train_loader, val_loader, test_loader = build_dataloaders(
        args.condition, args.dataset)

    model     = build_model(cfg["num_classes"]).to(device)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=LR,
        momentum=MOMENTUM,
        weight_decay=WEIGHT_DECAY,
        nesterov=NESTEROV,
    )
    criterion = nn.CrossEntropyLoss()

    out_dir = cfg["results_dir"] / args.condition / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    results_path = out_dir / "results.json"
    ckpt_path    = out_dir / "best.pt"

    history: list[dict] = []
    best_val_top1 = 0.0
    best_epoch    = -1
    start_time    = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        epoch_secs  = time.time() - t0

        row = {
            "epoch":        epoch,
            "train_loss":   round(train_loss, 6),
            "val_loss":     round(val_metrics["loss"], 6),
            "val_top1":     round(val_metrics["top1"], 6),
            "val_top5":     round(val_metrics["top5"], 6),
            "val_macro_f1": round(val_metrics["macro_f1"], 6),
            "epoch_time_s": round(epoch_secs, 2),
        }
        history.append(row)

        print(
            f"[{args.condition} s{args.seed}] "
            f"ep {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"top1={val_metrics['top1']:.4f} | "
            f"top5={val_metrics['top5']:.4f} | "
            f"f1={val_metrics['macro_f1']:.4f} | "
            f"{epoch_secs:.1f}s"
        )

        if val_metrics["top1"] > best_val_top1:
            best_val_top1 = val_metrics["top1"]
            best_epoch    = epoch
            torch.save(model.state_dict(), ckpt_path)

        # Save after every epoch so partial runs are recoverable
        _save_results(results_path, {
            "condition":     args.condition,
            "seed":          args.seed,
            "device":        str(device),
            "epochs_run":    len(history),
            "best_epoch":    best_epoch,
            "best_val_top1": round(best_val_top1, 6),
            "total_time_s":  round(time.time() - start_time, 2),
            "test":          None,
            "history":       history,
        })

    # Final evaluation on the test set using the best checkpoint
    print("\nRunning test evaluation with best checkpoint...")
    model.load_state_dict(
        torch.load(ckpt_path, map_location=device, weights_only=True)
    )
    test_metrics = evaluate(model, test_loader, criterion, device)

    print(
        f"Test | top1={test_metrics['top1']:.4f} | "
        f"top5={test_metrics['top5']:.4f} | "
        f"f1={test_metrics['macro_f1']:.4f}"
    )

    total_time = time.time() - start_time
    _save_results(results_path, {
        "condition":     args.condition,
        "seed":          args.seed,
        "device":        str(device),
        "epochs_run":    len(history),
        "best_epoch":    best_epoch,
        "best_val_top1": round(best_val_top1, 6),
        "total_time_s":  round(total_time, 2),
        "test": {
            "top1":      round(test_metrics["top1"], 6),
            "top5":      round(test_metrics["top5"], 6),
            "macro_f1":  round(test_metrics["macro_f1"], 6),
        },
        "history": history,
    })

    print(f"Results -> {results_path}")
    print(f"Done. Best val top1={best_val_top1:.4f} at epoch {best_epoch} | "
          f"total {total_time/60:.1f} min")


if __name__ == "__main__":
    main()
