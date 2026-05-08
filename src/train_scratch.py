"""
From-scratch training script for the color-vs-grayscale comparison.

Same structure as src/train.py but:
- NO pretrained weights (random initialization)
- SGD + momentum + Nesterov (classic ResNet-from-scratch recipe)
- Linear warmup -> cosine decay learning rate schedule
- More epochs (80) and longer early-stopping patience (20)
- Writes to results/experiments_scratch/ to keep separate from pretrained runs

Typical usage:
    python -m src.train_scratch --condition color     --seed 42
    python -m src.train_scratch --condition cie_lstar --seed 42
    python -m src.train_scratch --condition lstar_g0  --seed 42
"""

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.optim import SGD
from torchvision import models

from src.config import CONDITIONS, DATA_ROOTS, NUM_CLASSES, PROJECT_ROOT
from src.dataset import build_dataloaders


# ----------------------------------------------------------------------
# From-scratch-specific hyperparameters
# Kept here (not in src/config.py) so they don't collide with the
# pretrained experiment's settings.
# ----------------------------------------------------------------------
SCRATCH_EPOCHS        = 2
SCRATCH_LR            = 0.1
SCRATCH_MOMENTUM      = 0.9
SCRATCH_WEIGHT_DECAY  = 5e-4
SCRATCH_WARMUP_EPOCHS = 5
SCRATCH_EARLY_STOP    = 20  # more patience; from-scratch trajectories are noisier
SCRATCH_RESULTS_DIR   = PROJECT_ROOT / "results" / "experiments_scratch"


# ----------------------------------------------------------------------
# Device + seed (same as src/train.py)
# ----------------------------------------------------------------------
def pick_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() and torch.backends.mps.is_built():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if hasattr(torch, "mps"):
        try:
            torch.mps.manual_seed(seed)
        except Exception:
            pass
    torch.backends.cudnn.benchmark = True


# ----------------------------------------------------------------------
# Model - ResNet18 WITHOUT pretrained weights
# ----------------------------------------------------------------------
def build_model_from_scratch(num_classes: int) -> nn.Module:
    """
    ResNet18 with RANDOM initialization.
    `weights=None` is the key difference vs. the pretrained script.
    """
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model


def build_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    """
    Single learning rate for all parameters (no backbone/head distinction -
    everything starts from scratch).
    SGD + momentum + Nesterov is the classic ResNet recipe.
    """
    return SGD(
        model.parameters(),
        lr=SCRATCH_LR,
        momentum=SCRATCH_MOMENTUM,
        weight_decay=SCRATCH_WEIGHT_DECAY,
        nesterov=True,
    )


# ----------------------------------------------------------------------
# Learning rate schedule: linear warmup -> cosine decay
# ----------------------------------------------------------------------
def compute_lr(epoch: int, total_epochs: int, base_lr: float) -> float:
    """
    epoch is 1-indexed.
    - Epochs 1..SCRATCH_WARMUP_EPOCHS: linear warmup from 0 -> base_lr.
    - After warmup: cosine decay from base_lr -> ~0.
    """
    if epoch <= SCRATCH_WARMUP_EPOCHS:
        return base_lr * epoch / SCRATCH_WARMUP_EPOCHS

    progress = (epoch - SCRATCH_WARMUP_EPOCHS) / max(1, total_epochs - SCRATCH_WARMUP_EPOCHS)
    return base_lr * 0.5 * (1 + math.cos(math.pi * progress))


def set_lr(optimizer, lr: float) -> None:
    for g in optimizer.param_groups:
        g["lr"] = lr


# ----------------------------------------------------------------------
# Training / evaluation (identical to pretrained version)
# ----------------------------------------------------------------------
def train_one_epoch(model, loader, optimizer, criterion, device) -> float:
    model.train()
    running_loss, total = 0.0, 0
    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * images.size(0)
        total += images.size(0)
    return running_loss / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> dict:
    model.eval()
    running_loss, total = 0.0, 0
    top1_correct, top5_correct = 0, 0
    all_preds, all_targets = [], []

    for images, targets in loader:
        images  = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, targets)

        running_loss += loss.item() * images.size(0)
        total += images.size(0)

        _, top5 = logits.topk(5, dim=1)
        top1_correct += (top5[:, 0] == targets).sum().item()
        top5_correct += (top5 == targets.unsqueeze(1)).any(dim=1).sum().item()

        all_preds.append(top5[:, 0].cpu().numpy())
        all_targets.append(targets.cpu().numpy())

    preds   = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)

    return {
        "val_loss": running_loss / total,
        "top1":     top1_correct / total,
        "top5":     top5_correct / total,
        "macro_f1": f1_score(targets, preds, average="macro", zero_division=0),
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", required=True, choices=CONDITIONS)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--epochs", type=int, default=SCRATCH_EPOCHS)
    args = parser.parse_args()

    set_seed(args.seed)

    device = pick_device()
    print(f"[FROM SCRATCH] Condition: {args.condition} | Seed: {args.seed} | Device: {device}")

    data_root = DATA_ROOTS[args.condition]
    print(f"Data root: {data_root}")

    train_loader, val_loader, num_classes = build_dataloaders(data_root)
    if num_classes != NUM_CLASSES:
        print(f"Warning: found {num_classes} classes, config expects {NUM_CLASSES}")

    model = build_model_from_scratch(num_classes).to(device)
    optimizer = build_optimizer(model)
    criterion = nn.CrossEntropyLoss()

    out_dir = SCRATCH_RESULTS_DIR / args.condition / f"seed_{args.seed}"
    out_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_top1 = 0.0
    best_epoch = -1
    epochs_since_improve = 0
    start_time = time.time()

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        # Update LR for this epoch
        lr_now = compute_lr(epoch, args.epochs, SCRATCH_LR)
        set_lr(optimizer, lr_now)

        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, criterion, device)

        epoch_time = time.time() - t0
        row = {
            "epoch": epoch,
            "lr": lr_now,
            "train_loss": train_loss,
            "epoch_time_s": epoch_time,
            **val_metrics,
        }
        history.append(row)

        print(
            f"[scratch {args.condition} s{args.seed}] "
            f"epoch {epoch:02d}/{args.epochs} | "
            f"lr={lr_now:.4f} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['val_loss']:.4f} | "
            f"top1={val_metrics['top1']:.4f} | "
            f"top5={val_metrics['top5']:.4f} | "
            f"macro_f1={val_metrics['macro_f1']:.4f} | "
            f"t={epoch_time:.1f}s"
        )

        if val_metrics["top1"] > best_top1:
            best_top1 = val_metrics["top1"]
            best_epoch = epoch
            epochs_since_improve = 0
            torch.save(model.state_dict(), out_dir / "best.pt")
        else:
            epochs_since_improve += 1

        if epochs_since_improve >= SCRATCH_EARLY_STOP:
            print(f"Early stopping at epoch {epoch} (best top1={best_top1:.4f} at epoch {best_epoch})")
            break

        # Save progress after every epoch so partial results survive crashes
        partial = {
            "condition": args.condition,
            "seed": args.seed,
            "device": str(device),
            "training_mode": "from_scratch",
            "num_classes": num_classes,
            "epochs_run": len(history),
            "best_epoch": best_epoch,
            "best_top1": best_top1,
            "history": history,
            "completed": False,
        }
        with open(out_dir / "results.json", "w") as f:
            json.dump(partial, f, indent=2)

    total_time = time.time() - start_time

    final = {
        "condition": args.condition,
        "seed": args.seed,
        "device": str(device),
        "training_mode": "from_scratch",
        "num_classes": num_classes,
        "epochs_run": len(history),
        "best_epoch": best_epoch,
        "best_top1": best_top1,
        "best_top5": max(h["top5"] for h in history),
        "best_macro_f1": max(h["macro_f1"] for h in history),
        "final_top1": history[-1]["top1"],
        "total_time_s": total_time,
        "history": history,
        "completed": True,
    }
    with open(out_dir / "results.json", "w") as f:
        json.dump(final, f, indent=2)

    print(f"\nDone. Best top1={best_top1:.4f} at epoch {best_epoch}.")
    print(f"Results written to {out_dir}")


if __name__ == "__main__":
    main()
