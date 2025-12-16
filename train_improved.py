#!/usr/bin/env python3
"""
Improved Kannada Handwriting Recognition Training Script
(Fixed: saves idx_to_class mapping correctly)
"""

import os
import sys
import time
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, OneCycleLR
from tqdm import tqdm

# Add src to path
sys.path.append("src")

from data.dataset import create_dataloaders
from data.kannada_mnist_csv import create_csv_dataloaders
from models.cnn import ImprovedKannadaCNN, KannadaCNN
from utils.transforms import build_transforms

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("Warning: wandb not available. Logging will be disabled.")


# -------------------------------------------------
# Utility functions
# -------------------------------------------------

def accuracy_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=1)
    return (preds == targets).float().mean().item()


def train_one_epoch(model, loader, optimizer, device, scaler, loss_fn, scheduler=None):
    model.train()
    total_loss = 0.0
    total_acc = 0.0

    for batch_idx, (imgs, labels) in enumerate(tqdm(loader, desc="train", leave=False)):
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=torch.float16):
            _, logits = model(imgs)
            loss = loss_fn(logits, labels)

        scaler.scale(loss).backward()

        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        scaler.step(optimizer)
        scaler.update()

        if scheduler is not None and isinstance(scheduler, OneCycleLR):
            scheduler.step()

        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)

    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


@torch.no_grad()
def evaluate(model, loader, device, loss_fn):
    model.eval()
    total_loss = 0.0
    total_acc = 0.0

    for imgs, labels in tqdm(loader, desc="val", leave=False):
        imgs, labels = imgs.to(device), labels.to(device)
        _, logits = model(imgs)
        loss = loss_fn(logits, labels)

        total_loss += loss.item() * imgs.size(0)
        total_acc += accuracy_from_logits(logits, labels) * imgs.size(0)

    return total_loss / len(loader.dataset), total_acc / len(loader.dataset)


# -------------------------------------------------
# Main
# -------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--kaggle_csv", action="store_true")
    p.add_argument("--out_dir", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image_size", type=int, default=64)
    p.add_argument("--embedding_dim", type=int, default=256)
    p.add_argument("--grayscale", action="store_true")
    p.add_argument("--use_improved_model", action="store_true")
    p.add_argument("--scheduler", type=str, default="cosine",
                   choices=["cosine", "plateau", "onecycle"])
    p.add_argument("--weight_decay", type=float, default=1e-4)
    p.add_argument("--use_wandb", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_tfms, val_tfms = build_transforms(args.image_size, grayscale=True)

    if args.kaggle_csv:
        bundle = create_csv_dataloaders(
            args.data_dir, train_tfms, val_tfms,
            batch_size=args.batch_size, num_workers=0
        )
    else:
        bundle = create_dataloaders(
            args.data_dir, train_tfms, val_tfms,
            batch_size=args.batch_size, num_workers=0
        )

    print(f"Dataset loaded: {bundle.num_classes} classes")

    # -------------------------------------------------
    # 🔥 CRITICAL FIX: SAVE CLASS ↔ INDEX MAPPING
    # -------------------------------------------------

    train_dataset = bundle.train.dataset
    if hasattr(train_dataset, "dataset"):
        train_dataset = train_dataset.dataset

    class_to_idx = train_dataset.class_to_idx
    idx_to_class = {v: k for k, v in class_to_idx.items()}

    os.makedirs(args.out_dir, exist_ok=True)

    json_path = Path(args.out_dir) / "idx_to_class.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(idx_to_class, f, ensure_ascii=False, indent=2)

    print(f"✅ Saved idx_to_class.json ({len(idx_to_class)} classes)")

    # -------------------------------------------------
    # Model
    # -------------------------------------------------

    ModelClass = ImprovedKannadaCNN if args.use_improved_model else KannadaCNN

    model = ModelClass(
        in_channels=1,
        embedding_dim=args.embedding_dim,
        num_classes=bundle.num_classes
    ).to(device)

    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    if args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    elif args.scheduler == "plateau":
        scheduler = ReduceLROnPlateau(optimizer, mode="max")
    else:
        scheduler = OneCycleLR(
            optimizer,
            max_lr=args.lr * 10,
            epochs=args.epochs,
            steps_per_epoch=len(bundle.train)
        )

    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == "cuda")

    best_acc = 0.0
    best_path = Path(args.out_dir) / "best_improved.pt"

    # -------------------------------------------------
    # Training loop
    # -------------------------------------------------

    for epoch in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_one_epoch(
            model, bundle.train, optimizer, device, scaler, loss_fn, scheduler
        )

        va_loss, va_acc = evaluate(
            model, bundle.val, device, loss_fn
        )

        if args.scheduler in ["cosine", "plateau"]:
            scheduler.step(va_acc if args.scheduler == "plateau" else None)

        print(
            f"Epoch {epoch:03d} | "
            f"Train Acc: {tr_acc:.4f} | "
            f"Val Acc: {va_acc:.4f}"
        )

        if va_acc > best_acc:
            best_acc = va_acc

            torch.save({
                "model": model.state_dict(),
                "num_classes": bundle.num_classes,
                "embedding_dim": args.embedding_dim,
                "architecture": model.__class__.__name__,
                "epoch": epoch,
                "val_acc": va_acc,

                # 🔥 MUST HAVE
                "class_to_idx": class_to_idx,
                "idx_to_class": idx_to_class,
            }, best_path)

            print(f"✅ Best model saved ({best_acc:.4f})")

    print("🎉 Training completed")


if __name__ == "__main__":
    main()
