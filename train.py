"""
Training script for CLIPSeg text-conditioned segmentation.

Usage:
    python train.py [--epochs N] [--batch-size B] [--lr LR] [--smoke-test] [--subset N]
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingLR

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import set_seed
from data.dataset import get_dataloaders
from models.clipseg_model import CLIPSegModel
from models.losses import CombinedLoss
from evaluate import compute_metrics_batch


def train_one_epoch(model, loader, criterion, optimizer, scaler, device, use_amp):
    """Train for one epoch. Returns average loss."""
    model.train()
    total_loss = 0.0
    num_batches = 0

    for images, masks, prompts, metas in loader:
        images = images.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()

        if use_amp and device == "cuda":
            with autocast(device_type="cuda"):
                logits = model(images, prompts)
                loss = criterion(logits, masks)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images, prompts)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(num_batches, 1)


@torch.no_grad()
def validate(model, loader, criterion, device):
    """Validate and return loss + metrics."""
    model.eval()
    total_loss = 0.0
    all_iou = []
    all_dice = []
    num_batches = 0

    for images, masks, prompts, metas in loader:
        images = images.to(device)
        masks = masks.to(device)

        logits = model(images, prompts)
        loss = criterion(logits, masks)
        total_loss += loss.item()

        # Compute per-batch metrics
        probs = torch.sigmoid(logits)
        preds = (probs > config.THRESHOLD).float()
        iou, dice = compute_metrics_batch(preds, masks)
        all_iou.append(iou)
        all_dice.append(dice)
        num_batches += 1

    avg_loss = total_loss / max(num_batches, 1)
    avg_iou = sum(all_iou) / max(len(all_iou), 1)
    avg_dice = sum(all_dice) / max(len(all_dice), 1)

    return avg_loss, avg_iou, avg_dice


def main():
    parser = argparse.ArgumentParser(description="Train CLIPSeg for drywall QA")
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    parser.add_argument("--smoke-test", action="store_true", help="Quick 2-epoch test")
    parser.add_argument("--subset", type=int, default=0, help="Use N samples only")
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--resume", type=str, default="", help="Path to checkpoint")
    args = parser.parse_args()

    if args.smoke_test:
        args.epochs = 2
        args.subset = args.subset or 10

    set_seed()
    device = config.DEVICE
    print(f"\n{'='*60}")
    print(f"  Training CLIPSeg for Drywall QA Segmentation")
    print(f"  Device: {device}  |  Epochs: {args.epochs}  |  LR: {args.lr}")
    print(f"{'='*60}\n")

    # Data
    train_loader, val_loader, _ = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Optionally limit dataset size
    if args.subset > 0:
        train_loader.dataset.samples = train_loader.dataset.samples[:args.subset]
        val_loader.dataset.samples = val_loader.dataset.samples[:args.subset]
        print(f"  ⚠ Using subset: {args.subset} samples\n")

    # Model
    model = CLIPSegModel(freeze_backbone=True)
    model = model.to(device)

    # Loss
    criterion = CombinedLoss()

    # Optimizer — only trainable params
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=config.WEIGHT_DECAY,
    )

    # Scheduler
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    # AMP scaler
    scaler = GradScaler() if config.USE_AMP and device == "cuda" else None

    # Resume
    start_epoch = 0
    best_miou = 0.0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        best_miou = ckpt.get("best_miou", 0.0)
        print(f"  Resumed from {args.resume} (epoch {start_epoch})")

    # Checkpoint dir
    os.makedirs(config.CHECKPOINT_DIR, exist_ok=True)

    # Training loop
    patience_counter = 0
    train_start = time.time()

    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()

        # Train
        train_loss = train_one_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config.USE_AMP
        )

        # Validate
        val_loss, val_miou, val_dice = validate(model, val_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        epoch_time = time.time() - epoch_start
        current_lr = scheduler.get_last_lr()[0]

        print(
            f"  Epoch [{epoch+1}/{args.epochs}]  "
            f"Train Loss: {train_loss:.4f}  |  "
            f"Val Loss: {val_loss:.4f}  |  "
            f"Val mIoU: {val_miou:.4f}  |  "
            f"Val Dice: {val_dice:.4f}  |  "
            f"LR: {current_lr:.2e}  |  "
            f"Time: {epoch_time:.1f}s"
        )

        # Save best model
        if val_miou > best_miou:
            best_miou = val_miou
            patience_counter = 0
            ckpt_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_miou": best_miou,
                "val_dice": val_dice,
            }, ckpt_path)
            print(f"    ✓ Best model saved (mIoU={best_miou:.4f})")
        else:
            patience_counter += 1

        # Save latest
        latest_path = os.path.join(config.CHECKPOINT_DIR, "latest_model.pth")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_miou": best_miou,
        }, latest_path)

        # Early stopping
        if patience_counter >= config.EARLY_STOP_PATIENCE:
            print(f"\n  ⏹ Early stopping at epoch {epoch+1} (patience={config.EARLY_STOP_PATIENCE})")
            break

    total_time = time.time() - train_start
    print(f"\n{'='*60}")
    print(f"  Training complete in {total_time:.1f}s ({total_time/60:.1f} min)")
    print(f"  Best validation mIoU: {best_miou:.4f}")
    print(f"  Model size: {model.get_model_size_mb():.1f} MB")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
