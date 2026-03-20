"""
Evaluation script — compute mIoU and Dice on validation/test sets.

Usage:
    python evaluate.py [--split test] [--checkpoint PATH] [--smoke-test]
"""

import argparse
import os
import sys
import time
from collections import defaultdict

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import set_seed
from data.dataset import get_dataloaders
from models.clipseg_model import CLIPSegModel


# ──────────────────────────────────────────────
# Metrics
# ──────────────────────────────────────────────

def compute_iou(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute IoU for a single sample.
    pred, target: binary tensors of same shape.
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    intersection = (pred * target).sum().item()
    union = pred.sum().item() + target.sum().item() - intersection
    return (intersection + smooth) / (union + smooth)


def compute_dice(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6) -> float:
    """
    Compute Dice coefficient for a single sample.
    pred, target: binary tensors of same shape.
    """
    pred = pred.view(-1).float()
    target = target.view(-1).float()
    intersection = (pred * target).sum().item()
    return (2.0 * intersection + smooth) / (pred.sum().item() + target.sum().item() + smooth)


def compute_metrics_batch(preds: torch.Tensor, targets: torch.Tensor):
    """
    Compute mean IoU and Dice for a batch.

    Args:
        preds: [B, 1, H, W] binary
        targets: [B, 1, H, W] binary

    Returns:
        (mean_iou, mean_dice)
    """
    batch_size = preds.shape[0]
    ious, dices = [], []

    for i in range(batch_size):
        ious.append(compute_iou(preds[i], targets[i]))
        dices.append(compute_dice(preds[i], targets[i]))

    return np.mean(ious), np.mean(dices)


# ──────────────────────────────────────────────
# Full evaluation
# ──────────────────────────────────────────────

@torch.no_grad()
def evaluate_model(model, loader, device, threshold=config.THRESHOLD):
    """
    Evaluate model on a dataloader.

    Returns:
        overall_metrics: dict with overall mIoU, Dice
        per_dataset_metrics: dict[dataset_key] → {miou, dice, count}
        per_sample_results: list of dicts per sample
    """
    model.eval()
    per_dataset = defaultdict(lambda: {"ious": [], "dices": []})
    per_sample = []
    total_inference_time = 0.0
    total_images = 0

    for images, masks, prompts, metas in loader:
        images = images.to(device)
        masks = masks.to(device)

        t0 = time.time()
        logits = model(images, prompts)
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()
        t1 = time.time()

        batch_time = t1 - t0
        total_inference_time += batch_time
        total_images += images.shape[0]

        for i in range(images.shape[0]):
            iou = compute_iou(preds[i], masks[i])
            dice = compute_dice(preds[i], masks[i])
            ds_key = metas[i]["dataset_key"]

            per_dataset[ds_key]["ious"].append(iou)
            per_dataset[ds_key]["dices"].append(dice)

            per_sample.append({
                "image_id": metas[i]["image_id"],
                "dataset_key": ds_key,
                "iou": iou,
                "dice": dice,
                "prompt": prompts[i],
            })

    # Aggregate
    per_dataset_metrics = {}
    all_ious, all_dices = [], []

    for ds_key, vals in per_dataset.items():
        miou = np.mean(vals["ious"])
        mdice = np.mean(vals["dices"])
        per_dataset_metrics[ds_key] = {
            "miou": miou,
            "dice": mdice,
            "count": len(vals["ious"]),
        }
        all_ious.extend(vals["ious"])
        all_dices.extend(vals["dices"])

    overall = {
        "miou": np.mean(all_ious) if all_ious else 0.0,
        "dice": np.mean(all_dices) if all_dices else 0.0,
        "count": len(all_ious),
        "avg_inference_ms": (total_inference_time / max(total_images, 1)) * 1000,
    }

    return overall, per_dataset_metrics, per_sample


def print_results(overall, per_dataset):
    """Pretty-print evaluation results."""
    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")

    for ds_key, m in per_dataset.items():
        prompt = config.PRIMARY_PROMPTS.get(ds_key, ds_key)
        print(f"\n  Dataset: {ds_key} (prompt: \"{prompt}\")")
        print(f"    Samples: {m['count']}")
        print(f"    mIoU:    {m['miou']:.4f}")
        print(f"    Dice:    {m['dice']:.4f}")

    print(f"\n  {'─'*40}")
    print(f"  Overall ({overall['count']} samples)")
    print(f"    mIoU:               {overall['miou']:.4f}")
    print(f"    Dice:               {overall['dice']:.4f}")
    print(f"    Avg inference time: {overall['avg_inference_ms']:.1f} ms/image")
    print(f"{'='*60}\n")


def main():
    parser = argparse.ArgumentParser(description="Evaluate segmentation model")
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--threshold", type=float, default=config.THRESHOLD)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE

    # Data
    _, val_loader, test_loader = get_dataloaders(
        batch_size=config.BATCH_SIZE,
        num_workers=args.num_workers,
    )
    loader = test_loader if args.split == "test" else val_loader

    if args.smoke_test:
        loader.dataset.samples = loader.dataset.samples[:10]
        print("  ⚠ Smoke test: using 10 samples only\n")

    # Model
    model = CLIPSegModel(freeze_backbone=True)

    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {args.checkpoint}")
        print(f"  Checkpoint mIoU: {ckpt.get('best_miou', 'N/A')}")
    else:
        print(f"  ⚠ No checkpoint found at {args.checkpoint}, using pretrained weights")

    model = model.to(device)

    # Evaluate
    overall, per_dataset, per_sample = evaluate_model(model, loader, device, args.threshold)
    print_results(overall, per_dataset)

    # Save results
    import json
    results_path = os.path.join(config.OUTPUT_DIR, f"eval_results_{args.split}.json")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    with open(results_path, "w") as f:
        json.dump({
            "overall": overall,
            "per_dataset": per_dataset,
            "per_sample": per_sample[:20],  # save first 20 for report
        }, f, indent=2, default=str)
    print(f"  Results saved to {results_path}")


if __name__ == "__main__":
    main()
