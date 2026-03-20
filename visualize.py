"""
Visualisation script — generate side-by-side comparison images.
Creates triplets: Original | Ground Truth | Prediction

Usage:
    python visualize.py [--checkpoint PATH] [--num-samples 4] [--split test]
"""

import argparse
import os
import sys
import random

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import set_seed
from data.dataset import DrywallSegDataset, collate_fn
from data.transforms import denormalize
from models.clipseg_model import CLIPSegModel


@torch.no_grad()
def generate_visualizations(
    model, dataset, device, output_dir, num_samples=4, threshold=config.THRESHOLD
):
    """Generate comparison visualizations: Original | GT | Prediction."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    # Sample indices (spread across datasets)
    indices = list(range(len(dataset)))
    random.shuffle(indices)

    # Try to balance across dataset keys
    by_key = {}
    for idx in indices:
        _, _, _, meta = dataset[idx]
        key = meta["dataset_key"]
        by_key.setdefault(key, []).append(idx)

    selected = []
    per_key = max(1, num_samples // len(by_key))
    for key, idxs in by_key.items():
        selected.extend(idxs[:per_key])
    selected = selected[:num_samples]

    fig, axes = plt.subplots(num_samples, 3, figsize=(15, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(selected):
        image_t, mask_t, prompt, meta = dataset[idx]

        # Predict
        image_batch = image_t.unsqueeze(0).to(device)
        logits = model(image_batch, [prompt])
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

        # Convert for display
        img_display = denormalize(image_t)
        gt_display = mask_t.squeeze().numpy() * 255
        pred_display = pred.squeeze().cpu().numpy() * 255

        # Original image
        axes[row, 0].imshow(img_display)
        axes[row, 0].set_title(f"Original\n{meta['image_id']}", fontsize=10)
        axes[row, 0].axis("off")

        # Ground truth
        axes[row, 1].imshow(gt_display, cmap="gray", vmin=0, vmax=255)
        axes[row, 1].set_title(f"Ground Truth\n({meta['dataset_key']})", fontsize=10)
        axes[row, 1].axis("off")

        # Prediction
        axes[row, 2].imshow(pred_display, cmap="gray", vmin=0, vmax=255)
        axes[row, 2].set_title(f"Prediction\n\"{prompt}\"", fontsize=10)
        axes[row, 2].axis("off")

    plt.suptitle("Segmentation Results: Original | Ground Truth | Prediction",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "visual_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved visual comparison → {save_path}")

    return save_path


def generate_overlay_visualization(
    model, dataset, device, output_dir, num_samples=4, threshold=config.THRESHOLD
):
    """Generate overlay visualizations with masks on top of images."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    indices = random.sample(range(len(dataset)), min(num_samples, len(dataset)))

    fig, axes = plt.subplots(num_samples, 2, figsize=(12, 5 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for row, idx in enumerate(indices):
        image_t, mask_t, prompt, meta = dataset[idx]
        image_batch = image_t.unsqueeze(0).to(device)
        logits = model(image_batch, [prompt])
        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()

        img = denormalize(image_t)
        gt = mask_t.squeeze().numpy()
        pr = pred.squeeze().cpu().numpy()

        # GT overlay (green)
        gt_overlay = img.copy()
        gt_overlay[gt > 0.5] = [0, 255, 0]
        gt_blend = cv2.addWeighted(img, 0.6, gt_overlay, 0.4, 0)

        # Pred overlay (red)
        pred_overlay = img.copy()
        pred_overlay[pr > 0.5] = [255, 0, 0]
        pred_blend = cv2.addWeighted(img, 0.6, pred_overlay, 0.4, 0)

        axes[row, 0].imshow(gt_blend)
        axes[row, 0].set_title(f"GT Overlay (green)\n{meta['image_id']}", fontsize=10)
        axes[row, 0].axis("off")

        axes[row, 1].imshow(pred_blend)
        axes[row, 1].set_title(f"Pred Overlay (red)\n\"{prompt}\"", fontsize=10)
        axes[row, 1].axis("off")

    plt.suptitle("Overlay Comparison: GT (green) | Prediction (red)",
                 fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()

    save_path = os.path.join(output_dir, "overlay_comparison.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved overlay comparison → {save_path}")

    return save_path


def main():
    parser = argparse.ArgumentParser(description="Generate visualizations")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--split", type=str, default="test", choices=["valid", "test"])
    parser.add_argument("--num-samples", type=int, default=4)
    parser.add_argument("--threshold", type=float, default=config.THRESHOLD)
    parser.add_argument("--output-dir", type=str, default=config.FIGURES_DIR)
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE

    print(f"\n{'='*60}")
    print(f"  Generating Visualizations")
    print(f"{'='*60}\n")

    # Model
    model = CLIPSegModel(freeze_backbone=True)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {args.checkpoint}")
    model = model.to(device)

    # Dataset
    dataset = DrywallSegDataset(split=args.split, use_synonyms=False)

    # Generate both types
    generate_visualizations(
        model, dataset, device, args.output_dir,
        num_samples=args.num_samples, threshold=args.threshold,
    )
    generate_overlay_visualization(
        model, dataset, device, args.output_dir,
        num_samples=args.num_samples, threshold=args.threshold,
    )

    print(f"\n  ✓ All visualizations saved to {args.output_dir}\n")


if __name__ == "__main__":
    main()
