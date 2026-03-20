"""
Generate prediction masks for test images.

Output format (as per assignment spec):
  - PNG, single-channel, same spatial size as source image
  - Values: {0, 255}
  - Naming: {image_id}__{prompt_with_underscores}.png

Usage:
    python predict.py [--checkpoint PATH] [--split test] [--output-dir DIR]
"""

import argparse
import os
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from config import set_seed
from data.dataset import DrywallSegDataset, collate_fn
from models.clipseg_model import CLIPSegModel


def sanitize_prompt(prompt: str) -> str:
    """Convert prompt to filename-safe string: 'segment crack' → 'segment_crack'."""
    return prompt.strip().replace(" ", "_").replace("/", "_")


@torch.no_grad()
def generate_predictions(model, dataset, device, output_dir, threshold=config.THRESHOLD):
    """Generate and save prediction masks for all images in the dataset."""
    model.eval()
    os.makedirs(output_dir, exist_ok=True)

    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
    )

    total_time = 0.0
    count = 0

    for images, masks, prompts, metas in loader:
        images = images.to(device)
        meta = metas[0]
        prompt = prompts[0]

        t0 = time.time()
        logits = model(images, prompts)
        t1 = time.time()
        total_time += (t1 - t0)

        probs = torch.sigmoid(logits)
        pred = (probs > threshold).float()  # [1, 1, H, W]

        # Resize back to original image size
        orig_h, orig_w = meta["original_size"]
        pred_resized = F.interpolate(
            pred,
            size=(orig_h, orig_w),
            mode="nearest",
        )

        # Convert to numpy uint8 {0, 255}
        mask_np = (pred_resized.squeeze().cpu().numpy() * 255).astype(np.uint8)

        # Save with required naming convention
        prompt_str = sanitize_prompt(prompt)
        image_id = meta["image_id"]
        filename = f"{image_id}__{prompt_str}.png"
        save_path = os.path.join(output_dir, filename)
        cv2.imwrite(save_path, mask_np)

        count += 1

    avg_time = (total_time / max(count, 1)) * 1000
    print(f"  Generated {count} prediction masks → {output_dir}")
    print(f"  Average inference time: {avg_time:.1f} ms/image")

    return count, avg_time


def main():
    parser = argparse.ArgumentParser(description="Generate prediction masks")
    parser.add_argument("--checkpoint", type=str,
                        default=os.path.join(config.CHECKPOINT_DIR, "best_model.pth"))
    parser.add_argument("--split", type=str, default="test", choices=["train", "valid", "test"])
    parser.add_argument("--output-dir", type=str, default=config.PREDICTION_DIR)
    parser.add_argument("--threshold", type=float, default=config.THRESHOLD)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    set_seed()
    device = config.DEVICE

    print(f"\n{'='*60}")
    print(f"  Generating Prediction Masks")
    print(f"  Split: {args.split}  |  Device: {device}")
    print(f"{'='*60}\n")

    # Load model
    model = CLIPSegModel(freeze_backbone=True)
    if os.path.exists(args.checkpoint):
        ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"  Loaded checkpoint: {args.checkpoint}")
    else:
        print(f"  ⚠ No checkpoint at {args.checkpoint}, using pretrained")
    model = model.to(device)

    # Dataset (no synonyms — use primary prompt)
    dataset = DrywallSegDataset(split=args.split, use_synonyms=False)

    # Generate
    count, avg_ms = generate_predictions(
        model, dataset, device, args.output_dir, args.threshold
    )

    # Summary
    model_mb = model.get_model_size_mb()
    print(f"\n  Summary:")
    print(f"    Masks generated:    {count}")
    print(f"    Avg inference:      {avg_ms:.1f} ms/image")
    print(f"    Model size:         {model_mb:.1f} MB")
    print(f"    Output directory:   {args.output_dir}")
    print()


if __name__ == "__main__":
    main()
