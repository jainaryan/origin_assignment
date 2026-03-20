import os
import sys
import json
import torch
import numpy as np
from tqdm import tqdm

# Ensure local modules can be imported
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from data.dataset import DrywallSegDataset
from models.clipseg_model import CLIPSegModel
from evaluate import compute_iou, compute_dice

def main():
    # Force CPU to completely bypass the Apple M4 MPS matrix/upsampling bug
    device = "cpu"
    print(f"Device enforced to: {device} (Bypassing MPS Bug)")
    
    # 1. Load ONLY the crack validation dataset
    print("Loading crack validation dataset...")
    val_ds = DrywallSegDataset(split="valid", dataset_keys=["crack"], use_synonyms=False)
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=4
    )
    
    # 2. Load the trained best model
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, "best_model.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        return
        
    print(f"Loading checkpoint from {checkpoint_path}...")
    model = CLIPSegModel(freeze_backbone=True)
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model = model.to(device)
    model.eval()
    
    # 3. Run inference once and collect probability maps
    all_probs = []
    all_masks = []
    
    print(f"Running inference on {len(val_ds)} validation samples...")
    with torch.no_grad():
        for images, masks, prompts, _ in tqdm(val_loader, desc="Inference"):
            images = images.to(device)
            logits = model(images, prompts)
            probs = torch.sigmoid(logits).cpu()  # Move to CPU to save VRAM
            
            all_probs.append(probs)
            all_masks.append(masks.cpu())
            
    # Concatenate all batches 
    all_probs = torch.cat(all_probs, dim=0) # [N, 1, H, W]
    all_masks = torch.cat(all_masks, dim=0) # [N, 1, H, W]
    
    # 4. Sweep thresholds from 0.20 to 0.60
    thresholds = np.arange(0.20, 0.65, 0.05)
    results = []
    
    print("\n" + "="*45)
    print(f"{'Threshold':<12} | {'mIoU':<12} | {'Dice':<12}")
    print("-" * 45)
    
    best_iou = -1.0
    best_thresh = 0.5
    
    for t in thresholds:
        t = float(t)
        
        # Binarize using current threshold
        preds = (all_probs > t).float()
        
        ious, dices = [], []
        # Calculate metrics per sample
        for i in range(preds.shape[0]):
            ious.append(compute_iou(preds[i], all_masks[i]))
            dices.append(compute_dice(preds[i], all_masks[i]))
            
        miou = np.mean(ious)
        mdice = np.mean(dices)
        
        results.append({
            "threshold": round(t, 3),
            "miou": float(miou),
            "dice": float(mdice)
        })
        
        print(f"{t:<12.2f} | {miou:<12.4f} | {mdice:<12.4f}")
        
        if miou > best_iou:
            best_iou = miou
            best_thresh = t
            
    print("=" * 45)
    print(f"\n🎯 Best Threshold: {best_thresh:.2f} (mIoU: {best_iou:.4f})")
    
    # 5. Save best threshold to JSON
    out_file = "best_crack_threshold.json"
    with open(out_file, "w") as f:
        json.dump({
            "best_threshold": round(best_thresh, 3),
            "best_miou": best_iou,
            "sweep_results": results
        }, f, indent=4)
        
    print(f"✅ Saved threshold tuning results to {out_file}")

if __name__ == "__main__":
    main()
