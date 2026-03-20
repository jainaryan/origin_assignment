"""
Download datasets from Roboflow and prepare binary masks.

Usage:
    python data/download_datasets.py --api-key YOUR_ROBOFLOW_API_KEY

The script will:
  1. Download both datasets (taping area + cracks) from Roboflow
  2. Convert annotations to binary segmentation masks
  3. Organise into processed/train, processed/valid, processed/test splits
"""

import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np
from roboflow import Roboflow

# Allow imports from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def download_dataset(api_key: str, workspace: str, project: str, version: int, fmt: str, dest: str):
    """Download a single dataset from Roboflow."""
    rf = Roboflow(api_key=api_key)
    ws = rf.workspace(workspace)
    proj = ws.project(project)
    ds = proj.version(version)
    ds.download(fmt, location=dest)
    print(f"  ✓ Downloaded {project} v{version} to {dest}")


def coco_annotations_to_masks(coco_json_path: str, images_dir: str, masks_dir: str):
    """
    Convert COCO-format polygon annotations to binary masks.
    Falls back to bounding-box fill if no segmentation polygons exist.
    """
    with open(coco_json_path, "r") as f:
        coco = json.load(f)

    os.makedirs(masks_dir, exist_ok=True)

    # Build lookup: image_id -> image info
    id_to_info = {img["id"]: img for img in coco["images"]}

    # Group annotations by image id
    from collections import defaultdict
    anns_by_image = defaultdict(list)
    for ann in coco["annotations"]:
        anns_by_image[ann["image_id"]].append(ann)

    for img_id, img_info in id_to_info.items():
        h, w = img_info["height"], img_info["width"]
        mask = np.zeros((h, w), dtype=np.uint8)

        for ann in anns_by_image.get(img_id, []):
            # Try segmentation polygons first
            if "segmentation" in ann and ann["segmentation"]:
                for seg in ann["segmentation"]:
                    if isinstance(seg, list) and len(seg) >= 6:
                        pts = np.array(seg, dtype=np.int32).reshape(-1, 2)
                        cv2.fillPoly(mask, [pts], 255)
            # Fallback: fill bounding box
            elif "bbox" in ann:
                x, y, bw, bh = [int(v) for v in ann["bbox"]]
                mask[y : y + bh, x : x + bw] = 255

        fname = os.path.splitext(img_info["file_name"])[0] + ".png"
        cv2.imwrite(os.path.join(masks_dir, fname), mask)

    return len(id_to_info)


def yolo_annotations_to_masks(labels_dir: str, images_dir: str, masks_dir: str):
    """
    Convert YOLOv8 segmentation annotations to binary masks.
    YOLO seg format: class_id x1 y1 x2 y2 ... (normalised polygon coords)
    Falls back to bounding-box if coords look like bbox (4 values).
    """
    os.makedirs(masks_dir, exist_ok=True)
    count = 0

    for img_path in glob.glob(os.path.join(images_dir, "*")):
        ext = os.path.splitext(img_path)[1].lower()
        if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
            continue

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        mask = np.zeros((h, w), dtype=np.uint8)

        base = os.path.splitext(os.path.basename(img_path))[0]
        label_path = os.path.join(labels_dir, base + ".txt")

        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    coords = [float(v) for v in parts[1:]]

                    if len(coords) >= 6:
                        # Polygon segmentation
                        pts = np.array(coords).reshape(-1, 2)
                        pts[:, 0] *= w
                        pts[:, 1] *= h
                        pts = pts.astype(np.int32)
                        cv2.fillPoly(mask, [pts], 255)
                    elif len(coords) == 4:
                        # Bounding box (cx, cy, bw, bh)
                        cx, cy, bw, bh = coords
                        x1 = int((cx - bw / 2) * w)
                        y1 = int((cy - bh / 2) * h)
                        x2 = int((cx + bw / 2) * w)
                        y2 = int((cy + bh / 2) * h)
                        mask[max(0, y1):min(h, y2), max(0, x1):min(w, x2)] = 255

        fname = base + ".png"
        cv2.imwrite(os.path.join(masks_dir, fname), mask)
        count += 1

    return count


def process_dataset(raw_dir: str, processed_dir: str, dataset_key: str):
    """
    Detect annotation format and convert to binary masks.
    Supports COCO JSON and YOLO txt formats.
    """
    print(f"\n  Processing {dataset_key} from {raw_dir}")

    for split in ("train", "valid", "test"):
        split_dir = os.path.join(raw_dir, split)
        if not os.path.isdir(split_dir):
            print(f"    ⚠ Split '{split}' not found, skipping")
            continue

        out_images = os.path.join(processed_dir, split, "images")
        out_masks = os.path.join(processed_dir, split, "masks")
        os.makedirs(out_images, exist_ok=True)
        os.makedirs(out_masks, exist_ok=True)

        # Detect format
        coco_json = os.path.join(split_dir, "_annotations.coco.json")
        labels_dir = os.path.join(split_dir, "labels")
        images_dir = os.path.join(split_dir, "images") if os.path.isdir(os.path.join(split_dir, "images")) else split_dir

        count = 0
        if os.path.exists(coco_json):
            print(f"    [{split}] Detected COCO JSON format")
            count = coco_annotations_to_masks(coco_json, images_dir, out_masks)
        elif os.path.isdir(labels_dir):
            print(f"    [{split}] Detected YOLO format")
            count = yolo_annotations_to_masks(labels_dir, images_dir, out_masks)
        else:
            print(f"    [{split}] ⚠ No recognised annotation format found")
            # Still copy images so the pipeline doesn't break
            count = 0

        # Copy / symlink images
        for img_path in glob.glob(os.path.join(images_dir, "*")):
            ext = os.path.splitext(img_path)[1].lower()
            if ext in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                dst = os.path.join(out_images, os.path.basename(img_path))
                if not os.path.exists(dst):
                    # Copy to keep things self-contained
                    import shutil
                    shutil.copy2(img_path, dst)

        print(f"    [{split}] Processed {count} images → {out_masks}")


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets")
    parser.add_argument("--api-key", type=str, required=True, help="Roboflow API key")
    parser.add_argument("--format", type=str, default="coco-segmentation",
                        choices=["coco-segmentation", "coco", "yolov8"],
                        help="Annotation format to download (default: coco-segmentation)")
    args = parser.parse_args()

    os.makedirs(config.RAW_DATA_DIR, exist_ok=True)
    os.makedirs(config.PROCESSED_DATA_DIR, exist_ok=True)

    for key, ds_info in config.ROBOFLOW_DATASETS.items():
        raw_dest = os.path.join(config.RAW_DATA_DIR, key)
        processed_dest = os.path.join(config.PROCESSED_DATA_DIR, key)

        if os.path.isdir(raw_dest) and os.listdir(raw_dest):
            print(f"[{key}] Raw data already exists at {raw_dest}, skipping download")
        else:
            print(f"[{key}] Downloading from Roboflow ...")
            download_dataset(
                api_key=args.api_key,
                workspace=ds_info["workspace"],
                project=ds_info["project"],
                version=ds_info["version"],
                fmt=args.format,
                dest=raw_dest,
            )

        process_dataset(raw_dest, processed_dest, key)

    print("\n✅ All datasets downloaded and processed.")


if __name__ == "__main__":
    main()
