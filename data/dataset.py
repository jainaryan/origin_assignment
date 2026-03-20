"""
PyTorch Dataset class for prompted segmentation.

Each sample returns:
  - image: tensor [3, H, W], normalised
  - mask:  tensor [1, H, W], binary {0, 1}
  - prompt: str, randomly sampled from synonyms (train) or primary (eval)
  - meta:  dict with image_id, original_size, dataset_key
"""

import glob
import os
import random

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config
from data.transforms import get_train_transforms, get_val_transforms


class DrywallSegDataset(Dataset):
    """
    Combined dataset for both taping-area and crack segmentation.
    Supports text-prompt conditioning.
    """

    def __init__(
        self,
        split: str = "train",
        dataset_keys: list = None,
        use_synonyms: bool = True,
        image_size: int = config.IMAGE_SIZE,
    ):
        """
        Args:
            split: one of 'train', 'valid', 'test'
            dataset_keys: list of dataset keys to include, e.g. ['taping', 'crack']
                          If None, include all.
            use_synonyms: if True, randomly sample prompt synonyms (for training)
            image_size: resize target
        """
        super().__init__()
        self.split = split
        self.use_synonyms = use_synonyms
        self.image_size = image_size

        if dataset_keys is None:
            dataset_keys = list(config.PROMPT_SYNONYMS.keys())

        self.samples = []  # list of (image_path, mask_path, dataset_key)

        for key in dataset_keys:
            img_dir = os.path.join(config.PROCESSED_DATA_DIR, key, split, "images")
            mask_dir = os.path.join(config.PROCESSED_DATA_DIR, key, split, "masks")

            if not os.path.isdir(img_dir):
                print(f"⚠ Image dir not found: {img_dir}")
                continue

            for img_path in sorted(glob.glob(os.path.join(img_dir, "*"))):
                ext = os.path.splitext(img_path)[1].lower()
                if ext not in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
                    continue

                base = os.path.splitext(os.path.basename(img_path))[0]
                mask_path = os.path.join(mask_dir, base + ".png")

                if not os.path.exists(mask_path):
                    # Try other extensions
                    for mext in (".jpg", ".jpeg", ".png"):
                        alt = os.path.join(mask_dir, base + mext)
                        if os.path.exists(alt):
                            mask_path = alt
                            break

                self.samples.append((img_path, mask_path, key))

        # Set transforms
        if split == "train":
            self.transform = get_train_transforms(image_size)
        else:
            self.transform = get_val_transforms(image_size)

        print(f"  DrywallSegDataset [{split}]: {len(self.samples)} samples "
              f"from {dataset_keys}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, mask_path, dataset_key = self.samples[idx]

        # Load image (BGR → RGB)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Cannot read image: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # Load mask (grayscale)
        if os.path.exists(mask_path):
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if mask is None:
                mask = np.zeros((original_h, original_w), dtype=np.uint8)
        else:
            mask = np.zeros((original_h, original_w), dtype=np.uint8)

        # Binarise mask → {0, 1}
        mask = (mask > 127).astype(np.uint8)

        # Apply transforms (both image & mask)
        transformed = self.transform(image=image, mask=mask)
        image_t = transformed["image"]          # [3, H, W] float
        mask_t = transformed["mask"]            # [H, W] uint8

        # Mask to float tensor [1, H, W]
        mask_t = mask_t.float().unsqueeze(0)

        # Select prompt
        if self.use_synonyms:
            prompt = random.choice(config.PROMPT_SYNONYMS[dataset_key])
        else:
            prompt = config.PRIMARY_PROMPTS[dataset_key]

        # Image ID from filename
        image_id = os.path.splitext(os.path.basename(img_path))[0]

        meta = {
            "image_id": image_id,
            "original_size": (original_h, original_w),
            "dataset_key": dataset_key,
            "image_path": img_path,
        }

        return image_t, mask_t, prompt, meta


def collate_fn(batch):
    """
    Custom collate to handle variable-length prompts and meta dicts.
    """
    images = torch.stack([b[0] for b in batch])
    masks = torch.stack([b[1] for b in batch])
    prompts = [b[2] for b in batch]
    metas = [b[3] for b in batch]
    return images, masks, prompts, metas


def get_dataloaders(batch_size: int = config.BATCH_SIZE, num_workers: int = 4):
    """Create train, validation, and test dataloaders."""
    train_ds = DrywallSegDataset(split="train", use_synonyms=True)
    val_ds = DrywallSegDataset(split="valid", use_synonyms=False)
    test_ds = DrywallSegDataset(split="test", use_synonyms=False)

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
