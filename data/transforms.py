"""
Data augmentation and preprocessing transforms.
Uses albumentations for spatial transforms applied to both image and mask.
"""

import albumentations as A
import numpy as np
from albumentations.pytorch import ToTensorV2

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


def get_train_transforms(image_size: int = config.IMAGE_SIZE):
    """Training transforms with augmentation."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3),
        A.Affine(
            translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)},
            scale=(0.9, 1.1),
            rotate=(-15, 15),
            p=0.5,
        ),
        A.RandomBrightnessContrast(
            brightness_limit=0.2,
            contrast_limit=0.2,
            p=0.5,
        ),
        A.GaussNoise(p=0.2),
        A.GaussianBlur(blur_limit=(3, 5), p=0.1),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def get_val_transforms(image_size: int = config.IMAGE_SIZE):
    """Validation/test transforms — resize and normalise only."""
    return A.Compose([
        A.Resize(image_size, image_size),
        A.Normalize(
            mean=(0.485, 0.456, 0.406),
            std=(0.229, 0.224, 0.225),
        ),
        ToTensorV2(),
    ])


def denormalize(tensor):
    """Reverse ImageNet normalisation for visualisation."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    if hasattr(tensor, "cpu"):
        img = tensor.cpu().numpy()
    else:
        img = np.array(tensor)
    if img.ndim == 3 and img.shape[0] == 3:
        img = img.transpose(1, 2, 0)
    img = img * std + mean
    img = np.clip(img * 255, 0, 255).astype(np.uint8)
    return img
