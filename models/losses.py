"""
Loss functions for binary segmentation:
  - Dice Loss
  - Binary Cross-Entropy (with logits)
  - Combined (weighted sum)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class DiceLoss(nn.Module):
    """
    Soft Dice loss for binary segmentation.
    Operates on logits (applies sigmoid internally).
    """

    def __init__(self, smooth: float = 1.0):
        super().__init__()
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [B, 1, H, W] raw model output
            targets: [B, 1, H, W] binary {0, 1}
        """
        probs = torch.sigmoid(logits)
        probs = probs.view(-1)
        targets = targets.view(-1)

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class FocalLoss(nn.Module):
    """
    Focal loss for handling class imbalance.
    Reduces loss for well-classified examples.
    """

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        focal_weight = self.alpha * (1 - pt) ** self.gamma
        loss = focal_weight * bce
        return loss.mean()


class CombinedLoss(nn.Module):
    """
    Weighted combination of BCE and Dice loss.
    Default: 0.5 * BCE + 0.5 * Dice
    """

    def __init__(
        self,
        bce_weight: float = config.BCE_WEIGHT,
        dice_weight: float = config.DICE_WEIGHT,
        dice_smooth: float = 1.0,
    ):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss(smooth=dice_smooth)

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss
