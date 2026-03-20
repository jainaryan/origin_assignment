"""
CLIPSeg model wrapper for text-conditioned binary segmentation.

Fine-tunes only the segmentation decoder while keeping the
CLIP vision and text encoders frozen.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import CLIPSegProcessor, CLIPSegForImageSegmentation

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config


class CLIPSegModel(nn.Module):
    """
    Wrapper around HuggingFace CLIPSegForImageSegmentation.

    Freezes CLIP backbone and only trains the decoder.
    Provides a clean forward(images, prompts) → logits interface.
    """

    def __init__(self, model_name: str = config.MODEL_NAME, freeze_backbone: bool = True):
        super().__init__()
        self.processor = CLIPSegProcessor.from_pretrained(model_name)
        self.model = CLIPSegForImageSegmentation.from_pretrained(model_name)

        if freeze_backbone:
            self._freeze_backbone()

        # Count parameters
        total = sum(p.numel() for p in self.model.parameters())
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"  CLIPSeg loaded: {total:,} total params, {trainable:,} trainable")

    def _freeze_backbone(self):
        """Freeze the CLIP vision and text encoders."""
        # Freeze the CLIP model (vision + text encoders)
        for param in self.model.clip.parameters():
            param.requires_grad = False

        # Keep decoder trainable
        for param in self.model.decoder.parameters():
            param.requires_grad = True

        # Keep the film layers trainable if they exist (visual conditioning)
        if hasattr(self.model, "film_mul"):
            for param in self.model.film_mul.parameters():
                param.requires_grad = True
        if hasattr(self.model, "film_add"):
            for param in self.model.film_add.parameters():
                param.requires_grad = True

    def forward(self, images: torch.Tensor, prompts: list, target_size: int = config.IMAGE_SIZE):
        """
        Forward pass.

        Args:
            images: tensor [B, 3, H, W] — already normalised
            prompts: list of B text strings
            target_size: output mask spatial size

        Returns:
            logits: tensor [B, 1, target_size, target_size]
        """
        device = images.device

        # Tokenise text directly (avoids dummy image warnings)
        text_inputs = self.processor.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
        )

        # Build model inputs
        model_inputs = {
            "pixel_values": images,
            "input_ids": text_inputs["input_ids"].to(device),
            "attention_mask": text_inputs["attention_mask"].to(device),
        }

        outputs = self.model(**model_inputs)
        logits = outputs.logits  # [B, H', W'] — decoder output resolution

        # Resize to target size
        if logits.dim() == 3:
            logits = logits.unsqueeze(1)  # [B, 1, H', W']

        if logits.shape[-2:] != (target_size, target_size):
            logits = F.interpolate(
                logits,
                size=(target_size, target_size),
                mode="bilinear",
                align_corners=False,
            )

        return logits

    def predict(self, images: torch.Tensor, prompts: list, threshold: float = config.THRESHOLD):
        """
        Predict binary masks.

        Returns:
            masks: tensor [B, 1, H, W], binary {0, 1}
            probs: tensor [B, 1, H, W], probabilities [0, 1]
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(images, prompts)
            probs = torch.sigmoid(logits)
            masks = (probs > threshold).float()
        return masks, probs

    def get_model_size_mb(self) -> float:
        """Return model size in MB."""
        param_size = sum(p.nelement() * p.element_size() for p in self.model.parameters())
        buffer_size = sum(b.nelement() * b.element_size() for b in self.model.buffers())
        return (param_size + buffer_size) / (1024 ** 2)


class CLIPSegInference:
    """
    Simplified inference wrapper that handles image preprocessing internally.
    Use this for final prediction / evaluation when images aren't pre-normalised.
    """

    def __init__(self, model: CLIPSegModel, device: str = config.DEVICE):
        self.model = model.to(device)
        self.device = device
        self.processor = model.processor

    @torch.no_grad()
    def segment(self, images_pil: list, prompts: list, threshold: float = config.THRESHOLD):
        """
        Segment PIL images with text prompts.

        Args:
            images_pil: list of PIL.Image
            prompts: list of text prompts (same length as images)
            threshold: binarisation threshold

        Returns:
            masks: list of numpy arrays [H, W], binary {0, 255}
        """
        self.model.eval()

        inputs = self.processor(
            text=prompts,
            images=images_pil,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        outputs = self.model.model(**inputs)
        logits = outputs.logits

        if logits.dim() == 3:
            logits = logits.unsqueeze(1)

        probs = torch.sigmoid(logits)
        binary = (probs > threshold).float()

        masks = []
        for i, img in enumerate(images_pil):
            w, h = img.size
            mask = F.interpolate(
                binary[i:i+1],
                size=(h, w),
                mode="nearest",
            )
            mask_np = (mask.squeeze().cpu().numpy() * 255).astype("uint8")
            masks.append(mask_np)

        return masks
