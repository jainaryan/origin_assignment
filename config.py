"""
Configuration for Prompted Segmentation for Drywall QA.
All hyperparameters, paths, seeds, and prompt mappings.
"""

import os
import random
import numpy as np
import torch

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42


def set_seed(seed: int = SEED):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
CHECKPOINT_DIR = os.path.join(OUTPUT_DIR, "checkpoints")
PREDICTION_DIR = os.path.join(OUTPUT_DIR, "predictions")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")

# ──────────────────────────────────────────────
# Model
# ──────────────────────────────────────────────
MODEL_NAME = "CIDAS/clipseg-rd64-refined"
IMAGE_SIZE = 352  # CLIPSeg native resolution

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 8
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
NUM_EPOCHS = 50
EARLY_STOP_PATIENCE = 7
SCHEDULER = "cosine"  # cosine annealing
USE_AMP = True  # mixed precision

# Loss weights
BCE_WEIGHT = 0.5
DICE_WEIGHT = 0.5

# ──────────────────────────────────────────────
# Inference
# ──────────────────────────────────────────────
THRESHOLD = 0.5  # sigmoid threshold for binarisation

# ──────────────────────────────────────────────
# Prompt mappings
# ──────────────────────────────────────────────
# Each dataset maps to multiple prompt synonyms for training augmentation.
# During evaluation, the PRIMARY prompt is used.

PROMPT_SYNONYMS = {
    "taping": [
        "segment taping area",
        "segment joint tape",
        "segment drywall seam",
        "segment drywall joint",
        "segment tape line",
    ],
    "crack": [
        "segment crack",
        "segment wall crack",
        "segment surface crack",
        "segment drywall crack",
    ],
}

PRIMARY_PROMPTS = {
    "taping": "segment taping area",
    "crack": "segment crack",
}

# ──────────────────────────────────────────────
# Roboflow dataset identifiers
# ──────────────────────────────────────────────
ROBOFLOW_DATASETS = {
    "taping": {
        "workspace": "objectdetect-pu6rn",
        "project": "drywall-join-detect",
        "version": 1,
    },
    "crack": {
        "workspace": "kyawoo",
        "project": "newcracks",
        "version": 2,
    },
}

# ──────────────────────────────────────────────
# Device
# ──────────────────────────────────────────────
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
