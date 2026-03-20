# Prompted Segmentation for Drywall QA

Text-conditioned binary segmentation model for drywall quality assurance — detects **cracks** and **taping areas** from natural-language prompts.

## Overview

This project fine-tunes **CLIPSeg** (CLIP + segmentation decoder) to produce binary masks conditioned on text prompts:
- `"segment crack"` — detects wall cracks
- `"segment taping area"` — detects drywall joints/seams

## Pre-Trained Model

Due to GitHub's file size limits, the fully fine-tuned `best_model.pth` checkpoint (575 MB) is hosted externally on Google Drive.

🔗 **[Download the best_model.pth Checkpoint Here](https://drive.google.com/file/d/1BZRgjdJtLIQINn6U-0kWFwpqIVHTSh90/view?usp=sharing)**

To test inferences without re-training, download the `.pth` file and place it exactly at `outputs/checkpoints/best_model.pth`.

## Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

### 1. Download & Prepare Data

```bash
python data/download_datasets.py --api-key YOUR_ROBOFLOW_API_KEY
```

### 2. Train

```bash
python train.py
```

Options:
- `--epochs N` — number of epochs (default: 50)
- `--batch-size B` — batch size (default: 8)
- `--lr LR` — learning rate (default: 1e-4)
- `--smoke-test` — quick 2-epoch test run
- `--resume PATH` — resume from checkpoint

### 3. Evaluate

```bash
python evaluate.py --split test
```

### 4. Generate Predictions

```bash
python predict.py --split test
```

Output masks are saved to `outputs/predictions/` as `{image_id}__{prompt}.png`.

### 5. Generate Visualizations

```bash
python visualize.py --split test --num-samples 4
```

## Project Structure

```
├── config.py                  # Hyperparameters, paths, seeds
├── data/
│   ├── download_datasets.py   # Roboflow download + mask conversion
│   ├── dataset.py             # PyTorch Dataset + DataLoader
│   └── transforms.py          # Augmentations (albumentations)
├── models/
│   ├── clipseg_model.py       # CLIPSeg wrapper (frozen backbone)
│   └── losses.py              # Dice + BCE combined loss
├── train.py                   # Training loop
├── evaluate.py                # mIoU & Dice evaluation
├── predict.py                 # Prediction mask generation
├── visualize.py               # Visual comparison figures
├── report/
│   └── report.md              # Assignment report
└── outputs/
    ├── predictions/           # Generated masks
    ├── checkpoints/           # Model checkpoints
    └── figures/               # Visualizations
```

## Reproducibility

- **Seed:** 42 (set for Python, NumPy, PyTorch, CUDA)
- **Python:** 3.10+
- **Key deps:** PyTorch ≥ 2.0, Transformers ≥ 4.30
- All configurations centralised in `config.py`

## Datasets

| Dataset | Source | Prompt |
|---------|--------|--------|
| Drywall-Join-Detect | [Roboflow](https://universe.roboflow.com/objectdetect-pu6rn/drywall-join-detect) | `"segment taping area"` |
| Cracks | [Roboflow](https://universe.roboflow.com/fyp-ny1jt/cracks-3ii36) | `"segment crack"` |
