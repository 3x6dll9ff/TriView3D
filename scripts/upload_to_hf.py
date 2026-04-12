#!/usr/bin/env python3
"""Upload trained models to HuggingFace Hub.

Usage:
    python3 scripts/upload_to_hf.py
    python3 scripts/upload_to_hf.py --repo_id 3x6dll9ff/diplom
    python3 scripts/upload_to_hf.py --models-dir results --dry-run
"""

import argparse
import os
import sys
from pathlib import Path

from huggingface_hub import HfApi

REPO_ID = "3x6dll9ff/diplom"

MODEL_FILES = [
    "best_autoencoder.pt",
    "best_autoencoder_patched.pt",
    "best_refiner.pt",
    "best_vae.pt",
    "best_classifier.pt",
]

METRICS_DIR = "metrics"

MODEL_CARD = """# TriView3D — Multi-View 3D Cell Reconstruction

Reconstruction of 3D cell morphology from 3 orthogonal projections (top, bottom, side).

## Architecture

### Base CNN (`best_autoencoder.pt`)
- **Encoder2D**: 2D ConvNet [3, 64, 64] → latent (256)
- **Decoder3D**: 3D ConvTranspose → [1, 64, 64, 64] with skip connections from lifted views
- **Loss**: BCE + Dice + Projection Consistency (BCE) + Boundary-aware BCE
- **Training**: AdamW + 5-epoch LR warmup + complexity-weighted sampling + augmentation

### Detail Refiner (`best_refiner.pt`)
- Uncertainty-gated residual refinement on top of coarse CNN prediction
- **Input**: coarse logits + lifted views → gating * residual
- **Skip connection version**: compatible with skip-connected autoencoder

### Conditional VAE (`best_vae.pt`)
- **Encoder**: 2D ConvNet → mu (256) + logvar (256)
- **Decoder**: 3D ConvTranspose with skip connections
- **Inference**: best-of-K sampling (K=8) by reprojection error
- **Loss**: Reconstruction + KL divergence

### Classifier (`best_classifier.pt`)
- MLP on latent vector (256) + morphometric features (5: volume, sphericity, convexity, surface_area, compactness)
- **Task**: normal vs anomaly classification

## Input Modes
- **tri**: 3 channels (top_proj, bottom_proj, side_proj) — primary mode
- **quad**: 4 channels (top_proj, bottom_proj, side_proj, front_proj)

## Metrics (previous version, before skip connections + bug fixes)
- Dice: ~0.90
- IoU: ~0.82

**Note**: Models uploaded before the pipeline overhaul may have lower quality. Retrained models with skip connections, TTA, augmentation, and bug fixes are expected to achieve significantly higher Dice.

## Usage

```python
import torch
from src.autoencoder import TriViewAutoencoder
from src.reconstruction_utils import infer_in_channels_from_state_dict, infer_skip_channels_from_state_dict

state_dict = torch.load("best_autoencoder.pt", map_location="cpu")
if "model_state_dict" in state_dict:
    state_dict = state_dict["model_state_dict"]

in_channels = infer_in_channels_from_state_dict(state_dict)
skip_channels = infer_skip_channels_from_state_dict(state_dict)
latent_dim = int(state_dict["encoder.fc.1.weight"].shape[0])

model = TriViewAutoencoder(latent_dim=latent_dim, in_channels=in_channels, skip_channels=skip_channels)
model.load_state_dict(state_dict)
model.eval()

# Inference with TTA
import torch.nn.functional as F
x = ...  # [batch, 3, 64, 64] projections
pred = model(x)
pred_flipped = model(x.flip(-1)).flip(-1)
pred_avg = (pred + pred_flipped) / 2.0
volume = torch.sigmoid(pred_avg)  # [batch, 1, 64, 64, 64]
```

## Pipeline Order
1. `train_colab.ipynb` → `best_autoencoder.pt`
2. `train_refiner_colab.ipynb` → `best_refiner.pt` (needs stage 1)
3. `train_vae_colab.ipynb` → `best_vae.pt` (independent)
4. `train_classifier_colab.ipynb` → `best_classifier.pt` (needs stage 1)
"""


def upload_models(
    models_dir: str = "results",
    repo_id: str = REPO_ID,
    dry_run: bool = False,
) -> None:
    api = HfApi()
    user = api.whoami()
    print(f"Authenticated as: {user['name']}")
    print(f"Target repo: {repo_id}")
    print(f"Models dir: {models_dir}")
    print()

    uploaded = []
    skipped = []

    for filename in MODEL_FILES:
        filepath = Path(models_dir) / filename
        if not filepath.exists():
            skipped.append((filename, "file not found"))
            continue

        size_mb = filepath.stat().st_size / (1024 * 1024)
        print(f"Uploading {filename} ({size_mb:.1f} MB)...", end=" ", flush=True)

        if dry_run:
            print("[DRY RUN]")
            uploaded.append((filename, size_mb))
        else:
            try:
                api.upload_file(
                    path_or_fileobj=str(filepath),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model",
                )
                print("OK")
                uploaded.append((filename, size_mb))
            except Exception as e:
                print(f"FAILED: {e}")
                skipped.append((filename, str(e)))

    metrics_path = Path(models_dir) / METRICS_DIR
    if metrics_path.exists():
        for f in sorted(metrics_path.iterdir()):
            if f.is_file() and f.suffix in (".json", ".csv"):
                print(f"Uploading metrics/{f.name}...", end=" ", flush=True)
                if dry_run:
                    print("[DRY RUN]")
                else:
                    try:
                        api.upload_file(
                            path_or_fileobj=str(f),
                            path_in_repo=f"metrics/{f.name}",
                            repo_id=repo_id,
                            repo_type="model",
                        )
                        print("OK")
                        uploaded.append((f"metrics/{f.name}", f.stat().st_size / 1024))
                    except Exception as e:
                        print(f"FAILED: {e}")
                        skipped.append((f"metrics/{f.name}", str(e)))

    print(f"\nUploading README.md...", end=" ", flush=True)
    if dry_run:
        print("[DRY RUN]")
    else:
        try:
            api.upload_file(
                path_or_fileobj=MODEL_CARD.encode("utf-8"),
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
        )
            print("OK")
        except Exception as e:
            print(f"FAILED: {e}")

    print(f"\n{'='*50}")
    print(f"Uploaded: {len(uploaded)} files")
    for name, size in uploaded:
        unit = "MB" if size > 1 else "KB"
        print(f"  ✓ {name} ({size:.1f} {unit})")
    if skipped:
        print(f"Skipped: {len(skipped)} files")
        for name, reason in skipped:
            print(f"  ✗ {name}: {reason}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload models to HuggingFace Hub")
    parser.add_argument("--models-dir", type=str, default="results")
    parser.add_argument("--repo_id", type=str, default=REPO_ID)
    parser.add_argument("--dry-run", action="store_true", help="Show what would be uploaded without actually uploading")
    args = parser.parse_args()

    upload_models(
        models_dir=args.models_dir,
        repo_id=args.repo_id,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
