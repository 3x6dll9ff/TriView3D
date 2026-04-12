"""
Evaluation: overall + hard subset + per-cell-type + TTA.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

try:
    from src.autoencoder import TriViewAutoencoder
    from src.dataset import CellTriViewDataset
    from src.reconstruction_utils import (
        infer_in_channels_from_state_dict,
        infer_skip_channels_from_state_dict,
        lift_views_to_volume,
        project_volume_batch,
    )
    from src.refiner import DetailRefiner
except ImportError:
    from autoencoder import TriViewAutoencoder
    from dataset import CellTriViewDataset
    from reconstruction_utils import (
        infer_in_channels_from_state_dict,
        infer_skip_channels_from_state_dict,
        lift_views_to_volume,
        project_volume_batch,
    )
    from refiner import DetailRefiner


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unwrap_state_dict(checkpoint: dict[str, object]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]  # type: ignore[return-value]
    return checkpoint  # type: ignore[return-value]


def infer_latent_dim(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["encoder.fc.1.weight"].shape[0])


def compute_metrics(pred: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    pred_bin = (torch.sigmoid(pred) > 0.5).float()
    intersection = float((pred_bin * target).sum().item())
    union = float(pred_bin.sum().item() + target.sum().item())
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    iou = (intersection + 1.0) / (union - intersection + 1.0)
    mse = float(((torch.sigmoid(pred) - target) ** 2).mean().item())
    return {"dice": dice, "iou": iou, "mse": mse}


def projection_l1(pred_volume: torch.Tensor, inputs: torch.Tensor, view_names: tuple[str, ...]) -> float:
    projected = project_volume_batch(pred_volume, view_names)
    return float(torch.abs(projected - inputs).mean().item())


@torch.no_grad()
def tta_predict(
    model: TriViewAutoencoder,
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Test-Time Augmentation: горизонтальный flip + average.

    Для tri-view: горизонтальный flip всех проекций = mirror 3D по width (последняя ось).
    """
    pred_original = model(inputs)

    inputs_flipped = inputs.flip(-1)
    pred_flipped = model(inputs_flipped)
    pred_unflipped = pred_flipped.flip(-1)

    return (pred_original + pred_unflipped) / 2.0


def build_prediction(
    base_model: TriViewAutoencoder,
    inputs: torch.Tensor,
    refiner: DetailRefiner | None,
    view_names: tuple[str, ...],
    use_tta: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if use_tta:
        coarse = tta_predict(base_model, inputs)
    else:
        coarse = base_model(inputs)
    if refiner is None:
        return coarse, coarse
    lifted = lift_views_to_volume(inputs, view_names)
    refined = refiner(coarse, lifted)
    return coarse, refined


@torch.no_grad()
def evaluate_dataset(
    dataset: CellTriViewDataset,
    base_model: TriViewAutoencoder,
    refiner: DetailRefiner | None,
    device: torch.device,
    use_tta: bool = True,
) -> pd.DataFrame:
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    records: list[dict[str, object]] = []

    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target_3d"].to(device)

        coarse, final = build_prediction(
            base_model, inputs, refiner, dataset.view_names, use_tta=use_tta
        )

        coarse_proj = projection_l1(coarse, inputs, dataset.view_names)
        final_proj = projection_l1(final, inputs, dataset.view_names)

        coarse_metrics = compute_metrics(coarse, target)
        final_metrics = compute_metrics(final, target)
        records.append(
            {
                "name": batch["name"][0],
                "cell_type": batch["cell_type"][0],
                "complexity_score": float(batch["complexity_score"][0]),
                "coarse_dice": coarse_metrics["dice"],
                "coarse_iou": coarse_metrics["iou"],
                "coarse_mse": coarse_metrics["mse"],
                "coarse_projection_l1": coarse_proj,
                "final_dice": final_metrics["dice"],
                "final_iou": final_metrics["iou"],
                "final_mse": final_metrics["mse"],
                "final_projection_l1": final_proj,
            }
        )

    return pd.DataFrame(records)


def summarize_results(df: pd.DataFrame, hard_quantile: float) -> dict[str, object]:
    hard_threshold = float(df["complexity_score"].quantile(hard_quantile))
    hard_df = df[df["complexity_score"] >= hard_threshold].copy()

    overall = {
        "coarse_dice": float(df["coarse_dice"].mean()),
        "final_dice": float(df["final_dice"].mean()),
        "coarse_iou": float(df["coarse_iou"].mean()),
        "final_iou": float(df["final_iou"].mean()),
        "coarse_projection_l1": float(df["coarse_projection_l1"].mean()),
        "final_projection_l1": float(df["final_projection_l1"].mean()),
    }
    hard = {
        "threshold": hard_threshold,
        "count": int(len(hard_df)),
        "coarse_dice": float(hard_df["coarse_dice"].mean()),
        "final_dice": float(hard_df["final_dice"].mean()),
        "coarse_iou": float(hard_df["coarse_iou"].mean()),
        "final_iou": float(hard_df["final_iou"].mean()),
        "coarse_projection_l1": float(hard_df["coarse_projection_l1"].mean()),
        "final_projection_l1": float(hard_df["final_projection_l1"].mean()),
    }

    per_cell_type = (
        df.groupby("cell_type")[["coarse_dice", "final_dice", "coarse_iou", "final_iou"]]
        .mean()
        .sort_index()
        .round(4)
        .to_dict(orient="index")
    )
    return {"overall": overall, "hard_subset": hard, "per_cell_type": per_cell_type}


def plot_training_history(history_path: Path, save_path: Path) -> None:
    with history_path.open("r", encoding="utf-8") as file:
        history = json.load(file)

    train_loss = history.get("train_loss", [])
    val_loss = history.get("val_loss") or history.get("test_loss", [])
    val_dice = history.get("val_dice") or history.get("test_dice", [])
    val_hard_dice = history.get("val_hard_dice", [])

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(train_loss, label="Train")
    axes[0].plot(val_loss, label="Validation")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(val_dice, label="Val Dice")
    axes[1].set_title("Validation Dice")
    axes[1].legend()

    axes[2].plot(val_hard_dice, label="Hard Dice")
    axes[2].set_title("Hard Subset Dice")
    axes[2].legend()

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--autoencoder", type=str, default="results/best_autoencoder.pt")
    parser.add_argument("--refiner", type=str, default="")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--input_mode", type=str, default="tri", choices=["tri", "quad"])
    parser.add_argument("--hard_quantile", type=float, default=0.8)
    parser.add_argument("--no_tta", action="store_true", help="Disable TTA")
    args = parser.parse_args()

    device = select_device()
    test_ds = CellTriViewDataset(args.data_dir, split="test", input_mode=args.input_mode)

    base_checkpoint = torch.load(args.autoencoder, map_location=device)
    base_state_dict = unwrap_state_dict(base_checkpoint)
    in_channels = infer_in_channels_from_state_dict(base_state_dict)
    latent_dim = infer_latent_dim(base_state_dict)
    skip_channels = infer_skip_channels_from_state_dict(base_state_dict)
    if in_channels != len(test_ds.view_names):
        raise ValueError(
            f"Checkpoint expects {in_channels} channels, dataset input_mode={args.input_mode} gives {len(test_ds.view_names)}"
        )

    base_model = TriViewAutoencoder(
        latent_dim=latent_dim, in_channels=in_channels, skip_channels=skip_channels
    ).to(device)
    base_model.load_state_dict(base_state_dict)
    base_model.eval()

    refiner = None
    if args.refiner:
        refiner = DetailRefiner(view_channels=len(test_ds.view_names)).to(device)
        refiner.load_state_dict(torch.load(args.refiner, map_location=device))
        refiner.eval()

    result_df = evaluate_dataset(
        test_ds, base_model, refiner, device, use_tta=not args.no_tta
    )
    summary = summarize_results(result_df, hard_quantile=args.hard_quantile)

    output_dir = Path(args.output_dir)
    metrics_dir = output_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    evaluation_path = metrics_dir / "reconstruction_metrics.json"
    result_df.to_csv(metrics_dir / "reconstruction_metrics_per_sample.csv", index=False)
    with evaluation_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print("=== Reconstruction Metrics ===")
    for section, values in summary.items():
        print(f"[{section}]")
        print(json.dumps(values, indent=2, ensure_ascii=False))

    history_path = metrics_dir / "reconstruction_history.json"
    if history_path.exists():
        plot_training_history(history_path, output_dir / "figures" / "training_history.png")
        print("Training history plot saved")


if __name__ == "__main__":
    main()
