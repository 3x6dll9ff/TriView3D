"""
Обучение второго этапа refinement поверх base autoencoder.

Использование:
  python3 src/train_refiner.py --data_dir data/processed --base_model results/best_autoencoder.pt
"""

from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from src.autoencoder import TriViewAutoencoder, reconstruction_loss
    from src.dataset import CellTriViewDataset
    from src.reconstruction_utils import infer_in_channels_from_state_dict, lift_views_to_volume
    from src.refiner import DetailRefiner
except ImportError:
    from autoencoder import TriViewAutoencoder, reconstruction_loss
    from dataset import CellTriViewDataset
    from reconstruction_utils import infer_in_channels_from_state_dict, lift_views_to_volume
    from refiner import DetailRefiner


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def unwrap_state_dict(checkpoint: dict[str, torch.Tensor] | dict[str, object]) -> dict[str, torch.Tensor]:
    if "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    return checkpoint  # type: ignore[return-value]


def infer_latent_dim_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    return int(state_dict["encoder.fc.1.weight"].shape[0])


def build_loader(
    dataset: CellTriViewDataset,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
    )


def compute_overlap(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    pred_bin = (pred > 0.5).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3, 4))
    union = pred_bin.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    iou = (intersection + 1.0) / (union - intersection + 1.0)
    return dice, iou


def forward_refiner(
    base_model: TriViewAutoencoder,
    refiner: DetailRefiner,
    inputs: torch.Tensor,
    target_shape: torch.Size,
    view_names: tuple[str, ...],
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.no_grad():
        coarse = base_model(inputs)
    lifted = lift_views_to_volume(
        inputs,
        view_names,
        depth=int(target_shape[2]),
        height=int(target_shape[3]),
        width=int(target_shape[4]),
    )
    refined = refiner(coarse, lifted)
    return coarse, refined


def train_one_epoch(
    base_model: TriViewAutoencoder,
    refiner: DetailRefiner,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    view_names: tuple[str, ...],
    bce_weight: float,
    dice_weight: float,
    projection_weight: float,
    surface_weight: float,
    boundary_boost: float,
    scaler: GradScaler | None,
) -> dict[str, float]:
    refiner.train()
    totals = {"total": 0.0, "projection": 0.0, "surface": 0.0}
    n_samples = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target = batch["target_3d"].to(device, non_blocking=True)
        batch_size = inputs.size(0)

        optimizer.zero_grad(set_to_none=True)
        if scaler is not None and device.type == "cuda":
            with autocast():
                _, refined = forward_refiner(base_model, refiner, inputs, target.shape, view_names)
                loss, components = reconstruction_loss(
                    refined,
                    target,
                    inputs=inputs,
                    view_names=view_names,
                    bce_weight=bce_weight,
                    dice_weight=dice_weight,
                    projection_weight=projection_weight,
                    surface_weight=surface_weight,
                    boundary_boost=boundary_boost,
                    return_components=True,
                )
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            _, refined = forward_refiner(base_model, refiner, inputs, target.shape, view_names)
            loss, components = reconstruction_loss(
                refined,
                target,
                inputs=inputs,
                view_names=view_names,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                projection_weight=projection_weight,
                surface_weight=surface_weight,
                boundary_boost=boundary_boost,
                return_components=True,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), max_norm=1.0)
            optimizer.step()

        totals["total"] += components["total"] * batch_size
        totals["projection"] += components["projection"] * batch_size
        totals["surface"] += components["surface"] * batch_size
        n_samples += batch_size

    return {key: value / max(n_samples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    base_model: TriViewAutoencoder,
    refiner: DetailRefiner,
    loader: DataLoader,
    device: torch.device,
    view_names: tuple[str, ...],
    hard_threshold: float,
    bce_weight: float,
    dice_weight: float,
    projection_weight: float,
    surface_weight: float,
    boundary_boost: float,
) -> dict[str, float]:
    refiner.eval()
    totals = {"loss": 0.0, "dice": 0.0, "iou": 0.0, "hard_dice": 0.0, "projection": 0.0}
    n_samples = 0
    n_hard = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target = batch["target_3d"].to(device, non_blocking=True)
        complexity = torch.as_tensor(batch["complexity_score"], dtype=torch.float32, device=device)

        _, refined = forward_refiner(base_model, refiner, inputs, target.shape, view_names)
        loss, components = reconstruction_loss(
            refined,
            target,
            inputs=inputs,
            view_names=view_names,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            projection_weight=projection_weight,
            surface_weight=surface_weight,
            boundary_boost=boundary_boost,
            return_components=True,
        )
        dice, iou = compute_overlap(refined, target)
        batch_size = inputs.size(0)

        totals["loss"] += float(loss.item()) * batch_size
        totals["dice"] += float(dice.sum().item())
        totals["iou"] += float(iou.sum().item())
        totals["projection"] += components["projection"] * batch_size
        n_samples += batch_size

        hard_mask = complexity >= hard_threshold
        if hard_mask.any():
            totals["hard_dice"] += float(dice[hard_mask].sum().item())
            n_hard += int(hard_mask.sum().item())

    metrics = {
        "loss": totals["loss"] / max(n_samples, 1),
        "dice": totals["dice"] / max(n_samples, 1),
        "iou": totals["iou"] / max(n_samples, 1),
        "projection": totals["projection"] / max(n_samples, 1),
        "hard_dice": totals["hard_dice"] / max(n_hard, 1),
    }
    if n_hard == 0:
        metrics["hard_dice"] = metrics["dice"]
    return metrics


def train(
    data_dir: str,
    base_model_path: str,
    output_dir: str = "results",
    batch_size: int = 4,
    epochs: int = 50,
    lr: float = 3e-4,
    input_mode: str = "quad",
    bce_weight: float = 0.35,
    dice_weight: float = 0.25,
    projection_weight: float = 0.25,
    surface_weight: float = 0.15,
    boundary_boost: float = 4.0,
    complexity_boost: float = 1.0,
    num_workers: int = 0,
    seed: int = 42,
) -> None:
    set_seed(seed)
    device = select_device()
    print(f"Device: {device}")

    checkpoint = torch.load(base_model_path, map_location=device)
    state_dict = unwrap_state_dict(checkpoint)
    base_in_channels = infer_in_channels_from_state_dict(state_dict)
    latent_dim = infer_latent_dim_from_state_dict(state_dict)

    train_ds = CellTriViewDataset(data_dir, split="train", seed=seed, input_mode=input_mode)
    test_ds = CellTriViewDataset(data_dir, split="test", seed=seed, input_mode=input_mode)
    view_names = train_ds.view_names
    if len(view_names) != base_in_channels:
        raise ValueError(
            f"Base model expects {base_in_channels} channels, but dataset input_mode={input_mode} gives {len(view_names)}"
        )

    weights = train_ds.build_sample_weights(complexity_boost=complexity_boost)
    sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)
    train_loader = build_loader(train_ds, batch_size, num_workers, sampler=sampler)
    test_loader = build_loader(test_ds, batch_size, num_workers)

    base_model = TriViewAutoencoder(latent_dim=latent_dim, in_channels=base_in_channels).to(device)
    base_model.load_state_dict(state_dict)
    base_model.eval()
    for parameter in base_model.parameters():
        parameter.requires_grad = False

    refiner = DetailRefiner(view_channels=len(view_names)).to(device)
    optimizer = torch.optim.Adam(refiner.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )
    scaler = GradScaler() if device.type == "cuda" else None
    hard_threshold = test_ds.hard_threshold(quantile=0.8)

    output_path = Path(output_dir)
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history = {
        "train_loss": [],
        "train_projection": [],
        "train_surface": [],
        "val_loss": [],
        "val_projection": [],
        "val_dice": [],
        "val_iou": [],
        "val_hard_dice": [],
        "config": {
            "base_model_path": base_model_path,
            "input_mode": input_mode,
            "view_names": list(view_names),
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "seed": seed,
        },
    }

    best_score = -1.0
    patience_counter = 0
    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            base_model,
            refiner,
            train_loader,
            optimizer,
            device,
            view_names,
            bce_weight,
            dice_weight,
            projection_weight,
            surface_weight,
            boundary_boost,
            scaler,
        )
        val_metrics = evaluate(
            base_model,
            refiner,
            test_loader,
            device,
            view_names,
            hard_threshold,
            bce_weight,
            dice_weight,
            projection_weight,
            surface_weight,
            boundary_boost,
        )

        score = 0.7 * val_metrics["dice"] + 0.3 * val_metrics["hard_dice"]
        scheduler.step(score)

        history["train_loss"].append(train_metrics["total"])
        history["train_projection"].append(train_metrics["projection"])
        history["train_surface"].append(train_metrics["surface"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_projection"].append(val_metrics["projection"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_hard_dice"].append(val_metrics["hard_dice"])

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:3d}/{epochs} | train_loss: {train_metrics['total']:.4f} | "
            f"val_dice: {val_metrics['dice']:.4f} | val_hard_dice: {val_metrics['hard_dice']:.4f} | "
            f"val_proj: {val_metrics['projection']:.4f} | score: {score:.4f} | {elapsed:.1f}s"
        )

        if score > best_score:
            best_score = score
            patience_counter = 0
            torch.save(refiner.state_dict(), output_path / "best_refiner.pt")
        else:
            patience_counter += 1

        if patience_counter >= 10:
            print("\nEarly stopping для refiner")
            break

    history_path = metrics_dir / "refiner_history.json"
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(history, file, indent=2)

    print(f"\nRefiner training complete. Best score: {best_score:.4f}")
    print(f"Refiner weights: {output_path / 'best_refiner.pt'}")
    print(f"History: {history_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение detail refiner")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--base_model", type=str, default="results/best_autoencoder.pt")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--input_mode", type=str, default="quad", choices=["tri", "quad"])
    parser.add_argument("--bce_weight", type=float, default=0.35)
    parser.add_argument("--dice_weight", type=float, default=0.25)
    parser.add_argument("--projection_weight", type=float, default=0.25)
    parser.add_argument("--surface_weight", type=float, default=0.15)
    parser.add_argument("--boundary_boost", type=float, default=4.0)
    parser.add_argument("--complexity_boost", type=float, default=1.0)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        base_model_path=args.base_model,
        output_dir=args.output_dir,
        batch_size=args.batch_size,
        epochs=args.epochs,
        lr=args.lr,
        input_mode=args.input_mode,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        projection_weight=args.projection_weight,
        surface_weight=args.surface_weight,
        boundary_boost=args.boundary_boost,
        complexity_boost=args.complexity_boost,
        num_workers=args.num_workers,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
