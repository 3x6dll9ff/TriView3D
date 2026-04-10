"""
Обучение базового multi-view 3D autoencoder.

Использование:
  python3 src/train_reconstruction.py --data_dir data/processed --input_mode quad --epochs 80
"""

from __future__ import annotations

import argparse
import json
import random
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader, WeightedRandomSampler

try:
    from src.autoencoder import TriViewAutoencoder, reconstruction_loss
    from src.dataset import CellTriViewDataset
except ImportError:
    from autoencoder import TriViewAutoencoder, reconstruction_loss
    from dataset import CellTriViewDataset


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def select_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def current_git_hash() -> str | None:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def compute_overlap(
    pred: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    pred_bin = (pred > 0).float()
    intersection = (pred_bin * target).sum(dim=(1, 2, 3, 4))
    union = pred_bin.sum(dim=(1, 2, 3, 4)) + target.sum(dim=(1, 2, 3, 4))
    dice = (2.0 * intersection + 1.0) / (union + 1.0)
    iou = (intersection + 1.0) / (union - intersection + 1.0)
    return dice, iou


def make_loader(
    dataset: CellTriViewDataset,
    batch_size: int,
    num_workers: int,
    sampler: WeightedRandomSampler | None = None,
) -> DataLoader:
    use_pin_memory = torch.cuda.is_available()
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=use_pin_memory,
        persistent_workers=num_workers > 0,
    )


def train_one_epoch(
    model: TriViewAutoencoder,
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
    model.train()
    totals = {"total": 0.0, "bce": 0.0, "dice_loss": 0.0, "projection": 0.0, "surface": 0.0}
    n_samples = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target_3d = batch["target_3d"].to(device, non_blocking=True)
        batch_size = inputs.size(0)

        optimizer.zero_grad(set_to_none=True)

        if scaler is not None and device.type == "cuda":
            with autocast():
                pred = model(inputs)
                loss, components = reconstruction_loss(
                    pred,
                    target_3d,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(inputs)
            loss, components = reconstruction_loss(
                pred,
                target_3d,
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
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        for key in totals:
            totals[key] += components[key] * batch_size
        n_samples += batch_size

    return {key: value / max(n_samples, 1) for key, value in totals.items()}


@torch.no_grad()
def evaluate(
    model: TriViewAutoencoder,
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
    model.eval()
    totals = {
        "loss": 0.0,
        "projection": 0.0,
        "dice": 0.0,
        "iou": 0.0,
        "hard_dice": 0.0,
        "hard_iou": 0.0,
    }
    n_samples = 0
    n_hard = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target_3d = batch["target_3d"].to(device, non_blocking=True)
        complexity = torch.as_tensor(batch["complexity_score"], dtype=torch.float32, device=device)

        pred = model(inputs)
        loss, components = reconstruction_loss(
            pred,
            target_3d,
            inputs=inputs,
            view_names=view_names,
            bce_weight=bce_weight,
            dice_weight=dice_weight,
            projection_weight=projection_weight,
            surface_weight=surface_weight,
            boundary_boost=boundary_boost,
            return_components=True,
        )
        dice, iou = compute_overlap(pred, target_3d)

        batch_size = inputs.size(0)
        totals["loss"] += float(loss.item()) * batch_size
        totals["projection"] += components["projection"] * batch_size
        totals["dice"] += float(dice.sum().item())
        totals["iou"] += float(iou.sum().item())
        n_samples += batch_size

        hard_mask = complexity >= hard_threshold
        if hard_mask.any():
            totals["hard_dice"] += float(dice[hard_mask].sum().item())
            totals["hard_iou"] += float(iou[hard_mask].sum().item())
            n_hard += int(hard_mask.sum().item())

    metrics = {
        "loss": totals["loss"] / max(n_samples, 1),
        "projection": totals["projection"] / max(n_samples, 1),
        "dice": totals["dice"] / max(n_samples, 1),
        "iou": totals["iou"] / max(n_samples, 1),
        "hard_dice": totals["hard_dice"] / max(n_hard, 1),
        "hard_iou": totals["hard_iou"] / max(n_hard, 1),
        "hard_count": float(n_hard),
    }
    if n_hard == 0:
        metrics["hard_dice"] = metrics["dice"]
        metrics["hard_iou"] = metrics["iou"]
    return metrics


def train(
    data_dir: str,
    output_dir: str = "results",
    epochs: int = 80,
    batch_size: int = 8,
    lr: float = 1e-3,
    latent_dim: int = 256,
    input_mode: str = "tri",
    bce_weight: float = 0.35,
    dice_weight: float = 0.25,
    projection_weight: float = 0.25,
    surface_weight: float = 0.15,
    boundary_boost: float = 4.0,
    complexity_sampling: bool = True,
    complexity_boost: float = 1.0,
    num_workers: int = 0,
    early_stopping_patience: int = 12,
    seed: int = 42,
) -> None:
    set_seed(seed)
    device = select_device()
    print(f"Device: {device}")

    train_ds = CellTriViewDataset(data_dir, split="train", seed=seed, input_mode=input_mode)
    test_ds = CellTriViewDataset(data_dir, split="test", seed=seed, input_mode=input_mode)
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    train_sampler = None
    if complexity_sampling:
        weights = train_ds.build_sample_weights(complexity_boost=complexity_boost)
        train_sampler = WeightedRandomSampler(weights, num_samples=len(weights), replacement=True)

    train_loader = make_loader(train_ds, batch_size, num_workers, sampler=train_sampler)
    test_loader = make_loader(test_ds, batch_size, num_workers)

    view_names = train_ds.view_names
    model = TriViewAutoencoder(latent_dim=latent_dim, in_channels=len(view_names)).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=4,
    )
    scaler = GradScaler() if device.type == "cuda" else None

    n_params = sum(parameter.numel() for parameter in model.parameters())
    print(f"Параметры модели: {n_params:,}")
    print(f"Views: {view_names}")

    output_path = Path(output_dir)
    metrics_dir = output_path / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    history_path = metrics_dir / "reconstruction_history.json"
    if history_path.exists():
        print(f"ПРЕДУПРЕЖДЕНИЕ: {history_path} будет перезаписан")

    history: dict[str, list[float]] = {
        "train_loss": [],
        "train_bce": [],
        "train_dice_loss": [],
        "train_projection": [],
        "train_surface": [],
        "val_loss": [],
        "val_projection": [],
        "val_dice": [],
        "val_iou": [],
        "val_hard_dice": [],
        "val_hard_iou": [],
    }
    best_score = -1.0
    patience_counter = 0
    hard_threshold = test_ds.hard_threshold(quantile=0.8)

    run_config = {
        "git_hash": current_git_hash(),
        "data_dir": data_dir,
        "input_mode": input_mode,
        "view_names": list(view_names),
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "latent_dim": latent_dim,
        "bce_weight": bce_weight,
        "dice_weight": dice_weight,
        "projection_weight": projection_weight,
        "surface_weight": surface_weight,
        "boundary_boost": boundary_boost,
        "complexity_sampling": complexity_sampling,
        "complexity_boost": complexity_boost,
        "seed": seed,
    }

    for epoch in range(1, epochs + 1):
        start_time = time.time()
        train_metrics = train_one_epoch(
            model,
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
            model,
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

        composite_score = 0.7 * val_metrics["dice"] + 0.3 * val_metrics["hard_dice"]
        scheduler.step(composite_score)

        history["train_loss"].append(train_metrics["total"])
        history["train_bce"].append(train_metrics["bce"])
        history["train_dice_loss"].append(train_metrics["dice_loss"])
        history["train_projection"].append(train_metrics["projection"])
        history["train_surface"].append(train_metrics["surface"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_projection"].append(val_metrics["projection"])
        history["val_dice"].append(val_metrics["dice"])
        history["val_iou"].append(val_metrics["iou"])
        history["val_hard_dice"].append(val_metrics["hard_dice"])
        history["val_hard_iou"].append(val_metrics["hard_iou"])

        elapsed = time.time() - start_time
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_metrics['total']:.4f} | "
            f"val_dice: {val_metrics['dice']:.4f} | "
            f"val_hard_dice: {val_metrics['hard_dice']:.4f} | "
            f"val_proj: {val_metrics['projection']:.4f} | "
            f"score: {composite_score:.4f} | "
            f"{elapsed:.1f}s"
        )

        if composite_score > best_score:
            best_score = composite_score
            patience_counter = 0
            torch.save(model.state_dict(), output_path / "best_autoencoder.pt")
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "run_config": run_config,
                },
                output_path / "checkpoint_best_autoencoder.pt",
            )
        else:
            patience_counter += 1

        if epoch % 10 == 0:
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "run_config": run_config,
                },
                output_path / f"checkpoint_epoch{epoch}.pt",
            )

        if patience_counter >= early_stopping_patience:
            print(f"\nEarly stopping: нет улучшения {early_stopping_patience} эпох")
            break

    payload = {**history, "config": run_config}
    with history_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print(f"\nОбучение завершено. Best score: {best_score:.4f}")
    print(f"Модель: {output_path / 'best_autoencoder.pt'}")
    print(f"История: {history_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Обучение base multi-view 3D autoencoder")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--input_mode", type=str, default="tri", choices=["tri", "quad"])
    parser.add_argument("--bce_weight", type=float, default=0.35)
    parser.add_argument("--dice_weight", type=float, default=0.25)
    parser.add_argument("--projection_weight", type=float, default=0.25)
    parser.add_argument("--surface_weight", type=float, default=0.15)
    parser.add_argument("--boundary_boost", type=float, default=4.0)
    parser.add_argument("--complexity_boost", type=float, default=1.0)
    parser.add_argument("--no_complexity_sampling", action="store_true")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--early_stopping_patience", type=int, default=12)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        latent_dim=args.latent_dim,
        input_mode=args.input_mode,
        bce_weight=args.bce_weight,
        dice_weight=args.dice_weight,
        projection_weight=args.projection_weight,
        surface_weight=args.surface_weight,
        boundary_boost=args.boundary_boost,
        complexity_sampling=not args.no_complexity_sampling,
        complexity_boost=args.complexity_boost,
        num_workers=args.num_workers,
        early_stopping_patience=args.early_stopping_patience,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
