"""
Обучение Tri-View 3D Autoencoder (реконструкция 2D проекций → 3D).

Использование:
  python3 src/train_reconstruction.py --data_dir data/processed --epochs 50
"""

import argparse
import json
import os
import sys
import time

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast

from src.autoencoder import TriViewAutoencoder, SingleViewAutoencoder, reconstruction_loss
from src.dataset import CellTriViewDataset


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: GradScaler | None = None,
) -> float:
    model.train()
    total_loss = 0.0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target_3d = batch["target_3d"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Использование AMP для ускорения GPU
        if scaler is not None and device.type == "cuda":
            with autocast('cuda'):
                pred = model(inputs)
            # Вычисление лосса должно происходить в fp32, чтобы избежать ошибки BCE
            loss = reconstruction_loss(pred.float(), target_3d.float())
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred = model(inputs)
            loss = reconstruction_loss(pred, target_3d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item() * inputs.size(0)

    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target_3d = batch["target_3d"].to(device, non_blocking=True)

        pred = model(inputs)
        loss = reconstruction_loss(pred, target_3d)
        total_loss += loss.item() * inputs.size(0)

        # Dice и IoU
        pred_bin = (pred > 0.5).float()
        intersection = (pred_bin * target_3d).sum(dim=(1, 2, 3, 4))
        union = pred_bin.sum(dim=(1, 2, 3, 4)) + target_3d.sum(dim=(1, 2, 3, 4))

        dice = (2 * intersection + 1) / (union + 1)
        iou = (intersection + 1) / (union - intersection + 1)

        total_dice += dice.sum().item()
        total_iou += iou.sum().item()
        n += inputs.size(0)

    return {
        "loss": total_loss / n,
        "dice": total_dice / n,
        "iou": total_iou / n,
    }


def train(
    data_dir: str,
    output_dir: str = "results",
    epochs: int = 50,
    batch_size: int = 8,
    lr: float = 1e-3,
    latent_dim: int = 256,
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Данные
    train_ds = CellTriViewDataset(data_dir, split="train")
    test_ds = CellTriViewDataset(data_dir, split="test")

    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

    # Оптимизация DataLoader: воркеры и pin_memory
    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, 
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False, 
        num_workers=num_workers, pin_memory=True
    )

    # Модель
    model = TriViewAutoencoder(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметры модели: {n_params:,}")

    scaler = GradScaler('cuda') if device.type == "cuda" else None

    # Обучение
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    history = {"train_loss": [], "test_loss": [], "test_dice": [], "test_iou": []}
    best_dice = 0.0
    
    # Early Stopping setup
    patience_counter = 0
    patience_limit = 10

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(model, train_loader, optimizer, device, scaler)
        test_metrics = evaluate(model, test_loader, device)

        scheduler.step(test_metrics["loss"])

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_metrics["loss"])
        history["test_dice"].append(test_metrics["dice"])
        history["test_iou"].append(test_metrics["iou"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"train_loss: {train_loss:.4f} | "
            f"test_loss: {test_metrics['loss']:.4f} | "
            f"dice: {test_metrics['dice']:.4f} | "
            f"iou: {test_metrics['iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        # Сохранение лучшей модели
        if test_metrics["dice"] > best_dice:
            best_dice = test_metrics["dice"]
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_autoencoder.pt"),
            )
            # Чекоинт для дообучения
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                os.path.join(output_dir, "checkpoint_best.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"\\nEarly stopping triggered: 0 improvement for {patience_limit} epochs.")
            break

    # Сохранение истории
    history_path = os.path.join(output_dir, "metrics", "reconstruction_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nОбучение завершено. Best Dice: {best_dice:.4f}")
    print(f"Модель: {output_dir}/best_autoencoder.pt")
    print(f"История: {history_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Обучение Tri-View 3D Autoencoder"
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=256)
    args = parser.parse_args()

    train(
        args.data_dir, args.output_dir, args.epochs,
        args.batch_size, args.lr, args.latent_dim,
    )


if __name__ == "__main__":
    main()
