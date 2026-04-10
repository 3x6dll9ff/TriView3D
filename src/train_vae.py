"""
Обучение Conditional VAE для tri-view 3D реконструкции.

Использование:
  python3 src/train_vae.py --data_dir data/processed --epochs 50

Отличия от train_reconstruction.py:
  - Модель: TriViewCVAE вместо TriViewAutoencoder
  - Loss: BCE + Dice + KL divergence
  - Результат: best_vae.pt
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

from src.vae import TriViewCVAE, vae_loss
from src.dataset import CellTriViewDataset


def train_one_epoch(
    model: TriViewCVAE,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    kl_weight: float,
    scaler: GradScaler | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"total": 0.0, "recon": 0.0, "kl": 0.0}
    n = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target_3d = batch["target_3d"].to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        # Вычисление с AMP
        if scaler is not None and device.type == "cuda":
            with autocast('cuda'):
                pred, mu, logvar = model(inputs)
            # Loss считаем в fp32
            loss, components = vae_loss(pred.float(), target_3d.float(), mu.float(), logvar.float(), kl_weight=kl_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            pred, mu, logvar = model(inputs)
            loss, components = vae_loss(pred, target_3d, mu, logvar, kl_weight=kl_weight)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        bs = inputs.size(0)
        totals["total"] += components["total"] * bs
        totals["recon"] += components["recon"] * bs
        totals["kl"] += components["kl"] * bs
        n += bs

    return {k: v / n for k, v in totals.items()}


@torch.no_grad()
def evaluate(
    model: TriViewCVAE,
    loader: DataLoader,
    device: torch.device,
    kl_weight: float,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0
    n = 0

    for batch in loader:
        inputs = batch["input"].to(device, non_blocking=True)
        target_3d = batch["target_3d"].to(device, non_blocking=True)

        # Inference: используем generate (только mu, без шума)
        pred = model.generate(inputs)
        loss, _ = vae_loss(pred, target_3d, *model.encode(inputs), kl_weight=kl_weight)
        total_loss += loss.item() * inputs.size(0)

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
    kl_weight: float = 0.001,
) -> None:
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # Данные — тот же split что и CNN
    train_ds = CellTriViewDataset(data_dir, split="train")
    test_ds = CellTriViewDataset(data_dir, split="test")
    print(f"Train: {len(train_ds)}, Test: {len(test_ds)}")

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
    model = TriViewCVAE(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=5
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметры модели: {n_params:,}")
    print(f"KL weight: {kl_weight}")

    scaler = GradScaler('cuda') if device.type == "cuda" else None

    # Обучение
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    history = {
        "train_loss": [], "train_recon": [], "train_kl": [],
        "test_loss": [], "test_dice": [], "test_iou": [],
    }
    
    best_dice = 0.0
    patience_counter = 0
    patience_limit = 10

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        train_metrics = train_one_epoch(model, train_loader, optimizer, device, kl_weight, scaler)
        test_metrics = evaluate(model, test_loader, device, kl_weight)

        scheduler.step(test_metrics["loss"])

        history["train_loss"].append(train_metrics["total"])
        history["train_recon"].append(train_metrics["recon"])
        history["train_kl"].append(train_metrics["kl"])
        history["test_loss"].append(test_metrics["loss"])
        history["test_dice"].append(test_metrics["dice"])
        history["test_iou"].append(test_metrics["iou"])

        elapsed = time.time() - t0
        print(
            f"Epoch {epoch:3d}/{epochs} | "
            f"loss: {train_metrics['total']:.4f} "
            f"(recon: {train_metrics['recon']:.4f} kl: {train_metrics['kl']:.1f}) | "
            f"test_dice: {test_metrics['dice']:.4f} | "
            f"test_iou: {test_metrics['iou']:.4f} | "
            f"{elapsed:.1f}s"
        )

        if test_metrics["dice"] > best_dice:
            best_dice = test_metrics["dice"]
            patience_counter = 0
            torch.save(
                model.state_dict(),
                os.path.join(output_dir, "best_vae.pt"),
            )
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_dice": best_dice,
                },
                os.path.join(output_dir, "checkpoint_vae_best.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= patience_limit:
            print(f"\\nEarly stopping triggered: 0 improvement for {patience_limit} epochs.")
            break

    history_path = os.path.join(output_dir, "metrics", "vae_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)

    print(f"\nОбучение VAE завершено. Best Dice: {best_dice:.4f}")
    print(f"Модель: {output_dir}/best_vae.pt")
    print(f"История: {history_path}")


def main():
    parser = argparse.ArgumentParser(
        description="Обучение Conditional VAE для tri-view 3D реконструкции"
    )
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--kl_weight", type=float, default=0.001)
    args = parser.parse_args()

    train(
        args.data_dir, args.output_dir, args.epochs,
        args.batch_size, args.lr, args.latent_dim, args.kl_weight,
    )


if __name__ == "__main__":
    main()
