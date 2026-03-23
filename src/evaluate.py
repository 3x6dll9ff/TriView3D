"""
Evaluation: метрики реконструкции и классификации.

Использование:
  python3 src/evaluate.py --data_dir data/processed --autoencoder results/best_autoencoder.pt
"""

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from torch.utils.data import DataLoader

from autoencoder import TriViewAutoencoder
from dataset import CellTriViewDataset


@torch.no_grad()
def compute_reconstruction_metrics(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    """Считает Dice, IoU, MSE по всему датасету."""
    model.eval()
    all_dice, all_iou, all_mse = [], [], []

    for batch in loader:
        inputs = batch["input"].to(device)
        target = batch["target_3d"].to(device)

        pred = model(inputs)
        pred_bin = (pred > 0.5).float()

        for i in range(len(pred)):
            p = pred_bin[i].flatten()
            t = target[i].flatten()

            intersection = (p * t).sum().item()
            union = p.sum().item() + t.sum().item()

            dice = (2 * intersection + 1) / (union + 1)
            iou = (intersection + 1) / (union - intersection + 1)
            mse = ((pred[i] - target[i]) ** 2).mean().item()

            all_dice.append(dice)
            all_iou.append(iou)
            all_mse.append(mse)

    return {
        "dice_mean": float(np.mean(all_dice)),
        "dice_std": float(np.std(all_dice)),
        "iou_mean": float(np.mean(all_iou)),
        "iou_std": float(np.std(all_iou)),
        "mse_mean": float(np.mean(all_mse)),
    }


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str,
    title: str = "Confusion Matrix",
) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal", "Anomaly"])
    ax.set_yticklabels(["Normal", "Anomaly"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=16)

    fig.colorbar(im)
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str,
    title: str = "ROC Curve",
) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, color="crimson", lw=2, label=f"AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def plot_training_history(
    history_path: str,
    save_path: str,
) -> None:
    with open(history_path) as f:
        history = json.load(f)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].plot(history["train_loss"], label="Train")
    axes[0].plot(history["test_loss"], label="Test")
    axes[0].set_title("Loss")
    axes[0].legend()

    axes[1].plot(history["test_dice"])
    axes[1].set_title("Test Dice")

    axes[2].plot(history["test_iou"])
    axes[2].set_title("Test IoU")

    for ax in axes:
        ax.set_xlabel("Epoch")
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Evaluation")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--autoencoder", type=str, default="results/best_autoencoder.pt")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--latent_dim", type=int, default=256)
    args = parser.parse_args()

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Датасет
    test_ds = CellTriViewDataset(args.data_dir, split="test")
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    # Загружаем autoencoder
    if os.path.exists(args.autoencoder):
        model = TriViewAutoencoder(latent_dim=args.latent_dim).to(device)
        model.load_state_dict(
            torch.load(args.autoencoder, map_location=device)
        )

        # Метрики реконструкции
        print("=== Reconstruction Metrics ===")
        recon_metrics = compute_reconstruction_metrics(model, test_loader, device)
        for k, v in recon_metrics.items():
            print(f"  {k}: {v:.4f}")

        # Сохранение
        metrics_dir = os.path.join(args.output_dir, "metrics")
        os.makedirs(metrics_dir, exist_ok=True)
        with open(os.path.join(metrics_dir, "reconstruction_metrics.json"), "w") as f:
            json.dump(recon_metrics, f, indent=2)

        # Графики обучения
        history_path = os.path.join(metrics_dir, "reconstruction_history.json")
        if os.path.exists(history_path):
            plot_training_history(
                history_path,
                os.path.join(args.output_dir, "figures", "training_history.png"),
            )
            print("  Training history plot saved")
    else:
        print(f"Autoencoder не найден: {args.autoencoder}")


if __name__ == "__main__":
    main()
