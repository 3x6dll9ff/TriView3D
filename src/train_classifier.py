"""
Обучение классификатора морфологии: normal vs anomaly.

Два режима:
  1. --mode latent: MLP на latent vectors от обученного autoencoder
  2. --mode rf: Random Forest на морфометрических метриках

Использование:
  python3 src/train_classifier.py --data_dir data/processed --mode rf
  python3 src/train_classifier.py --data_dir data/processed --mode latent --autoencoder results/best_autoencoder.pt
"""

import argparse
import json
import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from sklearn.model_selection import StratifiedShuffleSplit

from src.autoencoder import TriViewAutoencoder
from src.classifier import LatentClassifier, MorphometryRFClassifier
from src.dataset import CellTriViewDataset
from src.morphometrics import extract_all_metrics
from src.reconstruction_utils import infer_in_channels_from_state_dict


def train_random_forest(data_dir: str, output_dir: str) -> None:
    """Обучение Random Forest на морфометрических метриках."""
    print("=== Random Forest на морфометрии ===")

    df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=42)
    stratify_col = df["cell_type"].astype(str) if df["cell_type"].nunique() >= 2 else df["label"].astype(str)
    train_idx, test_idx = next(splitter.split(df, stratify_col))
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    print(f"Train: {len(train_df)}, Test: {len(test_df)}")

    # Обучение
    rf = MorphometryRFClassifier(n_estimators=200)
    rf.fit(train_df)

    # Оценка
    train_metrics = rf.evaluate(train_df)
    test_metrics = rf.evaluate(test_df)

    print(f"\nTrain accuracy: {train_metrics['accuracy']:.4f}")
    print(f"Test accuracy:  {test_metrics['accuracy']:.4f}")
    print(f"Test AUROC:     {test_metrics['auroc']:.4f}")
    print(f"\nFeature importance:")
    for feat, imp in sorted(
        rf.feature_importance().items(), key=lambda x: -x[1]
    ):
        print(f"  {feat}: {imp:.4f}")

    # Сохранение
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    results = {
        "model": "RandomForest",
        "train_accuracy": train_metrics["accuracy"],
        "test_accuracy": test_metrics["accuracy"],
        "test_auroc": test_metrics["auroc"],
        "feature_importance": rf.feature_importance(),
    }
    with open(os.path.join(output_dir, "metrics", "rf_results.json"), "w") as f:
        json.dump(results, f, indent=2)

    # Сохраняем модель RF через joblib
    try:
        import joblib
    except ImportError:
        import pickle as joblib
    rf_model_path = os.path.join(output_dir, "best_rf_classifier.pkl")
    joblib.dump(rf.model, rf_model_path)
    print(f"RF модель сохранена: {rf_model_path}")
    print(f"Результаты: {output_dir}/metrics/rf_results.json")


# Морфометрические признаки, совпадающие с MorphometryRFClassifier.FEATURE_COLS
MORPHO_KEYS = ["volume", "sphericity", "convexity", "eccentricity", "surface_roughness"]


def _extract_morpho_features(vol_3d: np.ndarray) -> list[float]:
    """Извлекает 5 морфометрических признаков из 3D объёма."""
    m = extract_all_metrics(vol_3d, threshold=0.5)
    return [m[k] for k in MORPHO_KEYS]


def train_latent_classifier(
    data_dir: str,
    autoencoder_path: str,
    output_dir: str,
    input_mode: str = "quad",
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 256,
    patience: int = 8,
) -> None:
    print("=== Latent Classifier (MLP) ===")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    checkpoint = torch.load(autoencoder_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    in_channels = infer_in_channels_from_state_dict(state_dict)
    latent_dim = None
    for key in state_dict.keys():
        if "encoder.fc" in key and "weight" in key:
            latent_dim = int(state_dict[key].shape[0])
            break
    if latent_dim is None:
        latent_dim = 256
    autoencoder = TriViewAutoencoder(latent_dim=latent_dim, in_channels=in_channels).to(device)
    autoencoder.load_state_dict(state_dict)
    autoencoder.eval()
    print(f"Autoencoder загружен: {autoencoder_path}")

    train_ds = CellTriViewDataset(data_dir, split="train", input_mode=input_mode)
    test_ds = CellTriViewDataset(data_dir, split="test", input_mode=input_mode)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    n_morpho = 5
    classifier = LatentClassifier(latent_dim=latent_dim + n_morpho).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)

    # Динамический расчёт весов классов для балансировки
    labels_arr = train_ds.df["label"].values
    counts = np.bincount(labels_arr)
    class_weights = len(labels_arr) / (len(counts) * counts)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=device)

    criterion = torch.nn.CrossEntropyLoss(weight=class_weights_t)

    best_acc = 0.0
    patience_counter = 0

    for epoch in range(1, epochs + 1):
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)
            target_3d = batch["target_3d"]

            with torch.no_grad():
                z = autoencoder.encode(inputs)

            morpho_list = []
            for i in range(target_3d.size(0)):
                vol = target_3d[i, 0].cpu().numpy()
                morpho_list.append(_extract_morpho_features(vol))
            morpho_tensor = torch.tensor(morpho_list, dtype=torch.float32, device=device)

            combined = torch.cat([z, morpho_tensor], dim=1)
            logits = classifier(combined)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_acc = correct / total

        classifier.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)
                target_3d = batch["target_3d"]

                z = autoencoder.encode(inputs)

                morpho_list = []
                for i in range(target_3d.size(0)):
                    vol = target_3d[i, 0].cpu().numpy()
                    morpho_list.append(_extract_morpho_features(vol))
                morpho_tensor = torch.tensor(morpho_list, dtype=torch.float32, device=device)

                combined = torch.cat([z, morpho_tensor], dim=1)
                logits = classifier(combined)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += len(labels)

        test_acc = test_correct / test_total
        scheduler.step(test_acc)

        if epoch % 2 == 0 or epoch == 1:
            current_lr = optimizer.param_groups[0]['lr']
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f} | lr: {current_lr:.2e}"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            patience_counter = 0
            torch.save(
                classifier.state_dict(),
                os.path.join(output_dir, "best_classifier.pt"),
            )
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print(f"Early stopping: no improvement for {patience} epochs")
            break

    print(f"\nBest test accuracy: {best_acc:.4f}")

    # Сохранение метрик
    os.makedirs(os.path.join(output_dir, "metrics"), exist_ok=True)
    with open(os.path.join(output_dir, "metrics", "mlp_results.json"), "w") as f:
        json.dump({
            "model": "LatentClassifier_MLP",
            "latent_dim": latent_dim,
            "n_morpho_features": n_morpho,
            "morpho_keys": MORPHO_KEYS,
            "best_test_accuracy": best_acc,
        }, f, indent=2)
    print(f"Метрики: {output_dir}/metrics/mlp_results.json")


def main():
    parser = argparse.ArgumentParser(description="Обучение классификатора")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--mode", type=str, default="rf", choices=["rf", "latent"]
    )
    parser.add_argument("--autoencoder", type=str, default="results/best_autoencoder.pt")
    parser.add_argument("--input_mode", type=str, default="quad", choices=["tri", "quad"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "rf":
        train_random_forest(args.data_dir, args.output_dir)
    else:
        train_latent_classifier(
            args.data_dir, args.autoencoder, args.output_dir, args.input_mode,
            args.epochs, args.batch_size, args.lr, args.latent_dim,
        )


if __name__ == "__main__":
    main()
