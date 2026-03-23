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

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from autoencoder import TriViewAutoencoder
from classifier import LatentClassifier, MorphometryRFClassifier
from dataset import CellTriViewDataset


def train_random_forest(data_dir: str, output_dir: str) -> None:
    """Обучение Random Forest на морфометрических метриках."""
    print("=== Random Forest на морфометрии ===")

    df = pd.read_csv(os.path.join(data_dir, "metadata.csv"))

    # Split
    rng = np.random.RandomState(42)
    idx = rng.permutation(len(df))
    n_train = int(len(df) * 0.8)
    train_df = df.iloc[idx[:n_train]]
    test_df = df.iloc[idx[n_train:]]

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

    print(f"\nРезультаты: {output_dir}/metrics/rf_results.json")


def train_latent_classifier(
    data_dir: str,
    autoencoder_path: str,
    output_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 1e-3,
    latent_dim: int = 256,
) -> None:
    """Обучение MLP классификатора на latent vectors."""
    print("=== Latent Classifier (MLP) ===")

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    # Загрузка autoencoder
    autoencoder = TriViewAutoencoder(latent_dim=latent_dim).to(device)
    autoencoder.load_state_dict(torch.load(autoencoder_path, map_location=device))
    autoencoder.eval()
    print(f"Autoencoder загружен: {autoencoder_path}")

    # Данные
    train_ds = CellTriViewDataset(data_dir, split="train")
    test_ds = CellTriViewDataset(data_dir, split="test")
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    # Классификатор
    classifier = LatentClassifier(latent_dim=latent_dim).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()

    best_acc = 0.0

    for epoch in range(1, epochs + 1):
        # Train
        classifier.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in train_loader:
            inputs = batch["input"].to(device)
            labels = batch["label"].to(device)

            with torch.no_grad():
                z = autoencoder.encode(inputs)

            logits = classifier(z)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * len(labels)
            correct += (logits.argmax(1) == labels).sum().item()
            total += len(labels)

        train_acc = correct / total

        # Test
        classifier.eval()
        test_correct = 0
        test_total = 0

        with torch.no_grad():
            for batch in test_loader:
                inputs = batch["input"].to(device)
                labels = batch["label"].to(device)
                z = autoencoder.encode(inputs)
                logits = classifier(z)
                test_correct += (logits.argmax(1) == labels).sum().item()
                test_total += len(labels)

        test_acc = test_correct / test_total

        if epoch % 5 == 0 or epoch == 1:
            print(
                f"Epoch {epoch:3d}/{epochs} | "
                f"train_acc: {train_acc:.4f} | test_acc: {test_acc:.4f}"
            )

        if test_acc > best_acc:
            best_acc = test_acc
            torch.save(
                classifier.state_dict(),
                os.path.join(output_dir, "best_classifier.pt"),
            )

    print(f"\nBest test accuracy: {best_acc:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Обучение классификатора")
    parser.add_argument("--data_dir", type=str, default="data/processed")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument(
        "--mode", type=str, default="rf", choices=["rf", "latent"]
    )
    parser.add_argument("--autoencoder", type=str, default="results/best_autoencoder.pt")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--latent_dim", type=int, default=256)
    args = parser.parse_args()

    if args.mode == "rf":
        train_random_forest(args.data_dir, args.output_dir)
    else:
        train_latent_classifier(
            args.data_dir, args.autoencoder, args.output_dir,
            args.epochs, args.batch_size, args.lr, args.latent_dim,
        )


if __name__ == "__main__":
    main()
