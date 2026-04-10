"""
Визуализация примеров подготовленного датасета (multi-view).

Генерирует:
    1. Multi-view примеры для нескольких типов клеток
  2. 3D ground truth срезы (XY/XZ/YZ)
  3. Распределение морфометрических метрик по классам
  4. Распределение типов клеток (bar chart)

Сохраняет всё в results/figures/

Использование:
  python3 src/visualize_examples.py
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = "data/processed"
OUTPUT_DIR = "results/figures"


def plot_tri_view_examples(df: pd.DataFrame, n_per_class: int = 3) -> None:
    """Примеры входов: top/bottom/side/front для normal и anomaly."""
    fig, axes = plt.subplots(n_per_class * 2, 4, figsize=(16, 4 * n_per_class))

    normals = df[df["label"] == 0].sample(n=n_per_class, random_state=42)
    anomalies = df[df["label"] == 1].sample(n=n_per_class, random_state=42)

    proj_names = ["top_proj", "bottom_proj", "side_proj", "front_proj"]
    proj_labels = ["Top (сверху)", "Bottom (снизу)", "Side (сбоку)", "Front (спереди)"]

    for i, (_, row) in enumerate(normals.iterrows()):
        for j, (pname, plabel) in enumerate(zip(proj_names, proj_labels)):
            proj = np.load(os.path.join(DATA_DIR, pname, f"{row['name']}.npy"))
            axes[i, j].imshow(proj, cmap="hot")
            axes[i, j].set_title(f"{plabel} — {row['cell_type']} (NORMAL)", fontsize=9)

    for i, (_, row) in enumerate(anomalies.iterrows()):
        r = n_per_class + i
        for j, (pname, plabel) in enumerate(zip(proj_names, proj_labels)):
            proj = np.load(os.path.join(DATA_DIR, pname, f"{row['name']}.npy"))
            axes[r, j].imshow(proj, cmap="hot")
            axes[r, j].set_title(f"{plabel} — {row['cell_type']} (ANOMALY)", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("Multi-View Input Examples (Sum Projection)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "tri_view_examples.png"), dpi=150)
    plt.close()
    print("  ✓ tri_view_examples.png")


def plot_3d_slices(df: pd.DataFrame) -> None:
    """3D ground truth: XY/XZ/YZ срезы для 6 типов клеток."""
    cell_types = ["discocyte", "spherocyte", "stomatocyte_I",
                  "echinocyte_I", "echinocyte_II", "echinocyte_III"]

    fig, axes = plt.subplots(len(cell_types), 3, figsize=(12, 3 * len(cell_types)))

    for i, ct in enumerate(cell_types):
        subset = df[df["cell_type"] == ct]
        if len(subset) == 0:
            continue
        row = subset.iloc[0]
        obj = np.load(os.path.join(DATA_DIR, "obj", f"{row['name']}.npy"))

        d, h, w = obj.shape
        axes[i, 0].imshow(obj[d // 2], cmap="viridis")
        axes[i, 0].set_title(f"{ct} — XY (z={d // 2})", fontsize=9)
        axes[i, 1].imshow(obj[:, h // 2, :], cmap="viridis")
        axes[i, 1].set_title(f"{ct} — XZ (y={h // 2})", fontsize=9)
        axes[i, 2].imshow(obj[:, :, w // 2], cmap="viridis")
        axes[i, 2].set_title(f"{ct} — YZ (x={w // 2})", fontsize=9)

    for ax in axes.ravel():
        ax.axis("off")

    fig.suptitle("3D Ground Truth Slices (per cell type)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "3d_slices_by_type.png"), dpi=150)
    plt.close()
    print("  ✓ 3d_slices_by_type.png")


def plot_class_distribution(df: pd.DataFrame) -> None:
    """Bar chart: количество клеток по типам + normal/anomaly."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    counts = df["cell_type"].value_counts().sort_index()
    colors = ["#2ecc71" if ct == "discocyte" else "#e74c3c" for ct in counts.index]
    axes[0].bar(counts.index, counts.values, color=colors)
    axes[0].set_title("Распределение по типам клеток")
    axes[0].set_ylabel("Кол-во")
    axes[0].tick_params(axis="x", rotation=45)

    label_counts = df["label"].value_counts().sort_index()
    labels = ["Normal", "Anomaly"]
    axes[1].bar(labels, [label_counts.get(0, 0), label_counts.get(1, 0)],
                color=["#2ecc71", "#e74c3c"])
    axes[1].set_title("Normal vs Anomaly")
    axes[1].set_ylabel("Кол-во")

    for ax in axes:
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "class_distribution.png"), dpi=150)
    plt.close()
    print("  ✓ class_distribution.png")


def plot_morphometrics(df: pd.DataFrame) -> None:
    """Гистограммы морфометрических метрик по классам."""
    features = ["volume", "sphericity", "convexity", "eccentricity", "surface_roughness"]
    available = [f for f in features if f in df.columns]

    fig, axes = plt.subplots(1, len(available), figsize=(4 * len(available), 4))
    if len(available) == 1:
        axes = [axes]

    for ax, feat in zip(axes, available):
        for label, color, name in [(0, "#2ecc71", "Normal"), (1, "#e74c3c", "Anomaly")]:
            vals = df[df["label"] == label][feat]
            ax.hist(vals, bins=20, alpha=0.6, color=color, label=name)
        ax.set_title(feat, fontsize=11)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Morphometric Features by Class", fontsize=14, fontweight="bold")
    plt.tight_layout()
    fig.savefig(os.path.join(OUTPUT_DIR, "morphometrics_distribution.png"), dpi=150)
    plt.close()
    print("  ✓ morphometrics_distribution.png")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    csv_path = os.path.join(DATA_DIR, "metadata.csv")

    if not os.path.exists(csv_path):
        print(f"metadata.csv не найден: {csv_path}")
        print("Сначала запусти: python3 src/prepare_dataset.py")
        sys.exit(1)

    df = pd.read_csv(csv_path)
    print(f"Загружено {len(df)} записей из metadata.csv")
    print(f"Генерирую визуализации в {OUTPUT_DIR}/...\n")

    plot_class_distribution(df)
    plot_morphometrics(df)
    plot_tri_view_examples(df)
    plot_3d_slices(df)

    print(f"\nГотово! Все графики в {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
