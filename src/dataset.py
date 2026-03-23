"""
PyTorch Dataset для tri-view 3D cell reconstruction + classification.

Загружает подготовленные данные из data/processed/:
  - top_proj/    — Sum Projection верхней половины Z (вид сверху, 64×64)
  - bottom_proj/ — Sum Projection нижней половины Z (вид снизу, 64×64)
  - side_proj/   — Sum Projection по Y (профиль сбоку, 64×64)
  - obj/        — 3D ground truth (64×64×64)
  - metadata.csv — имена файлов + метки
"""

import os

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class CellTriViewDataset(Dataset):
    """Dataset для tri-view 3D реконструкции и классификации.

    Каждый элемент:
      - tri_input: tensor [3, H, W] — top + front + side projections
      - target_3d: tensor [1, D, H, W] — 3D ground truth
      - label: int (0=normal, 1=anomaly)
      - name: str
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "all",
        train_ratio: float = 0.8,
        seed: int = 42,
        transform=None,
    ):
        self.data_dir = data_dir
        self.transform = transform

        csv_path = os.path.join(data_dir, "metadata.csv")
        self.df = pd.read_csv(csv_path)

        # Train/test split
        if split in ("train", "test"):
            np.random.seed(seed)
            indices = np.random.permutation(len(self.df))
            n_train = int(len(self.df) * train_ratio)
            if split == "train":
                self.df = self.df.iloc[indices[:n_train]].reset_index(drop=True)
            else:
                self.df = self.df.iloc[indices[n_train:]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        name = row["name"]

        # Загрузка 3 проекций
        top = np.load(os.path.join(self.data_dir, "top_proj", f"{name}.npy"))
        bottom = np.load(os.path.join(self.data_dir, "bottom_proj", f"{name}.npy"))
        side = np.load(os.path.join(self.data_dir, "side_proj", f"{name}.npy"))
        obj_3d = np.load(os.path.join(self.data_dir, "obj", f"{name}.npy"))

        # Tri input: [3, H, W] — top + bottom + side projections
        tri_input = np.stack([top, bottom, side], axis=0).astype(np.float32)

        # Target 3D: [1, D, H, W]
        target_3d = obj_3d[np.newaxis, ...].astype(np.float32)

        label = int(row["label"])

        if self.transform:
            tri_input = self.transform(tri_input)

        return {
            "input": torch.from_numpy(tri_input),
            "target_3d": torch.from_numpy(target_3d),
            "label": label,
            "name": name,
        }


class CellSingleViewDataset(Dataset):
    """Dataset baseline: только 1 проекция (top) → 3D.

    Для сравнения: 1-view vs 3-view.
    """

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "all",
        train_ratio: float = 0.8,
        seed: int = 42,
        view: str = "top_proj",
    ):
        self.data_dir = data_dir
        self.view = view

        csv_path = os.path.join(data_dir, "metadata.csv")
        self.df = pd.read_csv(csv_path)

        if split in ("train", "test"):
            np.random.seed(seed)
            indices = np.random.permutation(len(self.df))
            n_train = int(len(self.df) * train_ratio)
            if split == "train":
                self.df = self.df.iloc[indices[:n_train]].reset_index(drop=True)
            else:
                self.df = self.df.iloc[indices[n_train:]].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        name = row["name"]

        proj = np.load(os.path.join(self.data_dir, self.view, f"{name}.npy"))
        obj_3d = np.load(os.path.join(self.data_dir, "obj", f"{name}.npy"))

        # Single input: [1, H, W]
        single_input = proj[np.newaxis, ...].astype(np.float32)
        target_3d = obj_3d[np.newaxis, ...].astype(np.float32)
        label = int(row["label"])

        return {
            "input": torch.from_numpy(single_input),
            "target_3d": torch.from_numpy(target_3d),
            "label": label,
            "name": name,
        }
