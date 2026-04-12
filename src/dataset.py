"""
PyTorch Dataset для multi-view 3D cell reconstruction + classification.

Поддерживает два режима входа:
  - tri  = top + bottom + side
  - quad = top + bottom + side + front
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedShuffleSplit
from torch.utils.data import Dataset

try:
    from src.reconstruction_utils import get_view_names, load_view_stack
except ImportError:
    from reconstruction_utils import get_view_names, load_view_stack


class RandomHFlip2D3D:
    """Горизонтальный flip 2D проекций + соответствующий flip 3D volume."""

    def __init__(self, p: float = 0.5) -> None:
        self.p = p

    def __call__(
        self, tri_input: np.ndarray, target_3d: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        if np.random.random() < self.p:
            tri_input = tri_input[:, :, ::-1].copy()
            target_3d = target_3d[..., ::-1].copy()
        return tri_input, target_3d


class GaussianNoise:
    """Гауссов шум на проекции (input-only)."""

    def __init__(self, sigma: float = 0.03) -> None:
        self.sigma = sigma

    def __call__(self, tri_input: np.ndarray) -> np.ndarray:
        noise = np.random.randn(*tri_input.shape).astype(np.float32) * self.sigma
        return np.clip(tri_input + noise, 0.0, 1.0).astype(np.float32)


class BrightnessJitter:
    """Яркость/контраст на проекции (input-only)."""

    def __init__(self, brightness: float = 0.1, contrast: float = 0.1) -> None:
        self.brightness = brightness
        self.contrast = contrast

    def __call__(self, tri_input: np.ndarray) -> np.ndarray:
        if self.brightness > 0:
            delta = np.random.uniform(-self.brightness, self.brightness)
            tri_input = np.clip(tri_input + delta, 0.0, 1.0)
        if self.contrast > 0:
            alpha = np.random.uniform(1.0 - self.contrast, 1.0 + self.contrast)
            tri_input = np.clip((tri_input - 0.5) * alpha + 0.5, 0.0, 1.0)
        return tri_input.astype(np.float32)


class CellTriViewDataset(Dataset):
    """Dataset для multi-view 3D реконструкции и классификации."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "all",
        train_ratio: float = 0.8,
        seed: int = 42,
        transform=None,
        input_mode: str = "quad",
        augment: bool = False,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.input_mode = input_mode
        self.view_names = get_view_names(input_mode)
        self.augment = augment

        if augment:
            self._flip = RandomHFlip2D3D(p=0.5)
            self._noise = GaussianNoise(sigma=0.03)
            self._brightness = BrightnessJitter(brightness=0.1, contrast=0.1)

        csv_path = self.data_dir / "metadata.csv"
        self.df = pd.read_csv(csv_path)
        self._validate_metadata()

        if split in ("train", "test"):
            self.df = self._split_dataframe(split, train_ratio, seed)
        elif split != "all":
            raise ValueError(f"Unsupported split: {split}")

        self.df = self.df.reset_index(drop=True)

    def _validate_metadata(self) -> None:
        required_columns = {"name", "label", "cell_type"}
        missing_columns = required_columns.difference(self.df.columns)
        if missing_columns:
            raise ValueError(f"metadata.csv missing columns: {sorted(missing_columns)}")
        if self.df["name"].isna().any():
            raise ValueError("metadata.csv contains NaN in name column")
        if self.df["label"].isna().any():
            raise ValueError("metadata.csv contains NaN in label column")

        for view_name in (*self.view_names, "obj"):
            view_dir = self.data_dir / view_name
            if not view_dir.exists():
                hint = ""
                if view_name == "front_proj":
                    hint = " (возможно, вам нужно использовать --input_mode tri для старых данных)"
                raise FileNotFoundError(f"Required directory not found: {view_dir}{hint}")

    def _split_dataframe(
        self,
        split: str,
        train_ratio: float,
        seed: int,
    ) -> pd.DataFrame:
        stratify_labels = self.df["cell_type"].astype(str)
        if stratify_labels.value_counts().min() < 2:
            stratify_labels = self.df["label"].astype(str)

        splitter = StratifiedShuffleSplit(
            n_splits=1,
            train_size=train_ratio,
            random_state=seed,
        )
        train_idx, test_idx = next(splitter.split(self.df, stratify_labels))
        selected_idx = train_idx if split == "train" else test_idx
        return self.df.iloc[selected_idx].copy()

    def build_sample_weights(self, complexity_boost: float = 1.0) -> torch.Tensor:
        label_counts = self.df["label"].value_counts()
        class_weights = self.df["label"].map(lambda label: 1.0 / label_counts[label]).to_numpy(np.float32)

        if "complexity_score" not in self.df.columns or complexity_boost <= 0:
            return torch.as_tensor(class_weights, dtype=torch.double)

        complexity = self.df["complexity_score"].to_numpy(np.float32)
        complexity = complexity - float(complexity.min())
        complexity = complexity / (float(complexity.max()) + 1e-6)
        weights = class_weights * (1.0 + complexity_boost * complexity)
        return torch.as_tensor(weights, dtype=torch.double)

    def hard_threshold(self, quantile: float = 0.8) -> float:
        if "complexity_score" not in self.df.columns:
            return float("inf")
        return float(self.df["complexity_score"].quantile(quantile))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | float | str]:
        row = self.df.iloc[idx]
        name = str(row["name"])

        tri_input = load_view_stack(self.data_dir, name, self.input_mode)
        target_3d = np.load(self.data_dir / "obj" / f"{name}.npy").astype(np.float32)[np.newaxis, ...]

        if self.augment:
            tri_input, target_3d = self._flip(tri_input, target_3d)
            tri_input = self._noise(tri_input)
            tri_input = self._brightness(tri_input)

        if self.transform:
            tri_input = self.transform(tri_input)

        return {
            "input": torch.from_numpy(tri_input),
            "target_3d": torch.from_numpy(target_3d),
            "label": int(row["label"]),
            "name": name,
            "complexity_score": float(row.get("complexity_score", 0.0)),
            "cell_type": str(row["cell_type"]),
        }


class CellSingleViewDataset(Dataset):
    """Baseline: только 1 проекция (top) -> 3D."""

    def __init__(
        self,
        data_dir: str = "data/processed",
        split: str = "all",
        train_ratio: float = 0.8,
        seed: int = 42,
        view: str = "top_proj",
    ) -> None:
        self.data_dir = Path(data_dir)
        self.view = view
        self.df = pd.read_csv(self.data_dir / "metadata.csv")

        if split in ("train", "test"):
            splitter = StratifiedShuffleSplit(
                n_splits=1,
                train_size=train_ratio,
                random_state=seed,
            )
            train_idx, test_idx = next(splitter.split(self.df, self.df["label"].astype(str)))
            selected_idx = train_idx if split == "train" else test_idx
            self.df = self.df.iloc[selected_idx].reset_index(drop=True)
        elif split != "all":
            raise ValueError(f"Unsupported split: {split}")

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        row = self.df.iloc[idx]
        name = str(row["name"])

        proj = np.load(self.data_dir / self.view / f"{name}.npy").astype(np.float32)
        obj_3d = np.load(self.data_dir / "obj" / f"{name}.npy").astype(np.float32)

        return {
            "input": torch.from_numpy(proj[np.newaxis, ...]),
            "target_3d": torch.from_numpy(obj_3d[np.newaxis, ...]),
            "label": int(row["label"]),
            "name": name,
        }
