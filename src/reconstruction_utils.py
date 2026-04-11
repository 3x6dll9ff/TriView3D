from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch


VIEW_NAMES_BY_MODE: dict[str, tuple[str, ...]] = {
    "tri": ("top_proj", "bottom_proj", "side_proj"),
    "quad": ("top_proj", "bottom_proj", "side_proj", "front_proj"),
}


def get_view_names(input_mode: str) -> tuple[str, ...]:
    if input_mode not in VIEW_NAMES_BY_MODE:
        raise ValueError(f"Unsupported input_mode: {input_mode}")
    return VIEW_NAMES_BY_MODE[input_mode]


def infer_in_channels_from_state_dict(state_dict: dict[str, torch.Tensor]) -> int:
    for key in state_dict.keys():
        if "conv.0.weight" in key:
            return int(state_dict[key].shape[1])
    raise KeyError("Не удалось найти входной сверточный слой (conv.0.weight) в state_dict")


def _safe_zscore(values: np.ndarray) -> np.ndarray:
    values = values.astype(np.float32)
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(values, dtype=np.float32)
    return (values - float(values.mean())) / std


def add_complexity_score(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    surface_ratio = result["surface_area"].to_numpy(np.float32) / np.clip(
        result["volume"].to_numpy(np.float32), 1.0, None
    )
    complexity = (
        _safe_zscore(result["surface_roughness"].to_numpy(np.float32))
        + _safe_zscore(surface_ratio)
        + _safe_zscore(1.0 - result["convexity"].to_numpy(np.float32))
    )
    result["complexity_score"] = complexity.astype(np.float32)
    return result


def extract_all_views(volume_3d: np.ndarray) -> dict[str, np.ndarray]:
    binary = (volume_3d > 0).astype(np.float32)
    depth, height, width = binary.shape
    z_mid = depth // 2

    top_proj = binary[:z_mid].sum(axis=0) / max(z_mid, 1)
    bottom_proj = binary[z_mid:].sum(axis=0) / max(depth - z_mid, 1)
    side_proj = binary.sum(axis=1) / max(height, 1)
    front_proj = binary.sum(axis=2) / max(width, 1)

    return {
        "top_proj": top_proj.astype(np.float32),
        "bottom_proj": bottom_proj.astype(np.float32),
        "side_proj": side_proj.astype(np.float32),
        "front_proj": front_proj.astype(np.float32),
    }


def load_view_stack(
    data_dir: str | Path,
    sample_name: str,
    input_mode: str,
) -> np.ndarray:
    data_dir = Path(data_dir)
    views = [
        np.load(data_dir / view_name / f"{sample_name}.npy").astype(np.float32)
        for view_name in get_view_names(input_mode)
    ]
    return np.stack(views, axis=0).astype(np.float32)


def project_volume_batch(
    volume: torch.Tensor,
    view_names: tuple[str, ...],
) -> torch.Tensor:
    if volume.ndim != 5:
        raise ValueError(f"Expected 5D tensor [B,1,D,H,W], got {tuple(volume.shape)}")

    # volume: (B, 1, D, H, W)
    _, _, depth, height, width = volume.shape
    z_mid = depth // 2
    projections: list[torch.Tensor] = []

    for view_name in view_names:
        if view_name == "top_proj":
            proj = volume[:, :, :z_mid].sum(dim=2) / max(z_mid, 1)
        elif view_name == "bottom_proj":
            proj = volume[:, :, z_mid:].sum(dim=2) / max(depth - z_mid, 1)
        elif view_name == "side_proj":
            proj = volume.sum(dim=3) / max(height, 1)
        elif view_name == "front_proj":
            proj = volume.sum(dim=4) / max(width, 1)
        else:
            raise ValueError(f"Unsupported view_name: {view_name}")

        projections.append(proj.squeeze(1))

    return torch.stack(projections, dim=1)


def lift_views_to_volume(
    views: torch.Tensor,
    view_names: tuple[str, ...],
    depth: int = 64,
    height: int = 64,
    width: int = 64,
) -> torch.Tensor:
    if views.ndim != 4:
        raise ValueError(f"Expected 4D tensor [B,C,H,W], got {tuple(views.shape)}")
    if views.shape[1] != len(view_names):
        raise ValueError("Number of channels does not match view_names")

    batch_size = views.shape[0]
    z_mid = depth // 2
    lifted: list[torch.Tensor] = []

    for channel_idx, view_name in enumerate(view_names):
        view = views[:, channel_idx]

        if view_name == "top_proj":
            volume = views.new_zeros((batch_size, 1, depth, height, width))
            volume[:, :, :z_mid] = view.unsqueeze(1).unsqueeze(2).expand(-1, -1, z_mid, -1, -1)
        elif view_name == "bottom_proj":
            volume = views.new_zeros((batch_size, 1, depth, height, width))
            volume[:, :, z_mid:] = view.unsqueeze(1).unsqueeze(2).expand(
                -1, -1, depth - z_mid, -1, -1
            )
        elif view_name == "side_proj":
            volume = view.unsqueeze(1).unsqueeze(3).expand(-1, -1, depth, height, width)
        elif view_name == "front_proj":
            volume = view.unsqueeze(1).unsqueeze(4).expand(-1, -1, depth, height, width)
        else:
            raise ValueError(f"Unsupported view_name: {view_name}")

        lifted.append(volume)

    return torch.cat(lifted, dim=1)


def load_dataset_config(data_dir: str | Path) -> dict[str, object] | None:
    config_path = Path(data_dir) / "dataset_config.json"
    if not config_path.exists():
        return None
    import json

    with config_path.open("r", encoding="utf-8") as file:
        return json.load(file)
