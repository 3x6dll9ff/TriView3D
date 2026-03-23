"""
3D визуализация voxel-кубов и результатов.

Функции:
  - plot_voxel_3d: интерактивный Plotly 3D scatter
  - plot_slices: срезы xy/xz/yz через matplotlib
  - plot_reconstruction: predicted vs ground truth overlay
  - plot_dual_view_input: визуализация входных top/bottom срезов
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def plot_voxel_3d(
    voxel: np.ndarray,
    title: str = "3D Voxel",
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = True,
    max_points: int = 15000,
    colorscale: str = "Viridis",
) -> go.Figure:
    """Plotly 3D scatter: voxel'и с intensity > threshold."""
    x, y, z = np.where(voxel > threshold)
    values = voxel[x, y, z]

    if len(x) > max_points:
        idx = np.random.choice(len(x), max_points, replace=False)
        x, y, z, values = x[idx], y[idx], z[idx], values[idx]

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode="markers",
                marker=dict(
                    size=2, color=values, colorscale=colorscale,
                    opacity=0.6, colorbar=dict(title="Intensity"),
                ),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X", yaxis_title="Y", zaxis_title="Z",
            aspectmode="data",
        ),
        width=800, height=700, template="plotly_dark",
    )

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def plot_slices(
    voxel: np.ndarray,
    title: str = "Voxel Slices",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Три среза voxel-куба: XY, XZ, YZ по центру."""
    d, h, w = voxel.shape
    mid_x, mid_y, mid_z = d // 2, h // 2, w // 2

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    slices = [
        (voxel[mid_x, :, :], f"XY (z={mid_x})"),
        (voxel[:, mid_y, :], f"XZ (y={mid_y})"),
        (voxel[:, :, mid_z], f"YZ (x={mid_z})"),
    ]
    for ax, (s, subtitle) in zip(axes, slices):
        im = ax.imshow(s, cmap="viridis", vmin=0, vmax=1)
        ax.set_title(subtitle)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_reconstruction(
    predicted: np.ndarray,
    ground_truth: np.ndarray,
    title: str = "Predicted vs Ground Truth",
    threshold: float = 0.5,
    save_path: str | None = None,
    show: bool = True,
) -> go.Figure:
    """Side-by-side 3D: predicted vs ground truth."""
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=("Predicted", "Ground Truth"),
        specs=[[{"type": "scatter3d"}, {"type": "scatter3d"}]],
    )
    max_pts = 10000

    for col, (voxel, cscale) in enumerate(
        [(predicted, "Reds"), (ground_truth, "Blues")], start=1
    ):
        x, y, z = np.where(voxel > threshold)
        vals = voxel[x, y, z]
        if len(x) > max_pts:
            idx = np.random.choice(len(x), max_pts, replace=False)
            x, y, z, vals = x[idx], y[idx], z[idx], vals[idx]

        fig.add_trace(
            go.Scatter3d(
                x=x, y=y, z=z, mode="markers",
                marker=dict(size=2, color=vals, colorscale=cscale, opacity=0.6),
            ),
            row=1, col=col,
        )

    fig.update_layout(
        title=title, width=1400, height=700, template="plotly_dark"
    )
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.write_html(save_path)
    if show:
        fig.show()
    return fig


def plot_dual_view_input(
    top: np.ndarray,
    bottom: np.ndarray,
    title: str = "Dual-View Input",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Визуализация top и bottom срезов рядом."""
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    axes[0].imshow(top, cmap="gray", vmin=0, vmax=1)
    axes[0].set_title("Top slice")
    axes[1].imshow(bottom, cmap="gray", vmin=0, vmax=1)
    axes[1].set_title("Bottom slice")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_metrics_distribution(
    metrics_df,
    label_col: str = "label",
    save_path: str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Распределение морфометрических метрик по классам."""
    feature_cols = [
        "volume", "sphericity", "convexity",
        "eccentricity", "surface_roughness",
    ]
    available = [c for c in feature_cols if c in metrics_df.columns]

    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    if n == 1:
        axes = [axes]

    for ax, col in zip(axes, available):
        for label, color in [(0, "steelblue"), (1, "crimson")]:
            subset = metrics_df[metrics_df[label_col] == label][col]
            label_name = "Normal" if label == 0 else "Anomaly"
            ax.hist(subset, bins=20, alpha=0.6, color=color, label=label_name)
        ax.set_title(col)
        ax.legend()

    fig.suptitle("Metrics Distribution", fontsize=14, fontweight="bold")
    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    if show:
        plt.show()
    return fig
