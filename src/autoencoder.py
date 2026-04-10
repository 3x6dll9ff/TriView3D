"""
Multi-View 3D Autoencoder: из 3/4 проекций предсказывает 3D форму.

Loss включает:
  - voxel BCE
  - Dice
  - projection consistency
  - boundary-aware BCE
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from src.reconstruction_utils import project_volume_batch
except ImportError:
    from reconstruction_utils import project_volume_batch


class Encoder2D(nn.Module):
    """2D Encoder: [batch, C, 64, 64] → [batch, latent_dim]."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            # C×64×64 → 32×32×32
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            # 32×32×32 → 64×16×16
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 64×16×16 → 128×8×8
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 128×8×8 → 256×4×4
            nn.Conv2d(128, 256, 3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(256 * 4 * 4, latent_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.conv(x)
        latent = self.fc(features)
        return latent


class Decoder3D(nn.Module):
    """3D Decoder: [batch, latent_dim] → [batch, 1, 64, 64, 64]."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )
        self.deconv = nn.Sequential(
            # 256×4×4×4 → 128×8×8×8
            nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
            # 128×8×8×8 → 64×16×16×16
            nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            # 64×16×16×16 → 32×32×32×32
            nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            # 32×32×32×32 → 1×64×64×64
            nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1),
            # Sigmoid удален для стабильности (используем BCEWithLogitsLoss)
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4, 4)
        x = self.deconv(x)
        return x


class TriViewAutoencoder(nn.Module):
    """Tri-View 3D Autoencoder.

    Input:  [batch, 3, 64, 64]     — top + front + side projections
    Output: [batch, 1, 64, 64, 64] — predicted 3D shape
    """

    def __init__(self, latent_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self.encoder = Encoder2D(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder3D(latent_dim=latent_dim)
        self.latent_dim = latent_dim
        self.in_channels = in_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        recon = self.decoder(z)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Только encoder → latent vector (для классификации)."""
        return self.encoder(x)


class SingleViewAutoencoder(nn.Module):
    """Baseline: single-view autoencoder (1 проекция → 3D).

    Для сравнения с tri-view.
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder2D(in_channels=1, latent_dim=latent_dim)
        self.decoder = Decoder3D(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


# ── Losses ──────────────────────────────────────────────────────────


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    """Dice loss для бинарной 3D сегментации."""
    pred_prob = torch.sigmoid(pred)
    pred_flat = pred_prob.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    return 1.0 - dice


def boundary_mask(target: torch.Tensor) -> torch.Tensor:
    """Выделяет boundary voxels через разность dilate/erode."""
    target_bin = (target > 0.5).float()
    dilated = F.max_pool3d(target_bin, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool3d(-target_bin, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded > 0).float()


def boundary_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_boost: float = 4.0,
) -> torch.Tensor:
    mask = boundary_mask(target)
    weights = 1.0 + boundary_boost * mask
    return F.binary_cross_entropy_with_logits(pred, target, weight=weights, reduction="mean")


def projection_consistency_loss(
    pred: torch.Tensor,
    inputs: torch.Tensor,
    view_names: tuple[str, ...],
) -> torch.Tensor:
    projected = project_volume_batch(torch.sigmoid(pred), view_names)
    return F.l1_loss(projected, inputs, reduction="mean")


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    inputs: torch.Tensor | None = None,
    view_names: tuple[str, ...] | None = None,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
    projection_weight: float = 0.0,
    surface_weight: float = 0.0,
    boundary_boost: float = 4.0,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
    """Комбинированный loss для multi-view 3D reconstruction."""
    bce = F.binary_cross_entropy_with_logits(pred, target, reduction="mean")
    dl = dice_loss(pred, target)
    proj = pred.new_tensor(0.0)
    surface = pred.new_tensor(0.0)

    if inputs is not None and view_names is not None and projection_weight > 0:
        proj = projection_consistency_loss(pred, inputs, view_names)

    if surface_weight > 0:
        surface = boundary_bce_loss(pred, target, boundary_boost=boundary_boost)

    total = (
        bce_weight * bce
        + dice_weight * dl
        + projection_weight * proj
        + surface_weight * surface
    )

    if return_components:
        return total, {
            "bce": float(bce.item()),
            "dice_loss": float(dl.item()),
            "projection": float(proj.item()),
            "surface": float(surface.item()),
            "total": float(total.item()),
        }
    return total
