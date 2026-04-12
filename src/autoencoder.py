"""
Multi-View 3D Autoencoder: из 3/4 проекций предсказывает 3D форму.

Loss включает:
  - voxel BCE
  - Dice
  - projection consistency (BCE)
  - boundary-aware BCE

Архитектура:
  - Decoder3D с skip connections от lifted views на каждом уровне
  - Skip connections — additive residuals (Pix2Vox-style):
    downsample skip_volume → 3x3 conv → add к decoder features
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.reconstruction_utils import lift_views_to_volume, project_volume_batch


class Encoder2D(nn.Module):
    """2D Encoder: [batch, C, 64, 64] → [batch, latent_dim]."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256) -> None:
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
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
    """3D Decoder: [batch, latent_dim] → [batch, 1, 64, 64, 64].

    С skip connections от lifted views на каждом уровне декодера:
      skip_volume downsampling → 3x3 conv → additive residual к decoder features.
    При skip_channels=0 ведёт себя как обычный decoder (без skip).
    """

    def __init__(self, latent_dim: int = 256, skip_channels: int = 0) -> None:
        super().__init__()
        self.skip_channels = skip_channels

        self.fc = nn.Sequential(
            nn.Linear(latent_dim, 256 * 4 * 4 * 4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
        )

        self.deconv0 = nn.ConvTranspose3d(256, 128, 4, stride=2, padding=1)
        self.bn0 = nn.BatchNorm3d(128)

        self.deconv1 = nn.ConvTranspose3d(128, 64, 4, stride=2, padding=1)
        self.bn1 = nn.BatchNorm3d(64)

        self.deconv2 = nn.ConvTranspose3d(64, 32, 4, stride=2, padding=1)
        self.bn2 = nn.BatchNorm3d(32)

        self.deconv3 = nn.ConvTranspose3d(32, 1, 4, stride=2, padding=1)

        if skip_channels > 0:
            self.skip0 = nn.Sequential(
                nn.Conv3d(skip_channels, 256, 3, padding=1),
                nn.BatchNorm3d(256), nn.ReLU(inplace=True),
            )
            self.skip1 = nn.Sequential(
                nn.Conv3d(skip_channels, 128, 3, padding=1),
                nn.BatchNorm3d(128), nn.ReLU(inplace=True),
            )
            self.skip2 = nn.Sequential(
                nn.Conv3d(skip_channels, 64, 3, padding=1),
                nn.BatchNorm3d(64), nn.ReLU(inplace=True),
            )
            self.skip3 = nn.Sequential(
                nn.Conv3d(skip_channels, 32, 3, padding=1),
                nn.BatchNorm3d(32), nn.ReLU(inplace=True),
            )

    def _downsample_skip(
        self, skip: torch.Tensor, size: tuple[int, int, int]
    ) -> torch.Tensor:
        if skip.shape[2:] == size:
            return skip
        return F.interpolate(skip, size=size, mode="trilinear", align_corners=False)

    def forward(
        self, z: torch.Tensor, skip_volume: torch.Tensor | None = None
    ) -> torch.Tensor:
        x = self.fc(z)
        x = x.view(-1, 256, 4, 4, 4)

        use_skip = skip_volume is not None and self.skip_channels > 0

        if use_skip:
            x = x + self.skip0(self._downsample_skip(skip_volume, (4, 4, 4)))
        x = F.relu(self.bn0(self.deconv0(x)))

        if use_skip:
            x = x + self.skip1(self._downsample_skip(skip_volume, (8, 8, 8)))
        x = F.relu(self.bn1(self.deconv1(x)))

        if use_skip:
            x = x + self.skip2(self._downsample_skip(skip_volume, (16, 16, 16)))
        x = F.relu(self.bn2(self.deconv2(x)))

        if use_skip:
            x = x + self.skip3(self._downsample_skip(skip_volume, (32, 32, 32)))
        x = self.deconv3(x)

        return x


class TriViewAutoencoder(nn.Module):
    """Tri-View 3D Autoencoder with skip connections.

    Input:  [batch, 3, 64, 64]     — top + bottom + side projections
    Output: [batch, 1, 64, 64, 64] — predicted 3D shape (logits)
    """

    def __init__(
        self,
        latent_dim: int = 256,
        in_channels: int = 3,
        skip_channels: int | None = None,
    ) -> None:
        super().__init__()
        if skip_channels is None:
            skip_channels = in_channels
        self.encoder = Encoder2D(in_channels=in_channels, latent_dim=latent_dim)
        self.decoder = Decoder3D(latent_dim=latent_dim, skip_channels=skip_channels)
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.view_names = _view_names_for_channels(in_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        skip = None
        if self.skip_channels > 0:
            skip = lift_views_to_volume(x, self.view_names)
        recon = self.decoder(z, skip)
        return recon

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class SingleViewAutoencoder(nn.Module):
    """Baseline: single-view autoencoder (1 проекция → 3D)."""

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder2D(in_channels=1, latent_dim=latent_dim)
        self.decoder = Decoder3D(latent_dim=latent_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.encoder(x)
        return self.decoder(z)


def _view_names_for_channels(in_channels: int) -> tuple[str, ...]:
    from src.reconstruction_utils import get_view_names

    return get_view_names("quad" if in_channels == 4 else "tri")


# ── Losses ──────────────────────────────────────────────────────────


def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1.0) -> torch.Tensor:
    pred_prob = torch.sigmoid(pred)
    pred_flat = pred_prob.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    return 1.0 - dice


def boundary_mask(target: torch.Tensor) -> torch.Tensor:
    target_bin = (target > 0.5).float()
    dilated = F.max_pool3d(target_bin, kernel_size=3, stride=1, padding=1)
    eroded = -F.max_pool3d(-target_bin, kernel_size=3, stride=1, padding=1)
    return (dilated - eroded > 0).float()


def boundary_bce_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    boundary_boost: float = 2.0,
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
    projected = torch.clamp(projected, min=1e-6, max=1.0 - 1e-6)
    return F.binary_cross_entropy(projected, inputs, reduction="mean")


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    inputs: torch.Tensor | None = None,
    view_names: tuple[str, ...] | None = None,
    bce_weight: float = 0.35,
    dice_weight: float = 0.25,
    projection_weight: float = 0.5,
    surface_weight: float = 0.15,
    boundary_boost: float = 2.0,
    return_components: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, float]]:
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
