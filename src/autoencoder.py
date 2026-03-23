"""
Tri-View 3D Autoencoder: из 3 проекций (top + front + side) предсказывает 3D форму.

Архитектура:
  Encoder: 2D ConvNet [3, 64, 64] → latent vector (256-dim)
  Decoder: 3D ConvTranspose → output [1, 64, 64, 64]

Loss: BCE + Dice для бинарной 3D реконструкции.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


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
            nn.Sigmoid(),
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

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.encoder = Encoder2D(in_channels=3, latent_dim=latent_dim)
        self.decoder = Decoder3D(latent_dim=latent_dim)
        self.latent_dim = latent_dim

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
    pred_flat = pred.contiguous().view(-1)
    target_flat = target.contiguous().view(-1)

    intersection = (pred_flat * target_flat).sum()
    dice = (2.0 * intersection + smooth) / (
        pred_flat.sum() + target_flat.sum() + smooth
    )
    return 1.0 - dice


def reconstruction_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> torch.Tensor:
    """Комбинированный loss: BCE + Dice."""
    bce = F.binary_cross_entropy(pred, target, reduction="mean")
    dl = dice_loss(pred, target)
    return bce_weight * bce + dice_weight * dl
