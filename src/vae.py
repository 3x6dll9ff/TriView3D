"""
Conditional VAE для 3D реконструкции клетки из tri-view проекций.

Отличие от TriViewAutoencoder (CNN):
  - Latent space: mu + logvar → reparameterization trick
  - Loss: BCE + Dice + KL divergence
  - Результат: более гладкие, biologically plausible формы
  - Trade-off: меньше деталей, но лучше общая форма

Архитектура:
  Encoder: 2D ConvNet [3, 64, 64] → mu (256), logvar (256)
  Decoder: 3D ConvTranspose → output [1, 64, 64, 64]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.autoencoder import Encoder2D, Decoder3D, dice_loss


class TriViewCVAE(nn.Module):
    """Conditional Variational Autoencoder для tri-view 3D реконструкции.

    Input:  [batch, 3, 64, 64]     — top + bottom + side projections
    Output: [batch, 1, 64, 64, 64] — generated 3D shape
    """

    def __init__(self, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim

        # Encoder: проекции → features (без финального ReLU — нужны mu/logvar)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1),
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
        self.flatten = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.3)
        )

        # VAE heads: mu и logvar вместо единого latent
        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        # Decoder: тот же 3D decoder
        self.decoder = Decoder3D(latent_dim=latent_dim)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode проекции → (mu, logvar)."""
        h = self.flatten(self.conv(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick: z = mu + eps * std."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode latent → 3D shape."""
        return self.decoder(z)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            recon: [batch, 1, 64, 64, 64] — reconstructed 3D
            mu: [batch, latent_dim]
            logvar: [batch, latent_dim]
        """
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        """Inference: используем только mu (без шума) для стабильного результата."""
        mu, _ = self.encode(x)
        return self.decode(mu)


# ── VAE Loss ────────────────────────────────────────────────────────


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
    """KL divergence: D_KL(q(z|x) || p(z)) где p(z) = N(0, I)."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())


def vae_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mu: torch.Tensor,
    logvar: torch.Tensor,
    kl_weight: float = 0.001,
    bce_weight: float = 0.5,
    dice_weight: float = 0.5,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Комбинированный VAE loss: BCE + Dice + KL.

    Args:
        pred: predicted 3D shape
        target: ground truth 3D shape
        mu, logvar: VAE latent parameters
        kl_weight: вес KL (маленький чтобы не доминировал)
        bce_weight: вес BCE reconstruction loss
        dice_weight: вес Dice loss

    Returns:
        total_loss, dict с компонентами для логирования
    """
    bce = F.binary_cross_entropy(pred, target, reduction="mean")
    dl = dice_loss(pred, target)
    kl = kl_divergence(mu, logvar)

    recon = bce_weight * bce + dice_weight * dl
    total = recon + kl_weight * kl

    return total, {
        "bce": bce.item(),
        "dice_loss": dl.item(),
        "kl": kl.item(),
        "recon": recon.item(),
        "total": total.item(),
    }
