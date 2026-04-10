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

try:
    from src.autoencoder import Decoder3D, reconstruction_loss
    from src.reconstruction_utils import project_volume_batch
except ImportError:
    from autoencoder import Decoder3D, reconstruction_loss
    from reconstruction_utils import project_volume_batch


class TriViewCVAE(nn.Module):
    """Conditional Variational Autoencoder для tri-view 3D реконструкции.

    Input:  [batch, 3, 64, 64]     — top + bottom + side projections
    Output: [batch, 1, 64, 64, 64] — generated 3D shape
    """

    def __init__(self, latent_dim: int = 256, in_channels: int = 3) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.in_channels = in_channels

        # Encoder: проекции → features (без финального ReLU — нужны mu/logvar)
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
    inputs: torch.Tensor | None = None,
    view_names: tuple[str, ...] | None = None,
    bce_weight: float = 0.35,
    dice_weight: float = 0.25,
    projection_weight: float = 0.25,
    surface_weight: float = 0.15,
    boundary_boost: float = 4.0,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Комбинированный VAE loss: reconstruction + KL."""
    recon, recon_components = reconstruction_loss(
        pred,
        target,
        inputs=inputs,
        view_names=view_names,
        bce_weight=bce_weight,
        dice_weight=dice_weight,
        projection_weight=projection_weight,
        surface_weight=surface_weight,
        boundary_boost=boundary_boost,
        return_components=True,
    )
    kl = kl_divergence(mu, logvar)
    total = recon + kl_weight * kl

    return total, {
        **recon_components,
        "kl": kl.item(),
        "recon": recon.item(),
        "total": total.item(),
    }


@torch.no_grad()
def best_of_k_generate(
    model: TriViewCVAE,
    inputs: torch.Tensor,
    view_names: tuple[str, ...],
    num_samples: int = 8,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Сэмплирует несколько кандидатов и выбирает лучший по reprojection error."""
    mu, logvar = model.encode(inputs)
    candidates: list[torch.Tensor] = [model.decode(mu)]
    scores: list[torch.Tensor] = [
        F.l1_loss(project_volume_batch(candidates[0], view_names), inputs, reduction="none")
        .mean(dim=(1, 2, 3))
    ]

    for _ in range(max(num_samples - 1, 0)):
        z = model.reparameterize(mu, logvar)
        candidate = model.decode(z)
        score = F.l1_loss(project_volume_batch(candidate, view_names), inputs, reduction="none")
        scores.append(score.mean(dim=(1, 2, 3)))
        candidates.append(candidate)

    candidate_tensor = torch.stack(candidates, dim=1)
    score_tensor = torch.stack(scores, dim=1)
    best_indices = score_tensor.argmin(dim=1)
    batch_indices = torch.arange(inputs.size(0), device=inputs.device)
    best_pred = candidate_tensor[batch_indices, best_indices]
    best_score = score_tensor[batch_indices, best_indices]
    return best_pred, best_score
