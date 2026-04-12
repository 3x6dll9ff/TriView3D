"""
Conditional VAE для 3D реконструкции клетки из tri-view проекций.

Архитектура:
  Encoder: 2D ConvNet [3, 64, 64] → mu (256), logvar (256)
  Decoder: 3D ConvTranspose + skip connections → output [1, 64, 64, 64]
  Skip: lifted views → additive residuals на каждом уровне decoder
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.autoencoder import Decoder3D, reconstruction_loss
from src.reconstruction_utils import lift_views_to_volume, project_volume_batch


def _view_names_for_channels(in_channels: int) -> tuple[str, ...]:
    from src.reconstruction_utils import get_view_names

    return get_view_names("quad" if in_channels == 4 else "tri")


class TriViewCVAE(nn.Module):
    """Conditional Variational Autoencoder для tri-view 3D реконструкции.

    Input:  [batch, 3, 64, 64]     — top + bottom + side projections
    Output: [batch, 1, 64, 64, 64] — generated 3D shape (logits)
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
        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.view_names = _view_names_for_channels(in_channels)

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

        self.fc_mu = nn.Linear(256 * 4 * 4, latent_dim)
        self.fc_logvar = nn.Linear(256 * 4 * 4, latent_dim)

        self.decoder = Decoder3D(latent_dim=latent_dim, skip_channels=skip_channels)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.flatten(self.conv(x))
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, skip_volume: torch.Tensor | None = None) -> torch.Tensor:
        return self.decoder(z, skip_volume)

    def forward(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        skip = lift_views_to_volume(x, self.view_names) if self.skip_channels > 0 else None
        recon = self.decode(z, skip)
        return recon, mu, logvar

    def generate(self, x: torch.Tensor) -> torch.Tensor:
        mu, _ = self.encode(x)
        skip = lift_views_to_volume(x, self.view_names) if self.skip_channels > 0 else None
        return torch.sigmoid(self.decode(mu, skip))


# ── VAE Loss ────────────────────────────────────────────────────────


def kl_divergence(mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
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
    projection_weight: float = 0.5,
    surface_weight: float = 0.15,
    boundary_boost: float = 2.0,
) -> tuple[torch.Tensor, dict[str, float]]:
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
    """Сэмплирует несколько кандидатов и выбирает лучший по reprojection error.

    Returns logits (not probabilities) for compatibility with reconstruction_loss.
    """
    mu, logvar = model.encode(inputs)
    skip = lift_views_to_volume(inputs, view_names) if model.skip_channels > 0 else None

    candidate_logits_0 = model.decode(mu, skip)
    candidate_probs_0 = torch.sigmoid(candidate_logits_0)
    scores: list[torch.Tensor] = [
        F.l1_loss(project_volume_batch(candidate_probs_0, view_names), inputs, reduction="none")
        .mean(dim=(1, 2, 3))
    ]
    all_logits: list[torch.Tensor] = [candidate_logits_0]

    for _ in range(max(num_samples - 1, 0)):
        z = model.reparameterize(mu, logvar)
        candidate_logits = model.decode(z, skip)
        candidate_probs = torch.sigmoid(candidate_logits)
        score = F.l1_loss(project_volume_batch(candidate_probs, view_names), inputs, reduction="none")
        scores.append(score.mean(dim=(1, 2, 3)))
        all_logits.append(candidate_logits)

    logits_tensor = torch.stack(all_logits, dim=1)
    score_tensor = torch.stack(scores, dim=1)
    best_indices = score_tensor.argmin(dim=1)
    batch_indices = torch.arange(inputs.size(0), device=inputs.device)
    best_logits = logits_tensor[batch_indices, best_indices]
    best_score = score_tensor[batch_indices, best_indices]
    return best_logits, best_score
