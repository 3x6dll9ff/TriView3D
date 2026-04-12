from __future__ import annotations

import torch
import torch.nn as nn


class ResidualBlock3D(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(channels),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))


class DetailRefiner(nn.Module):
    """3D residual refiner поверх coarse reconstruction."""

    def __init__(self, view_channels: int = 4, hidden_channels: int = 32) -> None:
        super().__init__()
        in_channels = 1 + view_channels + 1
        self.stem = nn.Sequential(
            nn.Conv3d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            ResidualBlock3D(hidden_channels),
            ResidualBlock3D(hidden_channels),
            ResidualBlock3D(hidden_channels),
        )
        self.head = nn.Sequential(
            nn.Conv3d(hidden_channels, hidden_channels // 2, kernel_size=3, padding=1),
            nn.BatchNorm3d(hidden_channels // 2),
            nn.ReLU(inplace=True),
            nn.Conv3d(hidden_channels // 2, 1, kernel_size=1),
        )

    def forward(self, coarse_logits: torch.Tensor, lifted_views: torch.Tensor) -> torch.Tensor:
        # Теперь coarse — это логиты. Для расчета неопределенности нам нужны вероятности (0...1)
        coarse_prob = torch.sigmoid(coarse_logits)
        
        # Неопределенность: выше там, где вероятность близка к 0.5 (в логитах это около 0)
        uncertainty = 1.0 - torch.abs(2.0 * coarse_prob - 1.0)
        
        x = torch.cat([coarse_prob, lifted_views, uncertainty], dim=1)
        features = self.stem(x)
        features = self.blocks(features)
        residual_logits = self.head(features)

        # Раньше тут был torch.logit(coarse), но теперь это и есть coarse_logits
        gating = uncertainty * 1.25
        refined_logits = coarse_logits + gating * residual_logits
        return refined_logits
