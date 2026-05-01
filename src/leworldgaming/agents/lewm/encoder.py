"""JEPA encoder: pixels -> latent embedding.

Targets ~15M params total across encoder+predictor (per gemini_research.md §6).
Reference impl in external/le-wm — port over the weekend.
"""

from __future__ import annotations

import torch
from torch import nn


class Encoder(nn.Module):
    """Tiny placeholder encoder. Replace with a small ViT / CNN ported from external/le-wm."""

    def __init__(self, in_channels: int = 3, latent_dim: int = 256) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, stride=2, padding=1),  # 224 -> 112
            nn.GELU(),
            nn.Conv2d(32, 64, 4, stride=2, padding=1),  # 112 -> 56
            nn.GELU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),  # 56 -> 28
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, latent_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
