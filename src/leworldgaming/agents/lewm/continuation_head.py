"""Continuation head: ``z → BCE logit``. Predicts P(round still ongoing)."""

from __future__ import annotations

import torch
from torch import nn


class ContinuationHead(nn.Module):
    def __init__(self, latent_dim: int = 192, hidden_dim: int = 512) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
