"""Value head: ``z → twohot logits`` over discounted return bins."""

from __future__ import annotations

import torch
from torch import nn


class ValueHead(nn.Module):
    def __init__(self, latent_dim: int = 192, hidden_dim: int = 512, num_bins: int = 41) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
