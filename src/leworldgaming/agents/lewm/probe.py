"""Linear probe: latent z -> physical signals (HP self/opp, energy, etc.).

Used both as a planning value head and as a metric (R^2) of how much
physical state survives in the JEPA latent. See gemini_research.md §7.1, §8.
"""

from __future__ import annotations

import torch
from torch import nn


class LinearProbe(nn.Module):
    def __init__(self, latent_dim: int = 192, target_dim: int = 4) -> None:
        super().__init__()
        self.head = nn.Linear(latent_dim, target_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.head(z)
