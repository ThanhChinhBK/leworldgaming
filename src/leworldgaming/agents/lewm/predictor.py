"""Latent transition model: (z_t, a_t) -> z_{t+1}.

Pure-latent rollout means no pixel decoder — this is the JEPA cost saving
called out in gemini_research.md §6.
"""

from __future__ import annotations

import torch
from torch import nn


class Predictor(nn.Module):
    def __init__(self, latent_dim: int = 256, action_dim: int = 56, hidden: int = 512) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.net = nn.Sequential(
            nn.Linear(latent_dim + action_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, latent_dim),
        )

    def forward(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a], dim=-1))
