"""Reward head: ``(z, action_emb) → twohot logits``.

Conditioned on action because MCTS scores edges, not nodes. Action input
is the same ``latent_dim``-sized embedding that ``ActionEncoder`` produces
for the predictor — sharing the embedding space lets the head re-use the
already-trained action representation.
"""

from __future__ import annotations

import torch
from torch import nn


class RewardHead(nn.Module):
    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512, num_bins: int = 41) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_bins),
        )

    def forward(self, z: torch.Tensor, a_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([z, a_emb], dim=-1))
