"""Action encoder matching ``external/le-wm/module.py::Embedder``."""

from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    """Stacked raw actions -> action embedding."""

    def __init__(
        self,
        action_dim: int = 56,
        emb_dim: int = 192,
        smoothed_dim: int = 10,
        mlp_scale: int = 4,
    ) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.patch_embed = nn.Conv1d(
            action_dim, smoothed_dim, kernel_size=1, stride=1
        )
        self.embed = nn.Sequential(
            nn.Linear(smoothed_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, action_dim) one-hot or continuous, returns (B, T, emb_dim)."""
        x = x.float().permute(0, 2, 1)
        x = self.patch_embed(x).permute(0, 2, 1)
        return self.embed(x)
