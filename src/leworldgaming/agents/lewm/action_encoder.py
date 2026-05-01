"""Action encoder — projects (one-hot) discrete actions into a continuous
embedding compatible with the AR predictor's conditioning channel.

Mirrors ``external/le-wm/module.py::Embedder`` but skips the per-step Conv1d
smoothing (which is only useful when stacking ``frameskip`` actions per step).
For our single-step replay we just need a small MLP.
"""

from __future__ import annotations

import torch
from torch import nn


class ActionEncoder(nn.Module):
    """One-hot action -> action embedding (matches predictor latent_dim)."""

    def __init__(self, action_dim: int = 56, emb_dim: int = 256, mlp_scale: int = 4) -> None:
        super().__init__()
        self.action_dim = action_dim
        self.emb_dim = emb_dim
        self.embed = nn.Sequential(
            nn.Linear(action_dim, mlp_scale * emb_dim),
            nn.SiLU(),
            nn.Linear(mlp_scale * emb_dim, emb_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: (B, T, action_dim) one-hot or continuous, returns (B, T, emb_dim)."""
        return self.embed(x)
