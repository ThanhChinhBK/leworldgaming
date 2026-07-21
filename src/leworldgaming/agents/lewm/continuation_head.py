"""Continuation head: ``z → BCE logit``. Predicts P(round still ongoing)."""

from __future__ import annotations

import torch
from torch import nn


class ContinuationHead(nn.Module):
    def __init__(self, latent_dim: int = 192, hidden_dim: int = 512, dropout: float = 0.0) -> None:
        super().__init__()
        # ``dropout`` defaults to 0.0 so existing checkpoints (whose saved
        # state_dict has no dropout params -- dropout is param-free anyway)
        # keep loading/behaving identically. Stage-B training sets this >0
        # (see heads.cont_dropout in configs/lewm_heads*.yaml) to fight the
        # continuation head's chronic overfitting: terminal (done) windows
        # are extremely scarce (~300 train / ~34 val out of >1M frames), so
        # without regularization the head quickly memorizes those few
        # examples and its held-out val loss blows up (0.55->4.7 observed
        # over 20k steps) even while train loss stays low. At inference,
        # ``LewmAgent`` puts every module in ``.eval()`` mode, so dropout is
        # always a no-op there regardless of this value.
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).squeeze(-1)
