"""Projector head — anti-collapse MLP applied between encoder and loss.

Ported from ``external/le-wm/module.py::MLP``. Two roles:

* ``Projector`` after the encoder: applied to both the context and target
  branches with shared weights.
* ``Projector`` after the predictor (``pred_proj``): a separate instance that
  aligns predictions back into the projected embedding space.

Why BatchNorm: the BN inside the hidden layer is the actual anti-collapse
mechanism (BYOL/SimSiam): the gradient through batch-normalized features
depends on inter-sample relationships, so the projector cannot drive all
samples in a batch to the same value without paying a cost. LayerNorm
lacks this property and lets the encoder collapse on small batches.

We bump BN momentum to 0.5 (PyTorch default 0.1) so the running stats
converge fast — important for short runs where the model gets used for
single-frame inference (B=1) before BN running stats have many updates.
"""

from __future__ import annotations

import torch
from torch import nn


class Projector(nn.Module):
    """Two-layer MLP with optional normalization on the hidden activation.

    Input/output dim default to ``latent_dim``; hidden width 2048 matches the
    LeWM reference. ``norm="batch"`` (default) gives the BN-based
    anti-collapse; ``norm="layer"`` is a fallback that doesn't depend on
    batch size but is weaker; ``norm="none"`` is for unit tests.
    """

    def __init__(
        self,
        latent_dim: int = 256,
        hidden_dim: int = 2048,
        norm: str = "batch",
        bn_momentum: float = 0.5,
    ) -> None:
        super().__init__()
        if norm == "batch":
            mid: nn.Module = nn.BatchNorm1d(hidden_dim, momentum=bn_momentum)
        elif norm == "layer":
            mid = nn.LayerNorm(hidden_dim)
        elif norm == "none":
            mid = nn.Identity()
        else:
            raise ValueError(f"unknown norm: {norm!r}")
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            mid,
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z)
