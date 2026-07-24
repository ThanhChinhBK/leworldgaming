"""Opponent-action head: ``z -> logits over num_actions``.

Behavior-cloned from the *real* opponent action recorded in fresh
Dreamer-opponent data (``obs/opp/action``, JVM-populated ``char.action`` —
see ``docs/lewm_opp_action_head_2026-07-23.md``), as opposed to
``policy_head.PolicyHead`` (which clones *our own* recorded action from the
old mixed-policy corpus as a CEM warm-start prior).

Purpose: predict what the opponent is *about to do* from our own encoded
latent ``z`` (a frozen-encoder embedding of *our* observation, which
implicitly contains the opponent's currently-visible state/position/
attack-frames -- but not their true internal decision). This is intended to
feed ``online_opponent_model``-style overlays or a joint-conditioned CEM
scoring pass with *real* opponent-behavior data instead of the online
logistic threat proxy or hand-picked geometric features.

Kept deliberately tiny and separate from ``PolicyHead`` (different
supervision target, different intended consumer) rather than parameterizing
one class for both, to keep each concern's checkpoint key and docstring
unambiguous.
"""

from __future__ import annotations

import torch
from torch import nn


class OppActionHead(nn.Module):
    def __init__(self, latent_dim: int = 192, hidden_dim: int = 256, num_actions: int = 56) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_actions),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """``z``: (..., latent_dim) -> (..., num_actions) raw logits."""
        return self.net(z)
