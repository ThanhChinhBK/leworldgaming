"""Policy-prior head: ``z -> logits over num_actions``.

Behavior-cloned from the recorded dataset's executed actions (whatever mix
of scripted/MCTS/etc. policies collected the replay data). Purpose is NOT to
imitate a specific opponent or to be used stand-alone as a policy -- it is a
**CEM warm-start prior** (TD-MPC2 / Sampled-MuZero style): instead of
initializing the per-timestep categorical `dist` in `planner.cem_shooting`
to uniform-over-valid-actions (current behavior, see `cem_shooting`'s
`init_dist=None` branch), seed it from this head's softmax so the very
first CEM iteration already samples action sequences concentrated on
"generally plausible" actions (dashes, guards, pokes) instead of wasting
elite-refinement iterations climbing out of a uniform prior over the full
~40-56-way discrete action space every single decision.

Cheap and safe by construction: it only ever affects the *initial* sampling
distribution fed into CEM's iterative elite-refinement loop -- CEM still
scores every sampled trajectory with the exact same (frozen) predictor +
reward/value heads and refits based on real scores, so a bad/biased prior
can only cost a couple of extra elite-refinement iterations to correct, not
silently override the planner's judgment the way the online-opponent-model
first-action overlay did (that fully replaced dist[0]; this only seeds
dist's initial value, which the CEM loop then updates every iteration from
real scores).
"""

from __future__ import annotations

import torch
from torch import nn


class PolicyHead(nn.Module):
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
