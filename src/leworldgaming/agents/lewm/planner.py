"""Latent-space planner — random shooting / CEM over short horizons.

Plans entirely in latent space using `Predictor`, scoring trajectories with
the probe's HP-difference output. Designed to fit inside 16.67 ms on RTX 3080.
"""

from __future__ import annotations

import torch


@torch.no_grad()
def random_shooting(
    z0: torch.Tensor,
    predictor: torch.nn.Module,
    probe: torch.nn.Module,
    num_actions: int,
    horizon: int = 5,
    num_samples: int = 64,
) -> int:
    """Sample `num_samples` action sequences of length `horizon`, roll out in
    latent space, score the final state via `probe`, return the first action
    of the best sequence."""
    device = z0.device
    batch_z = z0.unsqueeze(0).expand(num_samples, -1).clone()
    actions = torch.randint(0, num_actions, (num_samples, horizon), device=device)
    one_hot = torch.nn.functional.one_hot(actions, num_classes=num_actions).float()
    for t in range(horizon):
        batch_z = predictor(batch_z, one_hot[:, t])
    scores = probe(batch_z)[:, 0]  # convention: probe[..., 0] = HP diff
    best = int(scores.argmax().item())
    return int(actions[best, 0].item())
