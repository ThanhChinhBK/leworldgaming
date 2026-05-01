"""Latent-space planner — random shooting / CEM over short horizons.

Plans entirely in latent space using the AR ``Predictor``, scoring trajectories
with the ``probe``'s HP-difference output. The predictor is autoregressive
over a short history window — at each step we feed the last ``history_size``
embeddings + actions and take the last-position prediction as the next state.
``pred_proj`` is applied to predictor outputs so they live in the same
(post-projector) embedding space the predictor was trained on.

Designed to fit inside 16.67 ms on RTX 3080 with ``num_samples=64``,
``horizon=5``, ``history_size=3``.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


@torch.no_grad()
def random_shooting(
    z0: torch.Tensor,
    predictor: nn.Module,
    pred_proj: nn.Module,
    action_encoder: nn.Module,
    probe: nn.Module,
    num_actions: int,
    horizon: int = 5,
    num_samples: int = 64,
    history_size: int = 3,
) -> int:
    """Sample ``num_samples`` action sequences of length ``horizon``, roll out
    in latent space, score the final state via ``probe``, return the first
    action of the best sequence.

    At inference we don't have a real history (just one frame's ``z0``), so we
    seed the history by repeating ``z0`` ``history_size`` times. The action
    history is initialized to the no-op (action 0) one-hot vectors and
    populated by the sampled actions as we roll forward.
    """
    device = z0.device
    s = num_samples
    hs = history_size

    z_hist = z0.unsqueeze(0).unsqueeze(0).expand(s, hs, -1).contiguous()  # (S, HS, D)

    actions = torch.randint(0, num_actions, (s, horizon), device=device)  # (S, H)
    a_oh_seq = F.one_hot(actions, num_classes=num_actions).float()  # (S, H, A)

    a_hist_oh = torch.zeros(s, hs, num_actions, device=device)  # (S, HS, A)
    a_hist_oh[:, -1, 0] = 1.0  # placeholder no-op for the most recent slot

    for t in range(horizon):
        a_t_oh = a_oh_seq[:, t : t + 1]  # (S, 1, A)
        a_hist_oh = torch.cat([a_hist_oh[:, 1:], a_t_oh], dim=1)  # slide window

        a_hist_emb = action_encoder(a_hist_oh)  # (S, HS, D)
        z_pred_seq = predictor(z_hist, a_hist_emb)  # (S, HS, D)
        z_next = pred_proj(z_pred_seq[:, -1])  # (S, D)
        z_hist = torch.cat([z_hist[:, 1:], z_next.unsqueeze(1)], dim=1)

    final_z = z_hist[:, -1]  # (S, D)
    scores = probe(final_z)[:, 0]  # convention: probe[..., 0] = HP diff
    best = int(scores.argmax().item())
    return int(actions[best, 0].item())
