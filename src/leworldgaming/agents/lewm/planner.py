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

from leworldgaming.agents.lewm.twohot import twohot_decode


def _repeat_action_blocks(
    actions: torch.Tensor,
    num_actions: int,
    temporal_stride: int,
) -> torch.Tensor:
    one_hot = F.one_hot(actions, num_classes=num_actions).float()
    return (
        one_hot.unsqueeze(-2)
        .expand(*one_hot.shape[:-1], temporal_stride, num_actions)
        .reshape(*one_hot.shape[:-1], temporal_stride * num_actions)
    )


@torch.no_grad()
def random_shooting(
    z_context: torch.Tensor,
    predictor: nn.Module,
    pred_proj: nn.Module,
    action_encoder: nn.Module,
    probe: nn.Module,
    num_actions: int,
    horizon: int = 5,
    num_samples: int = 64,
    history_size: int = 3,
    temporal_stride: int = 1,
    past_actions: torch.Tensor | None = None,
    reward_head: nn.Module | None = None,
    continuation_head: nn.Module | None = None,
    value_head: nn.Module | None = None,
    reward_bins: torch.Tensor | None = None,
    value_bins: torch.Tensor | None = None,
    gamma: float = 0.997,
) -> int:
    """Sample ``num_samples`` action sequences of length ``horizon``, roll out
    in latent space, score the final state via ``probe``, return the first
    action of the best sequence.

    At inference we don't have a real history (just one frame's ``z0``), so we
    seed the history by repeating ``z0`` ``history_size`` times. The action
    history is initialized to the no-op (action 0) one-hot vectors and
    populated by the sampled actions as we roll forward.
    """
    device = z_context.device
    s = num_samples
    hs = history_size

    if z_context.ndim == 1:
        z_context = z_context.unsqueeze(0)
    if z_context.shape[0] > hs:
        z_context = z_context[-hs:]
    if z_context.shape[0] < hs:
        pad = z_context[:1].expand(hs - z_context.shape[0], -1)
        z_context = torch.cat([pad, z_context], dim=0)
    z_hist = z_context.unsqueeze(0).expand(s, -1, -1).contiguous()

    actions = torch.randint(0, num_actions, (s, horizon), device=device)  # (S, H)
    if hs == 1:
        past_actions = torch.empty(0, dtype=torch.long, device=device)
    elif past_actions is None:
        past_actions = torch.zeros(hs - 1, dtype=torch.long, device=device)
    else:
        past_actions = past_actions.to(device=device, dtype=torch.long)[-(hs - 1) :]
        if past_actions.numel() < hs - 1:
            pad = torch.zeros(
                hs - 1 - past_actions.numel(), dtype=torch.long, device=device
            )
            past_actions = torch.cat([pad, past_actions])
    action_hist = past_actions.unsqueeze(0).expand(s, -1).contiguous()
    scores = torch.zeros(s, device=device)
    discount = torch.ones(s, device=device)

    for t in range(horizon):
        action_window = torch.cat([action_hist, actions[:, t : t + 1]], dim=1)
        action_blocks = _repeat_action_blocks(
            action_window, num_actions, temporal_stride
        )
        a_hist_emb = action_encoder(action_blocks)
        if reward_head is not None and reward_bins is not None:
            reward_logits = reward_head(z_hist[:, -1], a_hist_emb[:, -1])
            scores.add_(discount * twohot_decode(reward_logits, reward_bins))
        z_pred_seq = predictor(z_hist, a_hist_emb)  # (S, HS, D)
        z_next = pred_proj(z_pred_seq[:, -1])  # (S, D)
        if continuation_head is not None:
            discount.mul_(
                gamma * torch.sigmoid(continuation_head(z_next))
            )
        else:
            discount.mul_(gamma)
        z_hist = torch.cat([z_hist[:, 1:], z_next.unsqueeze(1)], dim=1)
        action_hist = action_window[:, 1:]

    final_z = z_hist[:, -1]  # (S, D)
    if value_head is not None and value_bins is not None:
        scores.add_(discount * twohot_decode(value_head(final_z), value_bins))
    elif reward_head is None:
        scores = probe(final_z)[:, 0]
    best = int(scores.argmax().item())
    return int(actions[best, 0].item())
