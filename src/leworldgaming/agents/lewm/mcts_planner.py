"""Discrete MCTS/PUCT planner — a genuinely different search algorithm from
``planner.cem_shooting``, not a variant of it.

CEM (``planner.py``) is a *population-resampling* method: every iteration it
draws a flat batch of ``num_samples`` full-horizon trajectories, scores them
all, and refits a per-timestep categorical distribution toward the elites.
It has no notion of a search *tree* — nothing is shared or reused between
trajectories that agree on their first few actions, and simulation budget is
spent uniformly across the whole horizon regardless of which branches look
promising.

MuZero-style MCTS/PUCT (Schrittwieser et al. 2020, arXiv:1911.08265) is
structurally different: it builds an explicit tree of visited
``(latent, action)`` edges, and allocates each new simulation's compute to
whichever branch currently has the best ``Q(s,a) + U(s,a)`` PUCT score —
concentrating simulations on promising lines instead of resampling
everything uniformly. This module implements exactly that, reusing the same
frozen predictor/reward/value/continuation heads as CEM, with PUCT's
per-edge prior ``P(s,a)`` falling back to a flat (masked) uniform prior.

Design notes specific to this codebase:

- A "node" is a ``(z_hist, action_hist)`` pair (the same rolling
  context the AR ``Predictor`` needs — see ``planner._prepare_context``),
  not a single latent vector, since the predictor is autoregressive over a
  short history window.
- Edges lazily compute their reward (via ``reward_head(z, a_emb)``,
  exactly as ``planner._score_action_sequences`` does) and their child
  latent (one predictor step) the FIRST time that action is selected from a
  node — cheap because most edges from most nodes are never explored at a
  realistic simulation budget (tens, not thousands, per real-time decision).
- Leaf value bootstrap uses ``value_head`` (optionally continuation-head
  discounted), matching CEM's scoring convention exactly, so results are
  comparable across planners.
- The executed action is *sampled* (not argmaxed) from the root's final
  visit-count distribution, for the same "avoid deterministic lock-in"
  reason documented in ``cem_shooting``'s docstring point 3.

Wave/virtual-loss batching (2026-07-21): the first version of this module
ran one fully-sequential root-to-leaf simulation at a time (one Python-level
predictor/reward/value call per simulation) and measured ~9ms/simulation —
far too slow for the ~33-83ms/decision real-time budget at any useful
``num_simulations`` (24 sims cost ~208ms). This version instead runs
simulations in "waves" of ``sim_batch_size``: within a wave, each
simulation's root-to-leaf *selection* still happens sequentially in Python
(cheap — just tensor indexing, no NN calls), but a temporary **virtual
loss** is applied to every traversed edge immediately after selecting it
(standard AlphaZero/leaf-parallel MCTS technique, Silver et al. 2016) so
that later simulations in the same wave are pushed away from edges already
claimed this wave, without waiting for a real backup. All of the wave's
newly-reached leaves (both brand-new edges needing expansion, and
already-expanded nodes reached again at ``max_depth``) are then evaluated
in ONE batched forward pass each through ``action_encoder``/``predictor``/
``reward_head``/``value_head`` — turning
``num_simulations`` sequential tiny NN calls into
``num_simulations / sim_batch_size`` batched ones — before the virtual loss
is removed and the real backup is applied along every path in the wave.
Duplicate requests within a wave (two simulations reaching the same
not-yet-expanded edge, or the same already-expanded leaf, because virtual
loss didn't fully separate them) are de-duplicated before the batched call
so each edge/leaf is only evaluated once per wave.
"""

from __future__ import annotations

import math

import torch
from torch import nn

from leworldgaming.agents.lewm.planner import (
    _decode_pessimistic,
    _prepare_context,
    _repeat_action_blocks,
)


class _MCTSNode:
    """One node in the search tree: a ``(z_hist, action_hist)`` context plus
    per-action PUCT statistics. Children/edges are created lazily — an
    action's reward/child-latent/prior are only computed the first time
    that action is selected from this node.
    """

    __slots__ = (
        "z_hist", "action_hist", "prior", "N", "W", "reward", "discount",
        "children", "expanded", "valid_mask",
    )

    def __init__(self, z_hist: torch.Tensor, action_hist: torch.Tensor, num_actions: int, device: torch.device):
        self.z_hist = z_hist        # (history_size, D)
        self.action_hist = action_hist  # (history_size - 1,)
        self.prior: torch.Tensor | None = None       # (num_actions,) set on expand()
        self.N = torch.zeros(num_actions, device=device)
        self.W = torch.zeros(num_actions, device=device)
        self.reward = torch.zeros(num_actions, device=device)
        self.discount = torch.zeros(num_actions, device=device)
        self.children: dict[int, _MCTSNode] = {}
        self.expanded = False
        self.valid_mask: torch.Tensor | None = None  # (num_actions,) bool, None = all valid

    def q(self, a: int) -> float:
        n = self.N[a].item()
        return self.W[a].item() / n if n > 0 else 0.0


@torch.no_grad()
def mcts_search(
    z_context: torch.Tensor,
    predictor: nn.Module,
    pred_proj: nn.Module,
    action_encoder: nn.Module,
    probe: nn.Module,
    num_actions: int,
    num_simulations: int = 24,
    max_depth: int = 5,
    history_size: int = 3,
    temporal_stride: int = 1,
    past_actions: torch.Tensor | None = None,
    reward_head: nn.Module | nn.ModuleList | None = None,
    continuation_head: nn.Module | None = None,
    value_head: nn.Module | nn.ModuleList | None = None,
    reward_bins: torch.Tensor | None = None,
    value_bins: torch.Tensor | None = None,
    gamma: float = 0.997,
    valid_actions: torch.Tensor | None = None,
    uncertainty_penalty: float = 0.0,
    c_puct: float = 1.25,
    dirichlet_alpha: float = 0.3,
    dirichlet_frac: float = 0.25,
    temperature: float = 1.0,
    sim_batch_size: int = 16,
    virtual_loss: float = 1.0,
) -> tuple[int, torch.Tensor]:
    """MuZero-style discrete MCTS/PUCT over the frozen latent world model.

    Returns ``(action, root_visit_distribution)`` — mirrors
    ``cem_shooting``'s ``(action, dist)`` return shape so callers (e.g.
    ``LewmAgent.act``) can treat both planners uniformly, though unlike
    ``cem_shooting``'s per-timestep ``dist``, ``root_visit_distribution`` is
    a single ``(num_actions,)`` distribution over the very next action only
    (MCTS's tree does not commit to a fixed further plan the way CEM's
    warm-started ``dist`` does — every real decision restarts a fresh tree
    from the actually-observed state).

    ``c_puct``: exploration constant in the PUCT score
    ``Q(s,a) + c_puct * P(s,a) * sqrt(sum_b N(s,b)) / (1 + N(s,a))``.

    ``dirichlet_alpha``/``dirichlet_frac``: standard AlphaZero/MuZero root
    exploration noise — blends ``dirichlet_frac`` of Dirichlet(alpha) noise
    into the ROOT's prior only (not descendants), so repeated real-time
    calls don't always expand the same first action even from a peaked
    prior. ``dirichlet_frac=0`` disables it.

    ``temperature``: softens the final root visit-count distribution before
    sampling the executed action (``N(a)^(1/temperature)``, normalized).
    ``1.0`` is the standard proportional-to-visits choice; lower sharpens
    toward the most-visited action, higher flattens it (more exploration).

    ``sim_batch_size``: number of simulations selected (with virtual loss)
    before their leaves are expanded/evaluated in one batched NN call — see
    module docstring. Larger values amortize NN overhead better (fewer,
    bigger calls) but make within-wave selection slightly less "informed"
    (all sims in a wave act on the same pre-wave tree statistics, modulated
    only by virtual loss, not by any of the wave's own real backups yet).
    ``virtual_loss``: the pessimistic bias subtracted from ``W`` for every
    edge a simulation claims mid-wave, used only to diversify wave-mates'
    selections; removed before the real backup is applied.
    """
    device = z_context.device
    z_hist0, action_hist0 = _prepare_context(
        z_context, past_actions, 1, history_size, device
    )
    z_hist0 = z_hist0[0]
    action_hist0 = action_hist0[0]

    valid_mask: torch.Tensor | None = None
    if valid_actions is not None:
        valid_mask = torch.zeros(num_actions, dtype=torch.bool, device=device)
        valid_mask[valid_actions.to(device=device, dtype=torch.long)] = True

    def _priors_for_batch(z_last_batch: torch.Tensor) -> torch.Tensor:
        """Batched (flat, masked-uniform) PUCT-prior for K freshly-created
        nodes. ``z_last_batch`` is accepted (unused beyond shape) so this
        stays a drop-in slot for a learned prior later without touching
        call sites.
        """
        k = z_last_batch.shape[0]
        p = torch.ones(k, num_actions, device=device)
        if valid_mask is not None:
            p = p * valid_mask.float().unsqueeze(0)
        return p / p.sum(dim=-1, keepdim=True).clamp_min(1e-8)

    def _add_root_noise(node: _MCTSNode) -> None:
        if dirichlet_frac <= 0.0:
            return
        idx = valid_actions.to(device=device, dtype=torch.long) if valid_actions is not None else torch.arange(num_actions, device=device)
        noise = torch.distributions.Dirichlet(
            torch.full((idx.numel(),), dirichlet_alpha, device=device)
        ).sample()
        blended = node.prior.clone()
        blended[idx] = (1.0 - dirichlet_frac) * blended[idx] + dirichlet_frac * noise
        node.prior = blended / blended.sum().clamp_min(1e-8)

    def _select_action(node: _MCTSNode) -> int:
        total_n = node.N.sum().clamp_min(0.0)
        sqrt_total = math.sqrt(float(total_n) + 1e-8)
        q = torch.where(node.N > 0, node.W / node.N.clamp_min(1.0), torch.zeros_like(node.N))
        u = c_puct * node.prior * sqrt_total / (1.0 + node.N)
        score = q + u
        if node.valid_mask is not None:
            score = score.masked_fill(~node.valid_mask, float("-inf"))
        return int(torch.argmax(score).item())

    def _leaf_value_fallback_batch(z_last_batch: torch.Tensor) -> torch.Tensor:
        if value_head is not None and value_bins is not None:
            return _decode_pessimistic(value_head, value_bins, uncertainty_penalty, z_last_batch)
        if reward_head is None:
            return probe(z_last_batch)[:, 0]
        return torch.zeros(z_last_batch.shape[0], device=device)

    root = _MCTSNode(z_hist0, action_hist0, num_actions, device)
    root.prior = _priors_for_batch(root.z_hist[-1].unsqueeze(0))[0]
    root.valid_mask = valid_mask
    root.expanded = True
    _add_root_noise(root)

    remaining = max(1, num_simulations)
    wave_size = max(1, int(sim_batch_size))

    while remaining > 0:
        this_wave = min(wave_size, remaining)
        remaining -= this_wave

        sim_records: list[dict] = []
        # key -> (node, action); shared by every sim in this wave that
        # reaches that exact not-yet-expanded edge.
        pending_expand: dict[tuple[int, int], tuple[_MCTSNode, int]] = {}
        # id(node) -> node; already-expanded nodes reached again at
        # max_depth (no new edge needed, just a value estimate).
        pending_value: dict[int, _MCTSNode] = {}

        for _ in range(this_wave):
            node = root
            path: list[tuple[_MCTSNode, int]] = []
            depth = 0
            expand_key: tuple[int, int] | None = None
            existing_leaf: _MCTSNode | None = None
            while depth < max_depth:
                action = _select_action(node)
                path.append((node, action))
                # Virtual loss: bias this edge pessimistically so later
                # sims in the same wave spread out instead of repeating
                # the identical path (no real backup has happened yet).
                node.N[action] += 1
                node.W[action] += -virtual_loss
                if action not in node.children:
                    expand_key = (id(node), action)
                    pending_expand.setdefault(expand_key, (node, action))
                    break
                node = node.children[action]
                depth += 1
            else:
                existing_leaf = node
                pending_value[id(existing_leaf)] = existing_leaf

            sim_records.append(
                {"path": path, "expand_key": expand_key, "existing_leaf": existing_leaf}
            )

        value_of_expand_key: dict[tuple[int, int], float] = {}
        if pending_expand:
            edges = list(pending_expand.values())
            k = len(edges)
            z_hist_b = torch.stack([n.z_hist for n, _ in edges], dim=0)          # (K, hs, D)
            action_hist_b = torch.stack([n.action_hist for n, _ in edges], dim=0)  # (K, hs-1)
            action_col = torch.tensor([[a] for _, a in edges], device=device, dtype=torch.long)  # (K, 1)
            action_window = torch.cat([action_hist_b, action_col], dim=1)        # (K, hs)
            action_blocks = _repeat_action_blocks(action_window, num_actions, temporal_stride)
            a_hist_emb = action_encoder(action_blocks)                           # (K, hs, Demb)

            if reward_head is not None and reward_bins is not None:
                reward_b = _decode_pessimistic(
                    reward_head, reward_bins, uncertainty_penalty,
                    z_hist_b[:, -1], a_hist_emb[:, -1],
                )
            else:
                reward_b = torch.zeros(k, device=device)

            z_pred_seq = predictor(z_hist_b, a_hist_emb)
            z_next_b = pred_proj(z_pred_seq[:, -1])                              # (K, D)

            if continuation_head is not None:
                discount_b = gamma * torch.sigmoid(continuation_head(z_next_b)).squeeze(-1)
            else:
                discount_b = torch.full((k,), gamma, device=device)

            value_b = _leaf_value_fallback_batch(z_next_b)
            prior_b = _priors_for_batch(z_next_b)

            for i, (n, a) in enumerate(edges):
                n.reward[a] = reward_b[i]
                n.discount[a] = discount_b[i]
                new_z_hist = torch.cat([n.z_hist[1:], z_next_b[i : i + 1]], dim=0)
                new_action_hist = torch.cat([n.action_hist[1:], action_col[i]], dim=0)
                child = _MCTSNode(new_z_hist, new_action_hist, num_actions, device)
                child.prior = prior_b[i]
                child.valid_mask = valid_mask
                child.expanded = True
                n.children[a] = child
                value_of_expand_key[(id(n), a)] = float(value_b[i].item())

        value_of_existing_leaf: dict[int, float] = {}
        if pending_value:
            leaves = list(pending_value.values())
            z_last_b = torch.stack([n.z_hist[-1] for n in leaves], dim=0)
            value_b2 = _leaf_value_fallback_batch(z_last_b)
            for n, v in zip(leaves, value_b2, strict=True):
                value_of_existing_leaf[id(n)] = float(v.item())

        for rec in sim_records:
            path = rec["path"]
            if rec["expand_key"] is not None:
                value = value_of_expand_key[rec["expand_key"]]
            else:
                value = value_of_existing_leaf[id(rec["existing_leaf"])]

            for parent, action in reversed(path):
                # Undo the wave's temporary virtual loss on this edge, then
                # apply the real backup.
                parent.N[action] -= 1
                parent.W[action] += virtual_loss
                value = float(parent.reward[action].item()) + float(parent.discount[action].item()) * value
                parent.N[action] += 1
                parent.W[action] += value

    visit_dist = root.N.clone()
    if visit_dist.sum() <= 0:
        visit_dist = root.prior.clone()
    if temperature != 1.0 and visit_dist.sum() > 0:
        visit_dist = visit_dist.clamp_min(0.0).pow(1.0 / max(temperature, 1e-6))
    visit_dist = visit_dist / visit_dist.sum().clamp_min(1e-8)

    action0 = int(torch.multinomial(visit_dist, 1).item())
    return action0, visit_dist.detach()
