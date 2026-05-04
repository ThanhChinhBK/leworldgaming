"""Discrete-action CEM planner over an ensemble dynamics model.

Maintains per-timestep ``Categorical(num_actions)`` logits. Each iteration:
  1. Sample ``num_candidates`` action sequences ``(N, H)`` from the
     per-step categoricals.
  2. Roll forward through the ensemble (TS1: each particle picks one
     ensemble member at random for the whole rollout).
  3. Score by ``Σ γ^h r_h`` with ``r_h = analytic_reward(s_h, s_{h+1})``.
  4. Take top-K elites; refit each timestep's logits as
     ``log(softmax_count(elites[:, h]) + ε)``.

After ``num_iters`` iterations, return ``argmax(pi[0])`` as the action to
play at step 0. Wrapping in ``torch.no_grad()`` keeps the compute graph from
accumulating across the planning horizon (see ``docs/gemini_research.md``
§3 on VRAM budgeting).
"""

from __future__ import annotations

from typing import Callable

import torch

from leworldgaming.agents.pets.cost import analytic_reward
from leworldgaming.agents.pets.dynamics import EnsembleDynamics


class CEMPlannerDiscrete:
    def __init__(
        self,
        num_actions: int,
        horizon: int = 15,
        num_candidates: int = 200,
        num_elites: int = 20,
        num_iters: int = 4,
        gamma: float = 0.99,
        sample_dynamics: bool = True,
        device: str | torch.device = "cpu",
        max_hp: float = 400.0,
    ) -> None:
        self.num_actions = int(num_actions)
        self.horizon = int(horizon)
        self.num_candidates = int(num_candidates)
        self.num_elites = max(1, min(int(num_elites), int(num_candidates)))
        self.num_iters = int(num_iters)
        self.gamma = float(gamma)
        self.sample_dynamics = bool(sample_dynamics)
        self.device = torch.device(device)
        self.max_hp = float(max_hp)
        self.eps = 1e-6

    @torch.no_grad()
    def plan(
        self,
        state: torch.Tensor,
        dynamics: EnsembleDynamics,
        reward_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor] | None = None,
    ) -> int:
        """Return the integer action to take at step 0."""
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if reward_fn is None:
            reward_fn = lambda s, s_next: analytic_reward(s, s_next, max_hp=self.max_hp)

        # Per-timestep categorical logits, initialized uniform.
        logits = torch.zeros(self.horizon, self.num_actions, device=self.device)

        for _ in range(self.num_iters):
            # 1. Sample N action sequences. shape: (H, N) → transpose to (N, H).
            actions_per_step = torch.distributions.Categorical(logits=logits).sample(
                (self.num_candidates,)
            )  # (N, H)

            # 2. Roll forward. Each particle locks to one ensemble member.
            members = torch.randint(
                0, dynamics.ensemble_size, (self.num_candidates,), device=self.device
            )
            s = state.expand(self.num_candidates, -1).clone()  # (N, D)
            returns = torch.zeros(self.num_candidates, device=self.device)
            discount = 1.0
            for h in range(self.horizon):
                a_h = actions_per_step[:, h]
                s_next = dynamics.predict(s, a_h, members, sample=self.sample_dynamics)
                r = reward_fn(s, s_next)
                returns = returns + discount * r
                discount *= self.gamma
                s = s_next

            # 3. Top-K elites by total return.
            _, elite_idx = torch.topk(returns, self.num_elites)
            elites = actions_per_step[elite_idx]  # (K, H)

            # 4. Refit per-timestep logits as log(count_softmax + ε).
            new_logits = torch.zeros_like(logits)
            for h in range(self.horizon):
                counts = torch.bincount(elites[:, h], minlength=self.num_actions).float()
                probs = counts / max(self.num_elites, 1)
                new_logits[h] = torch.log(probs + self.eps)
            logits = new_logits

        return int(torch.argmax(logits[0]).item())
