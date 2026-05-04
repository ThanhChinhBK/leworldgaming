"""PETS agent — ensemble dynamics + discrete-action CEM planner.

Inference: ``act(obs)`` flattens the obs dict into the PETS state vector,
runs CEM under a 16.67 ms ``FrameBudget`` guard, and returns an integer
action.

Training is delegated to ``leworldgaming.training.train_pets.train`` — the
agent's ``learn`` method is a single forward+backward NLL step over a
transition batch.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from leworldgaming.agents.base import AgentBase
from leworldgaming.agents.pets.cem_planner import CEMPlannerDiscrete
from leworldgaming.agents.pets.cost import analytic_reward
from leworldgaming.agents.pets.dynamics import EnsembleDynamics
from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.state_vector import (
    PETS_STATE_DIM,
    obs_dict_to_pets_vector,
)
from leworldgaming.utils.timing import FrameBudget


class PETSAgent(AgentBase):
    def __init__(
        self,
        cfg: dict[str, Any] | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self._build_modules(cfg or {})

    def _build_modules(self, cfg: dict[str, Any]) -> None:
        self.cfg = dict(cfg)
        self.state_dim = int(cfg.get("state_dim", PETS_STATE_DIM))
        self.action_dim = int(cfg.get("action_dim", NUM_ACTIONS))
        self.max_hp = float(cfg.get("max_hp", 400.0))

        self.dynamics = EnsembleDynamics(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden=int(cfg.get("hidden", 200)),
            num_layers=int(cfg.get("num_layers", 3)),
            ensemble_size=int(cfg.get("ensemble_size", 5)),
            action_emb_dim=int(cfg.get("action_emb_dim", 16)),
        ).to(self.device)

        self.planner = CEMPlannerDiscrete(
            num_actions=self.action_dim,
            horizon=int(cfg.get("planner_horizon", 15)),
            num_candidates=int(cfg.get("planner_num_candidates", 200)),
            num_elites=int(cfg.get("planner_num_elites", 20)),
            num_iters=int(cfg.get("planner_num_iters", 4)),
            gamma=float(cfg.get("planner_gamma", 0.99)),
            sample_dynamics=bool(cfg.get("planner_sample_dynamics", True)),
            device=self.device,
            max_hp=self.max_hp,
        )

    def act(self, obs: dict[str, Any]) -> int:
        """Plan one frame.

        ``obs`` is the primitives dict from
        ``leworldgaming.env.frame_to_obs_dict``. Falls back to consuming a
        pre-flattened ``"state"`` tensor if present (used by smoke tests).
        """
        if "state" in obs:
            s_np = np.asarray(obs["state"], dtype=np.float32)
        else:
            s_np = obs_dict_to_pets_vector(obs)
        s = torch.from_numpy(s_np).to(self.device, dtype=torch.float32)
        self.dynamics.eval()
        budget = obs.get("_frame_budget")
        if budget is None:
            budget = FrameBudget()
        with budget:
            action = self.planner.plan(s, self.dynamics)
        return int(action)

    def learn(self, batch: dict[str, np.ndarray | torch.Tensor]) -> dict[str, float]:
        """One ensemble NLL step on a transition batch.

        Expects keys ``s, a, s_next`` (others ignored). Inputs may be numpy
        arrays or torch tensors.
        """
        s = self._to_tensor(batch["s"], dtype=torch.float32)
        a = self._to_tensor(batch["a"], dtype=torch.long)
        s_next = self._to_tensor(batch["s_next"], dtype=torch.float32)
        target_delta = s_next - s

        if not self.dynamics.scaler.fitted.item():
            with torch.no_grad():
                a_emb = self.dynamics.action_emb(a)
                self.dynamics.scaler.fit(torch.cat([s, a_emb], dim=-1))

        self.dynamics.train()
        loss, metrics = self.dynamics.nll(s, a, target_delta)
        return {"loss": loss, **metrics}  # caller does .backward() / .step()

    def _to_tensor(self, x: Any, dtype: torch.dtype) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x.to(self.device, dtype=dtype)
        return torch.as_tensor(x, device=self.device, dtype=dtype)

    def save(self, path: str) -> None:
        torch.save(
            {
                "dynamics": self.dynamics.state_dict(),
                "config": self.cfg,
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt.get("config")
        if cfg:
            self._build_modules(cfg)
        self.dynamics.load_state_dict(ckpt["dynamics"])
