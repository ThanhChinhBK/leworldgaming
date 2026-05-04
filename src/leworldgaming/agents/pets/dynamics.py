"""Ensemble of probabilistic dynamics models for PETS.

Each ensemble member is an MLP predicting ``(mean_Δs, logvar_Δs)`` for the
next-state delta given the current ``(state, action_embedding)``. Inputs are
normalized via a running standard scaler fit on the first replay batch
(mirrors ``handful-of-trials/BNN.py``). Discrete actions are projected to
continuous embeddings before concatenation.

The forward pass returns one prediction per ensemble member; ``nll`` averages
the per-member loss. ``predict`` is the inference-side entry point used by
the CEM planner — for trajectory sampling (TS1) the planner picks one member
per particle and reads ``mean[e]`` / ``logvar[e]`` for it.
"""

from __future__ import annotations

import torch
from torch import nn

LOGVAR_INIT_MAX = 0.5
LOGVAR_INIT_MIN = -10.0


class TensorStandardScaler(nn.Module):
    """Per-feature running mean/std. Fit once on the first batch, then frozen.

    Stored as buffers so ``state_dict()`` round-trips them.
    """

    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.register_buffer("mean", torch.zeros(dim))
        self.register_buffer("std", torch.ones(dim))
        self.register_buffer("fitted", torch.zeros((), dtype=torch.bool))
        self.eps = eps

    @torch.no_grad()
    def fit(self, x: torch.Tensor) -> None:
        # x: (..., D)
        flat = x.reshape(-1, x.shape[-1])
        self.mean.copy_(flat.mean(dim=0))
        std = flat.std(dim=0)
        std = torch.where(std < self.eps, torch.ones_like(std), std)
        self.std.copy_(std)
        self.fitted.fill_(True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.mean) / self.std


class _ProbMLP(nn.Module):
    """Single ensemble member: MLP outputting ``(mean, logvar)`` of Δs."""

    def __init__(self, in_dim: int, out_dim: int, hidden: int, num_layers: int) -> None:
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, hidden))
            layers.append(nn.SiLU())
            prev = hidden
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, 2 * out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(x)
        out = self.head(h)
        d = out.shape[-1] // 2
        mean = out[..., :d]
        logvar = out[..., d:]
        return mean, logvar


class EnsembleDynamics(nn.Module):
    """Ensemble of probabilistic MLPs predicting ``Δs`` distributions."""

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden: int = 200,
        num_layers: int = 3,
        ensemble_size: int = 5,
        action_emb_dim: int = 16,
    ) -> None:
        super().__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.ensemble_size = ensemble_size
        self.action_emb = nn.Embedding(action_dim, action_emb_dim)
        in_dim = state_dim + action_emb_dim
        self.members = nn.ModuleList([
            _ProbMLP(in_dim, state_dim, hidden=hidden, num_layers=num_layers)
            for _ in range(ensemble_size)
        ])
        self.scaler = TensorStandardScaler(in_dim)
        # Bounded logvar — see Chua et al. 2018 §B.
        self.max_logvar = nn.Parameter(torch.full((state_dim,), LOGVAR_INIT_MAX))
        self.min_logvar = nn.Parameter(torch.full((state_dim,), LOGVAR_INIT_MIN))

    def _bound_logvar(self, logvar: torch.Tensor) -> torch.Tensor:
        # Smooth soft-bounded logvar (paper-standard formulation).
        logvar = self.max_logvar - torch.nn.functional.softplus(self.max_logvar - logvar)
        logvar = self.min_logvar + torch.nn.functional.softplus(logvar - self.min_logvar)
        return logvar

    def _embed(self, s: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        a_emb = self.action_emb(a.long())
        x = torch.cat([s, a_emb], dim=-1)
        return self.scaler(x) if self.scaler.fitted.item() else x

    def forward(self, s: torch.Tensor, a: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """``s: (B, D)``, ``a: (B,)`` → ``(mean, logvar)`` each ``(E, B, D)``."""
        x = self._embed(s, a)
        means: list[torch.Tensor] = []
        logvars: list[torch.Tensor] = []
        for member in self.members:
            mean, logvar = member(x)
            logvar = self._bound_logvar(logvar)
            means.append(mean)
            logvars.append(logvar)
        return torch.stack(means, dim=0), torch.stack(logvars, dim=0)

    def nll(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        target_delta: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Mean-of-ensemble Gaussian NLL on the target Δs."""
        mean, logvar = self.forward(s, a)  # (E, B, D)
        target = target_delta.unsqueeze(0).expand_as(mean)
        inv_var = torch.exp(-logvar)
        per_dim = (mean - target) ** 2 * inv_var + logvar
        loss = 0.5 * per_dim.mean()
        # Encourage the soft logvar bounds to stay tight (small, paper-standard penalty).
        bound_reg = 0.01 * (self.max_logvar.sum() - self.min_logvar.sum())
        total = loss + bound_reg
        with torch.no_grad():
            mse = ((mean - target) ** 2).mean().item()
        return total, {
            "nll": float(loss.item()),
            "delta_mse": float(mse),
            "max_logvar_mean": float(self.max_logvar.mean().item()),
            "min_logvar_mean": float(self.min_logvar.mean().item()),
        }

    @torch.no_grad()
    def predict(
        self,
        s: torch.Tensor,
        a: torch.Tensor,
        member_ids: torch.Tensor,
        sample: bool = True,
    ) -> torch.Tensor:
        """TS1-style next-state prediction.

        Args:
            s: ``(B, D)`` current states.
            a: ``(B,)`` integer actions.
            member_ids: ``(B,)`` ensemble-member assignment per particle.
            sample: if True, sample from N(μ, σ²); else return the mean.

        Returns ``(B, D)`` next-state predictions.
        """
        mean, logvar = self.forward(s, a)  # (E, B, D)
        b_idx = torch.arange(s.shape[0], device=s.device)
        m = mean[member_ids, b_idx]
        lv = logvar[member_ids, b_idx]
        if sample:
            std = torch.exp(0.5 * lv)
            delta = m + std * torch.randn_like(m)
        else:
            delta = m
        return s + delta
