"""SIGReg — Sketch Isotropic Gaussian Regularizer for JEPA latents.

Ports the Epps-Pulley characteristic-function regularizer from
``external/le-wm/module.py::SIGReg``. Compares the empirical distribution of
random 1-D projections of ``z`` against N(0, 1) on a grid of frequencies,
integrating with trapezoidal weights and a Gaussian window.

Why this works on small batches: each random projection collapses ``D`` dims
to 1, and ``num_proj`` projections (default 1024) are averaged per call —
so we don't need a per-dim variance estimate from a tiny batch the way the
old moment-matching placeholder did.

See gemini_research.md §6.
"""

from __future__ import annotations

import torch
from torch import nn


class SIGReg(nn.Module):
    """Sketch Isotropic Gaussian Regularizer (single-GPU).

    Args:
        knots: number of frequency grid points in [0, 3]. 17 matches reference.
        num_proj: number of random unit-norm projections drawn each forward.

    Forward accepts ``(B, D)`` or ``(T, B, D)``; in the latter the statistic
    is averaged over T as well as over projections.
    """

    def __init__(self, knots: int = 17, num_proj: int = 1024) -> None:
        super().__init__()
        self.num_proj = num_proj
        t = torch.linspace(0.0, 3.0, knots, dtype=torch.float32)
        dt = 3.0 / (knots - 1)
        weights = torch.full((knots,), 2.0 * dt, dtype=torch.float32)
        weights[0] = dt
        weights[-1] = dt
        window = torch.exp(-t.square() / 2.0)
        self.register_buffer("t", t)
        self.register_buffer("phi", window)
        self.register_buffer("weights", weights * window)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        z = z.float()
        if z.dim() == 2:
            z = z.unsqueeze(0)  # (1, B, D)
        d = z.size(-1)
        a = torch.randn(d, self.num_proj, device=z.device, dtype=z.dtype)
        a = a / a.norm(p=2, dim=0, keepdim=True)
        x_t = (z @ a).unsqueeze(-1) * self.t  # (T, B, num_proj, knots)
        err = (x_t.cos().mean(-3) - self.phi).square() + x_t.sin().mean(-3).square()
        statistic = (err @ self.weights) * z.size(-2)
        return statistic.mean()


_DEFAULT: SIGReg | None = None


def sigreg_loss(z: torch.Tensor, knots: int = 17, num_proj: int = 1024) -> torch.Tensor:
    """Functional wrapper that lazily caches a default :class:`SIGReg` module.

    Useful for short scripts that don't want to wire a module through. For
    actual training, instantiate :class:`SIGReg` once and call it directly so
    the buffers participate in ``state_dict`` / device moves cleanly.
    """
    global _DEFAULT
    if _DEFAULT is None or _DEFAULT.num_proj != num_proj or _DEFAULT.t.numel() != knots:
        _DEFAULT = SIGReg(knots=knots, num_proj=num_proj)
    if _DEFAULT.t.device != z.device:
        _DEFAULT = _DEFAULT.to(z.device)
    return _DEFAULT(z)
