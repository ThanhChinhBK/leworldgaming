"""Twohot encoding/decoding with symexp-spaced bins (DreamerV3 / TD-MPC2).

For unbounded reward and value targets the network outputs a categorical
distribution over symlog-spaced bins; the loss is cross-entropy against a
two-hot encoding of the (symlog'd) target. Decoding takes the expectation
in symlog space and inverts via symexp. This is more stable than scalar
MSE when targets span several orders of magnitude.

References:
* DreamerV3, Hafner et al. 2023 — symlog two-hot.
* TD-MPC2, Hansen et al. 2024 — same trick, called "discrete regression".
"""

from __future__ import annotations

import torch
from torch.nn import functional as F


def symlog(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * (torch.expm1(x.abs()))


def make_bins(num_bins: int, low: float, high: float, device: torch.device | str) -> torch.Tensor:
    """``num_bins`` evenly-spaced bin centers in symlog space between symlog(low) and symlog(high)."""
    lo = symlog(torch.tensor(float(low), device=device))
    hi = symlog(torch.tensor(float(high), device=device))
    return torch.linspace(lo.item(), hi.item(), num_bins, device=device)


def twohot_encode(x: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Encode scalar targets ``x`` as twohot probabilities over ``bins``.

    ``x`` shape ``(*,)``; ``bins`` shape ``(K,)`` (sorted ascending in symlog
    space). Returns ``(*, K)``. Targets are first symlog-transformed, then
    placed between two adjacent bin centers with linear interpolation.
    """
    x_sl = symlog(x)
    x_sl = x_sl.clamp(min=bins[0].item(), max=bins[-1].item())
    # Find right-bin index k such that bins[k-1] <= x_sl <= bins[k].
    idx = torch.bucketize(x_sl, bins)
    idx = idx.clamp(min=1, max=bins.numel() - 1)
    lo = bins[idx - 1]
    hi = bins[idx]
    width = (hi - lo).clamp(min=1e-8)
    w_hi = (x_sl - lo) / width
    w_lo = 1.0 - w_hi

    out = torch.zeros(*x.shape, bins.numel(), device=x.device, dtype=torch.float32)
    out.scatter_add_(-1, (idx - 1).unsqueeze(-1), w_lo.unsqueeze(-1).float())
    out.scatter_add_(-1, idx.unsqueeze(-1), w_hi.unsqueeze(-1).float())
    return out


def twohot_decode(logits: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Inverse of :func:`twohot_encode`. Returns scalar in original (non-symlog) space."""
    probs = F.softmax(logits, dim=-1)
    expected_sl = (probs * bins).sum(dim=-1)
    return symexp(expected_sl)


def twohot_ce_loss(logits: torch.Tensor, target: torch.Tensor, bins: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of twohot(target) against softmax(logits). Mean over leading dims."""
    log_probs = F.log_softmax(logits, dim=-1)
    target_dist = twohot_encode(target, bins)
    return -(target_dist * log_probs).sum(dim=-1).mean()
