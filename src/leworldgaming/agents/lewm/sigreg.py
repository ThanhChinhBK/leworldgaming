"""SIGReg — anti-collapse regularizer for JEPA latents.

Forces the batch of embeddings z to roughly follow a unit Gaussian.
See gemini_research.md §6 and the reference in external/le-wm.

The placeholder below uses a coarse moment-matching surrogate so the demo
can compute a finite loss; replace with the proper SIGReg formulation
(characteristic-function based) when porting from external/le-wm.
"""

from __future__ import annotations

import torch


def sigreg_loss(z: torch.Tensor) -> torch.Tensor:
    mu = z.mean(dim=0)
    var = z.var(dim=0, unbiased=False)
    mean_term = mu.pow(2).mean()
    var_term = (var - 1.0).pow(2).mean()
    return mean_term + var_term
