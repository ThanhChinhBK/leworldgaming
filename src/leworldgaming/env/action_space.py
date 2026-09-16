"""Stable model action width and shared inference-time commandability.

Models retain all 56 output/embedding rows for checkpoint compatibility.
Inference excludes observation-only enum states, retaining NEUTRAL as a no-op.
This corrects historical unrestricted PETS/Dreamer scores; explicitly configure
``restrict_to_playable_actions=False`` only to reproduce that legacy behavior.
The mask is static: it does not impose energy, groundedness, or control rules.
"""

from __future__ import annotations

from functools import cache
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

NUM_ACTIONS = 56


@cache
def commandable_action_ids(num_actions: int = NUM_ACTIONS) -> tuple[int, ...] | None:
    """Return NEUTRAL plus playable enum IDs, or None for synthetic action spaces."""
    if num_actions != NUM_ACTIONS:
        return None
    from pyftg.models.enums.action import Action

    from leworldgaming.env.policies import PLAYABLE_ACTIONS

    return tuple(sorted({Action.NEUTRAL.to_int(), *(a.to_int() for a in PLAYABLE_ACTIONS)}))


def mask_action_logits(
    logits: torch.Tensor, restrict_to_playable_actions: bool = True,
) -> torch.Tensor:
    """Mask before categorical normalization/argmax without changing model shapes."""
    if not restrict_to_playable_actions:
        return logits
    valid = commandable_action_ids(logits.shape[-1])
    if valid is None:
        return logits
    import torch

    mask = torch.zeros(logits.shape[-1], device=logits.device, dtype=torch.bool)
    mask[list(valid)] = True
    # Dreamer's straight-through mode subtracts detached logits; -inf would
    # produce NaNs there. The finite minimum still yields zero categorical mass.
    minimum = torch.finfo(logits.dtype).min
    centered = logits - logits[..., list(valid)].amax(dim=-1, keepdim=True)
    return centered.clamp_min(minimum).masked_fill(~mask, minimum)
