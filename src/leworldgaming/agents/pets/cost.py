"""Analytic reward used by PETS' CEM scoring.

The fighting-game reward is HP differential change normalised by max HP, the
same formula ``recording_ai`` uses to log replay rewards. PETS recomputes it
on-the-fly from primitives so the planner doesn't need a learned reward head.

State indices ``PETS_OWN_HP_IDX`` and ``PETS_OPP_HP_IDX`` come from
``leworldgaming.env.state_vector`` — the layout the dynamics model trains on.
"""

from __future__ import annotations

import torch

from leworldgaming.env.state_vector import PETS_OPP_HP_IDX, PETS_OWN_HP_IDX


def analytic_reward(
    s_t: torch.Tensor,
    s_next: torch.Tensor,
    max_hp: float = 400.0,
) -> torch.Tensor:
    """``Δ(damage_dealt − damage_taken) / max_hp``. Shape ``(*B,)``.

    Damage dealt = ``opp_hp_t - opp_hp_{t+1}`` (positive when opp loses HP).
    Damage taken = ``own_hp_t - own_hp_{t+1}`` (positive when we lose HP).
    Result is unbounded but typically in ``[-0.05, 0.05]`` per frame.
    """
    own_t = s_t[..., PETS_OWN_HP_IDX]
    opp_t = s_t[..., PETS_OPP_HP_IDX]
    own_t1 = s_next[..., PETS_OWN_HP_IDX]
    opp_t1 = s_next[..., PETS_OPP_HP_IDX]
    damage_dealt = opp_t - opp_t1
    damage_taken = own_t - own_t1
    return (damage_dealt - damage_taken) / max(max_hp, 1.0)
