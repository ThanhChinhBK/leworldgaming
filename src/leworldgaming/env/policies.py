"""Action-selection policies used by data collectors.

A policy is a callable taking (frame_data, player_number) and returning a
pyftg `Action` enum value. Used by `RecordingAI` to drive scripted opponents
during data collection (gemini_research.md §7.2).
"""

from __future__ import annotations

import random
from collections.abc import Callable
from typing import Protocol

from pyftg.models.enums.action import Action

# Subset of Action that maps to actual playable inputs (CommandCenter knows them).
# Excludes pure-state actions like NEUTRAL/STAND/AIR/DOWN/RECOV which are observed but not commanded.
PLAYABLE_ACTIONS: list[Action] = [
    Action.FORWARD_WALK,
    Action.DASH,
    Action.BACK_STEP,
    Action.CROUCH,
    Action.JUMP,
    Action.FOR_JUMP,
    Action.BACK_JUMP,
    Action.STAND_GUARD,
    Action.CROUCH_GUARD,
    Action.AIR_GUARD,
    Action.THROW_A,
    Action.THROW_B,
    Action.STAND_A,
    Action.STAND_B,
    Action.CROUCH_A,
    Action.CROUCH_B,
    Action.AIR_A,
    Action.AIR_B,
    Action.AIR_DA,
    Action.AIR_DB,
    Action.STAND_FA,
    Action.STAND_FB,
    Action.CROUCH_FA,
    Action.CROUCH_FB,
    Action.AIR_FA,
    Action.AIR_FB,
    Action.AIR_UA,
    Action.AIR_UB,
    Action.STAND_D_DF_FA,
    Action.STAND_D_DF_FB,
    Action.STAND_F_D_DFA,
    Action.STAND_F_D_DFB,
    Action.STAND_D_DB_BA,
    Action.STAND_D_DB_BB,
    Action.AIR_D_DF_FA,
    Action.AIR_D_DF_FB,
    Action.AIR_F_D_DFA,
    Action.AIR_F_D_DFB,
    Action.AIR_D_DB_BA,
    Action.AIR_D_DB_BB,
    Action.STAND_D_DF_FC,
]


class Policy(Protocol):
    def __call__(self, frame_data: object, player_number: bool) -> Action: ...


class RandomPolicy:
    """Uniform random over `PLAYABLE_ACTIONS`. Picks a fresh action when control resumes.

    `sticky_frames` keeps the same action queued for that many frames to avoid
    spamming the CommandCenter input buffer (most attacks take 5-15 frames anyway).
    """

    def __init__(self, sticky_frames: int = 8, seed: int | None = None) -> None:
        self._sticky = sticky_frames
        self._counter = 0
        self._current: Action = Action.NEUTRAL
        self._rng = random.Random(seed)

    def __call__(self, frame_data: object, player_number: bool) -> Action:
        if self._counter <= 0:
            self._current = self._rng.choice(PLAYABLE_ACTIONS)
            self._counter = self._sticky
        self._counter -= 1
        return self._current


class NoOpPolicy:
    """Always returns NEUTRAL."""

    def __call__(self, frame_data: object, player_number: bool) -> Action:
        return Action.NEUTRAL


# ── Biased policies for training-data diversity ──────────────────────────

_ATTACK_ACTIONS: list[Action] = [
    a for a in PLAYABLE_ACTIONS if a.name.endswith(("_A", "_B", "_FA", "_FB",
    "_DA", "_DB", "_UA", "_UB", "_FC", "_DFA", "_DFB", "_DFA", "_DFB",
    "_BA", "_BB"))
]
_GUARD_MOVE_ACTIONS: list[Action] = [
    a for a in PLAYABLE_ACTIONS if a not in _ATTACK_ACTIONS
]


class AggressivePolicy:
    """80% attacks, 20% movement/guard."""

    def __init__(self, sticky_frames: int = 6, seed: int | None = None) -> None:
        self._sticky = sticky_frames
        self._counter = 0
        self._current: Action = Action.NEUTRAL
        self._rng = random.Random(seed)

    def __call__(self, frame_data: object, player_number: bool) -> Action:
        if self._counter <= 0:
            if self._rng.random() < 0.8:
                self._current = self._rng.choice(_ATTACK_ACTIONS)
            else:
                self._current = self._rng.choice(_GUARD_MOVE_ACTIONS)
            self._counter = self._sticky
        self._counter -= 1
        return self._current


class DefensivePolicy:
    """70% guard/movement, 30% attacks — plays cautiously."""

    def __init__(self, sticky_frames: int = 10, seed: int | None = None) -> None:
        self._sticky = sticky_frames
        self._counter = 0
        self._current: Action = Action.NEUTRAL
        self._rng = random.Random(seed)

    def __call__(self, frame_data: object, player_number: bool) -> Action:
        if self._counter <= 0:
            if self._rng.random() < 0.7:
                self._current = self._rng.choice(_GUARD_MOVE_ACTIONS)
            else:
                self._current = self._rng.choice(_ATTACK_ACTIONS)
            self._counter = self._sticky
        self._counter -= 1
        return self._current


class MixedPolicy:
    """Switches strategy every game: cycles through random → aggressive → defensive.

    Produces the broadest behavioral coverage from a single policy flag.
    """

    def __init__(self, seed: int | None = None, **kwargs: object) -> None:
        s = seed or 0
        self._sub_policies: list[Policy] = [
            RandomPolicy(seed=s),
            AggressivePolicy(seed=s + 100),
            DefensivePolicy(seed=s + 200),
        ]
        self._labels = ["random", "aggressive", "defensive"]
        self._game_count = 0
        self._active: Policy = self._sub_policies[0]

    def on_game_start(self) -> None:
        idx = self._game_count % len(self._sub_policies)
        self._active = self._sub_policies[idx]
        self._game_count += 1

    def __call__(self, frame_data: object, player_number: bool) -> Action:
        return self._active(frame_data, player_number)


def make_policy(name: str, **kwargs: object) -> Policy:
    name = name.lower()
    if name == "random":
        return RandomPolicy(**kwargs)  # type: ignore[arg-type]
    if name in ("noop", "neutral"):
        return NoOpPolicy()
    if name == "aggressive":
        return AggressivePolicy(**kwargs)  # type: ignore[arg-type]
    if name == "defensive":
        return DefensivePolicy(**kwargs)  # type: ignore[arg-type]
    if name == "mixed":
        return MixedPolicy(**kwargs)  # type: ignore[arg-type]
    raise ValueError(
        f"Unknown policy: {name!r}. "
        "Choose from: random, noop, aggressive, defensive, mixed"
    )


__all__ = [
    "Policy", "RandomPolicy", "NoOpPolicy", "AggressivePolicy",
    "DefensivePolicy", "MixedPolicy", "make_policy", "PLAYABLE_ACTIONS",
]


# Optional helper: hook into a callable for the record_callback signature
RecordCallback = Callable[[object, Action], None]
