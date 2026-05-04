"""Hand-engineered state representation extracted from pyftg `FrameData`.

Two complementary forms:

- ``frame_to_obs_dict(...)`` — the **canonical** form. A nested dict of named
  primitives in raw physical units (no clipping, no normalization). This is
  what gets written to the replay buffer; per-method dataloader views in
  ``leworldgaming.data.views`` materialize whatever shape each trainer needs.

- ``obs_dict_to_legacy_vector(...)`` and ``frame_to_state_vector(...)`` —
  the **legacy** flat 52-dim float vector kept for back-compat. Used as the
  LeWM linear-probe target and by older smoke tests. Not stored on disk
  anymore.

# Schema (legacy 52-feature flat layout)

Per character × 2 (own first, then opponent) = 44 features:
    0   hp / max_hp                       in [0, 1]
    1   energy / max_energy               in [0, 1]
    2   x / STAGE_W                       in [0, 1]
    3   y / STAGE_H                       in [0, 1]
    4   speed_x / SPEED_X_NORM            in [-1, 1] (clamped)
    5   speed_y / SPEED_Y_NORM            in [-1, 1] (clamped)
    6:10  state one-hot                   STAND, CROUCH, AIR, DOWN
    10  front                             ±1 (right=+1, left=-1)
    11  control                           {0, 1}
    12  remaining_frame / 60              in [0, 1] (clamped)
    13  hit_confirm                       {0, 1}
    14  attack.is_live                    {0, 1}
    15  attack.start_up / 30              in [0, 1] (clamped)
    16  attack.active / 30                in [0, 1] (clamped)
    17  attack.hit_damage / 30            in [0, 1] (clamped)
    18:22 attack.attack_type one-hot      HIGH, MIDDLE, LOW, THROW

Global / relative (8 features):
    44  dx, 45 dy, 46 distance, 47 hp_diff, 48 round_progress,
    49  frame_progress, 50 any_proj_self, 51 any_proj_opp

Total: 52 features.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# Stage dimensions (FightingICE default: 960x640 play area).
STAGE_W = 960.0
STAGE_H = 640.0
DIAG = float(np.sqrt(STAGE_W**2 + STAGE_H**2))

# Speed clamps — empirical bounds; outliers get clipped (legacy form only).
SPEED_X_NORM = 15.0
SPEED_Y_NORM = 25.0

# Round structure.
MAX_FRAMES_PER_ROUND = 3600  # 60 s at 60 FPS
MAX_ROUNDS = 3

# Frame counts for normalization (legacy form only).
REMAINING_FRAME_NORM = 60.0
ATTACK_FRAME_NORM = 30.0
HIT_DAMAGE_NORM = 30.0

# Schema dimensions (legacy form).
PER_CHAR_DIM = 22
GLOBAL_DIM = 8
STATE_VECTOR_DIM = PER_CHAR_DIM * 2 + GLOBAL_DIM  # 52

# Continuous primitives that PETS / TD-MPC2 / DC-MPC use as flat input.
# Order is the source of truth for `view_pets` and the analytic cost function.
PETS_PRIMITIVE_KEYS: tuple[str, ...] = (
    "hp", "energy", "x", "y", "speed_x", "speed_y",
    "remaining_frame", "atk_is_live",
    "atk_start_up", "atk_active", "atk_hit_damage",
)
PETS_GLOBAL_KEYS: tuple[str, ...] = (
    "current_round", "current_frame", "proj_self", "proj_opp",
)
# (own + opp) × len(PETS_PRIMITIVE_KEYS) + len(PETS_GLOBAL_KEYS)
PETS_STATE_DIM = 2 * len(PETS_PRIMITIVE_KEYS) + len(PETS_GLOBAL_KEYS)


def _state_to_int(state) -> int:
    """Map pyftg State enum to {0=STAND, 1=CROUCH, 2=AIR, 3=DOWN}."""
    name = state.name if hasattr(state, "name") else str(state)
    return {"STAND": 0, "CROUCH": 1, "AIR": 2, "DOWN": 3}.get(name, 0)


def _empty_char_dict() -> dict[str, Any]:
    """Neutral defaults for a missing character — STAND, facing right, zeros."""
    return {
        "hp": np.int32(0),
        "energy": np.int32(0),
        "x": np.float32(0.0),
        "y": np.float32(0.0),
        "speed_x": np.float32(0.0),
        "speed_y": np.float32(0.0),
        "state": np.int8(0),  # STAND
        "front": np.int8(1),  # facing right
        "control": np.int8(0),
        "remaining_frame": np.int16(0),
        "hit_confirm": np.int8(0),
        "atk_is_live": np.int8(0),
        "atk_start_up": np.int16(0),
        "atk_active": np.int16(0),
        "atk_hit_damage": np.int16(0),
        "atk_type": np.int8(0),
    }


def _encode_character_dict(char) -> dict[str, Any]:
    """Extract one character's primitives as a flat dict in raw physical units."""
    if char is None:
        return _empty_char_dict()

    atk = char.attack_data
    is_live = bool(getattr(atk, "is_live", False)) if atk is not None else False
    if is_live:
        atype = int(getattr(atk, "attack_type", 0))
        start_up = int(getattr(atk, "start_up", 0))
        active = int(getattr(atk, "active", 0))
        hit_damage = int(getattr(atk, "hit_damage", 0))
    else:
        atype = 0
        start_up = 0
        active = 0
        hit_damage = 0

    return {
        "hp": np.int32(char.hp),
        "energy": np.int32(char.energy),
        "x": np.float32(char.x),
        "y": np.float32(char.y),
        "speed_x": np.float32(char.speed_x),
        "speed_y": np.float32(char.speed_y),
        "state": np.int8(_state_to_int(char.state)),
        "front": np.int8(1 if char.front else -1),
        "control": np.int8(1 if char.control else 0),
        "remaining_frame": np.int16(char.remaining_frame),
        "hit_confirm": np.int8(1 if char.hit_confirm else 0),
        "atk_is_live": np.int8(1 if is_live else 0),
        "atk_start_up": np.int16(start_up),
        "atk_active": np.int16(active),
        "atk_hit_damage": np.int16(hit_damage),
        "atk_type": np.int8(atype),
    }


def frame_to_obs_dict(
    frame_data,
    player_number: bool,
    max_hp: float = 400.0,
    max_energy: float = 300.0,
) -> dict[str, dict[str, Any]]:
    """Extract a structured observation from a pyftg `FrameData`.

    Returns a nested dict::

        {
            "own": {hp, energy, x, y, speed_x, speed_y, state, front, control,
                    remaining_frame, hit_confirm,
                    atk_is_live, atk_start_up, atk_active, atk_hit_damage, atk_type},
            "opp": {... same fields ...},
            "global": {current_round, current_frame, proj_self, proj_opp,
                       max_hp, max_energy},
        }

    All values are NumPy scalars in raw physical units. No clipping, no
    normalization — that happens at the dataloader view per training method.
    """
    own = frame_data.get_character(player_number)
    opp = frame_data.get_character(not player_number)

    proj_self = 0
    proj_opp = 0
    projs = getattr(frame_data, "projectile_data", []) or []
    for p in projs:
        if not getattr(p, "is_live", False):
            continue
        if p.player_number == player_number:
            proj_self = 1
        else:
            proj_opp = 1

    return {
        "own": _encode_character_dict(own),
        "opp": _encode_character_dict(opp),
        "global": {
            "current_round": np.int8(max(int(frame_data.current_round), 0)),
            "current_frame": np.int16(max(int(frame_data.current_frame_number), 0)),
            "proj_self": np.int8(proj_self),
            "proj_opp": np.int8(proj_opp),
            "max_hp": np.int16(int(max_hp)),
            "max_energy": np.int16(int(max_energy)),
        },
    }


def _legacy_per_char(c: dict[str, Any], out: np.ndarray, offset: int,
                     max_hp: float, max_energy: float) -> None:
    """Encode a 22-dim per-character slice from primitives (legacy form)."""
    out[offset + 0] = float(c["hp"]) / max(max_hp, 1.0)
    out[offset + 1] = float(c["energy"]) / max(max_energy, 1.0)
    out[offset + 2] = float(c["x"]) / STAGE_W
    out[offset + 3] = float(c["y"]) / STAGE_H
    out[offset + 4] = float(np.clip(float(c["speed_x"]) / SPEED_X_NORM, -1.0, 1.0))
    out[offset + 5] = float(np.clip(float(c["speed_y"]) / SPEED_Y_NORM, -1.0, 1.0))

    state_idx = int(c["state"])
    if 0 <= state_idx <= 3:
        out[offset + 6 + state_idx] = 1.0  # one-hot STAND..DOWN

    out[offset + 10] = float(c["front"])  # already ±1
    out[offset + 11] = float(c["control"])
    out[offset + 12] = min(float(c["remaining_frame"]) / REMAINING_FRAME_NORM, 1.0)
    out[offset + 13] = float(c["hit_confirm"])

    if int(c["atk_is_live"]) == 1:
        out[offset + 14] = 1.0
        out[offset + 15] = min(float(c["atk_start_up"]) / ATTACK_FRAME_NORM, 1.0)
        out[offset + 16] = min(float(c["atk_active"]) / ATTACK_FRAME_NORM, 1.0)
        out[offset + 17] = min(float(c["atk_hit_damage"]) / HIT_DAMAGE_NORM, 1.0)
        atype = int(c["atk_type"])
        if 1 <= atype <= 4:
            out[offset + 17 + atype] = 1.0  # 18..21


def obs_dict_to_legacy_vector(obs: dict[str, dict[str, Any]]) -> np.ndarray:
    """Rebuild the legacy 52-dim float32 vector from a primitives dict.

    Used as the LeWM linear-probe target. Identical numerics to the original
    ``frame_to_state_vector`` so existing checkpoints / probes stay valid.
    """
    out = np.zeros(STATE_VECTOR_DIM, dtype=np.float32)
    own = obs["own"]
    opp = obs["opp"]
    g = obs["global"]
    max_hp = float(g["max_hp"]) or 400.0
    max_energy = float(g["max_energy"]) or 300.0

    _legacy_per_char(own, out, 0, max_hp, max_energy)
    _legacy_per_char(opp, out, PER_CHAR_DIM, max_hp, max_energy)

    gi = PER_CHAR_DIM * 2  # global block start = 44
    dx = (float(opp["x"]) - float(own["x"])) / STAGE_W
    dy = (float(opp["y"]) - float(own["y"])) / STAGE_H
    out[gi + 0] = float(np.clip(dx, -1.0, 1.0))
    out[gi + 1] = float(np.clip(dy, -1.0, 1.0))
    out[gi + 2] = float(np.sqrt(dx * dx + dy * dy) * STAGE_W / DIAG)
    out[gi + 3] = float(np.clip((float(own["hp"]) - float(opp["hp"])) / max_hp, -1.0, 1.0))
    out[gi + 4] = float(g["current_round"]) / MAX_ROUNDS
    out[gi + 5] = float(g["current_frame"]) / MAX_FRAMES_PER_ROUND
    out[gi + 6] = float(g["proj_self"])
    out[gi + 7] = float(g["proj_opp"])
    return out


def frame_to_state_vector(
    frame_data,
    player_number: bool,
    max_hp: float = 400.0,
    max_energy: float = 300.0,
) -> np.ndarray:
    """Legacy entry point — returns the 52-dim float32 vector directly.

    Equivalent to ``obs_dict_to_legacy_vector(frame_to_obs_dict(...))``.
    """
    obs = frame_to_obs_dict(frame_data, player_number, max_hp=max_hp, max_energy=max_energy)
    return obs_dict_to_legacy_vector(obs)


# --------------------------------------------------------------------------- #
# Side canonicalization (P1↔P2 symmetry)
# --------------------------------------------------------------------------- #
#
# `obs/own/x`, `speed_x`, `front` are in *stage coordinates*, so a model
# trained on P1 data (own at left, facing right) wouldn't transfer to P2
# (own at right, facing left) without seeing the mirrored distribution. We
# canonicalize to "own always on the left" before flattening so any
# state-vector method (PETS, vector-mode Dreamer) sees a side-invariant
# input. Done at view time, not at write time — the buffer keeps raw truth.

_SIDED_X_FIELDS: tuple[str, ...] = ("x",)
_SIDED_SIGN_FIELDS: tuple[str, ...] = ("speed_x", "front")


def _mirror_char_dict(c: dict[str, Any]) -> dict[str, Any]:
    """Side-flipped copy of a per-character primitives dict."""
    flipped = dict(c)
    for k in _SIDED_X_FIELDS:
        flipped[k] = type(c[k])(STAGE_W - float(c[k]))
    for k in _SIDED_SIGN_FIELDS:
        flipped[k] = type(c[k])(-float(c[k]))
    return flipped


def canonicalize_obs_dict(obs: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Mirror the obs along the x-axis when own is on the right of opp.

    Returns a new dict (caller's data is never mutated). When ``own.x <=
    opp.x`` already, returns the original dict unchanged for cheap pass-through.
    """
    if float(obs["own"]["x"]) <= float(obs["opp"]["x"]):
        return obs
    return {
        "own": _mirror_char_dict(obs["own"]),
        "opp": _mirror_char_dict(obs["opp"]),
        "global": dict(obs["global"]),  # globals untouched
    }


def canonicalize_sample(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Vectorised canonicalization for ``(B, L, ...)`` arrays from ``sample_window``.

    Returns a new flat dict (only the six side-leaking fields are replaced;
    other keys are aliased — caller must not mutate them).
    """
    own_x = sample["own/x"]
    opp_x = sample["opp/x"]
    on_right = own_x > opp_x  # broadcast-friendly mask, shape (B, L)

    out = dict(sample)
    # Mirror x: x → STAGE_W − x (apply on both sides).
    for side in ("own", "opp"):
        key = f"{side}/x"
        arr = sample[key]
        out[key] = np.where(on_right, np.float32(STAGE_W) - arr.astype(np.float32), arr)
    # Mirror signed fields (speed_x, front). dtype preserved via astype below.
    for side in ("own", "opp"):
        for field in _SIDED_SIGN_FIELDS:
            key = f"{side}/{field}"
            arr = sample[key]
            mirrored = np.where(on_right, -arr.astype(np.float32), arr.astype(np.float32))
            out[key] = mirrored.astype(arr.dtype)
    return out


# --------------------------------------------------------------------------- #
# Flat-vector views
# --------------------------------------------------------------------------- #


def obs_dict_to_pets_vector(obs: dict[str, dict[str, Any]]) -> np.ndarray:
    """Flat continuous-primitives view for PETS — side-canonicalized.

    Layout: ``[own.PETS_PRIMITIVE_KEYS] + [opp.PETS_PRIMITIVE_KEYS] + [PETS_GLOBAL_KEYS]``.
    Raw physical units (HP in points, x/y in pixels, speeds in px/frame, frames as ints).
    PETS' internal scaler handles normalization.
    """
    obs = canonicalize_obs_dict(obs)
    parts: list[float] = []
    for side in ("own", "opp"):
        c = obs[side]
        for k in PETS_PRIMITIVE_KEYS:
            parts.append(float(c[k]))
    g = obs["global"]
    for k in PETS_GLOBAL_KEYS:
        parts.append(float(g[k]))
    return np.asarray(parts, dtype=np.float32)


# Indices into the PETS flat vector — used by the analytic cost function.
PETS_OWN_HP_IDX = PETS_PRIMITIVE_KEYS.index("hp")
PETS_OPP_HP_IDX = len(PETS_PRIMITIVE_KEYS) + PETS_PRIMITIVE_KEYS.index("hp")


# Discrete enum cardinalities. ``state`` ∈ {0..3}, ``atk_type`` ∈ {1..4} when
# the attack is live (gated separately by ``atk_is_live``).
STATE_ENUM_DIM = 4   # STAND, CROUCH, AIR, DOWN
ATK_TYPE_DIM = 4     # HIGH, MIDDLE, LOW, THROW (atk_type 1..4)

# Per-side dimensions in the Dreamer flat vector.
_DREAMER_PER_CHAR_DIM = len(PETS_PRIMITIVE_KEYS) + STATE_ENUM_DIM + ATK_TYPE_DIM
DREAMER_STATE_DIM = 2 * _DREAMER_PER_CHAR_DIM + len(PETS_GLOBAL_KEYS)


def _one_hot(idx: int, dim: int) -> np.ndarray:
    out = np.zeros(dim, dtype=np.float32)
    if 0 <= idx < dim:
        out[idx] = 1.0
    return out


def obs_dict_to_dreamer_vector(obs: dict[str, dict[str, Any]]) -> np.ndarray:
    """Flat float32 vector for vector-mode Dreamer (proprio encoder).

    Layout per side (×2)::

        PETS_PRIMITIVE_KEYS continuous (11)
        state one-hot       (4: STAND, CROUCH, AIR, DOWN)
        atk_type one-hot    (4: HIGH, MIDDLE, LOW, THROW; all-zero when atk_is_live=0)

    Plus global block (PETS_GLOBAL_KEYS, 4) → ``DREAMER_STATE_DIM`` total.
    Side-canonicalized internally; symlog squashing happens inside the
    upstream MLP encoder.
    """
    obs = canonicalize_obs_dict(obs)
    parts: list[np.ndarray] = []
    for side in ("own", "opp"):
        c = obs[side]
        cont = np.asarray([float(c[k]) for k in PETS_PRIMITIVE_KEYS], dtype=np.float32)
        state_oh = _one_hot(int(c["state"]), STATE_ENUM_DIM)
        if int(c["atk_is_live"]) == 1 and 1 <= int(c["atk_type"]) <= 4:
            atk_oh = _one_hot(int(c["atk_type"]) - 1, ATK_TYPE_DIM)
        else:
            atk_oh = np.zeros(ATK_TYPE_DIM, dtype=np.float32)
        parts.extend([cont, state_oh, atk_oh])
    g = obs["global"]
    parts.append(
        np.asarray([float(g[k]) for k in PETS_GLOBAL_KEYS], dtype=np.float32)
    )
    return np.concatenate(parts, axis=0)
