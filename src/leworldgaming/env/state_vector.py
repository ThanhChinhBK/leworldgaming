"""Hand-engineered state vector extracted from pyftg `FrameData`.

This is the *primary observation* for state-vector MBRL agents (DreamerV3, PETS,
TD-MPC2, DC-MPC, etc.). LeWM consumes pixels directly and uses this only as a
linear-probe target (gemini_research.md §6, §8).

# Schema (52 features, all float32)

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
    44  dx = (opp.x - own.x) / STAGE_W    in [-1, 1]
    45  dy = (opp.y - own.y) / STAGE_H    in [-1, 1]
    46  distance / DIAG                   in [0, 1]
    47  hp_diff (own - opp) / max_hp      in [-1, 1]
    48  round_progress = current_round/3  in [0, 1]
    49  frame_progress = frame/MAX_FRAMES in [0, 1]
    50  any_proj_self (0/1)               own has live projectile
    51  any_proj_opp  (0/1)               opp has live projectile

Total: 52 features.
"""

from __future__ import annotations

import numpy as np

# Stage dimensions (FightingICE default: 960x640 play area).
STAGE_W = 960.0
STAGE_H = 640.0
DIAG = float(np.sqrt(STAGE_W**2 + STAGE_H**2))

# Speed clamps — empirical bounds; outliers get clipped.
SPEED_X_NORM = 15.0
SPEED_Y_NORM = 25.0

# Round structure.
MAX_FRAMES_PER_ROUND = 3600  # 60 s at 60 FPS
MAX_ROUNDS = 3

# Frame counts for normalization.
REMAINING_FRAME_NORM = 60.0
ATTACK_FRAME_NORM = 30.0
HIT_DAMAGE_NORM = 30.0

# Schema dimensions.
PER_CHAR_DIM = 22
GLOBAL_DIM = 8
STATE_VECTOR_DIM = PER_CHAR_DIM * 2 + GLOBAL_DIM  # 52


def _state_to_int(state) -> int:
    """Map pyftg State enum to {0=STAND, 1=CROUCH, 2=AIR, 3=DOWN}."""
    name = state.name if hasattr(state, "name") else str(state)
    return {"STAND": 0, "CROUCH": 1, "AIR": 2, "DOWN": 3}.get(name, 0)


def _encode_character(
    char,
    out: np.ndarray,
    offset: int,
    max_hp: float,
    max_energy: float,
) -> None:
    """Write 22 features for one character starting at out[offset]."""
    if char is None:
        # Defaults preserve neutrality: zeros except state[STAND]=1, front=+1.
        out[offset + 6] = 1.0  # STAND
        out[offset + 10] = 1.0  # front
        return

    out[offset + 0] = char.hp / max(max_hp, 1.0)
    out[offset + 1] = char.energy / max(max_energy, 1.0)
    out[offset + 2] = char.x / STAGE_W
    out[offset + 3] = char.y / STAGE_H
    out[offset + 4] = np.clip(char.speed_x / SPEED_X_NORM, -1.0, 1.0)
    out[offset + 5] = np.clip(char.speed_y / SPEED_Y_NORM, -1.0, 1.0)

    state_idx = _state_to_int(char.state)
    out[offset + 6 + state_idx] = 1.0  # one-hot STAND..DOWN

    out[offset + 10] = 1.0 if char.front else -1.0
    out[offset + 11] = 1.0 if char.control else 0.0
    out[offset + 12] = min(char.remaining_frame / REMAINING_FRAME_NORM, 1.0)
    out[offset + 13] = 1.0 if char.hit_confirm else 0.0

    atk = char.attack_data
    if atk is not None and getattr(atk, "is_live", False):
        out[offset + 14] = 1.0
        out[offset + 15] = min(atk.start_up / ATTACK_FRAME_NORM, 1.0)
        out[offset + 16] = min(atk.active / ATTACK_FRAME_NORM, 1.0)
        out[offset + 17] = min(atk.hit_damage / HIT_DAMAGE_NORM, 1.0)
        atype = getattr(atk, "attack_type", 0)
        if 1 <= atype <= 4:
            out[offset + 17 + atype] = 1.0  # 18..21


def frame_to_state_vector(
    frame_data,
    player_number: bool,
    max_hp: float = 400.0,
    max_energy: float = 300.0,
) -> np.ndarray:
    """Flatten a pyftg `FrameData` into a fixed-shape float32 vector.

    Args:
        frame_data: pyftg `FrameData`.
        player_number: True if "self" is player 1 (index 0), False if player 2.
        max_hp: HP cap, from `GameData.max_hps[i]` when known.
        max_energy: energy cap, from `GameData.max_energies[i]` when known.

    Returns:
        np.ndarray of shape (STATE_VECTOR_DIM,) and dtype float32.
        All values are finite and bounded in roughly [-1, 1] or [0, 1].
    """
    out = np.zeros(STATE_VECTOR_DIM, dtype=np.float32)

    own = frame_data.get_character(player_number)
    opp = frame_data.get_character(not player_number)

    _encode_character(own, out, 0, max_hp, max_energy)
    _encode_character(opp, out, PER_CHAR_DIM, max_hp, max_energy)

    g = PER_CHAR_DIM * 2  # global block start = 44
    if own is not None and opp is not None:
        dx = (opp.x - own.x) / STAGE_W
        dy = (opp.y - own.y) / STAGE_H
        out[g + 0] = np.clip(dx, -1.0, 1.0)
        out[g + 1] = np.clip(dy, -1.0, 1.0)
        out[g + 2] = float(np.sqrt(dx * dx + dy * dy) * STAGE_W / DIAG)
        out[g + 3] = np.clip((own.hp - opp.hp) / max(max_hp, 1.0), -1.0, 1.0)

    out[g + 4] = max(frame_data.current_round, 0) / MAX_ROUNDS
    out[g + 5] = max(frame_data.current_frame_number, 0) / MAX_FRAMES_PER_ROUND

    # Live projectile flags.
    projs = getattr(frame_data, "projectile_data", []) or []
    for p in projs:
        if not getattr(p, "is_live", False):
            continue
        # player_number on AttackData uses the same convention as CharacterData.
        if p.player_number == player_number:
            out[g + 6] = 1.0
        else:
            out[g + 7] = 1.0

    return out
