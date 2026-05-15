"""Per-method dataloader views over the shared replay buffer.

The buffer stores raw primitives (see ``replay_buffer.py``); each MBRL
training method materializes the shape it wants from the same source via
one of these three functions.

- ``view_lewm`` — returns ``(pixels, actions)`` matching the legacy
  ``train_lewm._sample_sequence_batch`` output exactly. The LeWM trainer
  is unaware of the schema change.
- ``view_dreamer`` — returns the dict format the vendored
  ``external/dreamerv3-torch`` trainer expects in vector-mode (proprio):
  side-canonicalized 42-d float vector under ``"vector"``, plus a tiny
  dummy ``"image"`` so the upstream ``WorldModel.preprocess`` doesn't
  crash on its hardcoded ``obs["image"] / 255.0`` line.
- ``view_pets`` — returns ``(s, a, s_next, r)`` flat-vector transitions.
  Reward is recomputed analytically from HP primitives — PETS upstream uses
  a known cost function rather than a learned reward head. State is
  side-canonicalized so a P1-collected transition trains a P2-deployable
  ensemble.
"""

from __future__ import annotations

import numpy as np

from leworldgaming.env.state_vector import (
    ATK_TYPE_DIM,
    DREAMER_STATE_DIM,
    PETS_GLOBAL_KEYS,
    PETS_OPP_HP_IDX,
    PETS_OWN_HP_IDX,
    PETS_PRIMITIVE_KEYS,
    STATE_ENUM_DIM,
    canonicalize_sample,
)


def view_lewm(sample: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """LeWM view: ``(pixels (B,L,C,H,W) uint8, actions (B,L) int)``.

    Matches the shape ``train_lewm._sample_sequence_batch`` already returns,
    so the LeWM trainer's downstream tensor pipeline is unchanged. No
    canonicalization — LeWM eats raw pixels and is P1-only by design.
    """
    if "pixels" not in sample:
        raise KeyError(
            "view_lewm requires pixel data — re-collect with --pixels."
        )
    return sample["pixels"], sample["action"]


def _flatten_primitives(
    sample: dict[str, np.ndarray],
    keys: tuple[str, ...],
    side: str,
) -> np.ndarray:
    """Stack named primitives from one obs side into ``(B, L, len(keys))``."""
    cols = [sample[f"{side}/{k}"].astype(np.float32) for k in keys]
    return np.stack(cols, axis=-1)


def _one_hot_int_array(values: np.ndarray, dim: int) -> np.ndarray:
    """Vectorised one-hot. ``values`` shape ``(B, L)`` int → ``(B, L, dim) float32``.

    Out-of-range values produce all-zero rows (matches ``state_vector._one_hot``).
    """
    b, length = values.shape
    out = np.zeros((b, length, dim), dtype=np.float32)
    flat = values.reshape(-1)
    in_range = (flat >= 0) & (flat < dim)
    flat_idx = np.arange(b * length)
    out_view = out.reshape(b * length, dim)
    out_view[flat_idx[in_range], flat[in_range]] = 1.0
    return out


def view_pets(sample: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """PETS view: continuous-only flat transitions in raw physical units.

    Returns a dict with keys::

        s        (B, state_dim) float32   — current state, side-canonicalized
        a        (B,)           int32     — discrete action id
        s_next   (B, state_dim) float32   — next state, side-canonicalized
        r        (B,)           float32   — analytic reward Δ(own_hp − opp_hp)/max_hp
        cont     (B,)           float32   — 1.0 if next is real (not episode boundary)

    The input ``sample`` must come from ``sample_sequences(seq_len=2)`` so
    each row contains a `(s_t, s_{t+1})` pair. State layout matches
    ``state_vector.obs_dict_to_pets_vector`` so the same indices apply at
    inference time.
    """
    if sample["action"].shape[1] < 2:
        raise ValueError(
            "view_pets requires sample_sequences(seq_len>=2) to form transitions."
        )
    sample = canonicalize_sample(sample)

    own_block = _flatten_primitives(sample, PETS_PRIMITIVE_KEYS, "own")  # (B, L, k)
    opp_block = _flatten_primitives(sample, PETS_PRIMITIVE_KEYS, "opp")
    glob_block = _flatten_primitives(sample, PETS_GLOBAL_KEYS, "global")
    state = np.concatenate([own_block, opp_block, glob_block], axis=-1).astype(np.float32)
    # state shape: (B, L, 2*len(prim) + len(global))

    s = state[:, 0]
    s_next = state[:, 1]

    max_hp = sample["global/max_hp"][:, 0].astype(np.float32)
    max_hp = np.where(max_hp > 0, max_hp, 400.0)

    own_hp_t = state[:, 0, PETS_OWN_HP_IDX]
    opp_hp_t = state[:, 0, PETS_OPP_HP_IDX]
    own_hp_t1 = state[:, 1, PETS_OWN_HP_IDX]
    opp_hp_t1 = state[:, 1, PETS_OPP_HP_IDX]
    damage_dealt = opp_hp_t - opp_hp_t1
    damage_taken = own_hp_t - own_hp_t1
    r = ((damage_dealt - damage_taken) / max_hp).astype(np.float32)

    cont = (1.0 - sample["done"][:, 0].astype(np.float32)).astype(np.float32)

    return {
        "s": s,
        "a": sample["action"][:, 0].astype(np.int32),
        "s_next": s_next,
        "r": r,
        "cont": cont,
    }


def _build_dreamer_vector(sample: dict[str, np.ndarray]) -> np.ndarray:
    """Construct ``(B, L, DREAMER_STATE_DIM) float32`` from a canonicalized sample.

    Layout matches ``state_vector.obs_dict_to_dreamer_vector`` so the
    inference path and the training path see byte-identical features.
    """
    own_cont = _flatten_primitives(sample, PETS_PRIMITIVE_KEYS, "own")
    opp_cont = _flatten_primitives(sample, PETS_PRIMITIVE_KEYS, "opp")

    own_state_oh = _one_hot_int_array(
        sample["own/state"].astype(np.int32), STATE_ENUM_DIM
    )
    opp_state_oh = _one_hot_int_array(
        sample["opp/state"].astype(np.int32), STATE_ENUM_DIM
    )

    # atk_type one-hot is gated by atk_is_live; map raw enum {1..4} → idx {0..3}.
    own_live = sample["own/atk_is_live"].astype(bool)
    opp_live = sample["opp/atk_is_live"].astype(bool)
    own_atk_idx = np.where(own_live, sample["own/atk_type"].astype(np.int32) - 1, -1)
    opp_atk_idx = np.where(opp_live, sample["opp/atk_type"].astype(np.int32) - 1, -1)
    own_atk_oh = _one_hot_int_array(own_atk_idx, ATK_TYPE_DIM)
    opp_atk_oh = _one_hot_int_array(opp_atk_idx, ATK_TYPE_DIM)

    glob = _flatten_primitives(sample, PETS_GLOBAL_KEYS, "global")

    parts = [
        own_cont, own_state_oh, own_atk_oh,
        opp_cont, opp_state_oh, opp_atk_oh,
        glob,
    ]
    out = np.concatenate(parts, axis=-1).astype(np.float32)
    assert out.shape[-1] == DREAMER_STATE_DIM, (
        f"Dreamer vector dim mismatch: built {out.shape[-1]}, expected {DREAMER_STATE_DIM}"
    )
    return out


def view_dreamer(
    sample: dict[str, np.ndarray],
    num_actions: int,
) -> dict[str, np.ndarray]:
    """Dreamer view: vector-mode (proprio) batch matching upstream's contract.

    Keys:

        vector      (B, L, DREAMER_STATE_DIM)  float32   side-canonicalized + one-hot
        image       (B, L, 1, 1, 3)            uint8     dummy — see B.3 in the plan
        action      (B, L, A)                  float32   one-hot
        reward      (B, L)                     float32
        is_first    (B, L)                     bool
        is_terminal (B, L)                     bool
    """
    sample = canonicalize_sample(sample)
    vec = _build_dreamer_vector(sample)
    b, length = vec.shape[:2]

    actions_int = sample["action"].astype(np.int64)
    one_hot = np.zeros((b, length, num_actions), dtype=np.float32)
    flat_idx = actions_int.reshape(-1)
    one_hot.reshape(-1, num_actions)[np.arange(b * length), flat_idx] = 1.0

    dummy_image = np.zeros((b, length, 1, 1, 3), dtype=np.uint8)

    return {
        "vector": vec,
        "image": dummy_image,
        "action": one_hot,
        "reward": sample["reward"].astype(np.float32),
        "is_first": sample["is_first"].astype(bool),
        "is_terminal": sample["done"].astype(bool),
    }
