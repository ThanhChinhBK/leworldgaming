"""Offline smoke test for the state-vector schema and replay buffer.

Builds a synthetic pyftg ``FrameData`` by hand, extracts the canonical
primitives dict, round-trips a small replay through HDF5 (named-group
schema with ``is_first`` / ``cont`` flags), then samples both transitions
and a length-2 sequence batch. Runs anywhere — does NOT need the game
container.

    uv run python scripts/demo_state_vector.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
from pyftg.models.attack_data import AttackData
from pyftg.models.character_data import CharacterData
from pyftg.models.enums.action import Action
from pyftg.models.enums.state import State
from pyftg.models.frame_data import FrameData

from leworldgaming.data.replay_buffer import BufferConfig, ReplayBuffer
from leworldgaming.env.state_vector import (
    DREAMER_STATE_DIM,
    PETS_STATE_DIM,
    STATE_VECTOR_DIM,
    canonicalize_obs_dict,
    frame_to_obs_dict,
    obs_dict_to_dreamer_vector,
    obs_dict_to_legacy_vector,
    obs_dict_to_pets_vector,
)


def make_synthetic_frame(frame_idx: int, hp_self: int, hp_opp: int) -> FrameData:
    own = CharacterData(
        player_number=True,
        hp=hp_self,
        energy=120,
        x=300,
        y=320,
        speed_x=4,
        speed_y=0,
        state=State.STAND,
        action=Action.STAND_A,
        front=True,
        control=True,
        remaining_frame=10,
        hit_confirm=False,
        attack_data=AttackData(
            is_live=True, start_up=4, active=6, hit_damage=10, attack_type=2
        ),
    )
    opp = CharacterData(
        player_number=False,
        hp=hp_opp,
        energy=80,
        x=620,
        y=320,
        speed_x=-3,
        speed_y=0,
        state=State.AIR,
        action=Action.AIR_B,
        front=False,
        control=False,
        remaining_frame=15,
        hit_confirm=False,
        attack_data=AttackData(empty_flag=True, is_live=False),
    )
    return FrameData(
        character_data=[own, opp],
        current_frame_number=frame_idx,
        current_round=1,
        empty_flag=False,
        front=[True, False],
    )


def _swap_sides_obs(obs: dict) -> dict:
    """Return an obs where own is at right (x mirrored across stage), facing left.

    Used to verify that ``canonicalize_obs_dict`` makes the two
    representations indistinguishable to the model.
    """
    from leworldgaming.env.state_vector import STAGE_W

    own = dict(obs["own"])
    opp = dict(obs["opp"])
    for c in (own, opp):
        c["x"] = type(c["x"])(STAGE_W - float(c["x"]))
        c["speed_x"] = type(c["speed_x"])(-float(c["speed_x"]))
        c["front"] = type(c["front"])(-float(c["front"]))
    return {"own": own, "opp": opp, "global": dict(obs["global"])}


def main() -> None:
    print(f"[demo] STATE_VECTOR_DIM (legacy)    = {STATE_VECTOR_DIM}")
    print(f"[demo] PETS_STATE_DIM (continuous) = {PETS_STATE_DIM}")
    print(f"[demo] DREAMER_STATE_DIM           = {DREAMER_STATE_DIM}")
    print()

    # 1. Extract primitives dict and inspect all views.
    fd = make_synthetic_frame(frame_idx=42, hp_self=350, hp_opp=200)
    obs = frame_to_obs_dict(fd, player_number=True)
    print(f"[demo] obs groups: {sorted(obs.keys())}")
    print(f"[demo] own.hp={int(obs['own']['hp'])} opp.hp={int(obs['opp']['hp'])} "
          f"own.x={float(obs['own']['x']):.1f} opp.x={float(obs['opp']['x']):.1f}")
    print(f"[demo] own.atk_is_live={int(obs['own']['atk_is_live'])} "
          f"atk_type={int(obs['own']['atk_type'])}")

    legacy = obs_dict_to_legacy_vector(obs)
    pets = obs_dict_to_pets_vector(obs)
    dreamer = obs_dict_to_dreamer_vector(obs)
    assert legacy.shape == (STATE_VECTOR_DIM,) and legacy.dtype == np.float32
    assert pets.shape == (PETS_STATE_DIM,) and pets.dtype == np.float32
    assert dreamer.shape == (DREAMER_STATE_DIM,) and dreamer.dtype == np.float32
    assert np.all(np.isfinite(legacy))
    assert np.all(np.isfinite(pets))
    assert np.all(np.isfinite(dreamer))
    print(f"[demo] legacy[:8]  = {legacy[:8]}")
    print(f"[demo] pets[:8]    = {pets[:8]}")
    print(f"[demo] dreamer[:8] = {dreamer[:8]}")
    print()

    # 1b. Side-canonicalization: swapping sides should yield identical
    # PETS / Dreamer vectors after canonicalize_obs_dict (which is called
    # internally by both view helpers).
    swapped = _swap_sides_obs(obs)
    assert obs["own"]["x"] != swapped["own"]["x"], "test fixture should differ pre-canonicalize"
    pets_swapped = obs_dict_to_pets_vector(swapped)
    dreamer_swapped = obs_dict_to_dreamer_vector(swapped)
    np.testing.assert_allclose(pets, pets_swapped, atol=1e-5, err_msg="PETS not side-invariant")
    np.testing.assert_allclose(dreamer, dreamer_swapped, atol=1e-5,
                               err_msg="Dreamer vector not side-invariant")
    # And canonicalize_obs_dict on a left-side obs is a pass-through.
    canon_left = canonicalize_obs_dict(obs)
    assert canon_left is obs, "left-side canonicalize should pass through"
    print("[demo] side canonicalization: PETS + Dreamer vectors invariant ✓")
    print()

    # 2. Round-trip an episode through the new HDF5 layout.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.h5"
        cfg = BufferConfig(path=str(path))
        with ReplayBuffer(cfg) as buf:
            for t in range(20):
                fd_t = make_synthetic_frame(
                    frame_idx=t, hp_self=400 - t, hp_opp=400 - 2 * t
                )
                obs_t = frame_to_obs_dict(fd_t, player_number=True)
                buf.add(
                    obs_dict=obs_t,
                    action=Action.STAND_A.to_int(),
                    reward=1.0 / 400.0,
                    done=False,
                    is_first=(t == 0),
                )
            buf.end_episode()
            print(f"[demo] wrote {len(buf)} transitions across {buf.num_episodes} episode(s)")

        # 3. Re-open in read mode and sample.
        with ReplayBuffer(BufferConfig(path=str(path), read_only=True)) as buf:
            assert len(buf) == 20
            tx = buf.sample(batch_size=8, rng=np.random.default_rng(0))
            print(f"[demo] sampled transitions: action={tx['action'].shape} "
                  f"reward.mean={tx['reward'].mean():.4f} "
                  f"is_first.sum={int(tx['is_first'].sum())} "
                  f"own.hp={tx['own/hp'].mean():.1f}")

            seq = buf.sample_sequences(batch_size=4, seq_len=2, rng=np.random.default_rng(1))
            print(f"[demo] sampled (B=4, L=2): action={seq['action'].shape} "
                  f"own/hp={seq['own/hp'].shape} cont={seq['cont'].shape}")

    print("[demo] OK")


if __name__ == "__main__":
    main()
