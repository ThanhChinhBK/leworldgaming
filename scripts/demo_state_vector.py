"""Offline smoke test for the state vector and replay buffer.

Builds a synthetic pyftg FrameData by hand, flattens it to the canonical state
vector, round-trips a small replay through HDF5, and prints the schema layout
and a sampled batch. Runs anywhere — does NOT need the game container.

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
    PER_CHAR_DIM,
    STATE_VECTOR_DIM,
    frame_to_state_vector,
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


def main() -> None:
    print(f"[demo] STATE_VECTOR_DIM = {STATE_VECTOR_DIM} (per-char {PER_CHAR_DIM} x 2 + 8 global)")
    print()

    # 1. flatten one frame and inspect the layout.
    fd = make_synthetic_frame(frame_idx=42, hp_self=350, hp_opp=200)
    sv = frame_to_state_vector(fd, player_number=True)
    print(f"[demo] state_vector dtype={sv.dtype} shape={sv.shape}")
    print(f"[demo] hp_self_norm={sv[0]:.3f}  hp_opp_norm={sv[PER_CHAR_DIM]:.3f}  "
          f"hp_diff_norm={sv[PER_CHAR_DIM*2 + 3]:.3f}")
    print(f"[demo] dx_norm={sv[PER_CHAR_DIM*2]:.3f}  dy_norm={sv[PER_CHAR_DIM*2+1]:.3f}  "
          f"distance={sv[PER_CHAR_DIM*2+2]:.3f}")
    assert np.all(np.isfinite(sv))
    assert sv.dtype == np.float32

    # 2. round-trip through HDF5.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "demo.h5"
        cfg = BufferConfig(path=str(path))
        with ReplayBuffer(cfg) as buf:
            for t in range(20):
                fd_t = make_synthetic_frame(
                    frame_idx=t, hp_self=400 - t, hp_opp=400 - 2 * t
                )
                sv_t = frame_to_state_vector(fd_t, player_number=True)
                buf.add(
                    state_vector=sv_t,
                    action=Action.STAND_A.to_int(),
                    reward=1.0 / 400.0,
                    done=False,
                    hp_self=400 - t,
                    hp_opp=400 - 2 * t,
                    frame_idx=t,
                )
            buf.end_episode()
            print(f"[demo] wrote {len(buf)} transitions across {buf.num_episodes} episode(s)")

        # Re-open and sample.
        with ReplayBuffer(cfg) as buf:
            assert len(buf) == 20
            batch = buf.sample(batch_size=8, rng=np.random.default_rng(0))
            print(f"[demo] sampled batch: state_vector={batch['state_vector'].shape}, "
                  f"action={batch['action'].shape}, reward.mean={batch['reward'].mean():.4f}")

    print("[demo] OK")


if __name__ == "__main__":
    main()
