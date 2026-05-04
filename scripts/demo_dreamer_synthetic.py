"""Offline smoke test for the DreamerV3 stack — runs without the game.

  1. Builds a tiny synthetic replay with ``--pixels``-style image data.
  2. Exports it to per-episode npz files via ``dreamer_export``.
  3. Runs ``train_dreamer.train`` for a few steps to confirm the full
     world-model + imagined-behaviour gradient path works on this host.

The vendored Dreamer is heavy (~10–30 M params at default config), so this
test is best run on a CUDA box. On CPU/MPS expect minutes per step.

    uv run python scripts/demo_dreamer_synthetic.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from leworldgaming.data.replay_buffer import BufferConfig, ReplayBuffer
from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.state_vector import frame_to_obs_dict
from leworldgaming.training.train_dreamer import train
from pyftg.models.attack_data import AttackData
from pyftg.models.character_data import CharacterData
from pyftg.models.enums.action import Action
from pyftg.models.enums.state import State
from pyftg.models.frame_data import FrameData


def _make_frame(t: int, hp_self: int, hp_opp: int) -> FrameData:
    own = CharacterData(
        player_number=True, hp=hp_self, energy=120, x=300, y=320,
        speed_x=0, speed_y=0, state=State.STAND, action=Action.STAND_A,
        front=True, control=True, remaining_frame=0, hit_confirm=False,
        attack_data=AttackData(empty_flag=True, is_live=False),
    )
    opp = CharacterData(
        player_number=False, hp=hp_opp, energy=80, x=620, y=320,
        speed_x=0, speed_y=0, state=State.STAND, action=Action.STAND_A,
        front=False, control=False, remaining_frame=0, hit_confirm=False,
        attack_data=AttackData(empty_flag=True, is_live=False),
    )
    return FrameData(
        character_data=[own, opp],
        current_frame_number=t, current_round=1, empty_flag=False,
        front=[True, False],
    )


def main() -> None:
    rng = np.random.default_rng(0)
    image_size = 64

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "replay.h5"
        ep_dir = Path(tmp) / "episodes"
        ckpt = Path(tmp) / "dreamer.pt"
        logdir = Path(tmp) / "logs"

        # 1. Synthetic replay with random pixels — 3 short episodes.
        cfg = BufferConfig(
            path=str(path),
            pixel_shape=(3, image_size, image_size),
        )
        with ReplayBuffer(cfg) as buf:
            for ep in range(3):
                for t in range(40):
                    fd = _make_frame(t, 400 - t, 400 - 2 * t)
                    obs = frame_to_obs_dict(fd, player_number=True)
                    pixels = rng.integers(
                        0, 256, size=(3, image_size, image_size), dtype=np.uint8
                    )
                    buf.add(
                        obs_dict=obs,
                        action=int(rng.integers(0, NUM_ACTIONS)),
                        reward=float(rng.normal(0, 0.01)),
                        done=(t == 39),
                        is_first=(t == 0),
                        pixels=pixels,
                    )
                buf.end_episode()
            print(f"[demo_dreamer] wrote {len(buf)} transitions, "
                  f"{buf.num_episodes} episodes (pixels {image_size}x{image_size})")

        # 2. Train a handful of steps. batch_length must be small for tiny
        # episodes; defaults are tuned for L=64.
        result = train(
            num_steps=2,
            config_path=None,
            data_path=str(path),
            episode_dir=str(ep_dir),
            ckpt_path=str(ckpt),
            logdir=str(logdir),
            image_size=image_size,
            batch_size=2,
            batch_length=16,
            log_every=1,
        )
        print(f"[demo_dreamer] final metrics: {result['final']}")

    print("[demo_dreamer] OK")


if __name__ == "__main__":
    main()
