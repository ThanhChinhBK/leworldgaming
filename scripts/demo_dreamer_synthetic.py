"""Offline smoke test for the DreamerV3 stack — runs without the game.

DreamerV3 is configured in **vector / proprio** mode here (per
``gemini_research.md §5``); pixels are LeWM-only. The smoke test:

  1. Builds a tiny synthetic replay with state primitives only (no pixels).
  2. Exports it to per-episode npz files via ``dreamer_export``.
  3. Runs ``train_dreamer.train`` for a few steps to confirm the full
     world-model + imagined-behaviour gradient path works on this host.

Also asserts that ``view_dreamer`` produces identical batches when the
input obs is mirrored across the stage x-axis (P1↔P2 symmetry).

The vendored Dreamer is heavy (~10–20 M params at default config), so this
test is best run on a CUDA box. On CPU/MPS expect minutes per step.

    uv run python scripts/demo_dreamer_synthetic.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from leworldgaming.data.replay_buffer import BufferConfig, ReplayBuffer
from leworldgaming.data.views import view_dreamer
from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.state_vector import (
    DREAMER_STATE_DIM,
    STAGE_W,
    frame_to_obs_dict,
)
from leworldgaming.training.train_dreamer import train
from pyftg.models.attack_data import AttackData
from pyftg.models.character_data import CharacterData
from pyftg.models.enums.action import Action
from pyftg.models.enums.state import State
from pyftg.models.frame_data import FrameData


def _make_frame(t: int, hp_self: int, hp_opp: int) -> FrameData:
    own = CharacterData(
        player_number=True, hp=hp_self, energy=120, x=300, y=320,
        speed_x=1, speed_y=0, state=State.STAND, action=Action.STAND_A,
        front=True, control=True, remaining_frame=0, hit_confirm=False,
        attack_data=AttackData(empty_flag=True, is_live=False),
    )
    opp = CharacterData(
        player_number=False, hp=hp_opp, energy=80, x=620, y=320,
        speed_x=-1, speed_y=0, state=State.STAND, action=Action.STAND_A,
        front=False, control=False, remaining_frame=0, hit_confirm=False,
        attack_data=AttackData(empty_flag=True, is_live=False),
    )
    return FrameData(
        character_data=[own, opp],
        current_frame_number=t, current_round=1, empty_flag=False,
        front=[True, False],
    )


def _symmetry_assertion() -> None:
    """view_dreamer(left-side sample) ≡ view_dreamer(right-side sample)."""
    fd = _make_frame(0, 400, 350)
    obs = frame_to_obs_dict(fd, player_number=True)
    mirrored = {
        side: {**vals,
               "x": type(vals["x"])(STAGE_W - float(vals["x"])) if "x" in vals else vals.get("x"),
               "speed_x": type(vals["speed_x"])(-float(vals["speed_x"])) if "speed_x" in vals else vals.get("speed_x"),
               "front": type(vals["front"])(-float(vals["front"])) if "front" in vals else vals.get("front"),
               }
        for side, vals in obs.items() if side in ("own", "opp")
    }
    mirrored["global"] = dict(obs["global"])

    def to_sample(o: dict) -> dict[str, np.ndarray]:
        # Materialize a (B=1, L=2) sample dict from a single obs.
        s: dict[str, np.ndarray] = {}
        for side in ("own", "opp", "global"):
            for k, v in o[side].items():
                arr = np.asarray([float(v), float(v)], dtype=np.float32)
                # cast back to original dtype family
                if isinstance(v, np.integer):
                    arr = arr.astype(v.dtype)
                s[f"{side}/{k}"] = arr[None, :]  # (B=1, L=2)
        s["action"] = np.asarray([[0, 0]], dtype=np.int32)
        s["reward"] = np.asarray([[0.0, 0.0]], dtype=np.float32)
        s["done"] = np.asarray([[0, 0]], dtype=np.uint8)
        s["is_first"] = np.asarray([[1, 0]], dtype=np.uint8)
        s["cont"] = np.asarray([[1, 1]], dtype=np.uint8)
        return s

    a = view_dreamer(to_sample(obs), num_actions=NUM_ACTIONS)
    b = view_dreamer(to_sample(mirrored), num_actions=NUM_ACTIONS)
    np.testing.assert_allclose(a["vector"], b["vector"], atol=1e-5,
                               err_msg="Dreamer view not side-invariant")
    print("[demo_dreamer] symmetry: view_dreamer invariant under side-swap ✓")


def main() -> None:
    rng = np.random.default_rng(0)

    print(f"[demo_dreamer] DREAMER_STATE_DIM = {DREAMER_STATE_DIM}")
    _symmetry_assertion()

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "replay.h5"
        ep_dir = Path(tmp) / "episodes"
        ckpt = Path(tmp) / "dreamer.pt"
        logdir = Path(tmp) / "logs"

        # 1. Synthetic replay — 3 short episodes, NO pixels (vector mode).
        cfg = BufferConfig(path=str(path))  # pixel_shape=None
        with ReplayBuffer(cfg) as buf:
            for ep in range(3):
                for t in range(40):
                    fd = _make_frame(t, 400 - t, 400 - 2 * t)
                    obs = frame_to_obs_dict(fd, player_number=True)
                    buf.add(
                        obs_dict=obs,
                        action=int(rng.integers(0, NUM_ACTIONS)),
                        reward=float(rng.normal(0, 0.01)),
                        done=(t == 39),
                        is_first=(t == 0),
                    )
                buf.end_episode()
            print(f"[demo_dreamer] wrote {len(buf)} transitions, "
                  f"{buf.num_episodes} episodes (vector mode)")

        # 2. Train a handful of steps. batch_length must be small for tiny
        # episodes; defaults are tuned for L=64.
        result = train(
            num_steps=2,
            config_path=None,
            data_path=str(path),
            episode_dir=str(ep_dir),
            ckpt_path=str(ckpt),
            logdir=str(logdir),
            batch_size=2,
            batch_length=16,
            log_every=1,
        )
        print(f"[demo_dreamer] final metrics keys: "
              f"{sorted(k for k in result['final'] if not k.startswith('beh_'))[:8]}")

    print("[demo_dreamer] OK")


if __name__ == "__main__":
    main()
