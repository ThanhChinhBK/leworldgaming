"""Offline smoke test for the PETS stack — runs without the game.

  1. Builds a tiny synthetic replay (one episode, random transitions).
  2. Trains the ensemble for a few NLL steps via ``train_pets.train``.
  3. Calls ``PETSAgent.act`` once with a sampled obs to confirm the CEM
     planner runs end-to-end (and prints the frame-budget cost).

    uv run python scripts/demo_pets_synthetic.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np

from leworldgaming.agents.pets.agent import PETSAgent
from leworldgaming.data.replay_buffer import BufferConfig, ReplayBuffer
from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.state_vector import (
    PETS_STATE_DIM,
    STAGE_W,
    frame_to_obs_dict,
    obs_dict_to_pets_vector,
)
from leworldgaming.training.train_pets import train
from leworldgaming.utils.timing import FrameBudget
from pyftg.models.attack_data import AttackData
from pyftg.models.character_data import CharacterData
from pyftg.models.enums.action import Action
from pyftg.models.enums.state import State
from pyftg.models.frame_data import FrameData


def _make_frame(t: int, hp_self: int, hp_opp: int) -> FrameData:
    own = CharacterData(
        player_number=True, hp=hp_self, energy=120, x=300 + t, y=320,
        speed_x=1, speed_y=0, state=State.STAND, action=Action.STAND_A,
        front=True, control=True, remaining_frame=10, hit_confirm=False,
        attack_data=AttackData(empty_flag=True, is_live=False),
    )
    opp = CharacterData(
        player_number=False, hp=hp_opp, energy=80, x=620 - t, y=320,
        speed_x=-1, speed_y=0, state=State.AIR, action=Action.AIR_B,
        front=False, control=False, remaining_frame=15, hit_confirm=False,
        attack_data=AttackData(empty_flag=True, is_live=False),
    )
    return FrameData(
        character_data=[own, opp],
        current_frame_number=t, current_round=1, empty_flag=False,
        front=[True, False],
    )


def _mirror_obs(obs: dict) -> dict:
    """Mirror an obs across the stage x-axis, swapping side roles."""
    own = dict(obs["own"])
    opp = dict(obs["opp"])
    for c in (own, opp):
        c["x"] = type(c["x"])(STAGE_W - float(c["x"]))
        c["speed_x"] = type(c["speed_x"])(-float(c["speed_x"]))
        c["front"] = type(c["front"])(-float(c["front"]))
    return {"own": own, "opp": opp, "global": dict(obs["global"])}


def main() -> None:
    rng = np.random.default_rng(0)

    # Symmetry assertion: training data must be side-invariant after
    # canonicalize. obs_dict_to_pets_vector(obs) ≡ obs_dict_to_pets_vector(mirror(obs)).
    fd_left = _make_frame(0, 400, 350)
    obs_left = frame_to_obs_dict(fd_left, player_number=True)
    obs_right = _mirror_obs(obs_left)
    v_left = obs_dict_to_pets_vector(obs_left)
    v_right = obs_dict_to_pets_vector(obs_right)
    np.testing.assert_allclose(v_left, v_right, atol=1e-5,
                               err_msg="PETS state vector is not side-invariant")
    print("[demo_pets] symmetry: PETS state invariant under side-swap ✓")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "replay.h5"
        ckpt = Path(tmp) / "pets.pt"

        # 1. Synthetic replay — two episodes, 60 steps each.
        cfg = BufferConfig(path=str(path))
        with ReplayBuffer(cfg) as buf:
            for ep in range(2):
                for t in range(60):
                    fd = _make_frame(t, 400 - t - ep * 20, 400 - 2 * t)
                    obs = frame_to_obs_dict(fd, player_number=True)
                    buf.add(
                        obs_dict=obs,
                        action=int(rng.integers(0, NUM_ACTIONS)),
                        reward=1.0 / 400.0,
                        done=(t == 59),
                        is_first=(t == 0),
                    )
                buf.end_episode()
            print(f"[demo_pets] wrote {len(buf)} transitions, {buf.num_episodes} episodes")

        # 2. Train a few steps.
        result = train(
            num_steps=5,
            config_path=None,  # use DEFAULTS
            data_path=str(path),
            ckpt_path=str(ckpt),
            batch_size=32,
            log_every=1,
            val_every=0,
            val_split=0.0,
            seed=0,
        )
        print(f"[demo_pets] final train metrics: {result['final_train']}")

        # 3. CEM plan call — assert the planner returns a valid action and
        # frame-budget overruns are reported (likely on CPU / synthetic).
        agent = PETSAgent(cfg=result.get("final_train", {}).get("config") or {
            "state_dim": PETS_STATE_DIM,
            "action_dim": NUM_ACTIONS,
            "planner_num_candidates": 64,
            "planner_horizon": 5,
            "planner_num_iters": 2,
        })
        agent.load(str(ckpt))
        obs = frame_to_obs_dict(_make_frame(0, 400, 400), player_number=True)
        budget = FrameBudget()
        a = agent.act({**obs, "_frame_budget": budget})
        print(
            f"[demo_pets] act -> {a} ({budget.last_ms:.1f} ms"
            f"{' OVER BUDGET' if budget.drops else ''})"
        )
        assert 0 <= a < NUM_ACTIONS

    print("[demo_pets] OK")


if __name__ == "__main__":
    main()
