"""Drive a trained agent (or a model-free baseline) through a live
DareFightingICE match via ``FightingIceEnv``.

Start the JVM game first in another terminal:

    make game-native            # Mac, with rendering (needed for --agent lewm)
    make game                   # Linux docker, headless (state agents)
    make game-pixels            # Linux docker, rendering (lewm)

Then:

    # No model required — validates the env loop end-to-end today:
    uv run python scripts/play.py --agent random --opponent MctsAi23i --games 1

    # Once checkpoints exist (train on the other machine, copy into data/):
    uv run python scripts/play.py --agent pets --ckpt data/pets_checkpoint.pt
    uv run python scripts/play.py --agent lewm --ckpt data/lewm_heads_checkpoint.pt

Two evaluation modes (``--pace``):

* ``sync`` (frame-by-frame / n-by-n): the JVM waits for the agent each
  decision (launch the server with ``--input-sync``). The agent gets
  unlimited time, so this measures **prediction quality only** — win-rate
  and HP-diff with no clock penalty. Pair with ``--frame-skip N`` for
  n-frame-by-n-frame.
* ``realtime``: each decision has a hard ``16.67 ms × frame_skip`` budget.
  Over-budget decisions arrive too late, so the game keeps the *previous*
  action and the decision is counted as a drop. Measures **overall
  quality** — fast *and* accurate.

Episode == one round. Reports win-rate, HP differential, and inference
frame-drop rate.
"""

from __future__ import annotations

import argparse
import logging
import random
from typing import Any

from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.fightingice_env import EnvConfig, FightingIceEnv
from leworldgaming.utils.timing import FRAME_BUDGET_MS, FrameBudget


class RandomAgent:
    """Model-free baseline so the env loop is exercisable without a checkpoint."""

    def __init__(self, num_actions: int = NUM_ACTIONS, seed: int = 0) -> None:
        self._n = num_actions
        self._rng = random.Random(seed)

    def act(self, obs: dict[str, Any]) -> int:
        return self._rng.randrange(self._n)


def build_agent(name: str, ckpt: str | None, device: str):
    name = name.lower()
    if name == "random":
        return RandomAgent()
    if name == "pets":
        from leworldgaming.agents.pets.agent import PETSAgent

        agent = PETSAgent(device=device)
        if ckpt:
            agent.load(ckpt)
        return agent
    if name == "lewm":
        from leworldgaming.agents.lewm.agent import LewmAgent

        agent = LewmAgent(device=device)
        if ckpt:
            agent.load(ckpt)
        agent.warmup()
        return agent
    if name == "dreamer":
        raise SystemExit(
            "Dreamer online play needs DreamerAgent.act() (still a stub). "
            "Use --agent pets/lewm/random for now."
        )
    raise SystemExit(f"Unknown agent: {name!r}. Choose: random, pets, lewm, dreamer.")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--agent", default="random", help="random | pets | lewm | dreamer")
    p.add_argument("--ckpt", default=None, help="checkpoint path (omit for --agent random)")
    p.add_argument("--opponent", default="MctsAi23i",
                   help="JVM AI class name (MctsAi23i, MctsAiZoning) or a python policy "
                        "(random, aggressive, defensive, ...)")
    p.add_argument("--player", default="P1", choices=["P1", "P2"])
    p.add_argument("--character", default="ZEN")
    p.add_argument("--games", type=int, default=1)
    p.add_argument("--pace", default="sync", choices=["sync", "realtime"],
                   help="sync: game waits for the agent (predict-only quality). "
                        "realtime: enforce the per-decision frame budget (overall quality).")
    p.add_argument("--frame-skip", type=int, default=None,
                   help="decide every N frames; defaults to the LeWM checkpoint's "
                        "temporal_stride for LeWM and 1 for other agents")
    p.add_argument("--obs-mode", default="auto", choices=["auto", "state", "pixel"],
                   help="auto: pixel for lewm, state otherwise")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=31415)
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    obs_mode = args.obs_mode
    if obs_mode == "auto":
        obs_mode = "pixel" if args.agent.lower() == "lewm" else "state"

    agent = build_agent(args.agent, args.ckpt, args.device)
    if args.agent.lower() == "lewm":
        expected_stride = int(agent.temporal_stride)
        if args.frame_skip is None:
            args.frame_skip = expected_stride
        elif args.frame_skip != expected_stride:
            raise SystemExit(
                f"LeWM checkpoint was trained with temporal_stride={expected_stride}, "
                f"but --frame-skip={args.frame_skip}. These must match."
            )
    elif args.frame_skip is None:
        args.frame_skip = 1

    env = FightingIceEnv(EnvConfig(
        host=args.host, port=args.port, character=args.character,
        obs_mode=obs_mode, agent_player=args.player, opponent=args.opponent,
        games=args.games, image_size=args.image_size, seed=args.seed,
        frame_skip=args.frame_skip,
    ))

    # In realtime mode a decision has frame_skip frames' worth of wall-clock.
    budget = FrameBudget(budget_ms=FRAME_BUDGET_MS * max(1, args.frame_skip))
    realtime = args.pace == "realtime"
    rounds = wins = 0
    hp_diffs: list[float] = []

    try:
        while True:
            obs, info = env.reset()
            if obs is None:  # match over
                break
            if hasattr(agent, "reset_episode"):
                agent.reset_episode()
            ep_reward = 0.0
            last_action = 0  # NEUTRAL fallback if the very first decision is late
            done = False
            while not done:
                with budget:
                    action = agent.act(obs)
                # Realtime: a late decision misses its slot — the game keeps the
                # previous action. Sync: always apply the fresh decision.
                late = realtime and budget.last_ms > budget.budget_ms
                send = last_action if late else action
                last_action = send
                obs, reward, terminated, truncated, info = env.step(send)
                ep_reward += reward
                done = terminated or truncated
            rounds += 1
            if info.get("win"):
                wins += 1
            if "hp_self" in info and "hp_opp" in info:
                hp_diffs.append(float(info["hp_self"]) - float(info["hp_opp"]))
            logging.info(
                "round %d done | reward=%.3f | hp_self=%s hp_opp=%s win=%s",
                rounds, ep_reward, info.get("hp_self"), info.get("hp_opp"), info.get("win"),
            )
    finally:
        env.close()

    avg_hp_diff = sum(hp_diffs) / len(hp_diffs) if hp_diffs else 0.0
    mean_ms = sum(budget.history_ms) / len(budget.history_ms) if budget.history_ms else 0.0
    drop_note = "enforced" if realtime else "informational — game waited"
    logging.info("=" * 60)
    logging.info("agent=%s opponent=%s pace=%s frame_skip=%d rounds=%d",
                 args.agent, args.opponent, args.pace, args.frame_skip, rounds)
    logging.info("win-rate      : %.1f%% (%d/%d)",
                 100.0 * wins / rounds if rounds else 0.0, wins, rounds)
    logging.info("avg HP diff   : %+.1f", avg_hp_diff)
    logging.info("frame budget  : %.1f%% drops (%d/%d over %.2f ms, mean %.2f ms) [%s]",
                 100.0 * budget.drop_rate, budget.drops, budget.total,
                 budget.budget_ms, mean_ms, drop_note)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
