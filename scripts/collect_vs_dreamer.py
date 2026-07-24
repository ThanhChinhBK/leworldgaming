"""Collect opponent-action-labelled data by recording P1's perspective of a
live match against a fixed opponent (typically Dreamer, the actual target
opponent LeWM needs to beat) — not a JVM built-in AI or a scripted policy.

``scripts/collect_data.py`` can only record games against the built-in JVM
AIs (MctsAi23i / MctsAiZoning) because it drives games through pyftg's
single-AI ``AIController`` + JVM AI-name resolution path. It cannot target
an arbitrary Python-side opponent like a loaded DreamerV3 checkpoint.

This script instead uses ``leworldgaming.env.agent_vs_agent.run_match``
(agent-vs-agent, both sides self-driving) with its new ``record_buffer``
parameter: P1's transitions (own action AND the opponent's true executed
action, via ``frame_to_obs_dict``'s existing ``obs/opp/action`` field) are
written to a ``ReplayBuffer`` while P1 and P2 play a real match.

Typical use — record a scripted/self-play-diverse P1 policy against Dreamer
(P2), so the resulting `obs/opp/action` column reflects Dreamer's actual
behavior distribution instead of the old corpus's MCTS/scripted-vs-scripted
opponents:

    uv run python scripts/collect_vs_dreamer.py \\
        --p1 mixed --p2-ckpt data/dreamer_checkpoint.pt \\
        --games 40 --out /media/jeovach/Hoctap/leword-opponent/01_mixed_v_dreamer.h5

``--p1`` accepts the same names as ``play.py --agent`` (random, pets, lewm,
dreamer) OR one of the scripted Python policies (random, noop, aggressive,
defensive, mixed) — the scripted ones are cheap/fast and diverse, useful for
covering a wide state distribution against Dreamer without needing a slow
LeWM-side planner call every frame.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from play import build_agent  # noqa: E402

from leworldgaming.data.replay_buffer import BufferConfig, ReplayBuffer  # noqa: E402
from leworldgaming.env.agent_vs_agent import run_match  # noqa: E402
from leworldgaming.env.policies import make_policy  # noqa: E402

SCRIPTED_POLICIES = {"random", "noop", "neutral", "aggressive", "defensive", "mixed"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p1", default="mixed",
                    help="random | pets | lewm | dreamer | (scripted: random, noop, "
                         "aggressive, defensive, mixed)")
    p.add_argument("--p1-ckpt", default=None)
    p.add_argument("--p1-frame-skip", type=int, default=None)
    p.add_argument("--p2", default="dreamer", help="usually 'dreamer'")
    p.add_argument("--p2-ckpt", default="data/dreamer_checkpoint.pt")
    p.add_argument("--p2-frame-skip", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=31415)
    p.add_argument("--character", default="ZEN")
    p.add_argument("--games", type=int, default=20)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--pixels", action="store_true",
                    help="Also record downsampled RGB frames (needed to train/retrain "
                         "LeWM's encoder-derived heads on this data). Requires the JVM "
                         "to be in render mode (native Linux/macOS, or docker MODE=pixels).")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO,
                         format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    args = parse_args()

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    if args.p1.lower() in SCRIPTED_POLICIES:
        from scripted_frame_agent import ScriptedFrameAgent  # noqa: E402

        p1_agent = ScriptedFrameAgent(make_policy(args.p1.lower(), seed=args.seed))
        p1_obs_mode = "state"  # unused by ScriptedFrameAgent but required by run_match
        p1_frame_skip = args.p1_frame_skip or 1
    else:
        p1_agent = build_agent(args.p1, args.p1_ckpt, args.device)
        p1_obs_mode = "pixel" if args.p1.lower() == "lewm" else "state"
        p1_frame_skip = args.p1_frame_skip or (
            int(p1_agent.temporal_stride) if args.p1.lower() == "lewm" else 1
        )

    p2_agent = build_agent(args.p2, args.p2_ckpt, args.device)
    p2_obs_mode = "pixel" if args.p2.lower() == "lewm" else "state"
    p2_frame_skip = args.p2_frame_skip or (
        int(p2_agent.temporal_stride) if args.p2.lower() == "lewm" else 1
    )

    pixel_shape = (3, args.image_size, args.image_size) if args.pixels else None
    cfg = BufferConfig(path=args.out, pixel_shape=pixel_shape)
    buffer = ReplayBuffer(cfg)
    buffer.open()

    logging.info("Recording P1=%s (perspective) vs P2=%s -> %s", args.p1, args.p2, args.out)
    try:
        result = run_match(
            p1_agent, p2_agent,
            p1_name="P1_REC", p2_name="P2_OPP",
            p1_obs_mode=p1_obs_mode, p2_obs_mode=p2_obs_mode,
            p1_frame_skip=p1_frame_skip, p2_frame_skip=p2_frame_skip,
            host=args.host, port=args.port, character=args.character,
            games=args.games, image_size=args.image_size,
            record_buffer=buffer, record_pixels=args.pixels,
        )
    finally:
        buffer.close()

    logging.info("=" * 60)
    for r in result.rounds:
        logging.info("round %d: hp_p1=%s hp_p2=%s winner=%s",
                      r.round_index, r.hp_p1, r.hp_p2, r.winner)
    logging.info("rounds=%d  P1 wins=%d (%.1f%%)  P2 wins=%d (%.1f%%)",
                 len(result.rounds), result.wins_p1,
                 100 * result.wins_p1 / max(1, len(result.rounds)),
                 result.wins_p2,
                 100 * result.wins_p2 / max(1, len(result.rounds)))
    logging.info("buffer: %d transitions, %d episodes -> %s",
                 len(buffer), buffer.num_episodes, args.out)
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
