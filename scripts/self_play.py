"""Pit two trained agents (or baselines) against each other directly.

Unlike ``scripts/play.py`` (one python agent vs a JVM AI / scripted policy),
both sides here are trained agents running their own inference — useful for
comparing LeWM vs Dreamer vs PETS head-to-head.

Start the JVM game first (rendering required if either side is ``lewm``):

    make game-native-linux       # Linux, native, GPU rendering
    make game-pixels             # Linux, docker, rendering
    make game                    # Linux, docker, headless (state agents only)

Then, e.g. LeWM (P1) vs Dreamer (P2):

    uv run python scripts/self_play.py \\
        --p1 lewm --p1-ckpt data/lewm_heads_checkpoint.pt \\
        --p2 dreamer --p2-ckpt data/dreamer_checkpoint.pt \\
        --games 3 --device cuda

Or PETS vs LeWM, or dreamer vs dreamer, etc. — any combination of
random | pets | lewm | dreamer works on both sides.

LeWM planner selection (``--p1-planner``/``--p2-planner``; ignored for
non-LeWM agents): ``random`` is one-shot random shooting (the planner used
in the original JEPA-planning paper); ``cem`` is the iCEM-style iterative
elite-refinement + warm-start planner (default). E.g. to reproduce the
paper's baseline planner for LeWM vs Dreamer:

    uv run python scripts/self_play.py \\
        --p1 lewm --p1-ckpt data/lewm_heads_checkpoint.pt --p1-planner random \\
        --p2 dreamer --p2-ckpt data/dreamer_checkpoint.pt \\
        --games 3 --device cuda
"""

from __future__ import annotations

import argparse
import logging

# Reuse the single-agent script's checkpoint-loading logic instead of
# duplicating it.
from play import build_agent  # noqa: E402  (sys.path shim below)

from leworldgaming.env.agent_vs_agent import run_match


def _obs_mode_for(agent_name: str) -> str:
    return "pixel" if agent_name.lower() == "lewm" else "state"


def _frame_skip_for(agent_name: str, agent, override: int | None) -> int:
    if override is not None:
        return override
    if agent_name.lower() == "lewm":
        return int(agent.temporal_stride)
    return 1


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--p1", default="random", help="random | pets | lewm | dreamer")
    p.add_argument("--p1-ckpt", default=None)
    p.add_argument("--p1-frame-skip", type=int, default=None)
    p.add_argument("--p1-planner", default=None, choices=[None, "random", "cem", "mcts"],
                   help="LeWM only. 'random' = original JEPA-planning-paper baseline "
                        "(one-shot random shooting); 'cem' = iCEM-style planner "
                        "(default set by the checkpoint's own config, normally 'cem').")
    p.add_argument("--p2", default="random", help="random | pets | lewm | dreamer")
    p.add_argument("--p2-ckpt", default=None)
    p.add_argument("--p2-frame-skip", type=int, default=None)
    p.add_argument("--p2-planner", default=None, choices=[None, "random", "cem", "mcts"],
                   help="Same as --p1-planner, applied to P2.")
    p.add_argument("--planner-horizon", type=int, default=None,
                   help="Applied to both sides' LeWM planner (if either is lewm). "
                        "Rollout length in decision-blocks (default 8); bigger = more "
                        "lookahead but linearly more latency.")
    p.add_argument("--planner-samples", type=int, default=None,
                   help="Candidate action sequences per iteration (default 24). "
                        "Measured on an RTX 5060 Ti: samples=64/iters=3/horizon=5 "
                        "~70ms/decision vs samples=24/iters=1/horizon=8 ~38ms/decision, "
                        "against an ~83ms real-time budget at temporal_stride=5 (60fps).")
    p.add_argument("--planner-iters", type=int, default=None,
                   help="CEM only: number of elite-refinement iterations (default 1). "
                        "Cost scales ~linearly with iters too.")
    p.add_argument("--planner-sticky-prob", type=float, default=None,
                   help="CEM only: prob. a rollout step repeats the previous step's action")
    p.add_argument("--planner-momentum", type=float, default=None,
                   help="CEM only: how much of the previous decision's warm-started "
                        "distribution carries over vs. the freshly-refit one (default "
                        "0.3 at stride=5, 0.1 at stride=2). Higher = slower to correct "
                        "a locked-in action; see planner.cem_shooting docstring point 3.")
    p.add_argument("--planner-min-prob", type=float, default=None,
                   help="CEM only: uniform exploration floor mixed into the refit "
                        "per-timestep distribution each iteration (default 0.02 at "
                        "stride=5, 0.05 at stride=2). Higher = faster escape from a "
                        "bad action lock-in at the cost of noisier action choice.")
    p.add_argument("--planner-use-cont-head", action="store_true", default=None,
                   help="Re-enable the (known miscalibrated) continuation head for "
                        "rollout discounting instead of a constant gamma")
    p.add_argument("--planner-allow-state-actions", action="store_true", default=None,
                   help="DON'T restrict sampling to the ~40 commandable actions; also "
                        "let the planner sample the ~16 unplayable 'state observation' "
                        "Action values CommandCenter can't execute. Off by default.")
    p.add_argument("--planner-no-value-head", action="store_true", default=None,
                   help="Disable the value-head bootstrap term (use only the finite-"
                        "horizon reward sum), e.g. if the value head is overfit/miscalibrated.")
    p.add_argument("--planner-uncertainty-penalty", type=float, default=None,
                   help="Pessimistic CEM scoring coefficient k: score uses "
                        "mean - k*std across a reward/value head ensemble "
                        "(only has an effect if the checkpoint's heads were "
                        "trained with reward_ensemble_size/value_ensemble_size > 1). "
                        "0.0 (default) = plain mean, no pessimism.")
    p.add_argument("--planner-chunk-size", type=int, default=None,
                   help="Only re-run the full CEM search once every N "
                        "decisions; the other N-1 decisions reuse actions "
                        "already sampled from that replan's multi-step "
                        "action distribution (near-zero extra compute -- "
                        "skips predictor/head rollout entirely). Does NOT "
                        "change how often the environment requests an "
                        "action (still every frame_skip*16.67ms) or relax "
                        "its per-call deadline -- only the 1-in-N replan "
                        "calls still do the expensive search and can still "
                        "be dropped if they overrun. 1 (default) = replan "
                        "every decision (original behavior).")
    p.add_argument("--planner-plan-raw-actions", action="store_true", default=None,
                   help="Search over temporal_stride genuinely distinct "
                        "actions per planned block instead of assuming one "
                        "action is held for the whole block (see "
                        "planner.cem_shooting's plan_raw_actions doc). Only "
                        "meaningful combined with --planner-chunk-size == "
                        "temporal_stride and --p1-frame-skip/--p2-frame-skip "
                        "1 (env must request a decision every raw frame, "
                        "otherwise frame_skip repetition flattens the "
                        "distinct actions back into one held action).")
    p.add_argument("--planner-num-simulations", type=int, default=None,
                   help="MCTS only: number of tree simulations per decision "
                        "(default 24). See mcts_planner.mcts_search.")
    p.add_argument("--planner-max-depth", type=int, default=None,
                   help="MCTS only: max tree depth per simulation (default = "
                        "--planner-horizon).")
    p.add_argument("--planner-c-puct", type=float, default=None,
                   help="MCTS only: PUCT exploration constant (default 1.25).")
    p.add_argument("--planner-dirichlet-frac", type=float, default=None,
                   help="MCTS only: fraction of Dirichlet noise blended into "
                        "the root's prior for exploration (default 0.25, 0 "
                        "disables).")
    p.add_argument("--planner-sim-batch-size", type=int, default=None,
                   help="MCTS only: simulations per virtual-loss-batched wave "
                        "(default 16). See mcts_planner module docstring "
                        "('Wave/virtual-loss batching').")
    p.add_argument("--planner-virtual-loss", type=float, default=None,
                   help="MCTS only: pessimistic bias applied to an edge's W "
                        "for the rest of its wave once claimed (default 1.0).")
    p.add_argument("--character", default="ZEN")
    p.add_argument("--games", type=int, default=1,
                   help="number of games to request from the JVM (each = several rounds)")
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=31415)
    p.add_argument("--device", default="cpu", help="cpu | mps | cuda (used for both agents)")
    p.add_argument("--debug", action="store_true",
                   help="Enable per-decision DEBUG logs from agent_vs_agent "
                        "(frame number + chosen action per agent). Also logs "
                        "a WARNING whenever either AI holds a blank key for "
                        ">=30 consecutive frames (~0.5s), naming the reason "
                        "(waiting on frame data, missing character, agent "
                        "exception, uncommandable action, or a genuine "
                        "NEUTRAL choice) -- use this to diagnose 'bot does "
                        "nothing' symptoms.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=logging.DEBUG if args.debug else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    shared_planner_kwargs = dict(
        horizon=args.planner_horizon,
        num_samples=args.planner_samples,
        num_iters=args.planner_iters,
        sticky_prob=args.planner_sticky_prob,
        momentum=args.planner_momentum,
        min_prob=args.planner_min_prob,
        use_continuation_head=args.planner_use_cont_head,
        use_value_head=(
            None if args.planner_no_value_head is None
            else not args.planner_no_value_head
        ),
        restrict_to_playable_actions=(
            None if args.planner_allow_state_actions is None
            else not args.planner_allow_state_actions
        ),
        uncertainty_penalty=args.planner_uncertainty_penalty,
        chunk_size=args.planner_chunk_size,
        plan_raw_actions=args.planner_plan_raw_actions,
        num_simulations=args.planner_num_simulations,
        max_depth=args.planner_max_depth,
        c_puct=args.planner_c_puct,
        dirichlet_frac=args.planner_dirichlet_frac,
        sim_batch_size=args.planner_sim_batch_size,
        virtual_loss=args.planner_virtual_loss,
    )
    p1_agent = build_agent(args.p1, args.p1_ckpt, args.device,
                            planner=args.p1_planner, **shared_planner_kwargs)
    p2_agent = build_agent(args.p2, args.p2_ckpt, args.device,
                            planner=args.p2_planner, **shared_planner_kwargs)
    if hasattr(p1_agent, "warmup"):
        p1_agent.warmup()
    if hasattr(p2_agent, "warmup"):
        p2_agent.warmup()

    result = run_match(
        p1_agent, p2_agent,
        p1_name=f"P1_{args.p1.upper()}", p2_name=f"P2_{args.p2.upper()}",
        p1_obs_mode=_obs_mode_for(args.p1), p2_obs_mode=_obs_mode_for(args.p2),
        p1_frame_skip=_frame_skip_for(args.p1, p1_agent, args.p1_frame_skip),
        p2_frame_skip=_frame_skip_for(args.p2, p2_agent, args.p2_frame_skip),
        host=args.host, port=args.port, character=args.character,
        games=args.games, image_size=args.image_size,
    )

    logging.info("=" * 60)
    logging.info("P1=%s (%s)  vs  P2=%s (%s)", args.p1, args.p1_ckpt, args.p2, args.p2_ckpt)
    for r in result.rounds:
        logging.info("round %d: hp_p1=%s hp_p2=%s winner=%s",
                      r.round_index, r.hp_p1, r.hp_p2, r.winner)
    n = len(result.rounds)
    logging.info("rounds=%d  P1 wins=%d (%.1f%%)  P2 wins=%d (%.1f%%)  draws=%d",
                 n, result.wins_p1, 100.0 * result.wins_p1 / n if n else 0.0,
                 result.wins_p2, 100.0 * result.wins_p2 / n if n else 0.0,
                 n - result.wins_p1 - result.wins_p2)
    for name, stats in result.latency.items():
        logging.info(
            "latency %-14s: mean=%.1fms p95=%.1fms budget=%.1fms drop_rate=%.1f%% (n=%d)",
            name, stats["mean_ms"], stats["p95_ms"], stats["budget_ms"],
            100.0 * stats["drop_rate"], int(stats["total"]),
        )
    logging.info("=" * 60)


if __name__ == "__main__":
    main()
