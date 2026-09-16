"""Side-balanced, synchronous head-to-head evaluation of three trained agents.

This is an end-to-end controller comparison, not the shared-planner four-arm
world-model experiment in the original benchmark plan.
"""

from __future__ import annotations

import itertools
import json
import os
import subprocess
from pathlib import Path
from typing import Any

from leworldgaming.eval.results import (
    checkpoint_identity,
    code_identity,
    dreamer_action_alignment,
    write_result,
)


def tournament_schedule(seeds: list[int], *, lewm_p1_only: bool = False) -> list[dict[str, Any]]:
    if not seeds or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least one seed, without duplicates.")
    schedule = [
        {"p1": p1, "p2": p2, "seed": seed}
        for seed in seeds
        for a, b in itertools.combinations(("lewm", "dreamer", "pets"), 2)
        for p1, p2 in ((a, b), (b, a))
    ]
    if lewm_p1_only:
        return [match for match in schedule if match["p2"] != "lewm"]
    return schedule


def summarize(records: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    totals: dict[str, dict[str, Any]] = {}
    for record in records:
        args = record["arguments"]
        rounds = record["result"]["rounds"]
        if not rounds:
            raise ValueError("Cannot summarize a match without rounds.")
        for own, opp, side, hp, hp_opp in (
            (args["p1"], args["p2"], "P1", "hp_p1", "hp_p2"),
            (args["p2"], args["p1"], "P2", "hp_p2", "hp_p1"),
        ):
            row = totals.setdefault(f"{own}_vs_{opp}", {
                "rounds": 0, "wins": 0, "losses": 0, "draws": 0, "hp_diff_sum": 0.0,
            })
            for outcome in rounds:
                winner = outcome["winner"]
                if winner not in {"P1", "P2", "draw"} or outcome[hp] is None or outcome[hp_opp] is None:
                    raise ValueError("Incomplete round outcome cannot count as a draw.")
                row["rounds"] += 1
                row["wins"] += winner == side
                row["draws"] += winner == "draw"
                row["losses"] += winner not in {side, "draw"}
                row["hp_diff_sum"] += outcome[hp] - outcome[hp_opp]
    for row in totals.values():
        row["win_rate"] = row["wins"] / row["rounds"]
        row["mean_hp_diff"] = row.pop("hp_diff_sum") / row["rounds"]
    return totals


def run_tournament(
    *,
    checkpoints: dict[str, str],
    output_dir: Path,
    seeds: list[int],
    games: int = 3,
    device: str = "cuda",
    character: str = "ZEN",
    dry_run: bool = False,
    allow_legacy_dreamer: bool = False,
    allow_unvalidated_lewm_p2: bool = False,
    pace: str = "sync",
    lewm_p1_only: bool = False,
) -> dict[str, Any]:
    if set(checkpoints) != {"lewm", "dreamer", "pets"}:
        raise ValueError("Exactly one trained checkpoint per model is required.")
    if games < 1:
        raise ValueError("games must be positive.")
    if pace not in {"sync", "realtime"}:
        raise ValueError("pace must be 'sync' or 'realtime'.")
    if not lewm_p1_only:
        raise ValueError("LeWM P2 pixel inference is unvalidated; use lewm_p1_only=True.")
    schedule = tournament_schedule(seeds, lewm_p1_only=True)
    root = Path(__file__).resolve().parents[3]
    output_dir = output_dir.resolve()
    identities = {name: checkpoint_identity(path) for name, path in checkpoints.items()}
    alignment = dreamer_action_alignment(checkpoints["dreamer"])
    blockers = []
    if alignment != "incoming_action_v1" and not allow_legacy_dreamer:
        blockers.append("Dreamer alignment is legacy/unknown; retrain on corrected exports or "
                        "explicitly label an exploratory run with --allow-legacy-dreamer.")
    plan = {
        "schema_version": 1,
        "evaluation": "head_to_head",
        "pace": pace,
        "control_rate_hz": 60,
        "games_per_match": games,
        "device": device,
        "character": character,
        "checkpoints": identities,
        "code": code_identity(root, exclude=output_dir),
        "jvm_extra": os.environ.get("JVM_EXTRA", ""),
        "schedule": schedule,
        "blockers": blockers,
        "dreamer_action_alignment": alignment,
        "allow_legacy_dreamer": allow_legacy_dreamer,
        "lewm_p1_only": True,
        "scope": "All controllers receive the current observation and choose an action every "
                 "raw game frame in lockstep. LeWM is P1-only and predicts five-frame latent "
                 "transitions conditioned on raw action blocks, but re-runs CEM every frame. "
                 "Not a side-balanced or equal-compute world-model comparison.",
        "reproducibility": "Seeds cover Python/NumPy/Torch, not JVM randomness or "
                           "concurrent GPU scheduling. Raw rounds are retained.",
    }
    if dry_run:
        return plan
    if blockers:
        raise ValueError("Benchmark is not cleared:\n" + "\n".join(blockers))
    output_dir.mkdir(parents=True, exist_ok=False)
    write_result(output_dir / "manifest.json", plan)
    records = []
    for index, match in enumerate(schedule):
        stem = f"{index:03d}_{match['p1']}_vs_{match['p2']}_seed{match['seed']}"
        result_path = output_dir / f"{stem}.json"
        command = [
            "bash", str(root / "scripts/run_eval.sh"), str(output_dir / f"{stem}.log"),
            "--p1", match["p1"], "--p1-ckpt", identities[match["p1"]]["path"],
            "--p2", match["p2"], "--p2-ckpt", identities[match["p2"]]["path"],
            "--seed", str(match["seed"]), "--games", str(games),
            "--device", device, "--character", character, "--output", str(result_path),
            "--pace", pace,
        ]
        if match["p1"] == "lewm":
            command.extend([
                "--p1-frame-skip", "1", "--p1-planner", "cem",
                "--planner-plan-raw-actions", "--planner-chunk-size", "1",
            ])
        if allow_legacy_dreamer:
            command.append("--allow-legacy-dreamer")
        subprocess.run(command, cwd=root, check=True, env={**os.environ, "EVAL_PACE": pace})
        with result_path.open() as stream:
            record = json.load(stream)
        if record["code"] != plan["code"]:
            raise RuntimeError("Source tree changed during tournament; results cannot be combined.")
        for side, name in (("P1", match["p1"]), ("P2", match["p2"])):
            if record["checkpoints"][side] != identities[name]:
                raise RuntimeError(f"Checkpoint changed during tournament: {name}")
        records.append(record)
        # A failed later match leaves successful raw records but never a final summary.
    summary = {"matches": len(records), "results": summarize(records)}
    write_result(output_dir / "summary.json", summary)
    return summary
