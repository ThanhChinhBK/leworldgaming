"""Run or preview the P1-fixed-LeWM, 60 Hz head-to-head benchmark."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from leworldgaming.eval.tournament import run_tournament


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("lewm", "dreamer", "pets"):
        parser.add_argument(f"--{name}-ckpt", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--games", type=int, default=3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--character", default="ZEN")
    parser.add_argument("--pace", default="sync", choices=["sync", "realtime"])
    parser.add_argument("--dry-run", action="store_true",
                        help="Print checkpoint identities, readiness blockers and schedule without starting games.")
    parser.add_argument("--allow-legacy-dreamer", action="store_true",
                        help="Exploratory runs only: acknowledge legacy/unknown Dreamer action alignment.")
    args = parser.parse_args()
    result = run_tournament(
        checkpoints={name: getattr(args, f"{name}_ckpt") for name in ("lewm", "dreamer", "pets")},
        output_dir=args.out_dir, seeds=args.seeds, games=args.games,
        device=args.device, character=args.character, dry_run=args.dry_run,
        allow_legacy_dreamer=args.allow_legacy_dreamer,
        pace=args.pace,
        lewm_p1_only=True,
    )
    print(json.dumps(result, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
