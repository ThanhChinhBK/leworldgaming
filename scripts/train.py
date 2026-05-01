"""Mid-week training driver. Dispatches to LeWM or Dreamer training loops.

Examples:
    uv run python scripts/train.py --agent lewm --steps 50 --batch-size 8
    uv run python scripts/train.py --agent lewm --config configs/lewm.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["lewm", "dreamer"], default="lewm")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--config", type=str, default="configs/lewm.yaml")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    overrides = {
        "batch_size": args.batch_size,
        "lr": args.lr,
        "sigreg_lambda": args.sigreg_lambda,
        "data_path": args.data_path,
        "ckpt_path": args.ckpt_path,
        "seed": args.seed,
    }

    if args.agent == "lewm":
        from leworldgaming.training.train_lewm import train

        train(num_steps=args.steps, config_path=args.config, **overrides)
    else:
        from leworldgaming.training.train_dreamer import train

        train(num_steps=args.steps)


if __name__ == "__main__":
    main()
