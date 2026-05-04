"""Mid-week training driver. Dispatches to LeWM, Dreamer, or PETS training loops.

Examples:
    uv run python scripts/train.py --agent lewm --steps 50 --batch-size 8
    uv run python scripts/train.py --agent dreamer --steps 100 --image-size 64
    uv run python scripts/train.py --agent pets --steps 500
    uv run python scripts/train.py --agent lewm --config configs/lewm.yaml
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["lewm", "dreamer", "pets"], default="lewm")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--config", type=str, default=None)
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

        train(num_steps=args.steps,
              config_path=args.config or "configs/lewm.yaml", **overrides)
    elif args.agent == "dreamer":
        from leworldgaming.training.train_dreamer import train

        # Dreamer doesn't use sigreg_lambda; drop it to avoid a stray override.
        overrides.pop("sigreg_lambda", None)
        overrides.pop("lr", None)  # use model_lr / actor_lr / critic_lr instead
        train(num_steps=args.steps,
              config_path=args.config or "configs/dreamer.yaml", **overrides)
    else:  # pets
        from leworldgaming.training.train_pets import train

        overrides.pop("sigreg_lambda", None)
        train(num_steps=args.steps,
              config_path=args.config or "configs/pets.yaml", **overrides)


if __name__ == "__main__":
    main()
