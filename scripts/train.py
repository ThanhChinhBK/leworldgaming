"""Mid-week training driver. Dispatches to LeWM (Stage A or B), Dreamer, or PETS.

``--data-path`` accepts either a single ``.h5`` file or a directory of them.
When a directory is given, all ``*.h5`` files inside are loaded and sampled
from jointly during training.

Examples:
    uv run python scripts/train.py --agent lewm --steps 50 --batch-size 8
    uv run python scripts/train.py --agent lewm --stage b --steps 20000
    uv run python scripts/train.py --agent dreamer --steps 100
    uv run python scripts/train.py --agent pets --steps 500
    uv run python scripts/train.py --agent pets --data-path /path/to/h5_folder/
"""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["lewm", "dreamer", "pets"], default="lewm")
    parser.add_argument(
        "--stage",
        choices=["a", "b"],
        default="a",
        help="LeWM only: 'a' = JEPA pretraining (default), 'b' = head training on top of a Stage-A ckpt.",
    )
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--sigreg-lambda", type=float, default=None)
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--ckpt-in", type=str, default=None,
                        help="LeWM Stage B: path to Stage-A checkpoint to load (overrides config).")
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

    if args.agent == "lewm" and args.stage == "b":
        from leworldgaming.training.train_lewm_heads import train

        # Stage B uses ckpt_in / ckpt_out, not the Stage-A ckpt_path.
        overrides.pop("sigreg_lambda", None)
        overrides.pop("ckpt_path", None)
        if args.ckpt_in is not None:
            overrides["ckpt_in"] = args.ckpt_in
        if args.ckpt_path is not None:
            overrides["ckpt_out"] = args.ckpt_path
        train(num_steps=args.steps,
              config_path=args.config or "configs/lewm_heads.yaml", **overrides)
    elif args.agent == "lewm":
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
