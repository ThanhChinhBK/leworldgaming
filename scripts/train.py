"""Mid-week training driver. Dispatches to LeWM or Dreamer training loops."""

from __future__ import annotations

import argparse


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=["lewm", "dreamer"], default="lewm")
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--config", type=str, default=None)
    args = parser.parse_args()

    if args.agent == "lewm":
        from leworldgaming.training.train_lewm import train

        train(num_steps=args.steps)
    else:
        from leworldgaming.training.train_dreamer import train

        train(num_steps=args.steps)


if __name__ == "__main__":
    main()
