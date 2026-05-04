"""Dump an HDF5 replay buffer into a human-browsable folder.

Writes:
    <out>/frames/NNNNN.png      — pixel frames (one per recorded frame)
    <out>/metadata.csv          — per-frame state info (action, reward, HP, ...)
    <out>/summary.txt           — top-level stats

Usage:
    uv run python scripts/extract_replay.py
    uv run python scripts/extract_replay.py --in data/replay.h5 --out data/extracted
    uv run python scripts/extract_replay.py --stride 30   # every 30th frame
    uv run python scripts/extract_replay.py --max 200     # cap at 200 frames
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from PIL import Image
from pyftg.models.enums.action import Action


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="src", type=str, default="data/replay.h5")
    p.add_argument("--out", type=str, default="data/extracted")
    p.add_argument("--stride", type=int, default=1,
                   help="Save every Nth frame (default 1 = all frames)")
    p.add_argument("--max", type=int, default=None,
                   help="Cap number of frames written")
    p.add_argument("--no-pixels", action="store_true",
                   help="Skip PNG export, only write metadata.csv")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    frames_dir = out / "frames"
    frames_dir.mkdir(exist_ok=True)

    if not src.exists():
        raise SystemExit(f"input file not found: {src}")

    with h5py.File(src, "r") as f:
        n_total = f["action"].shape[0]
        n_episodes = f["episode_starts"].shape[0]
        episode_starts = f["episode_starts"][:].tolist()
        has_pixels = "pixels" in f and not args.no_pixels

        print(f"reading {src}: {n_total} transitions, {n_episodes} episodes")
        if has_pixels:
            print(f"  pixels: {f['pixels'].shape} {f['pixels'].dtype}")
        else:
            print("  pixels: none (state-vector-only buffer)" if "pixels" not in f
                  else "  pixels: skipped (--no-pixels)")

        # Decide which indices to export.
        indices = list(range(0, n_total, args.stride))
        if args.max is not None:
            indices = indices[: args.max]

        # Per-frame metadata, read from named-group primitives.
        action_arr = f["action"][:]
        reward_arr = f["reward"][:]
        hp_self_arr = f["obs/own/hp"][:]
        hp_opp_arr = f["obs/opp/hp"][:]
        frame_idx_arr = f["obs/global/current_frame"][:]
        is_first_arr = f["is_first"][:]

        # Write CSV.
        csv_path = out / "metadata.csv"
        with csv_path.open("w", newline="") as fp:
            writer = csv.writer(fp)
            writer.writerow([
                "row_idx", "frame_idx", "episode",
                "action_id", "action_name", "reward",
                "hp_self", "hp_opp", "hp_diff",
                "is_first", "png_path",
            ])
            for i in indices:
                ep = sum(1 for s in episode_starts if s <= i) - 1
                a_id = int(action_arr[i])
                try:
                    a_name = Action.from_int(a_id).name
                except Exception:  # noqa: BLE001
                    a_name = f"UNKNOWN({a_id})"
                png_rel = f"frames/{i:06d}.png" if has_pixels else ""
                writer.writerow([
                    i, int(frame_idx_arr[i]), ep,
                    a_id, a_name, f"{float(reward_arr[i]):+.4f}",
                    int(hp_self_arr[i]), int(hp_opp_arr[i]),
                    int(hp_self_arr[i]) - int(hp_opp_arr[i]),
                    int(is_first_arr[i]),
                    png_rel,
                ])
        print(f"wrote {csv_path} ({len(indices)} rows)")

        # Write PNGs.
        if has_pixels:
            pixels = f["pixels"]
            for n_done, i in enumerate(indices):
                arr = pixels[i]  # (3, H, W) uint8
                img = Image.fromarray(arr.transpose(1, 2, 0))  # CHW -> HWC
                img.save(frames_dir / f"{i:06d}.png")
                if (n_done + 1) % 200 == 0:
                    print(f"  {n_done+1}/{len(indices)} frames written…")
            print(f"wrote {len(indices)} PNGs to {frames_dir}")

        # Summary.
        summary = (
            f"source:        {src}\n"
            f"transitions:   {n_total}\n"
            f"episodes:      {n_episodes} (starts: {episode_starts})\n"
            f"action range:  [{int(action_arr.min())}, {int(action_arr.max())}]\n"
            f"unique acts:   {len(np.unique(action_arr))}\n"
            f"reward sum:    {float(reward_arr.sum()):+.4f}\n"
            f"reward range:  [{float(reward_arr.min()):+.4f}, {float(reward_arr.max()):+.4f}]\n"
            f"hp_self:       start={int(hp_self_arr[0])}, end={int(hp_self_arr[-1])}, "
            f"min={int(hp_self_arr.min())}\n"
            f"hp_opp:        start={int(hp_opp_arr[0])}, end={int(hp_opp_arr[-1])}, "
            f"min={int(hp_opp_arr.min())}\n"
            f"is_first sum:  {int(is_first_arr.sum())} (should ≈ #episodes)\n"
            f"exported:      {len(indices)} frames "
            f"(stride={args.stride}, max={args.max})\n"
        )
        (out / "summary.txt").write_text(summary)
        print("\n" + summary)


if __name__ == "__main__":
    main()
