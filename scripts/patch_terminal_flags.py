"""One-time patch for a data-collection bug: ``done``/``cont`` were always
written as ``False``/``1`` for every frame (see ``RecordingAI.processing()``
and the old ``ReplayBuffer._do_end_episode()``), because the per-frame write
happens before the recorder knows a frame is the last one of the round.
``replay_buffer.py`` now patches the true terminal frame at ``end_episode()``
time going forward; this script retroactively fixes files collected before
that fix, using the already-correct ``episode_starts`` array to find each
episode's last frame.

Only the ``done``/``cont`` columns are touched, in place — pixels, actions,
rewards, state_vector, and episode_starts are untouched and were never wrong.

Usage:
    uv run python scripts/patch_terminal_flags.py /path/to/replay.h5
    uv run python scripts/patch_terminal_flags.py /path/to/directory/   # all .h5 files
    uv run python scripts/patch_terminal_flags.py --all                 # default DATA_DIR

Note: if DATA_DIR lives on an NTFS mount (e.g. via ntfs-3g/ntfs3), h5py's
r+ open can fail with "BlockingIOError: Resource temporarily unavailable"
because the filesystem doesn't support HDF5's file-locking. Work around it
with ``HDF5_USE_FILE_LOCKING=FALSE uv run python scripts/patch_terminal_flags.py ...``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np

DEFAULT_DATA_DIR = "/media/jeovach/New Volume/leworldgaming"


def patch_file(path: Path, dry_run: bool = False) -> None:
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as f:
        if "episode_starts" not in f or "done" not in f:
            print(f"  {path.name}: missing episode_starts/done — skipping")
            return
        starts = f["episode_starts"][:]
        total = f["done"].shape[0]
        if starts.size == 0 or total == 0:
            print(f"  {path.name}: empty — skipping")
            return

        # Last frame of episode i is (starts[i+1] - 1), or (total - 1) for the
        # final episode.
        last_idxs = np.empty(starts.size, dtype=np.int64)
        last_idxs[:-1] = starts[1:] - 1
        last_idxs[-1] = total - 1

        already_done = f["done"][()][last_idxs]
        n_already = int(already_done.sum())
        n_total = last_idxs.size

        if dry_run:
            print(
                f"  {path.name}: {n_total} episodes, "
                f"{n_total - n_already} frame(s) need patching "
                f"(already correct: {n_already})"
            )
            return

        done_ds = f["done"]
        cont_ds = f["cont"]
        for idx in last_idxs:
            done_ds[idx] = np.uint8(1)
            cont_ds[idx] = np.uint8(0)
        f.flush()
        print(
            f"  {path.name}: patched {n_total} episode terminal frame(s) "
            f"({n_total - n_already} were previously wrong)"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Patch done/cont terminal flags in existing replay HDF5 files"
    )
    parser.add_argument(
        "paths",
        nargs="*",
        help="HDF5 files or directories to patch (default: DATA_DIR)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help=f"Patch all .h5 files in {DEFAULT_DATA_DIR}",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what would change without writing anything",
    )
    args = parser.parse_args()

    files: list[Path] = []
    if args.all or not args.paths:
        data_dir = Path(DEFAULT_DATA_DIR)
        files = sorted(data_dir.glob("*.h5"))
    else:
        for p in args.paths:
            path = Path(p)
            if path.is_dir():
                files.extend(sorted(path.glob("*.h5")))
            elif path.is_file() and path.suffix == ".h5":
                files.append(path)
            else:
                print(f"Skipping: {p} (not a .h5 file or directory)", file=sys.stderr)

    if not files:
        print("No .h5 files found to patch.")
        return

    print(f"Found {len(files)} file(s) to {'check' if args.dry_run else 'patch'}\n")
    for f in files:
        patch_file(f, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
