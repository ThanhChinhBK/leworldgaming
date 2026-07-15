"""Repair requested-vs-executed action labels in existing replay files.

Older ``RecordingAI`` versions wrote the policy's newly requested action even
while ``CommandCenter`` was still executing a multi-frame command and ignored
that request. This script reconstructs the active high-level command from the
recorded request stream and stores the original labels in
``action_requested`` before replacing ``action``.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import h5py
import numpy as np
from pyftg.aiinterface.command_center import CommandCenter
from pyftg.models.enums.action import Action

DEFAULT_DATA_DIR = "/media/jeovach/New Volume/leworldgaming"


class _FrontFacingFrame:
    @staticmethod
    def is_front(player_number: bool) -> bool:
        return True


def _command_lengths() -> dict[int, int]:
    lengths: dict[int, int] = {}
    for action_id in range(56):
        action = Action.from_int(action_id)
        command_center = CommandCenter()
        command_center.set_frame_data(_FrontFacingFrame(), True)
        command_center.command_call(action.name)
        lengths[action_id] = max(1, len(command_center.get_skill_keys()))
    return lengths


def _reconstruct_executed_actions(
    requested: np.ndarray,
    command_lengths: dict[int, int],
) -> np.ndarray:
    executed = requested.copy()
    active_action = int(Action.NEUTRAL.to_int())
    remaining_keys = 0
    for index, requested_action in enumerate(requested):
        if remaining_keys == 0:
            active_action = int(requested_action)
            remaining_keys = command_lengths[active_action]
        executed[index] = active_action
        remaining_keys -= 1
    return executed


def patch_file(path: Path, dry_run: bool, command_lengths: dict[int, int]) -> None:
    mode = "r" if dry_run else "r+"
    with h5py.File(path, mode) as replay:
        if "action" not in replay:
            print(f"  {path.name}: missing action — skipping")
            return
        source_name = "action_requested" if "action_requested" in replay else "action"
        requested = replay[source_name][:]
        executed = _reconstruct_executed_actions(requested, command_lengths)
        changed = int(np.count_nonzero(executed != requested))
        percent = 100.0 * changed / max(requested.size, 1)
        if dry_run:
            print(
                f"  {path.name}: {changed:,}/{requested.size:,} labels "
                f"would change ({percent:.3f}%)"
            )
            return

        if "action_requested" not in replay:
            replay.create_dataset(
                "action_requested",
                data=requested,
                maxshape=(None,),
                chunks=replay["action"].chunks,
                compression=replay["action"].compression,
            )
        replay["action"][:] = executed
        replay.flush()
        print(
            f"  {path.name}: repaired {changed:,}/{requested.size:,} labels "
            f"({percent:.3f}%); original preserved in action_requested"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Repair ignored policy-action labels in replay HDF5 files"
    )
    parser.add_argument("paths", nargs="*", help="HDF5 files or directories")
    parser.add_argument("--all", action="store_true", help="Process the default data directory")
    parser.add_argument("--dry-run", action="store_true", help="Report without writing")
    args = parser.parse_args()

    files: list[Path] = []
    if args.all or not args.paths:
        files = sorted(Path(DEFAULT_DATA_DIR).glob("*.h5"))
    else:
        for raw_path in args.paths:
            path = Path(raw_path)
            if path.is_dir():
                files.extend(sorted(path.glob("*.h5")))
            elif path.is_file() and path.suffix == ".h5":
                files.append(path)
            else:
                print(f"Skipping: {raw_path}", file=sys.stderr)

    if not files:
        print("No .h5 files found.")
        return
    command_lengths = _command_lengths()
    for path in files:
        patch_file(path, args.dry_run, command_lengths)


if __name__ == "__main__":
    main()
