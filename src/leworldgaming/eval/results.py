"""Persistent, non-overwriting evaluation records."""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any


def checkpoint_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path).resolve(strict=True)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return {"path": str(path), "sha256": digest.hexdigest(), "bytes": path.stat().st_size}


def dreamer_action_alignment(path: str | Path) -> str | None:
    import torch

    checkpoint = torch.load(path, map_location="cpu", weights_only=False, mmap=True)
    return checkpoint.get("action_alignment")


def code_identity(root: Path, *, exclude: Path | None = None) -> dict[str, Any]:
    pathspec = ["."]
    if exclude is not None and exclude.resolve().is_relative_to(root.resolve()):
        relative = exclude.resolve().relative_to(root.resolve())
        if relative == Path("."):
            raise ValueError("Evaluation output must not be the source root.")
        pathspec.append(f":(top,literal,exclude){relative}")
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=root, capture_output=True, text=True, check=True,
    ).stdout.strip()
    diff = subprocess.run(
        ["git", "diff", "HEAD", "--binary", "--", *pathspec],
        cwd=root, capture_output=True, check=True,
    ).stdout
    status = subprocess.run(
        ["git", "status", "--porcelain", "--", *pathspec],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    untracked = subprocess.run(
        ["git", "ls-files", "--others", "--exclude-standard", "-z", "--", *pathspec],
        cwd=root, capture_output=True, text=True, check=True,
    ).stdout
    return {
        "revision": revision,
        "dirty": bool(status),
        "diff_sha256": hashlib.sha256(diff).hexdigest(),
        "status": status,
        "untracked_files": {
            name: checkpoint_identity(root / name)["sha256"]
            for name in untracked.split("\0") if name
        },
    }


def write_result(path: str | Path, result: dict[str, Any]) -> None:
    """Serialize before creating a file; never overwrite an earlier experiment."""
    payload = json.dumps(result, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        stream.write(payload)
