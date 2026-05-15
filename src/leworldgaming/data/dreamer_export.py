"""Convert an HDF5 replay buffer into per-episode .npz files for the
vendored ``external/dreamerv3-torch`` offline trainer.

The upstream trainer reads its dataset via ``tools.load_episodes(directory)``
which expects one ``.npz`` per episode. With Dreamer running in vector
(proprio) mode, the keys consumed by ``WorldModel.preprocess`` are::

    vector      (T, DREAMER_STATE_DIM)  float32   side-canonicalized
    image       (T, 1, 1, 3)            uint8     dummy (see views.py B.3)
    action      (T, A)                  float32   one-hot
    reward      (T,)                    float32
    is_first    (T,)                    bool
    is_terminal (T,)                    bool

The dummy ``image`` exists solely because ``WorldModel.preprocess`` at
``external/dreamerv3-torch/models.py:182`` unconditionally executes
``obs["image"] = obs["image"] / 255.0``. With ``encoder.cnn_keys: '$^'``
the MultiEncoder skips it; the placeholder costs 3 bytes per timestep.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

from leworldgaming.data.views import _build_dreamer_vector
from leworldgaming.data.replay_buffer import GROUPS as _BUFFER_GROUPS
from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.state_vector import (
    DREAMER_STATE_DIM,
    canonicalize_sample,
)

# Bumped when the on-disk npz layout changes — old caches get rebuilt instead
# of silently feeding stale shapes into the trainer.
_EXPORT_SCHEMA = "vector_v1"


def _episode_slices(episode_starts: np.ndarray, n_total: int) -> list[tuple[int, int]]:
    """Return ``[(start, stop)]`` half-open intervals per episode."""
    starts = list(int(s) for s in episode_starts)
    if not starts:
        return [(0, n_total)] if n_total > 0 else []
    stops = starts[1:] + [n_total]
    return list(zip(starts, stops))


def _read_episode_arrays(f: h5py.File, a: int, b: int) -> dict[str, np.ndarray]:
    """Read all primitives + flags for a slice ``[a, b)`` as a flat dict."""
    out: dict[str, np.ndarray] = {}
    for group_path, schema in _BUFFER_GROUPS.items():
        side_key = group_path.split("/")[-1]
        for name in schema:
            out[f"{side_key}/{name}"] = f[f"{group_path}/{name}"][a:b]
    return out


def export_episodes_to_npz(
    h5_path: str | Path,
    out_dir: str | Path,
    action_dim: int = NUM_ACTIONS,
    overwrite: bool = False,
) -> int:
    """Slice an HDF5 replay into per-episode npz files. Returns number written.

    Cache invalidation: writes a small ``_EXPORT_SCHEMA`` marker. If the
    marker is missing or stale and ``out_dir`` already has npz files,
    re-export. Otherwise skip silently.
    """
    h5_path = Path(h5_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".schema"

    existing_marker = marker.read_text().strip() if marker.exists() else ""
    has_episodes = any(out_dir.glob("*.npz"))

    if has_episodes and existing_marker == _EXPORT_SCHEMA and not overwrite:
        existing = len(list(out_dir.glob("*.npz")))
        print(f"[dreamer_export] {out_dir} already has {existing} episodes "
              f"(schema={_EXPORT_SCHEMA}) — skip")
        return existing

    if has_episodes and existing_marker != _EXPORT_SCHEMA:
        print(f"[dreamer_export] cache schema mismatch "
              f"('{existing_marker}' != '{_EXPORT_SCHEMA}') — clearing {out_dir}")
        for p in out_dir.glob("*.npz"):
            p.unlink()

    with h5py.File(h5_path, "r") as f:
        n = f["action"].shape[0]
        episode_starts = f["episode_starts"][:]
        slices = _episode_slices(episode_starts, n)
        print(f"[dreamer_export] {h5_path}: {n} steps, {len(slices)} episodes "
              f"-> {out_dir} (state_dim={DREAMER_STATE_DIM})")

        action_arr = f["action"][:]
        reward_arr = f["reward"][:]
        is_first_arr = f["is_first"][:].astype(bool)
        done_arr = f["done"][:].astype(bool)

        n_written = 0
        for ep_idx, (a, b) in enumerate(slices):
            length = b - a
            if length < 2:
                continue  # sample_episodes() in upstream skips these anyway

            ep_primitives = _read_episode_arrays(f, a, b)
            # Add a leading "L=1" axis so canonicalize_sample / _build_dreamer_vector
            # work on (B=1, L=length, ...) arrays. Squeeze out afterwards.
            ep_primitives = {k: v[None, :] for k, v in ep_primitives.items()}
            ep_primitives = canonicalize_sample(ep_primitives)
            vector = _build_dreamer_vector(ep_primitives)[0]  # (T, DREAMER_STATE_DIM)

            actions_int = action_arr[a:b].astype(np.int64)
            action_oh = np.zeros((length, action_dim), dtype=np.float32)
            action_oh[np.arange(length), actions_int] = 1.0

            is_first = is_first_arr[a:b].copy()
            is_first[0] = True  # Dreamer requires first step of every chunk to flag reset

            is_terminal = done_arr[a:b].copy()
            is_terminal[-1] = True  # mark episode end so cont-head sees signal

            ep = {
                "vector": vector.astype(np.float32),
                "image": np.zeros((length, 1, 1, 3), dtype=np.uint8),
                "action": action_oh,
                "reward": reward_arr[a:b].astype(np.float32),
                "is_first": is_first,
                "is_terminal": is_terminal,
            }
            # Upstream's load_episodes parses filename; the suffix encodes length.
            filename = out_dir / f"ep{ep_idx:06d}-{length}.npz"
            np.savez_compressed(filename, **ep)
            n_written += 1

        marker.write_text(_EXPORT_SCHEMA)
        print(f"[dreamer_export] wrote {n_written} episodes")
        return n_written
