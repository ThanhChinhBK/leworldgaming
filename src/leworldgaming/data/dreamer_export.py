"""Convert an HDF5 replay buffer into per-episode .npz files for the
vendored ``external/dreamerv3-torch`` offline trainer.

The upstream trainer reads its dataset via ``tools.load_episodes(directory)``
which expects one ``.npz`` per episode. Two obs modes are supported:

``obs_mode="vector"`` (proprio) — keys consumed by ``WorldModel.preprocess``::

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

``obs_mode="image"`` (vision) — full pixel-based DreamerV3. The replay
buffer's ``pixels (T, 3, 224, 224) uint8`` are down-sampled to a HWC
``image`` the upstream ``ConvEncoder`` consumes directly::

    image       (T, S, S, 3)            uint8     S = image_size (default 64)
    action      (T, A)                  float32   one-hot
    reward      (T,)                    float32
    is_first    (T,)                    bool
    is_terminal (T,)                    bool

Pixels are P1-perspective as collected (no side-canonicalization — the
frame is a rendered image, not a symmetric state vector), so an image
world model trained here is tied to the collecting player's viewpoint.
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
# of silently feeding stale shapes into the trainer. One marker per obs mode so
# vector and image caches never clobber each other.
_EXPORT_SCHEMA = {
    "vector": "vector_v1",
    "image": "image_v1",
}

# Default edge length for the CNN-mode image. DreamerV3's ConvEncoder needs a
# power-of-two side (stages = log2(size) - log2(minres=4)); 64 → 4 stages,
# matching the upstream ``dmc_vision`` config.
_DEFAULT_IMAGE_SIZE = 64


def _downsample_pixels(px_chw: np.ndarray, image_size: int) -> np.ndarray:
    """``(T, C, H, W) uint8`` → ``(T, image_size, image_size, C) uint8`` (HWC).

    DreamerV3's ``ConvEncoder.forward`` expects images laid out as
    ``(batch, time, h, w, ch)`` and permutes to CHW internally
    (``external/dreamerv3-torch/networks.py``). The replay buffer stores
    pixels CHW at 224², so we transpose to HWC and area-resize down to
    ``image_size`` per frame (INTER_AREA is the correct downsampling filter).
    """
    import cv2

    t, c, h, w = px_chw.shape
    out = np.empty((t, image_size, image_size, c), dtype=np.uint8)
    for i in range(t):
        hwc = np.ascontiguousarray(px_chw[i].transpose(1, 2, 0))  # CHW -> HWC
        if (h, w) != (image_size, image_size):
            hwc = cv2.resize(hwc, (image_size, image_size), interpolation=cv2.INTER_AREA)
        out[i] = hwc
    return out


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


def _resolve_h5_paths(data_path: str | Path) -> list[Path]:
    """Return a list of H5 file paths from either a single file or a directory."""
    data_path = Path(data_path)
    if data_path.is_dir():
        paths = sorted(data_path.glob("*.h5"))
        if not paths:
            raise FileNotFoundError(f"No .h5 files found in {data_path}")
        return paths
    return [data_path]


def export_episodes_to_npz(
    data_path: str | Path,
    out_dir: str | Path,
    action_dim: int = NUM_ACTIONS,
    overwrite: bool = False,
    obs_mode: str = "vector",
    image_size: int = _DEFAULT_IMAGE_SIZE,
) -> int:
    """Slice HDF5 replay file(s) into per-episode npz files. Returns number written.

    ``data_path`` may be a single ``.h5`` file or a directory of them.

    ``obs_mode`` selects the exported layout: ``"vector"`` (proprio, default)
    or ``"image"`` (full pixels down-sampled to ``image_size``²). Each mode
    has its own schema marker so caches never mix.

    Cache invalidation: writes a small ``_EXPORT_SCHEMA`` marker. If the
    marker is missing or stale and ``out_dir`` already has npz files,
    re-export. Otherwise skip silently.
    """
    if obs_mode not in _EXPORT_SCHEMA:
        raise ValueError(f"obs_mode must be one of {sorted(_EXPORT_SCHEMA)}, got {obs_mode!r}")
    schema = _EXPORT_SCHEMA[obs_mode]

    h5_paths = _resolve_h5_paths(data_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / ".schema"

    existing_marker = marker.read_text().strip() if marker.exists() else ""
    has_episodes = any(out_dir.glob("*.npz"))

    if has_episodes and existing_marker == schema and not overwrite:
        existing = len(list(out_dir.glob("*.npz")))
        print(f"[dreamer_export] {out_dir} already has {existing} episodes "
              f"(schema={schema}) — skip")
        return existing

    if has_episodes and existing_marker != schema:
        print(f"[dreamer_export] cache schema mismatch "
              f"('{existing_marker}' != '{schema}') — clearing {out_dir}")
        for p in out_dir.glob("*.npz"):
            p.unlink()

    n_written = 0
    global_ep_idx = 0

    for h5_path in h5_paths:
        with h5py.File(str(h5_path), "r") as f:
            n = f["action"].shape[0]
            episode_starts = f["episode_starts"][:]
            slices = _episode_slices(episode_starts, n)
            print(f"[dreamer_export] {h5_path}: {n} steps, {len(slices)} episodes "
                  f"(obs_mode={obs_mode}"
                  f"{f', img={image_size}' if obs_mode == 'image' else f', state_dim={DREAMER_STATE_DIM}'})")

            if obs_mode == "image" and "pixels" not in f:
                raise KeyError(
                    f"{h5_path} has no 'pixels' dataset — image-mode Dreamer needs "
                    "pixel data. Re-collect with pixels enabled (BufferConfig.pixel_shape)."
                )

            action_arr = f["action"][:]
            reward_arr = f["reward"][:]
            is_first_arr = f["is_first"][:].astype(bool)
            done_arr = f["done"][:].astype(bool)

            for _, (a, b) in enumerate(slices):
                length = b - a
                if length < 2:
                    continue

                actions_int = action_arr[a:b].astype(np.int64)
                action_oh = np.zeros((length, action_dim), dtype=np.float32)
                action_oh[np.arange(length), actions_int] = 1.0

                is_first = is_first_arr[a:b].copy()
                is_first[0] = True

                is_terminal = done_arr[a:b].copy()
                is_terminal[-1] = True

                ep = {
                    "action": action_oh,
                    "reward": reward_arr[a:b].astype(np.float32),
                    "is_first": is_first,
                    "is_terminal": is_terminal,
                }

                if obs_mode == "image":
                    px = f["pixels"][a:b]  # (length, C, H, W) uint8
                    ep["image"] = _downsample_pixels(px, image_size)
                else:
                    ep_primitives = _read_episode_arrays(f, a, b)
                    ep_primitives = {k: v[None, :] for k, v in ep_primitives.items()}
                    ep_primitives = canonicalize_sample(ep_primitives)
                    vector = _build_dreamer_vector(ep_primitives)[0]
                    ep["vector"] = vector.astype(np.float32)
                    ep["image"] = np.zeros((length, 1, 1, 3), dtype=np.uint8)

                filename = out_dir / f"ep{global_ep_idx:06d}-{length}.npz"
                np.savez_compressed(filename, **ep)
                n_written += 1
                global_ep_idx += 1

    marker.write_text(schema)
    print(f"[dreamer_export] wrote {n_written} episodes total from {len(h5_paths)} file(s)")
    return n_written
