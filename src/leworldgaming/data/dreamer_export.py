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

from leworldgaming.data.replay_buffer import GROUPS as _BUFFER_GROUPS
from leworldgaming.data.views import _build_dreamer_vector
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
    return list(zip(starts, stops, strict=True))


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


def _stride_block_starts(length: int, stride: int) -> np.ndarray:
    """Raw-frame indices where each ``stride``-sized decision block begins.

    Mirrors ``leworldgaming.training._replay_utils`` conventions for LeWM's
    own ``temporal_stride`` handling: block ``i`` "observes" raw frame
    ``i*stride`` (this is the frame whose action was actually chosen/held at
    that decision point) and "owns" the future reward span
    ``[i*stride+1, i*stride+1+stride)`` — i.e. rewards are shifted by one
    relative to actions, matching ``reduce_reward_seq``'s ``[t+1, t+stride+1)``
    alignment (see repo memory "reward timing": reward[t] is the HP-delta
    entering obs[t], not the reward earned by action[t]).

    Only block starts with at least one raw future reward row are kept
    (``s <= length - 2``), so every block has a non-empty reward window.
    """
    if length < 2:
        return np.empty(0, dtype=np.int64)
    return np.arange(0, length - 1, stride, dtype=np.int64)


def _stride_reduce_reward(
    reward_arr: np.ndarray, starts: np.ndarray, stride: int, length: int
) -> np.ndarray:
    """Sum raw per-frame reward over each block's shifted window.

    Block ``i`` (starting at raw frame ``starts[i]``) sums
    ``reward_arr[starts[i]+1 : min(starts[i]+1+stride, length)]`` — the last
    block's window is clipped (not truncated-and-dropped) to ``length`` so
    any leftover tail frames (when ``length-1`` isn't an exact multiple of
    ``stride``) still contribute to the final block instead of being
    silently discarded, conserving total episode reward exactly.
    """
    out = np.empty(len(starts), dtype=np.float32)
    for i, s in enumerate(starts):
        end = min(int(s) + 1 + stride, length)
        out[i] = reward_arr[int(s) + 1 : end].sum()
    return out


def _stride_terminal_flags(
    done_arr: np.ndarray, starts: np.ndarray, stride: int, length: int
) -> np.ndarray:
    """``is_terminal`` per block: True iff the raw frame at the end of that
    block's reward window is the episode's true terminal frame."""
    out = np.zeros(len(starts), dtype=bool)
    for i, s in enumerate(starts):
        end = min(int(s) + 1 + stride, length)
        out[i] = bool(done_arr[end - 1])
    return out


def export_episodes_to_npz(
    data_path: str | Path,
    out_dir: str | Path,
    action_dim: int = NUM_ACTIONS,
    overwrite: bool = False,
    obs_mode: str = "vector",
    image_size: int = _DEFAULT_IMAGE_SIZE,
    stride: int = 1,
) -> int:
    """Slice HDF5 replay file(s) into per-episode npz files. Returns number written.

    ``data_path`` may be a single ``.h5`` file or a directory of them.

    ``obs_mode`` selects the exported layout: ``"vector"`` (proprio, default)
    or ``"image"`` (full pixels down-sampled to ``image_size``²). Each mode
    has its own schema marker so caches never mix.

    ``stride`` (temporal frameskip, a.k.a. action repeat): when ``>1``, only
    every ``stride``-th raw frame is kept as an observation/decision point
    (matching LeWM's ``temporal_stride`` convention — see
    ``configs/lewm.yaml`` and ``_stride_block_starts`` above), so a Dreamer
    trained this way can be compared to LeWM at the same decision rate.
    ``stride=1`` (default) is the original unstrided/backward-compatible
    behavior — byte-identical to before this parameter was added. Reward is
    summed (not just kept at the block-start frame) over each block's
    shifted window via ``_stride_reduce_reward`` so total episode reward is
    conserved regardless of stride.

    Cache invalidation: writes a small ``_EXPORT_SCHEMA`` marker (stride is
    folded into the marker string so caches at different strides never mix).
    If the marker is missing or stale and ``out_dir`` already has npz files,
    re-export. Otherwise skip silently.
    """
    if obs_mode not in _EXPORT_SCHEMA:
        raise ValueError(f"obs_mode must be one of {sorted(_EXPORT_SCHEMA)}, got {obs_mode!r}")
    stride = int(stride) or 1
    schema = f"{_EXPORT_SCHEMA[obs_mode]}_stride{stride}"

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
                  f"(obs_mode={obs_mode}, stride={stride}"
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

                if stride == 1:
                    actions_int = action_arr[a:b].astype(np.int64)
                    out_len = length
                    is_first = is_first_arr[a:b].copy()
                    is_first[0] = True
                    is_terminal = done_arr[a:b].copy()
                    is_terminal[-1] = True
                    reward_out = reward_arr[a:b].astype(np.float32)
                    pick_idx = np.arange(length)
                else:
                    starts = _stride_block_starts(length, stride)
                    if len(starts) == 0:
                        continue
                    out_len = len(starts)
                    pick_idx = starts
                    actions_int = action_arr[a:b][starts].astype(np.int64)
                    is_first = np.zeros(out_len, dtype=bool)
                    is_first[0] = True
                    is_terminal = _stride_terminal_flags(
                        done_arr[a:b], starts, stride, length
                    )
                    reward_out = _stride_reduce_reward(
                        reward_arr[a:b], starts, stride, length
                    )

                action_oh = np.zeros((out_len, action_dim), dtype=np.float32)
                action_oh[np.arange(out_len), actions_int] = 1.0

                ep = {
                    "action": action_oh,
                    "reward": reward_out,
                    "is_first": is_first,
                    "is_terminal": is_terminal,
                }

                if obs_mode == "image":
                    px = f["pixels"][a:b][pick_idx]  # (out_len, C, H, W) uint8
                    ep["image"] = _downsample_pixels(px, image_size)
                else:
                    ep_primitives = _read_episode_arrays(f, a, b)
                    ep_primitives = {k: v[pick_idx][None, :] for k, v in ep_primitives.items()}
                    ep_primitives = canonicalize_sample(ep_primitives)
                    vector = _build_dreamer_vector(ep_primitives)[0]
                    ep["vector"] = vector.astype(np.float32)
                    ep["image"] = np.zeros((out_len, 1, 1, 3), dtype=np.uint8)

                filename = out_dir / f"ep{global_ep_idx:06d}-{out_len}.npz"
                np.savez_compressed(filename, **ep)
                n_written += 1
                global_ep_idx += 1

    marker.write_text(schema)
    print(f"[dreamer_export] wrote {n_written} episodes total from {len(h5_paths)} file(s)")
    return n_written
