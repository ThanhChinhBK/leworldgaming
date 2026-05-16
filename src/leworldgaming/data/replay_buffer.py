"""HDF5-backed episode-aware replay buffer with a background writer thread.

Layout: one HDF5 file with chunked, resizable datasets organised under named
groups so each MBRL method's dataloader (LeWM / Dreamer / PETS) materializes
its own view at sample time. Each ``add()`` call enqueues one transition onto
a bounded ``queue.Queue`` consumed by a single dedicated writer thread; the
thread performs the actual ``ds.resize()`` + LZF-compressed write. h5py and
LZF release the GIL during the heavy work, so the writer runs truly in
parallel with the asyncio loop driving the pyftg gateway. ``close()`` drains
the queue and joins the thread before closing the file.

Datasets:

  obs/own/{hp:int32, energy:int32, x:float32, y:float32,
           speed_x:float32, speed_y:float32, state:int8, front:int8,
           control:int8, remaining_frame:int16, hit_confirm:int8,
           atk_is_live:int8, atk_start_up:int16, atk_active:int16,
           atk_hit_damage:int16, atk_type:int8}
  obs/opp/...same fields, mirrored
  obs/global/{current_round:int8, current_frame:int16,
              proj_self:int8, proj_opp:int8,
              max_hp:int16, max_energy:int16}
  action          (N,)        int32
  reward          (N,)        float32
  done            (N,)        uint8
  is_first        (N,)        uint8     — Dreamer RSSM reset flag
  cont            (N,)        uint8     — = 1 - done; Dreamer-native
  state_vector    (N, 52)     float32   — legacy flat form, mirror of named groups
  episode_starts  (E,)        int64

Optional (only when ``BufferConfig.pixel_shape`` is set, e.g. for LeWM):
  pixels          (N, C, H, W) uint8

Threading caveat: h5py is not thread-safe, so the writer thread is the
*only* thread that touches the file once ``open()`` has started it. Reads
(``sample`` / ``sample_sequences``) go through the writer's idle window —
prefer opening a separate read-only ``h5py.File`` for heavy training reads
(see ``leworldgaming.training.train_lewm`` for the pattern).
"""

from __future__ import annotations

import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from leworldgaming.env.state_vector import STATE_VECTOR_DIM, obs_dict_to_legacy_vector

# Per-group primitive schema — the source of truth driving both dataset
# creation and per-row writes. Key order matters only for readability.
_PER_CHAR_SCHEMA: dict[str, str] = {
    "hp": "int32",
    "energy": "int32",
    "x": "float32",
    "y": "float32",
    "speed_x": "float32",
    "speed_y": "float32",
    "state": "int8",
    "front": "int8",
    "control": "int8",
    "remaining_frame": "int16",
    "hit_confirm": "int8",
    "atk_is_live": "int8",
    "atk_start_up": "int16",
    "atk_active": "int16",
    "atk_hit_damage": "int16",
    "atk_type": "int8",
}
_GLOBAL_SCHEMA: dict[str, str] = {
    "current_round": "int8",
    "current_frame": "int16",
    "proj_self": "int8",
    "proj_opp": "int8",
    "max_hp": "int16",
    "max_energy": "int16",
}
_TOP_LEVEL_SCHEMA: dict[str, str] = {
    "action": "int32",
    "reward": "float32",
    "done": "uint8",
    "is_first": "uint8",
    "cont": "uint8",
}

# Optional flat 52-dim state vector mirror (legacy form). Stored alongside the
# named groups so probe-style targets (LeWM Stage-B head trainer) and any
# tooling expecting the legacy layout keep working without re-derivation.
_STATE_VECTOR_NAME = "state_vector"

# Group prefix → schema lookup, used by sample_sequences to walk all primitives.
# Public — external consumers (dreamer_export, tests) read it to enumerate keys.
GROUPS: dict[str, dict[str, str]] = {
    "obs/own": _PER_CHAR_SCHEMA,
    "obs/opp": _PER_CHAR_SCHEMA,
    "obs/global": _GLOBAL_SCHEMA,
}


@dataclass
class BufferConfig:
    path: str = "data/replay.h5"
    chunk_size: int = 1024
    # Optional pixel storage. None disables it; (C, H, W) tuple enables it.
    pixel_shape: tuple[int, int, int] | None = None
    # Bounded queue between producer (asyncio) and writer thread. With ~150 KB
    # pixel rows × 4096 ≈ 600 MB worst-case buffering — fine, and chosen so
    # the writer should never throttle the producer in normal operation.
    write_queue_size: int = 4096
    # If True, open the file read-only and skip the writer thread.
    read_only: bool = False


# Marker objects for the writer queue. Identity comparison only.
_END_EPISODE = object()
_SHUTDOWN = object()


class ReplayBuffer:
    """Append-only writer + random-batch sampler.

    Use as a context manager so HDF5 flushes cleanly::

        with ReplayBuffer(cfg) as buf:
            for transition in episode:
                buf.add(...)
            buf.end_episode()

    For training-side reads, prefer opening a fresh read-only ``h5py.File``
    or pass ``BufferConfig(read_only=True)`` to skip the writer thread.
    """

    def __init__(self, cfg: BufferConfig | None = None) -> None:
        self.cfg = cfg or BufferConfig()
        Path(self.cfg.path).parent.mkdir(parents=True, exist_ok=True)
        self._f: h5py.File | None = None
        self._n: int = 0
        self._n_episodes: int = 0
        self._episode_start: int = 0

        self._write_queue: queue.Queue = queue.Queue(maxsize=int(self.cfg.write_queue_size))
        self._writer_thread: threading.Thread | None = None
        self._writer_exc: BaseException | None = None

    def __enter__(self) -> ReplayBuffer:
        self.open()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def open(self) -> None:
        if self._f is not None:
            return
        if self.cfg.read_only:
            self._f = h5py.File(self.cfg.path, "r")
            self._n = self._f["action"].shape[0]
            self._n_episodes = self._f["episode_starts"].shape[0]
            self._episode_start = self._n
            return

        self._f = h5py.File(self.cfg.path, "a")
        if "action" not in self._f:
            self._init_datasets()
        else:
            self._validate_existing()
            self._n = self._f["action"].shape[0]
            self._n_episodes = self._f["episode_starts"].shape[0]
            self._episode_start = self._n

        self._writer_thread = threading.Thread(
            target=self._writer_loop, name="replay-writer", daemon=False,
        )
        self._writer_thread.start()

    def _validate_existing(self) -> None:
        """Refuse to silently mismatch the requested pixel schema."""
        assert self._f is not None
        has_pixels = "pixels" in self._f
        wants_pixels = self.cfg.pixel_shape is not None
        path = self.cfg.path
        if wants_pixels and not has_pixels:
            self._f.close()
            self._f = None
            raise ValueError(
                f"{path} exists without a 'pixels' dataset but pixel_shape is set. "
                "Delete the file or use a different --out path."
            )
        if has_pixels and not wants_pixels:
            self._f.close()
            self._f = None
            raise ValueError(
                f"{path} has a 'pixels' dataset but pixel_shape is not set. "
                "Pass --pixels or use a different --out path."
            )
        if has_pixels:
            stored = tuple(self._f["pixels"].shape[1:])
            if stored != self.cfg.pixel_shape:
                self._f.close()
                self._f = None
                raise ValueError(
                    f"{path} has pixels {stored} but pixel_shape is {self.cfg.pixel_shape}."
                )

    def _init_datasets(self) -> None:
        assert self._f is not None
        chunk = self.cfg.chunk_size

        # Top-level transition columns.
        for name, dtype in _TOP_LEVEL_SCHEMA.items():
            self._f.create_dataset(
                name,
                shape=(0,),
                maxshape=(None,),
                chunks=(chunk,),
                dtype=dtype,
                compression="lzf",
            )

        # Named obs groups.
        for group_path, schema in GROUPS.items():
            grp = self._f.require_group(group_path)
            for name, dtype in schema.items():
                grp.create_dataset(
                    name,
                    shape=(0,),
                    maxshape=(None,),
                    chunks=(chunk,),
                    dtype=dtype,
                    compression="lzf",
                )

        self._f.create_dataset(
            _STATE_VECTOR_NAME,
            shape=(0, STATE_VECTOR_DIM),
            maxshape=(None, STATE_VECTOR_DIM),
            chunks=(chunk, STATE_VECTOR_DIM),
            dtype="float32",
            compression="lzf",
        )
        self._f.create_dataset(
            "episode_starts",
            shape=(0,),
            maxshape=(None,),
            chunks=(64,),
            dtype="int64",
        )
        if self.cfg.pixel_shape is not None:
            c, h, w = self.cfg.pixel_shape
            # No compression for pixels — LZF adds ~78ms/frame latency which
            # bottlenecks the entire --input-sync pipeline. Raw writes are
            # 0.2ms/frame. Storage cost: ~150KB/frame ≈ 540MB/round on disk.
            pixel_chunk = max(1, min(chunk, 64))
            self._f.create_dataset(
                "pixels",
                shape=(0, c, h, w),
                maxshape=(None, c, h, w),
                chunks=(pixel_chunk, c, h, w),
                dtype="uint8",
            )

    def close(self) -> None:
        if self._writer_thread is not None:
            # Finalize an in-progress episode and stop the worker. The
            # end-of-episode marker is a no-op if the episode was already
            # closed; cheap insurance.
            self._write_queue.put(_END_EPISODE)
            self._write_queue.put(_SHUTDOWN)
            self._writer_thread.join()
            self._writer_thread = None
            if self._writer_exc is not None:
                exc, self._writer_exc = self._writer_exc, None
                raise exc
        if self._f is not None:
            self._f.close()
            self._f = None

    def add(
        self,
        obs_dict: dict[str, dict[str, Any]],
        action: int,
        reward: float,
        done: bool,
        is_first: bool,
        pixels: np.ndarray | None = None,
    ) -> None:
        """Enqueue a transition for the writer thread.

        ``obs_dict`` follows the structure produced by
        ``leworldgaming.env.frame_to_obs_dict`` — nested ``{own, opp, global}``
        of named primitives. ``pixels`` is not copied — callers must pass
        arrays they no longer mutate (the existing producers
        — ``SpectatorRecorder.latest_pixels`` — already return fresh
        references that the spectator never overwrites in-place).
        """
        item = (
            obs_dict,
            int(action),
            float(reward),
            bool(done),
            bool(is_first),
            pixels,
        )
        self._write_queue.put(item)

    def end_episode(self) -> None:
        self._write_queue.put(_END_EPISODE)

    def _writer_loop(self) -> None:
        try:
            while True:
                item = self._write_queue.get()
                if item is _SHUTDOWN:
                    return
                if item is _END_EPISODE:
                    self._do_end_episode()
                    continue
                self._do_add(*item)
        except BaseException as exc:  # noqa: BLE001
            self._writer_exc = exc

    def _do_add(
        self,
        obs_dict: dict[str, dict[str, Any]],
        action: int,
        reward: float,
        done: bool,
        is_first: bool,
        pixels: np.ndarray | None,
    ) -> None:
        assert self._f is not None
        i = self._n
        new_n = i + 1
        cont = 0 if done else 1

        # Top-level columns.
        top_values: dict[str, Any] = {
            "action": np.int32(action),
            "reward": np.float32(reward),
            "done": np.uint8(1 if done else 0),
            "is_first": np.uint8(1 if is_first else 0),
            "cont": np.uint8(cont),
        }
        for name, value in top_values.items():
            ds = self._f[name]
            ds.resize((new_n,))
            ds[i] = value

        # Named groups.
        for group_path, schema in GROUPS.items():
            side_key = group_path.split("/")[-1]  # own / opp / global
            obs_side = obs_dict[side_key]
            grp = self._f[group_path]
            for name in schema:
                ds = grp[name]
                ds.resize((new_n,))
                ds[i] = obs_side[name]

        # Legacy flat 52-dim state vector. Stored as a mirror of the named
        # groups for back-compat (LeWM linear-probe target, older tooling).
        if _STATE_VECTOR_NAME in self._f:
            sv = obs_dict_to_legacy_vector(obs_dict)
            ds = self._f[_STATE_VECTOR_NAME]
            ds.resize((new_n, ds.shape[1]))
            ds[i] = sv

        if "pixels" in self._f:
            ds = self._f["pixels"]
            ds.resize((new_n, *ds.shape[1:]))
            ds[i] = (
                pixels.astype(np.uint8)
                if pixels is not None
                else np.zeros(ds.shape[1:], np.uint8)
            )
        self._n = new_n

    def _do_end_episode(self) -> None:
        assert self._f is not None
        if self._episode_start == self._n:
            return
        starts = self._f["episode_starts"]
        starts.resize((self._n_episodes + 1,))
        starts[self._n_episodes] = self._episode_start
        self._n_episodes += 1
        self._episode_start = self._n
        self._f.flush()

    def __len__(self) -> int:
        return self._n

    @property
    def num_episodes(self) -> int:
        return self._n_episodes

    def sample(self, batch_size: int, rng: np.random.Generator | None = None) -> dict[str, Any]:
        """Random-transition batch (PETS-style). Each value has shape ``(B, ...)``.

        Returns a flat dict — primitives are prefixed by their group:
        ``"own/hp"``, ``"opp/x"``, ``"global/proj_self"``, plus top-level
        ``action / reward / done / is_first / cont`` and ``pixels`` if stored.
        """
        seq = self.sample_sequences(batch_size, seq_len=1, rng=rng)
        return {k: v[:, 0] for k, v in seq.items()}

    def sample_sequences(
        self,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator | None = None,
    ) -> dict[str, np.ndarray]:
        """Sample ``batch_size`` contiguous-episode windows of length ``seq_len``.

        Returns a flat dict of ``np.ndarray``s shaped ``(B, L, ...)``. Primitive
        keys use their group prefix (``"own/hp"``, ``"global/current_round"``).
        """
        assert self._f is not None
        if self._n == 0:
            raise ValueError("Buffer is empty")
        rng = rng or np.random.default_rng()
        starts = valid_seq_starts(self._f, seq_len)
        if starts.size == 0:
            raise ValueError(
                f"No valid sequences of length {seq_len} (n={self._n}, "
                f"episodes={self._n_episodes})"
            )
        return sample_window(self._f, starts, batch_size, seq_len, rng)


# --------------------------------------------------------------------------- #
# Module-level helpers (used by training scripts that open h5py.File directly)
# --------------------------------------------------------------------------- #


def valid_seq_starts(f: h5py.File, seq_len: int) -> np.ndarray:
    """Indices ``i`` such that ``[i, i+seq_len-1]`` lies within one episode and
    contains no terminal frames before the last position. Replaces
    ``train_lewm._valid_seq_start_indices`` and now serves all trainers."""
    n = f["action"].shape[0]
    starts = f["episode_starts"][:]
    dones = f["done"][:]
    next_start = np.concatenate([starts[1:], [n]])
    is_last = np.zeros(n, dtype=bool)
    is_last[next_start - 1] = True

    candidates = np.arange(max(n - seq_len + 1, 0))
    valid = np.ones(candidates.size, dtype=bool)
    for k in range(seq_len):
        idx = candidates + k
        valid &= dones[idx] == 0
        if k < seq_len - 1:
            valid &= ~is_last[idx]
    return candidates[valid]


def _gather_seq(ds: h5py.Dataset, all_idx: np.ndarray, batch_size: int, seq_len: int) -> np.ndarray:
    """Read ``all_idx`` (flat) once, then unflatten to ``(B, L, ...)``.

    h5py fancy indexing requires strictly-increasing indices, so we go through
    a unique-then-inverse-map detour for free deduplication on overlapping
    windows."""
    union, inverse = np.unique(all_idx, return_inverse=True)
    block = ds[union]
    return block[inverse].reshape(batch_size, seq_len, *ds.shape[1:])


def sample_window(
    f: h5py.File,
    valid_starts: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
    """Pick ``batch_size`` window starts and read all primitives + pixels.

    Public entry point for trainers that open ``h5py.File`` directly (so
    they don't need a ``ReplayBuffer`` instance / writer thread).
    """
    pick = rng.choice(valid_starts, size=batch_size, replace=valid_starts.size < batch_size)
    all_idx = (pick[:, None] + np.arange(seq_len)[None, :]).reshape(-1)

    out: dict[str, np.ndarray] = {}

    # Top-level columns.
    for name in _TOP_LEVEL_SCHEMA:
        out[name] = _gather_seq(f[name], all_idx, batch_size, seq_len)

    # Named groups.
    for group_path, schema in GROUPS.items():
        side_key = group_path.split("/")[-1]
        for name in schema:
            out[f"{side_key}/{name}"] = _gather_seq(
                f[f"{group_path}/{name}"], all_idx, batch_size, seq_len
            )

    if "pixels" in f:
        out["pixels"] = _gather_seq(f["pixels"], all_idx, batch_size, seq_len)

    return out


def open_for_read(path: str) -> h5py.File:
    """Convenience: open the buffer file in read-only mode for training reads."""
    return h5py.File(path, "r")
