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
        # Mark the true terminal frame: processing() always writes done=False
        # since it can't know in advance which frame is last, so patch the
        # final frame of this just-ended episode here instead.
        last_idx = self._n - 1
        self._f["done"][last_idx] = np.uint8(1)
        self._f["cont"][last_idx] = np.uint8(0)
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
        # The window's LAST position is allowed to be a terminal frame (that's
        # the whole point of training on episode-ending transitions) — only
        # earlier positions must not be terminal/episode-final, otherwise the
        # window would silently cross into the next episode.
        if k < seq_len - 1:
            valid &= dones[idx] == 0
            valid &= ~is_last[idx]
    return candidates[valid]


def _gather_seq(ds: h5py.Dataset, picks: np.ndarray, seq_len: int) -> np.ndarray:
    """Read a ``seq_len``-length window per start in ``picks`` → ``(B, L, ...)``.

    Each window is read as a *contiguous* slice ``ds[s:s+seq_len]`` rather than
    via one scattered fancy-index gather. h5py point-selection on hundreds of
    non-contiguous indices is dominated by per-element selection overhead and
    is ~50× slower than contiguous slab reads here (measured 72 fps vs 3535 fps
    on the 224×224 pixel dataset), so this loop is the fast path even though it
    issues ``B`` reads instead of one."""
    batch_size = picks.shape[0]
    out = np.empty((batch_size, seq_len, *ds.shape[1:]), dtype=ds.dtype)
    for i in range(batch_size):
        s = int(picks[i])
        out[i] = ds[s : s + seq_len]
    return out


def _read_windows(
    f: h5py.File,
    picks: np.ndarray,
    seq_len: int,
    extra_keys: tuple[str, ...] = (),
) -> dict[str, np.ndarray]:
    """Read windows starting at ``picks`` (local start indices) from one file.

    Like ``sample_window`` but without the ``rng.choice`` step — the caller
    has already decided which starts to read.
    """
    out: dict[str, np.ndarray] = {}

    for name in _TOP_LEVEL_SCHEMA:
        out[name] = _gather_seq(f[name], picks, seq_len)

    for group_path, schema in GROUPS.items():
        side_key = group_path.split("/")[-1]
        for name in schema:
            out[f"{side_key}/{name}"] = _gather_seq(
                f[f"{group_path}/{name}"], picks, seq_len
            )

    if "pixels" in f:
        out["pixels"] = _gather_seq(f["pixels"], picks, seq_len)

    for key in extra_keys:
        if key in f and key not in out:
            out[key] = _gather_seq(f[key], picks, seq_len)

    return out


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
    return _read_windows(f, pick, seq_len)


def open_for_read(path: str) -> h5py.File:
    """Convenience: open the buffer file in read-only mode for training reads."""
    return h5py.File(path, "r")


# --------------------------------------------------------------------------- #
# Multi-file training reader
# --------------------------------------------------------------------------- #


class _MultiStarts:
    """Valid sequence start indices across one or more H5 files.

    Supports ``.size``, NumPy-style slicing, and batch sampling — a drop-in
    for the flat ``np.ndarray`` the single-file path used to return.
    """

    __slots__ = ("file_indices", "local_starts")

    def __init__(self, file_indices: np.ndarray, local_starts: np.ndarray) -> None:
        self.file_indices = np.asarray(file_indices, dtype=np.int32)
        self.local_starts = np.asarray(local_starts, dtype=np.int64)

    @property
    def size(self) -> int:
        return len(self.file_indices)

    def __len__(self) -> int:
        return len(self.file_indices)

    def __getitem__(self, idx):  # noqa: ANN001
        return _MultiStarts(self.file_indices[idx], self.local_starts[idx])


class DataReader:
    """Unified read interface for training: single H5 file or a directory of them.

    If ``path`` is a file, behaves like the old single-file path.
    If ``path`` is a directory, opens every ``*.h5`` file inside it and
    presents them as one logical dataset.

    Usage::

        with DataReader("data/replay.h5") as reader:     # single file
            starts = reader.valid_seq_starts(seq_len=4)
            batch = reader.sample_window(starts[:1000], 16, 4, rng)

        with DataReader("/data/replays/") as reader:      # directory
            ...
    """

    def __init__(self, path: str | Path) -> None:
        path = Path(path)
        if path.is_dir():
            h5_paths = sorted(path.glob("*.h5"))
            if not h5_paths:
                raise FileNotFoundError(f"No .h5 files found in {path}")
        elif path.is_file() or path.suffix == ".h5":
            h5_paths = [path]
        else:
            raise FileNotFoundError(f"{path} is not a file or directory")

        self._files: list[h5py.File] = [h5py.File(str(p), "r") for p in h5_paths]
        self._paths: list[str] = [str(p) for p in h5_paths]
        self._sizes: list[int] = [f["action"].shape[0] for f in self._files]

    def close(self) -> None:
        for f in self._files:
            f.close()
        self._files.clear()

    def __enter__(self) -> "DataReader":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    @property
    def num_files(self) -> int:
        return len(self._files)

    @property
    def paths(self) -> list[str]:
        return list(self._paths)

    @property
    def total_frames(self) -> int:
        return sum(self._sizes)

    @property
    def total_episodes(self) -> int:
        return sum(f["episode_starts"].shape[0] for f in self._files)

    def has_pixels(self) -> bool:
        """True only when **all** loaded files contain a ``pixels`` dataset."""
        return all("pixels" in f for f in self._files)

    def has_key(self, key: str) -> bool:
        """True when **all** loaded files contain the given top-level dataset."""
        return all(key in f for f in self._files)

    def valid_seq_starts(self, seq_len: int) -> _MultiStarts:
        """Compute valid sequence starts across all loaded files."""
        all_fi: list[np.ndarray] = []
        all_local: list[np.ndarray] = []
        for i, f in enumerate(self._files):
            local = valid_seq_starts(f, seq_len)
            if local.size > 0:
                all_fi.append(np.full(local.size, i, dtype=np.int32))
                all_local.append(local)
        if not all_fi:
            return _MultiStarts(np.array([], dtype=np.int32), np.array([], dtype=np.int64))
        return _MultiStarts(np.concatenate(all_fi), np.concatenate(all_local))

    def sample_window(
        self,
        starts: _MultiStarts,
        batch_size: int,
        seq_len: int,
        rng: np.random.Generator,
        extra_keys: tuple[str, ...] = (),
    ) -> dict[str, np.ndarray]:
        """Sample ``batch_size`` sequence windows. Returns the same dict
        format as the single-file ``sample_window``."""
        n = starts.size
        pick_idx = rng.choice(n, size=batch_size, replace=n < batch_size)
        picked_files = starts.file_indices[pick_idx]
        picked_locals = starts.local_starts[pick_idx]

        # Fast path: single file
        if self.num_files == 1:
            return _read_windows(self._files[0], picked_locals, seq_len, extra_keys)

        unique_files = np.unique(picked_files)

        # Read each file's subset independently
        file_batches: dict[int, dict[str, np.ndarray]] = {}
        local_pos = np.empty(batch_size, dtype=np.int64)

        for fi in unique_files:
            fi_int = int(fi)
            mask = picked_files == fi_int
            local_starts_for_file = picked_locals[mask]
            file_batches[fi_int] = _read_windows(
                self._files[fi_int], local_starts_for_file, seq_len, extra_keys
            )
            local_pos[np.where(mask)[0]] = np.arange(mask.sum())

        # Reassemble in original pick order
        sample_keys = list(next(iter(file_batches.values())).keys())
        result: dict[str, np.ndarray] = {}
        for k in sample_keys:
            ref = next(iter(file_batches.values()))[k]
            out = np.empty((batch_size, *ref.shape[1:]), dtype=ref.dtype)
            for fi in unique_files:
                fi_int = int(fi)
                mask = picked_files == fi_int
                out[mask] = file_batches[fi_int][k][local_pos[mask]]
            result[k] = out

        return result
