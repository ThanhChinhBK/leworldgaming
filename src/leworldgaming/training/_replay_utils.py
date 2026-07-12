"""Shared replay-buffer sequence sampling helpers.

Used by ``train_lewm.py`` (Stage A — JEPA pretraining) and
``train_lewm_heads.py`` (Stage B — reward / value / continuation heads).
Both trainers consume the same ``data/replay.h5`` written by
``data/replay_buffer.py``.

Three helpers:

* ``valid_seq_start_indices(f, seq_len)`` — vectorised episode-aware mask
  yielding indices ``i`` such that ``[i, i+seq_len-1]`` is a contiguous
  non-terminal slice within a single episode.
* ``sample_sequence_batch(f, valid_starts, batch_size, seq_len, rng)`` —
  fancy-indexed ``(B, T, ...)`` batch of pixels and actions.
* ``to_device_seq(pixels_np, actions_np, action_dim, device)`` — move to
  device, normalise pixels to ``[-1, 1]``, one-hot the actions.

Kept as free functions (not a class) so the existing ``train_lewm.py``
call-sites can swap to imports with no other refactor.
"""

from __future__ import annotations

import h5py
import numpy as np
import torch
from torch import nn


def valid_seq_start_indices(f: h5py.File, seq_len: int) -> np.ndarray:
    """Indices ``i`` such that ``[i, i+seq_len-1]`` are all in the same episode and non-terminal."""
    n = f["action"].shape[0]
    starts = f["episode_starts"][:]
    dones = f["done"][:]
    next_start = np.concatenate([starts[1:], [n]])
    is_last = np.zeros(n, dtype=bool)
    is_last[next_start - 1] = True

    candidates = np.arange(n - seq_len + 1)
    valid = np.ones(candidates.size, dtype=bool)
    for k in range(seq_len):
        idx = candidates + k
        valid &= dones[idx] == 0
        if k < seq_len - 1:
            valid &= ~is_last[idx]
    return candidates[valid]


def sample_sequence_batch(
    f: h5py.File,
    valid_starts: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(pixels, actions)`` shaped ``(B, T, C, H, W)`` and ``(B, T)``."""
    pick = rng.choice(valid_starts, size=batch_size, replace=valid_starts.size < batch_size)
    all_idx = (pick[:, None] + np.arange(seq_len)[None, :]).reshape(-1)
    union = np.unique(all_idx)
    pixels_block = f["pixels"][union]
    actions_block = f["action"][union]
    lookup = {int(v): k for k, v in enumerate(union)}
    flat_lookup = np.array([lookup[int(i)] for i in all_idx])
    pixels = pixels_block[flat_lookup].reshape(batch_size, seq_len, *pixels_block.shape[1:])
    actions = actions_block[flat_lookup].reshape(batch_size, seq_len)
    return pixels, actions


def sample_sequence_batch_with_extras(
    f: h5py.File,
    valid_starts: np.ndarray,
    batch_size: int,
    seq_len: int,
    rng: np.random.Generator,
    extra_keys: tuple[str, ...] = ("reward", "done"),
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    """Like ``sample_sequence_batch`` but also returns named top-level columns.

    Each entry in ``extra_keys`` must be a top-level dataset in ``f``. Works
    for both 1-D scalars (``reward``, ``done``) and 2-D rows
    (``state_vector`` of shape ``(N, 52)``); shape is preserved as
    ``(B, T)`` or ``(B, T, *trailing)`` respectively.
    """
    pick = rng.choice(valid_starts, size=batch_size, replace=valid_starts.size < batch_size)
    all_idx = (pick[:, None] + np.arange(seq_len)[None, :]).reshape(-1)
    union = np.unique(all_idx)
    pixels_block = f["pixels"][union]
    actions_block = f["action"][union]
    extras_blocks: dict[str, np.ndarray] = {k: f[k][union] for k in extra_keys}
    lookup = {int(v): k for k, v in enumerate(union)}
    flat_lookup = np.array([lookup[int(i)] for i in all_idx])
    pixels = pixels_block[flat_lookup].reshape(batch_size, seq_len, *pixels_block.shape[1:])
    actions = actions_block[flat_lookup].reshape(batch_size, seq_len)
    extras = {
        k: extras_blocks[k][flat_lookup].reshape(batch_size, seq_len, *extras_blocks[k].shape[1:])
        for k in extra_keys
    }
    return pixels, actions, extras


def to_device_seq(
    pixels_np: np.ndarray,
    actions_np: np.ndarray,
    action_dim: int,
    device: torch.device,
    stride: int = 1,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Move pixels (``[-1, 1]`` normalised float32) and one-hot actions to ``device``.

    ``stride`` (temporal frameskip): when >1, ``actions_np`` is expected to be
    the FULL raw (un-subsampled) span of shape ``(B, (steps)*stride[+1])`` —
    see ``replay_buffer._RAW_STRIDE_KEYS``. Every raw one-hot action inside
    each ``stride``-sized block is concatenated into one
    ``stride * action_dim``-wide vector per step, matching the original LeWM
    paper's ``effective_act_dim = frameskip * action_dim`` convention (the AR
    predictor needs to know everything that happened during the skipped
    frames, not just a single snapshot action).
    """
    pixels = (
        torch.from_numpy(pixels_np).to(device, dtype=torch.float32).div_(127.5).sub_(1.0)
    )
    actions = torch.from_numpy(actions_np.astype(np.int64)).to(device)
    a_oh_raw = nn.functional.one_hot(actions, num_classes=action_dim).float()
    if stride == 1:
        return pixels, a_oh_raw
    b, span = actions.shape
    steps = span // stride
    a_oh = a_oh_raw[:, : steps * stride].reshape(b, steps, stride * action_dim)
    return pixels, a_oh


def reduce_reward_seq(rewards_raw: torch.Tensor, stride: int) -> torch.Tensor:
    """Sum raw per-frame rewards into one value per ``stride``-sized block.

    ``rewards_raw``: ``(B, span)`` full raw span (see ``_RAW_STRIDE_KEYS``).
    Returns ``(B, steps)`` where ``steps = span // stride`` — the reward
    earned while executing one step's action-block, matching the frameskip
    convention where a "step" covers ``stride`` raw environment frames.
    ``stride=1`` returns the input unchanged (steps == span).
    """
    if stride == 1:
        return rewards_raw
    b, span = rewards_raw.shape
    steps = span // stride
    return rewards_raw[:, : steps * stride].reshape(b, steps, stride).sum(dim=-1)
