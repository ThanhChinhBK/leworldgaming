"""LeWM training loop — JEPA prediction loss + SIGReg over short sequences.

Pulls ``T+1``-frame sequences (default ``T=history_size=3``) from an HDF5
replay, runs::

    emb       = projector(encoder(o[:, :T+1]))         # (B, T+1, D)
    act_emb   = action_encoder(actions[:, :T])         # (B, T,   D)
    ctx_emb   = emb[:, :T]                             # history
    tgt_emb   = emb[:, 1:]                             # one-step ahead targets
    pred_emb  = pred_proj(predictor(ctx_emb, act_emb)) # (B, T,   D)
    L = ||pred_emb - sg(tgt_emb)||^2 + lambda * SIGReg(emb)

The AR predictor's causal self-attention turns one forward pass into ``T``
parallel next-step predictions per batch element — same compute, ``T``× the
supervision signal.

Architecture notes:

* ``Encoder`` is a compact ViT (default depth=6, ``encoder_depth: 12`` for
  full ViT-tiny parity).
* ``Projector`` / ``pred_proj`` are MLPs with BatchNorm1d at high momentum
  (0.5) for fast running-stat convergence — see projector.py for rationale.
* ``Predictor`` is the LeWM-style transformer with AdaLN-zero per-step action
  conditioning. ``ActionEncoder`` projects one-hot actions into the same
  ``latent_dim`` the predictor expects.
* SIGReg pulls the post-projector distribution toward N(0, I).
* Targets are detached, so loss flows through the prediction branch only.

Validation: a tail slice of valid sequence-start indices is held out;
``pred_loss`` / SIGReg are evaluated on it periodically with BN in train
mode (avoids running-stats lag — see projector.py).

See gemini_research.md §6, §7.2.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
import yaml
from torch import nn

from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.projector import Projector
from leworldgaming.agents.lewm.sigreg import SIGReg
from leworldgaming.utils.device import amp_autocast, best_device
from leworldgaming.utils.seed import set_seed

DEFAULTS: dict[str, Any] = {
    # Architecture
    "latent_dim": 256,
    "action_dim": 56,
    "projector_hidden": 2048,
    "history_size": 3,
    # Encoder (ViT)
    "encoder_image_size": 224,
    "encoder_patch_size": 16,
    "encoder_embed_dim": 192,
    "encoder_depth": 12,
    "encoder_heads": 3,
    "encoder_mlp_ratio": 4.0,
    "encoder_dropout": 0.0,
    # Predictor (AR transformer)
    "predictor_depth": 6,
    "predictor_heads": 16,
    "predictor_dim_head": 64,
    "predictor_mlp_dim": 2048,
    "predictor_dropout": 0.1,
    # Optimization
    "batch_size": 16,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "grad_clip": 1.0,
    # Loss
    "sigreg_lambda": 0.1,
    "sigreg_knots": 17,
    "sigreg_num_proj": 1024,
    # Data + I/O
    "data_path": "data/replay.h5",
    "ckpt_path": "data/lewm_checkpoint.pt",
    "log_every": 10,
    "val_split": 0.1,
    "val_every": 25,
    "seed": 0,
}


def _load_config(path: str | Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is not None and Path(path).exists():
        with open(path) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        cfg.update({k: v for k, v in file_cfg.items() if k in DEFAULTS})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _valid_seq_start_indices(f: h5py.File, seq_len: int) -> np.ndarray:
    """Indices ``i`` such that ``[i, i+seq_len-1]`` are all in the same episode and non-terminal."""
    n = f["action"].shape[0]
    starts = f["episode_starts"][:]
    dones = f["done"][:]
    next_start = np.concatenate([starts[1:], [n]])
    # last-of-episode mask
    is_last = np.zeros(n, dtype=bool)
    is_last[next_start - 1] = True

    # An index i is a valid start iff every position j in [i, i+seq_len-1]:
    #   - exists (i + seq_len - 1 < n)
    #   - is not a `done` flag (no terminal in the middle of the window)
    #   - is in the same episode (no episode boundary inside the window)
    candidates = np.arange(n - seq_len + 1)
    # Vectorise the "no done & same episode" check via a rolling window.
    valid = np.ones(candidates.size, dtype=bool)
    for k in range(seq_len):
        idx = candidates + k
        valid &= dones[idx] == 0
        # Skip "is_last" in the middle (positions 0..seq_len-2). Last position
        # is allowed to be last-of-episode since we don't predict beyond it.
        if k < seq_len - 1:
            valid &= ~is_last[idx]
    return candidates[valid]


def _sample_sequence_batch(
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


def _to_device_seq(
    pixels_np: np.ndarray,
    actions_np: np.ndarray,
    action_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    pixels = (
        torch.from_numpy(pixels_np).to(device, dtype=torch.float32).div_(127.5).sub_(1.0)
    )  # (B, T, C, H, W)
    actions = torch.from_numpy(actions_np.astype(np.int64)).to(device)  # (B, T)
    a_oh = nn.functional.one_hot(actions, num_classes=action_dim).float()  # (B, T, A)
    return pixels, a_oh


def train(
    num_steps: int = 1000,
    config_path: str | Path | None = "configs/lewm.yaml",
    **overrides: Any,
) -> dict[str, Any]:
    """Run the LeWM training loop. Returns a dict of final metrics."""
    cfg = _load_config(config_path, overrides)
    set_seed(int(cfg["seed"]))
    device = best_device()
    history_size = int(cfg["history_size"])
    seq_len = history_size + 1
    print(
        f"[train_lewm] device={device} steps={num_steps} batch_size={cfg['batch_size']} "
        f"history={history_size}"
    )
    print(f"[train_lewm] data={cfg['data_path']} ckpt={cfg['ckpt_path']}")

    if not Path(cfg["data_path"]).exists():
        raise FileNotFoundError(
            f"{cfg['data_path']} not found — run scripts/collect_data.py --pixels first"
        )

    encoder = Encoder(
        latent_dim=int(cfg["latent_dim"]),
        image_size=int(cfg["encoder_image_size"]),
        patch_size=int(cfg["encoder_patch_size"]),
        embed_dim=int(cfg["encoder_embed_dim"]),
        depth=int(cfg["encoder_depth"]),
        num_heads=int(cfg["encoder_heads"]),
        mlp_ratio=float(cfg["encoder_mlp_ratio"]),
        dropout=float(cfg["encoder_dropout"]),
    ).to(device)
    projector = Projector(
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=int(cfg["projector_hidden"]),
    ).to(device)
    action_encoder = ActionEncoder(
        action_dim=int(cfg["action_dim"]),
        emb_dim=int(cfg["latent_dim"]),
    ).to(device)
    predictor = Predictor(
        latent_dim=int(cfg["latent_dim"]),
        action_dim=int(cfg["latent_dim"]),
        history_size=history_size,
        depth=int(cfg["predictor_depth"]),
        num_heads=int(cfg["predictor_heads"]),
        dim_head=int(cfg["predictor_dim_head"]),
        mlp_dim=int(cfg["predictor_mlp_dim"]),
        dropout=float(cfg["predictor_dropout"]),
    ).to(device)
    pred_proj = Projector(
        latent_dim=int(cfg["latent_dim"]),
        hidden_dim=int(cfg["projector_hidden"]),
    ).to(device)
    sigreg = SIGReg(
        knots=int(cfg["sigreg_knots"]),
        num_proj=int(cfg["sigreg_num_proj"]),
    ).to(device)

    modules = nn.ModuleDict({
        "encoder": encoder,
        "projector": projector,
        "action_encoder": action_encoder,
        "predictor": predictor,
        "pred_proj": pred_proj,
    })
    n_params = sum(p.numel() for p in modules.parameters()) / 1e6
    print(
        f"[train_lewm] params: {n_params:.2f}M total — "
        f"enc={sum(p.numel() for p in encoder.parameters()) / 1e6:.2f}M "
        f"proj={sum(p.numel() for p in projector.parameters()) / 1e6:.2f}M "
        f"act_enc={sum(p.numel() for p in action_encoder.parameters()) / 1e6:.2f}M "
        f"pred={sum(p.numel() for p in predictor.parameters()) / 1e6:.2f}M "
        f"pred_proj={sum(p.numel() for p in pred_proj.parameters()) / 1e6:.2f}M"
    )

    optim = torch.optim.AdamW(
        modules.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    rng = np.random.default_rng(int(cfg["seed"]))
    sigreg_lambda = float(cfg["sigreg_lambda"])
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    val_every = int(cfg["val_every"])
    batch_size = int(cfg["batch_size"])
    action_dim = int(cfg["action_dim"])

    history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []

    def _step_forward(
        pixels: torch.Tensor, a_oh: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder/projector/predictor and return (pred_loss, sigreg, |z|)."""
        b, t = pixels.shape[:2]
        pixels_flat = pixels.reshape(b * t, *pixels.shape[2:])
        emb_flat = projector(encoder(pixels_flat))
        emb = emb_flat.reshape(b, t, -1)  # (B, T+1, D)
        ctx_emb = emb[:, :history_size]  # (B, T, D)
        tgt_emb = emb[:, 1:]  # (B, T, D)
        ctx_act = action_encoder(a_oh[:, :history_size])  # (B, T, D)
        pred = pred_proj(predictor(ctx_emb, ctx_act).reshape(b * history_size, -1)).reshape(
            b, history_size, -1
        )
        pred_loss = nn.functional.mse_loss(pred, tgt_emb.detach())
        # SIGReg expects (T, B, D) per the reference, averages over T.
        reg_loss = sigreg(emb.transpose(0, 1).reshape(t, b, -1))
        z_norm = ctx_emb.float().norm(dim=-1).mean()
        return pred_loss, reg_loss, z_norm

    with h5py.File(cfg["data_path"], "r") as f:
        valid_starts = _valid_seq_start_indices(f, seq_len)
        if valid_starts.size == 0:
            raise RuntimeError("No valid sequences in replay buffer.")

        n = valid_starts.size
        n_val = max(int(n * float(cfg["val_split"])), batch_size)
        n_val = min(n_val, n - batch_size)
        train_starts = valid_starts[: n - n_val]
        val_starts = valid_starts[n - n_val :]
        print(
            f"[train_lewm] frames={f['action'].shape[0]} "
            f"valid_seq_starts={n} train={train_starts.size} val={val_starts.size} "
            f"episodes={f['episode_starts'].shape[0]}"
        )

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            losses: list[float] = []
            sigregs: list[float] = []
            znorms: list[float] = []
            n_batches = max(1, val_starts.size // batch_size)
            val_rng = np.random.default_rng(12345)
            for _ in range(n_batches):
                pixels_np, actions_np = _sample_sequence_batch(
                    f, val_starts, batch_size, seq_len, val_rng
                )
                pixels, a_oh = _to_device_seq(pixels_np, actions_np, action_dim, device)
                pl, rl, zn = _step_forward(pixels, a_oh)
                losses.append(pl.item())
                sigregs.append(rl.item())
                znorms.append(zn.item())
            return {
                "val_pred_loss": float(np.mean(losses)),
                "val_sigreg": float(np.mean(sigregs)),
                "val_z_norm": float(np.mean(znorms)),
            }

        t0 = time.time()
        modules.train()
        for step in range(num_steps):
            pixels_np, actions_np = _sample_sequence_batch(
                f, train_starts, batch_size, seq_len, rng
            )
            pixels, a_oh = _to_device_seq(pixels_np, actions_np, action_dim, device)

            with amp_autocast(device):
                pred_loss, reg_loss, z_norm = _step_forward(pixels, a_oh)
                loss = pred_loss + sigreg_lambda * reg_loss

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(modules.parameters(), grad_clip)
            optim.step()

            metrics = {
                "step": step,
                "loss": float(loss.item()),
                "pred_loss": float(pred_loss.item()),
                "sigreg": float(reg_loss.item()),
                "z_norm": float(z_norm.item()),
                "grad_norm": float(grad_norm.item()),
            }
            history.append(metrics)
            if step % log_every == 0 or step == num_steps - 1:
                print(
                    f"[train_lewm] step={step:5d} train "
                    f"pred={metrics['pred_loss']:.4f} "
                    f"sigreg={metrics['sigreg']:.4f} "
                    f"|z|={metrics['z_norm']:.2f} "
                    f"grad={metrics['grad_norm']:.2f}"
                )

            if val_every > 0 and (step % val_every == 0 or step == num_steps - 1):
                val_metrics = evaluate()
                val_metrics["step"] = step
                val_history.append(val_metrics)
                print(
                    f"[train_lewm] step={step:5d}  val  "
                    f"pred={val_metrics['val_pred_loss']:.4f} "
                    f"sigreg={val_metrics['val_sigreg']:.4f} "
                    f"|z|={val_metrics['val_z_norm']:.2f}"
                )

        elapsed = time.time() - t0
        print(f"[train_lewm] done in {elapsed:.1f}s ({num_steps / elapsed:.1f} step/s)")

    ckpt_path = Path(cfg["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "encoder": encoder.state_dict(),
            "projector": projector.state_dict(),
            "action_encoder": action_encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "pred_proj": pred_proj.state_dict(),
            "config": cfg,
            "num_steps": num_steps,
        },
        ckpt_path,
    )
    print(f"[train_lewm] saved checkpoint -> {ckpt_path}")

    return {
        "ckpt_path": str(ckpt_path),
        "final_train": history[-1] if history else {},
        "final_val": val_history[-1] if val_history else {},
        "history": history,
        "val_history": val_history,
    }
