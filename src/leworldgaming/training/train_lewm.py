"""LeWM training loop — JEPA prediction loss + SIGReg over short sequences.

Pulls ``T+1``-frame sequences (default ``T=history_size=3``) from an HDF5
replay, runs::

    emb       = projector(encoder(o[:, :T+1]))         # (B, T+1, D)
    act_emb   = action_encoder(actions[:, :T])         # (B, T,   D)
    ctx_emb   = emb[:, :T]                             # history
    tgt_emb   = emb[:, 1:]                             # one-step ahead targets
    pred_emb  = pred_proj(predictor(ctx_emb, act_emb)) # (B, T,   D)
    L = ||pred_emb - tgt_emb||^2 + lambda * SIGReg(emb)

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
* As in the reference source, prediction loss flows through both branches.

Validation: a tail slice of valid sequence-start indices is held out;
``pred_loss`` / SIGReg are evaluated on it periodically with BN in train
mode (avoids running-stats lag — see projector.py).

See gemini_research.md §6, §7.2.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn

from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.projector import Projector
from leworldgaming.agents.lewm.sigreg import SIGReg
from leworldgaming.data.replay_buffer import DataReader
from leworldgaming.training._replay_utils import (
    to_device_seq as _to_device_seq,
)
from leworldgaming.utils.device import amp_autocast, best_device
from leworldgaming.utils.seed import set_seed

DEFAULTS: dict[str, Any] = {
    # Architecture
    "latent_dim": 192,
    "action_dim": 56,
    "projector_hidden": 2048,
    "history_size": 3,
    # Temporal frameskip: number of raw environment frames per training
    # "step". stride=1 (default) is unstrided/backward-compatible. stride=5
    # matches the original LeWM paper's frameskip convention — each step's
    # observation is the frame at the block's start, and its action is the
    # concatenation of all `stride` raw one-hot actions within the block
    # (effective ActionEncoder input dim becomes action_dim * temporal_stride).
    "temporal_stride": 1,
    # Encoder (ViT)
    "encoder_image_size": 224,
    "encoder_patch_size": 14,
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
    "batch_size": 128,
    "lr": 5.0e-5,
    "weight_decay": 1.0e-3,
    "grad_clip": 1.0,
    # Loss
    "sigreg_lambda": 0.09,
    "sigreg_knots": 17,
    "sigreg_num_proj": 1024,
    # Data + I/O
    "data_path": "data/replay.h5",
    "ckpt_path": "data/lewm_checkpoint.pt",
    "log_every": 10,
    "val_split": 0.1,
    "val_every": 25,
    "val_batches": 8,
    "ckpt_every": 1000,
    "seed": 3072,
    # Resume: if True and ckpt_path exists, load weights + optimizer state and
    # continue from the saved step. num_steps is treated as the TOTAL target.
    "resume": False,
}


def _load_config(path: str | Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is not None and Path(path).exists():
        with open(path) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        cfg.update({k: v for k, v in file_cfg.items() if k in DEFAULTS})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


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
    stride = int(cfg.get("temporal_stride", 1) or 1)
    print(
        f"[train_lewm] device={device} steps={num_steps} batch_size={cfg['batch_size']} "
        f"history={history_size} temporal_stride={stride}"
    )
    print(f"[train_lewm] data={cfg['data_path']} ckpt={cfg['ckpt_path']}")

    if not Path(cfg["data_path"]).exists():
        raise FileNotFoundError(
            f"{cfg['data_path']} not found — run scripts/collect_data.py --pixels first, "
            "or point --data-path at a directory of .h5 files"
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
    # ActionEncoder's input width scales with the frameskip: it needs to see
    # every raw one-hot action inside a stride block, concatenated, not just
    # one snapshot action (matches the original LeWM paper's
    # ``effective_act_dim = frameskip * action_dim``).
    action_encoder = ActionEncoder(
        action_dim=int(cfg["action_dim"]) * stride,
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

    # Optional resume: load weights + optimizer state and continue from the
    # saved step. num_steps is the TOTAL target, so re-running the same command
    # with a higher --steps extends training; an equal --steps is a no-op.
    ckpt_path = Path(cfg["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    start_step = 0
    resume_ckpt: dict[str, Any] | None = None
    if bool(cfg.get("resume", False)) and ckpt_path.exists():
        resume_ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        encoder.load_state_dict(resume_ckpt["encoder"])
        projector.load_state_dict(resume_ckpt["projector"])
        action_encoder.load_state_dict(resume_ckpt["action_encoder"])
        predictor.load_state_dict(resume_ckpt["predictor"])
        pred_proj.load_state_dict(resume_ckpt["pred_proj"])
        if "optim" in resume_ckpt:
            optim.load_state_dict(resume_ckpt["optim"])
        start_step = int(resume_ckpt.get("num_steps", 0))
        print(
            f"[train_lewm] resumed from {ckpt_path} at step={start_step} "
            f"-> training to {num_steps}"
        )
        if start_step >= num_steps:
            print(
                f"[train_lewm] nothing to do: start_step ({start_step}) >= "
                f"num_steps ({num_steps}). Raise --steps to continue."
            )
    elif bool(cfg.get("resume", False)):
        raise FileNotFoundError(
            f"--resume was requested but checkpoint does not exist: {ckpt_path}"
        )

    rng = np.random.default_rng(int(cfg["seed"]))
    if resume_ckpt is not None:
        if "numpy_rng_state" in resume_ckpt:
            rng.bit_generator.state = resume_ckpt["numpy_rng_state"]
        if "torch_rng_state" in resume_ckpt:
            torch.set_rng_state(resume_ckpt["torch_rng_state"].cpu())
        if torch.cuda.is_available() and "cuda_rng_state_all" in resume_ckpt:
            torch.cuda.set_rng_state_all(
                [state.cpu() for state in resume_ckpt["cuda_rng_state_all"]]
            )

    sigreg_lambda = float(cfg["sigreg_lambda"])
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    val_every = int(cfg["val_every"])
    batch_size = int(cfg["batch_size"])
    action_dim = int(cfg["action_dim"])
    ckpt_every = int(cfg.get("ckpt_every", 0) or 0)

    def _save_ckpt(step_done: int, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        torch.save(
            {
                "encoder": encoder.state_dict(),
                "projector": projector.state_dict(),
                "action_encoder": action_encoder.state_dict(),
                "predictor": predictor.state_dict(),
                "pred_proj": pred_proj.state_dict(),
                "optim": optim.state_dict(),
                "config": cfg,
                "num_steps": step_done,
                "format_version": 2,
                "pixel_normalization": "imagenet",
                "numpy_rng_state": rng.bit_generator.state,
                "torch_rng_state": torch.get_rng_state(),
                "cuda_rng_state_all": (
                    torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
                ),
            },
            tmp,
        )
        tmp.replace(dest)

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
        pred_loss = nn.functional.mse_loss(pred, tgt_emb)
        # SIGReg expects (T, B, D) per the reference, averages over T.
        reg_loss = sigreg(emb.transpose(0, 1).reshape(t, b, -1))
        z_norm = ctx_emb.float().norm(dim=-1).mean()
        return pred_loss, reg_loss, z_norm

    with DataReader(cfg["data_path"]) as reader:
        valid_starts = reader.valid_seq_starts(seq_len, stride=stride)
        if valid_starts.size == 0:
            raise RuntimeError("No valid sequences in replay buffer.")

        if not reader.has_pixels():
            raise RuntimeError(
                "LeWM requires pixel data — re-collect with --pixels or check your data directory."
            )

        train_starts, val_starts = valid_starts.split_by_episode(
            float(cfg["val_split"]), int(cfg["seed"])
        )
        print(
            f"[train_lewm] files={reader.num_files} frames={reader.total_frames} "
            f"valid_seq_starts={valid_starts.size} "
            f"train={train_starts.size} val={val_starts.size} "
            f"episodes={reader.total_episodes}"
        )

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            modules.eval()
            losses: list[float] = []
            sigregs: list[float] = []
            znorms: list[float] = []
            n_batches = max(1, val_starts.size // batch_size)
            val_batches = int(cfg.get("val_batches", 0) or 0)
            if val_batches > 0:
                n_batches = min(n_batches, val_batches)
            val_rng = np.random.default_rng(12345)
            for _ in range(n_batches):
                batch = reader.sample_window(
                    val_starts, batch_size, seq_len, val_rng, stride=stride
                )
                pixels, a_oh = _to_device_seq(
                    batch["pixels"], batch["action"], action_dim, device, stride=stride
                )
                pl, rl, zn = _step_forward(pixels, a_oh)
                losses.append(pl.item())
                sigregs.append(rl.item())
                znorms.append(zn.item())
            result = {
                "val_pred_loss": float(np.mean(losses)),
                "val_sigreg": float(np.mean(sigregs)),
                "val_z_norm": float(np.mean(znorms)),
            }
            modules.train()
            return result

        t0 = time.time()
        modules.train()
        for step in range(start_step, num_steps):
            batch = reader.sample_window(
                train_starts, batch_size, seq_len, rng, stride=stride
            )
            pixels, a_oh = _to_device_seq(
                batch["pixels"], batch["action"], action_dim, device, stride=stride
            )

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

            completed_steps = step + 1
            if ckpt_every > 0 and completed_steps % ckpt_every == 0:
                _save_ckpt(completed_steps, ckpt_path)
                print(
                    f"[train_lewm] step={completed_steps:5d}  "
                    f"saved checkpoint -> {ckpt_path}"
                )

        elapsed = time.time() - t0
        steps_run = max(num_steps - start_step, 0)
        rate = steps_run / elapsed if elapsed > 0 else 0.0
        print(f"[train_lewm] done in {elapsed:.1f}s ({rate:.1f} step/s, {steps_run} steps)")

    # Save with the highest step count reached, so a no-op resume (start_step >=
    # num_steps) never regresses the checkpoint's recorded progress.
    steps_done = max(start_step, num_steps)
    _save_ckpt(steps_done, ckpt_path)
    print(f"[train_lewm] saved checkpoint -> {ckpt_path}")

    return {
        "ckpt_path": str(ckpt_path),
        "final_train": history[-1] if history else {},
        "final_val": val_history[-1] if val_history else {},
        "history": history,
        "val_history": val_history,
    }
