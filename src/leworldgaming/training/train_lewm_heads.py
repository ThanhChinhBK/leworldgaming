"""LeWM Stage-B head trainer.

Loads a Stage-A checkpoint (from ``train_lewm.py``), freezes the JEPA
components (encoder, projector, action_encoder, predictor, pred_proj) and
trains four small heads on ``data/replay.h5`` so LeWM can be plugged into
MCTS like Dreamer / PETS::

    RewardHead(z, a_emb)       — twohot CE on reward[t]            (M2)
    ContinuationHead(z)        — BCE on (1 - done)                 (M2)
    ValueHead(z)               — twohot CE on λ-return + EMA target (M3)
    LinearProbe(z)             — MSE on physical state vector      (M5)

Imagined-rollout consistency over predictor-rolled latents is added in M4
(controlled by ``heads.imagined_horizon`` / ``heads.imagined_loss_weight``).

This module is built up incrementally (M2 → M5); the head_loss_weight
config keys gate each stage so partial features can be trained / disabled
without forking the file.

Architecture for the JEPA components is loaded from the input checkpoint's
stored ``config`` — never re-specify it here. Head dimensions come from
``heads.*`` in ``configs/lewm_heads.yaml``.

# Data caveat (M2)

``replay.h5`` from the current ``collect_data.py`` only contains a small
number of episodes (typically <10). Continuation labels are therefore
~constant 1 across most windows, which gives a weak training signal but
does not break the pipeline. M3+ should revisit with terminal-aware
sampling once data collection is scaled up.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.nn import functional as F

from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.continuation_head import ContinuationHead
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.probe import LinearProbe
from leworldgaming.agents.lewm.projector import Projector
from leworldgaming.agents.lewm.reward_head import RewardHead
from leworldgaming.agents.lewm.twohot import make_bins, twohot_ce_loss, twohot_decode
from leworldgaming.agents.lewm.value_head import ValueHead
from leworldgaming.data.replay_buffer import DataReader
from leworldgaming.training._replay_utils import (
    to_device_seq,
)
from leworldgaming.utils.device import amp_autocast, best_device
from leworldgaming.utils.seed import set_seed

DEFAULTS: dict[str, Any] = {
    "heads": {
        "reward_bins": 41,
        "reward_low": -1.0,
        "reward_high": 1.0,
        "value_bins": 41,
        "value_low": -10.0,
        "value_high": 10.0,
        "hidden_dim": 512,
        "imagined_horizon": 0,
        "lambda_return": 0.95,
        "gamma": 0.997,
        "target_ema": 0.99,
        "reward_loss_weight": 1.0,
        "cont_loss_weight": 1.0,
        "value_loss_weight": 0.0,
        "imagined_loss_weight": 0.0,
        "probe_loss_weight": 0.0,
        # Indices into the legacy 52-dim state_vector to use as probe targets.
        # Default matches planner.py:64 convention: probe[..., 0] = HP-diff.
        # 47 = hp_diff (signed), 0 = hp_self, 22 = hp_opp, 46 = distance.
        "probe_targets": [47, 0, 22, 46],
    },
    "batch_size": 16,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    "grad_clip": 1.0,
    "data_path": "data/replay.h5",
    "ckpt_in": "data/lewm_checkpoint.pt",
    "ckpt_out": "data/lewm_heads_checkpoint.pt",
    "log_every": 10,
    "val_split": 0.1,
    "val_every": 25,
    "seed": 0,
}


def _load_config(path: str | Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg: dict[str, Any] = {k: (dict(v) if isinstance(v, dict) else v) for k, v in DEFAULTS.items()}
    if path is not None and Path(path).exists():
        with open(path) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        for k, v in file_cfg.items():
            if k == "heads" and isinstance(v, dict):
                cfg["heads"].update(v)
            elif k in DEFAULTS:
                cfg[k] = v
    for k, v in overrides.items():
        if v is None:
            continue
        if k in DEFAULTS:
            cfg[k] = v
    return cfg


def _build_jepa_from_ckpt(ckpt: dict[str, Any], device: torch.device) -> dict[str, nn.Module]:
    """Rebuild Stage-A modules from the checkpoint's stored config and load weights frozen."""
    arch = ckpt.get("config")
    if arch is None:
        raise RuntimeError("Stage-A checkpoint is missing 'config' — cannot rebuild architecture.")

    history_size = int(arch["history_size"])
    latent_dim = int(arch["latent_dim"])
    action_dim = int(arch["action_dim"])
    projector_hidden = int(arch["projector_hidden"])

    encoder = Encoder(
        latent_dim=latent_dim,
        image_size=int(arch["encoder_image_size"]),
        patch_size=int(arch["encoder_patch_size"]),
        embed_dim=int(arch["encoder_embed_dim"]),
        depth=int(arch["encoder_depth"]),
        num_heads=int(arch["encoder_heads"]),
        mlp_ratio=float(arch["encoder_mlp_ratio"]),
        dropout=float(arch["encoder_dropout"]),
    ).to(device)
    projector = Projector(latent_dim=latent_dim, hidden_dim=projector_hidden).to(device)
    action_encoder = ActionEncoder(action_dim=action_dim, emb_dim=latent_dim).to(device)
    predictor = Predictor(
        latent_dim=latent_dim,
        action_dim=latent_dim,
        history_size=history_size,
        depth=int(arch["predictor_depth"]),
        num_heads=int(arch["predictor_heads"]),
        dim_head=int(arch["predictor_dim_head"]),
        mlp_dim=int(arch["predictor_mlp_dim"]),
        dropout=float(arch["predictor_dropout"]),
    ).to(device)
    pred_proj = Projector(latent_dim=latent_dim, hidden_dim=projector_hidden).to(device)

    encoder.load_state_dict(ckpt["encoder"])
    projector.load_state_dict(ckpt["projector"])
    action_encoder.load_state_dict(ckpt["action_encoder"])
    predictor.load_state_dict(ckpt["predictor"])
    pred_proj.load_state_dict(ckpt["pred_proj"])

    for m in (encoder, projector, action_encoder, predictor, pred_proj):
        for p in m.parameters():
            p.requires_grad_(False)
        m.eval()

    return {
        "encoder": encoder,
        "projector": projector,
        "action_encoder": action_encoder,
        "predictor": predictor,
        "pred_proj": pred_proj,
        "_arch": arch,
        "_history_size": history_size,
        "_latent_dim": latent_dim,
        "_action_dim": action_dim,
    }


def train(
    num_steps: int = 1000,
    config_path: str | Path | None = "configs/lewm_heads.yaml",
    **overrides: Any,
) -> dict[str, Any]:
    """Run the LeWM Stage-B head training loop. Returns a dict of final metrics."""
    cfg = _load_config(config_path, overrides)
    set_seed(int(cfg["seed"]))
    device = best_device()
    hcfg = cfg["heads"]

    print(f"[train_lewm_heads] device={device} steps={num_steps} batch_size={cfg['batch_size']}")
    print(f"[train_lewm_heads] data={cfg['data_path']} ckpt_in={cfg['ckpt_in']} ckpt_out={cfg['ckpt_out']}")

    if not Path(cfg["data_path"]).exists():
        raise FileNotFoundError(
            f"{cfg['data_path']} not found — run scripts/collect_data.py --pixels first, "
            "or point --data-path at a directory of .h5 files"
        )
    if not Path(cfg["ckpt_in"]).exists():
        raise FileNotFoundError(
            f"{cfg['ckpt_in']} not found — train Stage A first via scripts/train.py --agent lewm"
        )

    ckpt_in = torch.load(cfg["ckpt_in"], map_location=device, weights_only=False)
    jepa = _build_jepa_from_ckpt(ckpt_in, device)
    encoder = jepa["encoder"]
    projector = jepa["projector"]
    action_encoder = jepa["action_encoder"]
    predictor = jepa["predictor"]
    pred_proj = jepa["pred_proj"]
    history_size: int = jepa["_history_size"]
    latent_dim: int = jepa["_latent_dim"]
    action_dim: int = jepa["_action_dim"]
    imagined_horizon = int(hcfg["imagined_horizon"])
    # Window covers `history_size` real frames + `imagined_horizon` rolled
    # positions (+1 for the encoder-grounded last position when K=0).
    seq_len = history_size + max(1, imagined_horizon)

    reward_head = RewardHead(
        latent_dim=latent_dim,
        hidden_dim=int(hcfg["hidden_dim"]),
        num_bins=int(hcfg["reward_bins"]),
    ).to(device)
    continuation_head = ContinuationHead(
        latent_dim=latent_dim,
        hidden_dim=int(hcfg["hidden_dim"]),
    ).to(device)
    value_head = ValueHead(
        latent_dim=latent_dim,
        hidden_dim=int(hcfg["hidden_dim"]),
        num_bins=int(hcfg["value_bins"]),
    ).to(device)
    value_target_head = ValueHead(
        latent_dim=latent_dim,
        hidden_dim=int(hcfg["hidden_dim"]),
        num_bins=int(hcfg["value_bins"]),
    ).to(device)
    value_target_head.load_state_dict(value_head.state_dict())
    for p in value_target_head.parameters():
        p.requires_grad_(False)
    value_target_head.eval()

    probe_targets = list(hcfg["probe_targets"])
    probe = LinearProbe(latent_dim=latent_dim, target_dim=len(probe_targets)).to(device)

    reward_bins = make_bins(int(hcfg["reward_bins"]), float(hcfg["reward_low"]), float(hcfg["reward_high"]), device)
    value_bins = make_bins(int(hcfg["value_bins"]), float(hcfg["value_low"]), float(hcfg["value_high"]), device)

    head_modules = nn.ModuleDict({
        "reward_head": reward_head,
        "continuation_head": continuation_head,
        "value_head": value_head,
        "probe": probe,
    })
    n_params = sum(p.numel() for p in head_modules.parameters()) / 1e6
    print(f"[train_lewm_heads] head params: {n_params:.2f}M (frozen JEPA: {sum(p.numel() for p in jepa['encoder'].parameters() if not p.requires_grad) / 1e6:.2f}M+ ...)")

    optim = torch.optim.AdamW(
        head_modules.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    rng = np.random.default_rng(int(cfg["seed"]))
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    val_every = int(cfg["val_every"])
    batch_size = int(cfg["batch_size"])
    w_r = float(hcfg["reward_loss_weight"])
    w_c = float(hcfg["cont_loss_weight"])
    w_v = float(hcfg["value_loss_weight"])
    w_im = float(hcfg["imagined_loss_weight"])
    w_probe = float(hcfg["probe_loss_weight"])
    gamma = float(hcfg["gamma"])
    lam = float(hcfg["lambda_return"])
    target_ema = float(hcfg["target_ema"])
    K = imagined_horizon
    probe_idx = torch.as_tensor(probe_targets, dtype=torch.long, device=device)

    history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []

    @torch.no_grad()
    def _encode_grounded(pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, T, D) post-projector embeddings, JEPA frozen."""
        b, t = pixels.shape[:2]
        flat = pixels.reshape(b * t, *pixels.shape[2:])
        emb = projector(encoder(flat))
        return emb.reshape(b, t, -1)

    def _lambda_return(
        rewards: torch.Tensor,
        cont: torch.Tensor,
        v_boot: torch.Tensor,
    ) -> torch.Tensor:
        """Dreamer-style λ-return.

        ``rewards``: (B, T) reward earned at step t (i.e. for transition t→t+1
                     under our one-hot reward convention).
        ``cont``:    (B, T) continuation prob at step t (1 - done).
        ``v_boot``:  (B, T) bootstrap value at step t (from frozen target net).

        Returns ``G`` of shape (B, T):
            G_{T-1} = v_boot_{T-1}
            G_t     = r_t + γ · cont_{t+1} · ((1-λ) · v_boot_{t+1} + λ · G_{t+1})
        Computed without grads — used as twohot CE target.
        """
        B, T = rewards.shape
        G = torch.zeros_like(rewards)
        G[:, -1] = v_boot[:, -1]
        for t in range(T - 2, -1, -1):
            G[:, t] = rewards[:, t] + gamma * cont[:, t + 1] * (
                (1.0 - lam) * v_boot[:, t + 1] + lam * G[:, t + 1]
            )
        return G

    def _step_forward(
        pixels: torch.Tensor,
        a_oh: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        state_vec: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z = _encode_grounded(pixels)                    # (B, T, D)
        a_emb = action_encoder(a_oh)                    # (B, T, D)

        # Reward at index t: r_head(z[t], a_emb[t]) -> reward[t]
        r_logits = reward_head(z, a_emb)                # (B, T, K)
        loss_r = twohot_ce_loss(r_logits, rewards, reward_bins)

        # Continuation at index t (M2: predict cont[t] = 1 - done[t] from z[t]).
        # In valid windows from valid_seq_start_indices, dones is all-zero by
        # construction, so this is effectively a constant target — kept for
        # pipeline correctness. M3+ will sample terminal-tail windows.
        c_logits = continuation_head(z)                 # (B, T)
        cont_target = (1.0 - dones.float())
        loss_c = F.binary_cross_entropy_with_logits(c_logits, cont_target)

        loss = w_r * loss_r + w_c * loss_c
        loss_v_val = torch.zeros((), device=device)
        loss_im_val = torch.zeros((), device=device)
        loss_r_im_val = torch.zeros((), device=device)
        loss_c_im_val = torch.zeros((), device=device)

        if w_v > 0.0:
            v_logits = value_head(z)                     # (B, T, K_v)
            with torch.no_grad():
                v_tgt_logits = value_target_head(z)
                v_boot = twohot_decode(v_tgt_logits, value_bins)   # (B, T)
                G = _lambda_return(rewards, cont_target, v_boot)   # (B, T)
            loss_v = twohot_ce_loss(v_logits, G, value_bins)
            loss = loss + w_v * loss_v
            loss_v_val = loss_v.detach()

        if w_im > 0.0 and K > 0:
            # Roll the frozen predictor forward K steps starting from the
            # encoder-grounded history z[:, :history_size]. At each step we
            # take the last position of the AR predictor as the next-step
            # latent, push it onto the history, and slide the action window.
            with torch.no_grad():
                z_hist = z[:, :history_size].contiguous()           # (B, hs, D)
                imagined_list: list[torch.Tensor] = []
                for k in range(K):
                    a_window = a_emb[:, k : k + history_size]       # (B, hs, D)
                    z_pred = pred_proj(predictor(z_hist, a_window)[:, -1])  # (B, D)
                    imagined_list.append(z_pred)
                    z_hist = torch.cat([z_hist[:, 1:], z_pred.unsqueeze(1)], dim=1)
                z_im = torch.stack(imagined_list, dim=1)            # (B, K, D)

            # Heads see the imagined latents WITHOUT a stop-grad because we
            # want gradients to flow into the heads. The predictor itself is
            # frozen and its weights aren't updated.
            a_emb_im = a_emb[:, history_size : history_size + K]    # (B, K, D)
            r_logits_im = reward_head(z_im, a_emb_im)               # (B, K, R_bins)
            loss_r_im = twohot_ce_loss(
                r_logits_im,
                rewards[:, history_size : history_size + K],
                reward_bins,
            )
            c_logits_im = continuation_head(z_im)                   # (B, K)
            cont_im_target = 1.0 - dones[:, history_size : history_size + K].float()
            loss_c_im = F.binary_cross_entropy_with_logits(c_logits_im, cont_im_target)
            loss_imagined = loss_r_im + loss_c_im
            loss = loss + w_im * loss_imagined
            loss_im_val = loss_imagined.detach()
            loss_r_im_val = loss_r_im.detach()
            loss_c_im_val = loss_c_im.detach()

        loss_probe_val = torch.zeros((), device=device)
        if w_probe > 0.0 and state_vec is not None:
            # state_vec: (B, T, 52). Pick the configured target columns.
            phys = state_vec.index_select(dim=-1, index=probe_idx)   # (B, T, P)
            probe_pred = probe(z)                                    # (B, T, P)
            loss_probe = F.mse_loss(probe_pred, phys)
            loss = loss + w_probe * loss_probe
            loss_probe_val = loss_probe.detach()

        return {
            "loss": loss,
            "loss_r": loss_r.detach(),
            "loss_c": loss_c.detach(),
            "loss_v": loss_v_val,
            "loss_im": loss_im_val,
            "loss_r_im": loss_r_im_val,
            "loss_c_im": loss_c_im_val,
            "loss_probe": loss_probe_val,
            "z_norm": z.float().norm(dim=-1).mean().detach(),
        }

    @torch.no_grad()
    def _ema_update_target() -> None:
        for p, p_tgt in zip(value_head.parameters(), value_target_head.parameters()):
            p_tgt.data.mul_(target_ema).add_(p.data, alpha=1.0 - target_ema)

    with DataReader(cfg["data_path"]) as reader:
        valid_starts = reader.valid_seq_starts(seq_len)
        if valid_starts.size == 0:
            raise RuntimeError("No valid sequences in replay buffer.")

        n = valid_starts.size
        n_val = max(int(n * float(cfg["val_split"])), batch_size)
        n_val = min(n_val, n - batch_size)
        train_starts = valid_starts[: n - n_val]
        val_starts = valid_starts[n - n_val :]
        print(
            f"[train_lewm_heads] files={reader.num_files} frames={reader.total_frames} "
            f"valid_seq_starts={n} train={train_starts.size} val={val_starts.size} "
            f"episodes={reader.total_episodes}"
        )

        # Probe needs state_vector; only fetch it if enabled and available.
        extra_keys: tuple[str, ...] = ("reward", "done")
        if w_probe > 0.0 and reader.has_key("state_vector"):
            extra_keys = extra_keys + ("state_vector",)

        def _make_batch(
            starts, batch_rng: np.random.Generator
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
            batch = reader.sample_window(
                starts, batch_size, seq_len, batch_rng, extra_keys=extra_keys
            )
            pixels, a_oh = to_device_seq(batch["pixels"], batch["action"], action_dim, device)
            rewards = torch.from_numpy(batch["reward"]).to(device, dtype=torch.float32)
            dones = torch.from_numpy(batch["done"]).to(device, dtype=torch.float32)
            state_vec: torch.Tensor | None = None
            if "state_vector" in batch:
                state_vec = torch.from_numpy(batch["state_vector"]).to(device, dtype=torch.float32)
            return pixels, a_oh, rewards, dones, state_vec

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            head_modules.eval()
            losses_r: list[float] = []
            losses_c: list[float] = []
            losses_v: list[float] = []
            losses_r_im: list[float] = []
            losses_c_im: list[float] = []
            losses_probe: list[float] = []
            n_batches = max(1, val_starts.size // batch_size)
            val_rng = np.random.default_rng(12345)
            for _ in range(n_batches):
                pixels, a_oh, rewards, dones, state_vec = _make_batch(val_starts, val_rng)
                out = _step_forward(pixels, a_oh, rewards, dones, state_vec)
                losses_r.append(out["loss_r"].item())
                losses_c.append(out["loss_c"].item())
                losses_v.append(out["loss_v"].item())
                losses_r_im.append(out["loss_r_im"].item())
                losses_c_im.append(out["loss_c_im"].item())
                losses_probe.append(out["loss_probe"].item())
            head_modules.train()
            return {
                "val_loss_r": float(np.mean(losses_r)),
                "val_loss_c": float(np.mean(losses_c)),
                "val_loss_v": float(np.mean(losses_v)),
                "val_loss_r_im": float(np.mean(losses_r_im)),
                "val_loss_c_im": float(np.mean(losses_c_im)),
                "val_loss_probe": float(np.mean(losses_probe)),
            }

        t0 = time.time()
        head_modules.train()
        for step in range(num_steps):
            pixels, a_oh, rewards, dones, state_vec = _make_batch(train_starts, rng)

            with amp_autocast(device):
                out = _step_forward(pixels, a_oh, rewards, dones, state_vec)
                loss = out["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(head_modules.parameters(), grad_clip)
            optim.step()
            if w_v > 0.0:
                _ema_update_target()

            metrics = {
                "step": step,
                "loss": float(loss.item()),
                "loss_r": float(out["loss_r"].item()),
                "loss_c": float(out["loss_c"].item()),
                "loss_v": float(out["loss_v"].item()),
                "loss_r_im": float(out["loss_r_im"].item()),
                "loss_c_im": float(out["loss_c_im"].item()),
                "loss_probe": float(out["loss_probe"].item()),
                "z_norm": float(out["z_norm"].item()),
                "grad_norm": float(grad_norm.item()),
            }
            history.append(metrics)
            if step % log_every == 0 or step == num_steps - 1:
                print(
                    f"[train_lewm_heads] step={step:5d} train "
                    f"r={metrics['loss_r']:.4f} c={metrics['loss_c']:.4f} v={metrics['loss_v']:.4f} "
                    f"r_im={metrics['loss_r_im']:.4f} c_im={metrics['loss_c_im']:.4f} "
                    f"probe={metrics['loss_probe']:.4f} "
                    f"|z|={metrics['z_norm']:.2f} grad={metrics['grad_norm']:.2f}"
                )

            if val_every > 0 and (step % val_every == 0 or step == num_steps - 1):
                vm = evaluate()
                vm["step"] = step
                val_history.append(vm)
                print(
                    f"[train_lewm_heads] step={step:5d}  val  "
                    f"r={vm['val_loss_r']:.4f} c={vm['val_loss_c']:.4f} v={vm['val_loss_v']:.4f} "
                    f"r_im={vm['val_loss_r_im']:.4f} c_im={vm['val_loss_c_im']:.4f} "
                    f"probe={vm['val_loss_probe']:.4f}"
                )

        elapsed = time.time() - t0
        print(f"[train_lewm_heads] done in {elapsed:.1f}s ({num_steps / elapsed:.1f} step/s)")

    ckpt_out = Path(cfg["ckpt_out"])
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    save_dict: dict[str, Any] = {
        # Re-save Stage-A weights so this checkpoint is self-contained for inference.
        "encoder": ckpt_in["encoder"],
        "projector": ckpt_in["projector"],
        "action_encoder": ckpt_in["action_encoder"],
        "predictor": ckpt_in["predictor"],
        "pred_proj": ckpt_in["pred_proj"],
        "reward_head": reward_head.state_dict(),
        "continuation_head": continuation_head.state_dict(),
        "value_head": value_head.state_dict(),
        "value_target_head": value_target_head.state_dict(),
        "probe": probe.state_dict(),
        "config": jepa["_arch"],
        "heads_config": hcfg,
        "stage": "B",
        "num_steps": num_steps,
    }
    torch.save(save_dict, ckpt_out)
    print(f"[train_lewm_heads] saved checkpoint -> {ckpt_out}")

    return {
        "ckpt_path": str(ckpt_out),
        "final_train": history[-1] if history else {},
        "final_val": val_history[-1] if val_history else {},
        "history": history,
        "val_history": val_history,
    }
