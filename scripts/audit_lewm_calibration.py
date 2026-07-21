"""Offline audit: does LeWM's imagined-rollout latent/reward/value quality
degrade with planning depth (the "model exploitation" hypothesis)?

No live JVM match required — this reuses the exact same frozen JEPA +
Stage-B heads and the exact same predictor-rollout code path as
``train_lewm_heads.py``'s M4 imagined-loss branch, but breaks the result
down PER DEPTH (1..K) instead of aggregating over the whole horizon, and
adds a direct latent-space drift metric (imagined vs grounded latent),
which the trainer never computes.

For each depth k = 1..K, on held-out validation windows (same val split
as Stage-B training: same seed/val_split), reports:

  * latent cosine similarity / MSE between the predictor's imagined
    latent at depth k and the "ground truth" grounded (encoder) latent
    at the same timestep — measures pure world-model rollout drift.
  * reward-head prediction error (decoded MAE vs the real logged reward)
    when fed the IMAGINED latent at depth k, vs when fed the GROUNDED
    latent at the same timestep — isolates how much of the reward error
    is due to the world model's drift vs the reward head itself.
  * value-head decoded-value delta between imagined and grounded latents
    at depth k (no ground-truth value exists, so this is a calibration
    self-consistency metric, not an error).

Usage::

    .venv/bin/python scripts/audit_lewm_calibration.py \\
        --ckpt data/lewm_heads_checkpoint_stride2_m4_v3.pt \\
        --data "/media/jeovach/New Volume/leworldgaming" \\
        --num-windows 2048
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn

from leworldgaming.agents.lewm.continuation_head import ContinuationHead
from leworldgaming.agents.lewm.reward_head import RewardHead
from leworldgaming.agents.lewm.twohot import make_bins, twohot_decode
from leworldgaming.agents.lewm.value_head import ValueHead
from leworldgaming.data.replay_buffer import DataReader
from leworldgaming.training._replay_utils import reduce_reward_seq, to_device_seq
from leworldgaming.training.train_lewm_heads import _build_jepa_from_ckpt
from leworldgaming.utils.device import best_device


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ckpt", type=str, required=True, help="Stage-B checkpoint (self-contained).")
    p.add_argument("--data", type=str, required=True, help="replay.h5 file or directory.")
    p.add_argument("--num-windows", type=int, default=2048)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--val-split", type=float, default=0.1, help="Must match Stage-B training config.")
    p.add_argument("--seed", type=int, default=0, help="Must match Stage-B training config.")
    return p.parse_args()


@torch.no_grad()
def main() -> None:
    args = parse_args()
    device = best_device()
    print(f"[audit] device={device}")

    ckpt = torch.load(args.ckpt, map_location=device, weights_only=False)
    hcfg = ckpt["heads_config"]
    K = int(hcfg["imagined_horizon"])
    if K < 1:
        raise SystemExit(f"{args.ckpt} was trained with imagined_horizon={K}; nothing to audit.")

    jepa = _build_jepa_from_ckpt(ckpt, device)
    encoder, projector = jepa["encoder"], jepa["projector"]
    action_encoder, predictor, pred_proj = (
        jepa["action_encoder"], jepa["predictor"], jepa["pred_proj"]
    )
    history_size = jepa["_history_size"]
    latent_dim = jepa["_latent_dim"]
    action_dim = jepa["_action_dim"]
    stride = jepa["_temporal_stride"]

    reward_head = RewardHead(
        latent_dim=latent_dim, hidden_dim=int(hcfg["hidden_dim"]), num_bins=int(hcfg["reward_bins"])
    ).to(device)
    reward_head.load_state_dict(ckpt["reward_head"])
    value_head = ValueHead(
        latent_dim=latent_dim, hidden_dim=int(hcfg["hidden_dim"]), num_bins=int(hcfg["value_bins"])
    ).to(device)
    value_head.load_state_dict(ckpt["value_head"])
    continuation_head = ContinuationHead(
        latent_dim=latent_dim,
        hidden_dim=int(hcfg.get("cont_hidden_dim") or hcfg["hidden_dim"]),
        dropout=0.0,
    ).to(device)
    continuation_head.load_state_dict(ckpt["continuation_head"])
    for m in (reward_head, value_head, continuation_head):
        m.eval()

    reward_bins = make_bins(int(hcfg["reward_bins"]), float(hcfg["reward_low"]), float(hcfg["reward_high"]), device)
    value_bins = make_bins(int(hcfg["value_bins"]), float(hcfg["value_low"]), float(hcfg["value_high"]), device)

    transition_len = history_size + K
    state_len = transition_len + 1

    with DataReader(args.data) as reader:
        valid_starts = reader.valid_seq_starts(state_len, stride=stride)
        if valid_starts.size == 0:
            raise SystemExit("No valid sequences in replay buffer for this state_len/stride.")
        _, val_starts = valid_starts.split_by_episode(args.val_split, args.seed)
        print(
            f"[audit] files={reader.num_files} frames={reader.total_frames} "
            f"val_windows_available={val_starts.size} state_len={state_len} stride={stride}"
        )

        rng = np.random.default_rng(999)
        n_windows = 0
        # Per-depth accumulators (k=1..K)
        lat_cos = [[] for _ in range(K)]
        lat_mse = [[] for _ in range(K)]
        r_mae_grounded = [[] for _ in range(K)]
        r_mae_imagined = [[] for _ in range(K)]
        v_delta = [[] for _ in range(K)]
        r_true_all = [[] for _ in range(K)]  # for nonzero-reward (decisive-frame) breakdown

        while n_windows < args.num_windows:
            bs = min(args.batch_size, args.num_windows - n_windows)
            batch = reader.sample_window(
                val_starts, bs, state_len, rng, extra_keys=("reward", "done"), stride=stride
            )
            pixels, a_oh = to_device_seq(batch["pixels"], batch["action"], action_dim, device, stride=stride)
            rewards_raw = torch.from_numpy(batch["reward"]).to(device, dtype=torch.float32)
            rewards = reduce_reward_seq(rewards_raw, stride)

            b, t = pixels.shape[:2]
            flat = pixels.reshape(b * t, *pixels.shape[2:])
            z_states = projector(encoder(flat)).reshape(b, t, -1)  # (B, T+1, D) grounded latents
            a_emb = action_encoder(a_oh)  # (B, T, D)

            z_hist = z_states[:, :history_size].contiguous()
            for k in range(K):
                a_window = a_emb[:, k : k + history_size]
                z_pred = pred_proj(predictor(z_hist, a_window)[:, -1])  # (B, D) imagined latent, depth k+1
                z_hist = torch.cat([z_hist[:, 1:], z_pred.unsqueeze(1)], dim=1)

                depth_pos = history_size + k  # position in z_states this imagined latent targets
                z_true = z_states[:, depth_pos]

                cos = nn.functional.cosine_similarity(z_pred, z_true, dim=-1)
                mse = ((z_pred - z_true) ** 2).mean(dim=-1)
                lat_cos[k].append(cos.cpu())
                lat_mse[k].append(mse.cpu())

                a_here = a_emb[:, depth_pos]
                r_true = rewards[:, depth_pos]
                r_pred_im = twohot_decode(reward_head(z_pred, a_here), reward_bins)
                r_pred_gr = twohot_decode(reward_head(z_true, a_here), reward_bins)
                r_mae_imagined[k].append((r_pred_im - r_true).abs().cpu())
                r_mae_grounded[k].append((r_pred_gr - r_true).abs().cpu())
                r_true_all[k].append(r_true.cpu())

                v_im = twohot_decode(value_head(z_pred), value_bins)
                v_gr = twohot_decode(value_head(z_true), value_bins)
                v_delta[k].append((v_im - v_gr).abs().cpu())

            n_windows += bs

        print(f"[audit] evaluated {n_windows} held-out windows, K={K} depths, stride={stride}\n")
        header = (
            f"{'depth':>5} | {'lat_cos':>8} | {'lat_mse':>8} | "
            f"{'r_mae_im':>9} | {'r_mae_gr':>9} | {'r_gap':>7} | {'v_delta':>8}"
        )
        print(header)
        print("-" * len(header))
        for k in range(K):
            cos_m = torch.cat(lat_cos[k]).mean().item()
            mse_m = torch.cat(lat_mse[k]).mean().item()
            rmae_im = torch.cat(r_mae_imagined[k]).mean().item()
            rmae_gr = torch.cat(r_mae_grounded[k]).mean().item()
            vdelta_m = torch.cat(v_delta[k]).mean().item()
            print(
                f"{k + 1:>5} | {cos_m:>8.4f} | {mse_m:>8.5f} | "
                f"{rmae_im:>9.4f} | {rmae_gr:>9.4f} | {rmae_im - rmae_gr:>7.4f} | {vdelta_m:>8.4f}"
            )

        # Decisive-frame breakdown: sparse rewards (~1% of frames) are the
        # ones that actually matter for the planner; average MAE above is
        # dominated by the ~99% zero-reward frames and can mask a large
        # error specifically on the rare, high-magnitude events a CEM
        # planner needs to detect (e.g. imminent KO / big HP swing).
        print(
            "\n[audit] decisive-frame (|reward| > 0) breakdown "
            "-- this is what the planner actually needs to get right:"
        )
        header2 = f"{'depth':>5} | {'n_nz':>6} | {'r_mae_im':>9} | {'r_mae_gr':>9} | {'r_gap':>7}"
        print(header2)
        print("-" * len(header2))
        for k in range(K):
            r_true_k = torch.cat(r_true_all[k])
            mask = r_true_k.abs() > 1e-8
            n_nz = int(mask.sum())
            if n_nz == 0:
                print(f"{k + 1:>5} | {n_nz:>6} | {'n/a':>9} | {'n/a':>9} | {'n/a':>7}")
                continue
            rmae_im_nz = torch.cat(r_mae_imagined[k])[mask].mean().item()
            rmae_gr_nz = torch.cat(r_mae_grounded[k])[mask].mean().item()
            print(
                f"{k + 1:>5} | {n_nz:>6} | {rmae_im_nz:>9.4f} | {rmae_gr_nz:>9.4f} | "
                f"{rmae_im_nz - rmae_gr_nz:>7.4f}"
            )


if __name__ == "__main__":
    main()
