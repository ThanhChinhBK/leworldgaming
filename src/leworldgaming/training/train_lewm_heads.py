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
from leworldgaming.agents.lewm.policy_head import PolicyHead
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.probe import LinearProbe
from leworldgaming.agents.lewm.projector import Projector
from leworldgaming.agents.lewm.reward_head import RewardHead
from leworldgaming.agents.lewm.twohot import make_bins, twohot_ce_loss, twohot_decode
from leworldgaming.agents.lewm.value_head import ValueHead
from leworldgaming.data.replay_buffer import DataReader
from leworldgaming.training._replay_utils import (
    reduce_reward_seq,
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
        "terminal_window_fraction": 0.1,
        # Continuation head only: dropout to fight the chronic val-loss
        # blowup on the ~300 (train) terminal-window examples (see
        # ContinuationHead docstring). 0.0 preserves old behavior exactly.
        "cont_dropout": 0.0,
        # Continuation head only: optional smaller hidden width. None means
        # "use the shared heads.hidden_dim" (old behavior). With only ~300
        # train / ~34 val terminal windows, the shared 512-wide MLP has far
        # more capacity than the data supports and memorizes those few
        # examples; shrinking just this head reduces that capacity without
        # touching reward/value heads, which have plenty of (non-terminal)
        # training signal and benefit from the larger width.
        "cont_hidden_dim": None,
        # Indices into the legacy 52-dim state_vector to use as probe targets.
        # Default matches planner.py:64 convention: probe[..., 0] = HP-diff.
        # 47 = hp_diff (signed), 0 = hp_self, 22 = hp_opp, 46 = distance.
        "probe_targets": [47, 0, 22, 46],
        # Reward/value head ensembling (2026-07-20): train N independently
        # -initialized heads instead of one so the CEM planner can score
        # trajectories with mean-minus-uncertainty instead of a single
        # head's point estimate (see planner._decode_pessimistic). Targets
        # the "model exploitation" failure mode this config's own docstring
        # already anticipated. 1 = old behavior exactly (single head, saved
        # under the singular "reward_head"/"value_head" checkpoint keys for
        # full backward compatibility with LewmAgent/older checkpoints).
        "reward_ensemble_size": 1,
        "value_ensemble_size": 1,
        # Policy-prior head (BC warm-start for CEM, see policy_head.py):
        # cross-entropy on the recorded executed action at every grounded
        # step. 0.0 (default) disables it -- no architecture change to any
        # existing checkpoint/consumer when unused.
        "policy_loss_weight": 0.0,
        "policy_hidden_dim": 256,
    },
    "batch_size": 16,
    "lr": 3.0e-4,
    "weight_decay": 1.0e-3,
    # Continuation head only: optional stronger weight decay param group.
    # None means "use the shared weight_decay" (old behavior).
    "cont_weight_decay": None,
    "grad_clip": 1.0,
    "data_path": "data/replay.h5",
    "ckpt_in": "data/lewm_checkpoint.pt",
    "ckpt_out": "data/lewm_heads_checkpoint.pt",
    "log_every": 10,
    "val_split": 0.1,
    "val_every": 1000,
    "val_batches": 8,
    "ckpt_every": 1000,
    "resume": False,  # resume from ckpt_out if it exists; --steps is the TOTAL target
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
    temporal_stride = int(arch.get("temporal_stride", 1) or 1)
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
    # ActionEncoder's input width is `action_dim * temporal_stride` (matches
    # how train_lewm.py builds it) — `action_dim` here stays the RAW one-hot
    # count so callers can still one-hot raw per-frame actions correctly.
    action_encoder = ActionEncoder(action_dim=action_dim * temporal_stride, emb_dim=latent_dim).to(device)
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
        "_temporal_stride": temporal_stride,
    }


def _lambda_return(
    rewards: torch.Tensor,
    next_cont: torch.Tensor,
    next_values: torch.Tensor,
    gamma: float,
    lam: float,
) -> torch.Tensor:
    """Compute λ-returns for aligned ``(s, a, r, s_next)`` transitions."""
    if not (rewards.shape == next_cont.shape == next_values.shape):
        raise ValueError(
            "rewards, next_cont, and next_values must have equal shapes"
        )
    _, horizon = rewards.shape
    returns = torch.zeros_like(rewards)
    carry = next_values[:, -1]
    for t in range(horizon - 1, -1, -1):
        carry = rewards[:, t] + gamma * next_cont[:, t] * (
            (1.0 - lam) * next_values[:, t] + lam * carry
        )
        returns[:, t] = carry
    return returns


def _load_ensemble_or_singular(
    modules: nn.ModuleList,
    ckpt: dict[str, Any],
    plural_key: str,
    singular_key: str,
) -> None:
    """Load an ensemble's state dicts from a checkpoint.

    Prefers the plural (list-of-state-dicts) key written by ensemble-aware
    checkpoints. Falls back to the singular key (old, pre-ensembling
    checkpoints, or ensemble_size==1) loaded into the first/only member —
    keeps resuming from any pre-2026-07-20 Stage-B checkpoint working
    unchanged.
    """
    if plural_key in ckpt:
        saved = ckpt[plural_key]
        if len(saved) != len(modules):
            raise ValueError(
                f"{plural_key} has {len(saved)} members in checkpoint but "
                f"{len(modules)} were configured — ensemble size must match to resume."
            )
        for member, state in zip(modules, saved, strict=True):
            member.load_state_dict(state)
    elif singular_key in ckpt:
        if len(modules) != 1:
            raise ValueError(
                f"Checkpoint only has a singular '{singular_key}' but "
                f"{len(modules)} ensemble members were configured."
            )
        modules[0].load_state_dict(ckpt[singular_key])
    else:
        raise KeyError(f"Checkpoint has neither {plural_key!r} nor {singular_key!r}.")


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
    ckpt_out = Path(cfg["ckpt_out"])
    ckpt_out.parent.mkdir(parents=True, exist_ok=True)
    resume_ckpt: dict[str, Any] | None = None
    if bool(cfg.get("resume", False)):
        if not ckpt_out.exists():
            raise FileNotFoundError(
                f"--resume was requested but checkpoint does not exist: {ckpt_out}"
            )
        resume_ckpt = torch.load(ckpt_out, map_location=device, weights_only=False)
        saved_heads_config = resume_ckpt.get("heads_config")
        structural_head_keys = (
            "reward_bins",
            "reward_low",
            "reward_high",
            "value_bins",
            "value_low",
            "value_high",
            "hidden_dim",
            "probe_targets",
            "reward_ensemble_size",
            "value_ensemble_size",
        )
        if saved_heads_config is not None:
            ensemble_keys = ("reward_ensemble_size", "value_ensemble_size")
            changed_structure = [
                key
                for key in structural_head_keys
                if (
                    (saved_heads_config.get(key, 1) or 1) != (hcfg.get(key, 1) or 1)
                    if key in ensemble_keys
                    else saved_heads_config.get(key) != hcfg.get(key)
                )
            ]
            if changed_structure:
                raise ValueError(
                    "Stage-B resume cannot change head architecture keys: "
                    + ", ".join(changed_structure)
                )

    if resume_ckpt is not None:
        # Stage-B checkpoints are self-contained; resume the exact frozen JEPA
        # paired with these heads even if ckpt_in was moved or replaced.
        ckpt_in = resume_ckpt
    else:
        if not Path(cfg["ckpt_in"]).exists():
            raise FileNotFoundError(
                f"{cfg['ckpt_in']} not found — train Stage A first via "
                "scripts/train.py --agent lewm"
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
    stride: int = jepa["_temporal_stride"]
    imagined_horizon = int(hcfg["imagined_horizon"])
    # T transitions require T+1 grounded states. The endpoint is needed for
    # the final reward block, terminal label, and value bootstrap at every
    # stride (including stride=1).
    transition_len = history_size + max(1, imagined_horizon)
    state_len = transition_len + 1

    n_reward = max(1, int(hcfg.get("reward_ensemble_size", 1) or 1))
    n_value = max(1, int(hcfg.get("value_ensemble_size", 1) or 1))
    reward_heads = nn.ModuleList(
        [
            RewardHead(
                latent_dim=latent_dim,
                hidden_dim=int(hcfg["hidden_dim"]),
                num_bins=int(hcfg["reward_bins"]),
            )
            for _ in range(n_reward)
        ]
    ).to(device)
    continuation_head = ContinuationHead(
        latent_dim=latent_dim,
        hidden_dim=int(hcfg.get("cont_hidden_dim") or hcfg["hidden_dim"]),
        dropout=float(hcfg.get("cont_dropout", 0.0)),
    ).to(device)
    value_heads = nn.ModuleList(
        [
            ValueHead(
                latent_dim=latent_dim,
                hidden_dim=int(hcfg["hidden_dim"]),
                num_bins=int(hcfg["value_bins"]),
            )
            for _ in range(n_value)
        ]
    ).to(device)
    value_target_heads = nn.ModuleList(
        [
            ValueHead(
                latent_dim=latent_dim,
                hidden_dim=int(hcfg["hidden_dim"]),
                num_bins=int(hcfg["value_bins"]),
            )
            for _ in range(n_value)
        ]
    ).to(device)
    for vh, vth in zip(value_heads, value_target_heads, strict=True):
        vth.load_state_dict(vh.state_dict())
        for p in vth.parameters():
            p.requires_grad_(False)
        vth.eval()

    probe_targets = list(hcfg["probe_targets"])
    probe = LinearProbe(latent_dim=latent_dim, target_dim=len(probe_targets)).to(device)

    w_policy = float(hcfg.get("policy_loss_weight", 0.0))
    policy_head: PolicyHead | None = None
    if w_policy > 0.0:
        policy_head = PolicyHead(
            latent_dim=latent_dim,
            hidden_dim=int(hcfg.get("policy_hidden_dim", 256)),
            num_actions=action_dim,
        ).to(device)

    reward_bins = make_bins(int(hcfg["reward_bins"]), float(hcfg["reward_low"]), float(hcfg["reward_high"]), device)
    value_bins = make_bins(int(hcfg["value_bins"]), float(hcfg["value_low"]), float(hcfg["value_high"]), device)

    head_modules = nn.ModuleDict({
        "reward_heads": reward_heads,
        "continuation_head": continuation_head,
        "value_heads": value_heads,
        "probe": probe,
    })
    if policy_head is not None:
        head_modules["policy_head"] = policy_head
    n_params = sum(p.numel() for p in head_modules.parameters()) / 1e6
    print(f"[train_lewm_heads] head params: {n_params:.2f}M (reward_ensemble={n_reward} value_ensemble={n_value}) (frozen JEPA: {sum(p.numel() for p in jepa['encoder'].parameters() if not p.requires_grad) / 1e6:.2f}M+ ...)")

    cont_wd = cfg.get("cont_weight_decay")
    if cont_wd is None:
        optim = torch.optim.AdamW(
            head_modules.parameters(),
            lr=float(cfg["lr"]),
            weight_decay=float(cfg["weight_decay"]),
        )
    else:
        # Separate param group so the (data-starved) continuation head can
        # use stronger weight decay than reward/value/probe without
        # affecting them.
        cont_params = list(continuation_head.parameters())
        other_params = [p for n, p in head_modules.named_parameters() if not n.startswith("continuation_head.")]
        optim = torch.optim.AdamW(
            [
                {"params": other_params, "weight_decay": float(cfg["weight_decay"])},
                {"params": cont_params, "weight_decay": float(cont_wd)},
            ],
            lr=float(cfg["lr"]),
        )

    # Optional resume: load heads + optimizer state and continue from the
    # saved step. num_steps is the TOTAL target, so re-running the same command
    # with a higher --steps extends training; an equal --steps is a no-op.
    # Resuming restores the self-contained frozen JEPA, heads, optimizer, and
    # RNG states. Loss weights and imagined horizon may change between M2-M5;
    # architecture-changing head settings are rejected above.
    start_step = 0
    if resume_ckpt is not None:
        _load_ensemble_or_singular(reward_heads, resume_ckpt, "reward_heads", "reward_head")
        continuation_head.load_state_dict(resume_ckpt["continuation_head"])
        _load_ensemble_or_singular(value_heads, resume_ckpt, "value_heads", "value_head")
        _load_ensemble_or_singular(
            value_target_heads, resume_ckpt, "value_target_heads", "value_target_head"
        )
        probe.load_state_dict(resume_ckpt["probe"])
        if policy_head is not None and resume_ckpt.get("policy_head") is not None:
            policy_head.load_state_dict(resume_ckpt["policy_head"])
        _policy_added_fresh = policy_head is not None and resume_ckpt.get("policy_head") is None
        if "optim" in resume_ckpt and not _policy_added_fresh:
            optim.load_state_dict(resume_ckpt["optim"])
        elif _policy_added_fresh:
            print(
                "[train_lewm_heads] policy_head is new in this run (not present in "
                "resumed checkpoint) -- optimizer state NOT restored (param groups "
                "changed shape); all head params restart with a fresh AdamW state "
                "but keep their loaded weights."
            )
        start_step = int(resume_ckpt.get("num_steps", 0))
        print(
            f"[train_lewm_heads] resumed from {ckpt_out} at step={start_step} "
            f"-> training to {num_steps}"
        )
        if start_step >= num_steps:
            print(
                f"[train_lewm_heads] nothing to do: start_step ({start_step}) >= "
                f"num_steps ({num_steps}). Raise --steps to continue."
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
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    val_every = int(cfg["val_every"])
    ckpt_every = int(cfg.get("ckpt_every", 0) or 0)
    batch_size = int(cfg["batch_size"])
    w_r = float(hcfg["reward_loss_weight"])
    w_c = float(hcfg["cont_loss_weight"])
    w_v = float(hcfg["value_loss_weight"])
    w_im = float(hcfg["imagined_loss_weight"])
    w_probe = float(hcfg["probe_loss_weight"])
    gamma = float(hcfg["gamma"])
    lam = float(hcfg["lambda_return"])
    target_ema = float(hcfg["target_ema"])
    terminal_window_fraction = float(hcfg["terminal_window_fraction"])
    K = imagined_horizon
    probe_idx = torch.as_tensor(probe_targets, dtype=torch.long, device=device)

    history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []

    # Best-val continuation-head snapshot: the continuation head overfits
    # its handful of terminal-window examples much faster than the other
    # heads converge (val BCE observed climbing 0.55 -> 4.7 over 20k steps
    # while train stays ~0.06-0.6), so training to num_steps and saving
    # whatever the continuation head looks like *then* silently ships a
    # badly miscalibrated head even though reward/value/probe kept
    # improving over the same run. Track the step with the lowest
    # val_loss_c and snapshot the continuation head's weights there; the
    # final checkpoint swaps this snapshot in for continuation_head only,
    # keeping the fully-converged final weights for the other heads.
    best_val_loss_c = (
        float(resume_ckpt.get("best_val_loss_c", float("inf")))
        if resume_ckpt is not None else float("inf")
    )
    best_continuation_state: dict[str, torch.Tensor] = (
        {k: v.clone() for k, v in resume_ckpt["best_continuation_head"].items()}
        if resume_ckpt is not None and "best_continuation_head" in resume_ckpt
        else {k: v.clone() for k, v in continuation_head.state_dict().items()}
    )

    @torch.no_grad()
    def _encode_grounded(pixels: torch.Tensor) -> torch.Tensor:
        """(B, T, C, H, W) -> (B, T, D) post-projector embeddings, JEPA frozen."""
        b, t = pixels.shape[:2]
        flat = pixels.reshape(b * t, *pixels.shape[2:])
        emb = projector(encoder(flat))
        return emb.reshape(b, t, -1)

    def _balanced_continuation_loss(
        logits: torch.Tensor,
        targets: torch.Tensor,
    ) -> torch.Tensor:
        per_item = F.binary_cross_entropy_with_logits(
            logits, targets, reduction="none"
        )
        terminal = targets < 0.5
        nonterminal = ~terminal
        if terminal.any() and nonterminal.any():
            return 0.5 * per_item[terminal].mean() + 0.5 * per_item[nonterminal].mean()
        return per_item.mean()

    def _step_forward(
        pixels: torch.Tensor,
        a_oh: torch.Tensor,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        state_vec: torch.Tensor | None = None,
        policy_targets: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        z_states = _encode_grounded(pixels)             # (B, T+1, D)
        z = z_states[:, :transition_len]                # current states
        z_next = z_states[:, 1:]                        # successor states
        a_emb = action_encoder(a_oh)                    # (B, T, D)

        # Reward block t is earned after taking action block t from z[t].
        # Ensemble members share the batch but are otherwise independent —
        # averaging each member's own CE loss trains them jointly with no
        # cross-member gradient coupling (see planner._decode_pessimistic
        # for how the resulting ensemble is scored at inference time).
        loss_r = torch.stack(
            [twohot_ce_loss(head(z, a_emb), rewards, reward_bins) for head in reward_heads]
        ).mean()

        # Continuation is a state property. Include the endpoint so terminal
        # states are supervised, and balance terminal/nonterminal examples.
        c_logits = continuation_head(z_states)          # (B, T+1)
        cont_target = 1.0 - dones.float()
        loss_c = _balanced_continuation_loss(c_logits, cont_target)

        loss = w_r * loss_r + w_c * loss_c
        loss_v_val = torch.zeros((), device=device)
        loss_im_val = torch.zeros((), device=device)
        loss_r_im_val = torch.zeros((), device=device)
        loss_c_im_val = torch.zeros((), device=device)

        if w_v > 0.0:
            loss_v_per_member = []
            for value_head, value_target_head in zip(value_heads, value_target_heads, strict=True):
                v_logits = value_head(z)                  # (B, T, K_v)
                with torch.no_grad():
                    v_tgt_logits = value_target_head(z_next)
                    next_values = twohot_decode(v_tgt_logits, value_bins)
                    next_cont = cont_target[:, 1:]
                    G = _lambda_return(
                        rewards, next_cont, next_values, gamma=gamma, lam=lam
                    )
                loss_v_per_member.append(twohot_ce_loss(v_logits, G, value_bins))
            loss_v = torch.stack(loss_v_per_member).mean()
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
            loss_r_im = torch.stack(
                [
                    twohot_ce_loss(
                        head(z_im, a_emb_im),
                        rewards[:, history_size : history_size + K],
                        reward_bins,
                    )
                    for head in reward_heads
                ]
            ).mean()
            c_logits_im = continuation_head(z_im)                   # (B, K)
            cont_im_target = 1.0 - dones[:, history_size : history_size + K].float()
            loss_c_im = _balanced_continuation_loss(c_logits_im, cont_im_target)
            loss_imagined = loss_r_im + loss_c_im
            loss = loss + w_im * loss_imagined
            loss_im_val = loss_imagined.detach()
            loss_r_im_val = loss_r_im.detach()
            loss_c_im_val = loss_c_im.detach()

        loss_probe_val = torch.zeros((), device=device)
        if w_probe > 0.0 and state_vec is not None:
            # Probe every grounded state, including the endpoint.
            phys = state_vec.index_select(dim=-1, index=probe_idx)
            probe_pred = probe(z_states)
            loss_probe = F.mse_loss(probe_pred, phys)
            loss = loss + w_probe * loss_probe
            loss_probe_val = loss_probe.detach()

        loss_policy_val = torch.zeros((), device=device)
        if policy_head is not None and w_policy > 0.0 and policy_targets is not None:
            # Behavior-cloning CE on the recorded executed action at each
            # GROUNDED step z[t] (not the imagined rollout -- this head is a
            # CEM warm-start prior, not a planning-time-critical component,
            # so only supervising grounded latents keeps it simple and
            # avoids the imagined-rollout compute/complexity of the reward/
            # continuation heads' M4 branch above).
            policy_logits = policy_head(z)                    # (B, T, A)
            loss_policy = F.cross_entropy(
                policy_logits.reshape(-1, policy_logits.shape[-1]),
                policy_targets.reshape(-1),
            )
            loss = loss + w_policy * loss_policy
            loss_policy_val = loss_policy.detach()

        return {
            "loss": loss,
            "loss_r": loss_r.detach(),
            "loss_c": loss_c.detach(),
            "loss_v": loss_v_val,
            "loss_im": loss_im_val,
            "loss_r_im": loss_r_im_val,
            "loss_c_im": loss_c_im_val,
            "loss_probe": loss_probe_val,
            "loss_policy": loss_policy_val,
            "z_norm": z_states.float().norm(dim=-1).mean().detach(),
        }

    @torch.no_grad()
    def _ema_update_target() -> None:
        for value_head, value_target_head in zip(value_heads, value_target_heads, strict=True):
            for p, p_tgt in zip(
                value_head.parameters(), value_target_head.parameters(), strict=True
            ):
                p_tgt.data.mul_(target_ema).add_(p.data, alpha=1.0 - target_ema)

    with DataReader(cfg["data_path"]) as reader:
        valid_starts = reader.valid_seq_starts(state_len, stride=stride)
        if valid_starts.size == 0:
            raise RuntimeError("No valid sequences in replay buffer.")

        train_starts, val_starts = valid_starts.split_by_episode(
            float(cfg["val_split"]), int(cfg["seed"])
        )
        train_terminal_starts = reader.terminal_ending_starts(
            train_starts, state_len, stride
        )
        val_terminal_starts = reader.terminal_ending_starts(
            val_starts, state_len, stride
        )
        print(
            f"[train_lewm_heads] files={reader.num_files} frames={reader.total_frames} "
            f"valid_seq_starts={valid_starts.size} train={train_starts.size} "
            f"val={val_starts.size} terminal_train={train_terminal_starts.size} "
            f"terminal_val={val_terminal_starts.size} "
            f"episodes={reader.total_episodes} temporal_stride={stride}"
        )

        # Probe needs state_vector; only fetch it if enabled and available.
        extra_keys: tuple[str, ...] = ("reward", "done")
        if w_probe > 0.0 and reader.has_key("state_vector"):
            extra_keys = extra_keys + ("state_vector",)

        def _make_batch(
            starts,
            terminal_starts,
            batch_rng: np.random.Generator,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
            terminal_count = 0
            if terminal_starts.size > 0 and terminal_window_fraction > 0.0:
                terminal_count = min(
                    batch_size - 1,
                    max(1, int(round(batch_size * terminal_window_fraction))),
                )
            regular_count = batch_size - terminal_count
            batch = reader.sample_window(
                starts,
                regular_count,
                state_len,
                batch_rng,
                extra_keys=extra_keys,
                stride=stride,
            )
            if terminal_count:
                terminal_batch = reader.sample_window(
                    terminal_starts,
                    terminal_count,
                    state_len,
                    batch_rng,
                    extra_keys=extra_keys,
                    stride=stride,
                )
                batch = {
                    key: np.concatenate([value, terminal_batch[key]], axis=0)
                    for key, value in batch.items()
                }
            pixels, a_oh = to_device_seq(
                batch["pixels"], batch["action"], action_dim, device, stride=stride
            )
            rewards_raw = torch.from_numpy(batch["reward"]).to(device, dtype=torch.float32)
            rewards = reduce_reward_seq(rewards_raw, stride)
            dones = torch.from_numpy(batch["done"]).to(device, dtype=torch.float32)
            state_vec: torch.Tensor | None = None
            if "state_vector" in batch:
                state_vec = torch.from_numpy(batch["state_vector"]).to(device, dtype=torch.float32)
            policy_targets: torch.Tensor | None = None
            if policy_head is not None and w_policy > 0.0:
                # Behavior-cloning target: the block-ending (last raw-frame)
                # executed action id per action block -- matches how
                # `action_hist` conditions subsequent blocks in the planner
                # (see `planner._score_action_sequences`'s use of
                # `sub_actions[:, t, -1:]`).
                raw_actions = torch.from_numpy(batch["action"].astype(np.int64)).to(device)
                b_sz = raw_actions.shape[0]
                steps = transition_len
                block_actions = raw_actions[:, : steps * stride].reshape(b_sz, steps, stride)
                policy_targets = block_actions[:, :, -1]
            return pixels, a_oh, rewards, dones, state_vec, policy_targets

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            head_modules.eval()
            losses_r: list[float] = []
            losses_c: list[float] = []
            losses_v: list[float] = []
            losses_r_im: list[float] = []
            losses_c_im: list[float] = []
            losses_probe: list[float] = []
            losses_policy: list[float] = []
            n_batches = max(1, val_starts.size // batch_size)
            val_batches = int(cfg.get("val_batches", 0) or 0)
            if val_batches > 0:
                n_batches = min(n_batches, val_batches)
            val_rng = np.random.default_rng(12345)
            for _ in range(n_batches):
                pixels, a_oh, rewards, dones, state_vec, policy_targets = _make_batch(
                    val_starts, val_terminal_starts, val_rng
                )
                out = _step_forward(pixels, a_oh, rewards, dones, state_vec, policy_targets)
                losses_r.append(out["loss_r"].item())
                losses_c.append(out["loss_c"].item())
                losses_v.append(out["loss_v"].item())
                losses_r_im.append(out["loss_r_im"].item())
                losses_c_im.append(out["loss_c_im"].item())
                losses_probe.append(out["loss_probe"].item())
                losses_policy.append(out["loss_policy"].item())
            head_modules.train()
            return {
                "val_loss_r": float(np.mean(losses_r)),
                "val_loss_c": float(np.mean(losses_c)),
                "val_loss_v": float(np.mean(losses_v)),
                "val_loss_r_im": float(np.mean(losses_r_im)),
                "val_loss_c_im": float(np.mean(losses_c_im)),
                "val_loss_probe": float(np.mean(losses_probe)),
                "val_loss_policy": float(np.mean(losses_policy)),
            }

        def _save_ckpt(
            step_done: int,
            dest: Path,
            use_best_continuation: bool = False,
        ) -> None:
            """Atomic save (write to a temp file then rename) so a kill mid-write
            never corrupts the last good checkpoint.

            ``use_best_continuation``: swap in the lowest-val_loss_c
            continuation-head snapshot instead of its current (possibly
            overfit) weights, while everything else stays at ``step_done``'s
            fully-converged state. See the best-val tracking comment above
            ``history``/``val_history``.
            """
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_suffix(dest.suffix + ".tmp")
            cont_state = best_continuation_state if use_best_continuation else continuation_head.state_dict()
            torch.save(
                {
                    # Re-save Stage-A weights so this checkpoint is self-contained.
                    "encoder": ckpt_in["encoder"],
                    "projector": ckpt_in["projector"],
                    "action_encoder": ckpt_in["action_encoder"],
                    "predictor": ckpt_in["predictor"],
                    "pred_proj": ckpt_in["pred_proj"],
                    "reward_head": reward_heads[0].state_dict(),
                    "reward_heads": [h.state_dict() for h in reward_heads],
                    "continuation_head": cont_state,
                    "value_head": value_heads[0].state_dict(),
                    "value_heads": [h.state_dict() for h in value_heads],
                    "value_target_head": value_target_heads[0].state_dict(),
                    "value_target_heads": [h.state_dict() for h in value_target_heads],
                    "probe": probe.state_dict(),
                    "policy_head": (policy_head.state_dict() if policy_head is not None else None),
                    "optim": optim.state_dict(),
                    "config": jepa["_arch"],
                    "heads_config": hcfg,
                    "stage": "B",
                    "num_steps": step_done,
                    "format_version": 2,
                    "pixel_normalization": "imagenet",
                    "best_val_loss_c": best_val_loss_c,
                    "best_continuation_head": {k: v.clone() for k, v in best_continuation_state.items()},
                    "numpy_rng_state": rng.bit_generator.state,
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state_all": (
                        torch.cuda.get_rng_state_all()
                        if torch.cuda.is_available()
                        else []
                    ),
                },
                tmp,
            )
            tmp.replace(dest)

        t0 = time.time()
        head_modules.train()
        for step in range(start_step, num_steps):
            pixels, a_oh, rewards, dones, state_vec, policy_targets = _make_batch(
                train_starts, train_terminal_starts, rng
            )

            with amp_autocast(device):
                out = _step_forward(pixels, a_oh, rewards, dones, state_vec, policy_targets)
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
                "loss_policy": float(out["loss_policy"].item()),
                "z_norm": float(out["z_norm"].item()),
                "grad_norm": float(grad_norm.item()),
            }
            history.append(metrics)
            if step % log_every == 0 or step == num_steps - 1:
                print(
                    f"[train_lewm_heads] step={step:5d} train "
                    f"r={metrics['loss_r']:.4f} c={metrics['loss_c']:.4f} v={metrics['loss_v']:.4f} "
                    f"r_im={metrics['loss_r_im']:.4f} c_im={metrics['loss_c_im']:.4f} "
                    f"probe={metrics['loss_probe']:.4f} policy={metrics['loss_policy']:.4f} "
                    f"|z|={metrics['z_norm']:.2f} grad={metrics['grad_norm']:.2f}"
                )

            if val_every > 0 and (step % val_every == 0 or step == num_steps - 1):
                vm = evaluate()
                vm["step"] = step
                val_history.append(vm)
                if vm["val_loss_c"] < best_val_loss_c:
                    best_val_loss_c = vm["val_loss_c"]
                    best_continuation_state = {
                        k: v.clone() for k, v in continuation_head.state_dict().items()
                    }
                print(
                    f"[train_lewm_heads] step={step:5d}  val  "
                    f"r={vm['val_loss_r']:.4f} c={vm['val_loss_c']:.4f} v={vm['val_loss_v']:.4f} "
                    f"r_im={vm['val_loss_r_im']:.4f} c_im={vm['val_loss_c_im']:.4f} "
                    f"probe={vm['val_loss_probe']:.4f}  (best val c={best_val_loss_c:.4f})"
                )

            completed_steps = step + 1
            if ckpt_every > 0 and completed_steps % ckpt_every == 0:
                _save_ckpt(completed_steps, ckpt_out)
                print(
                    f"[train_lewm_heads] step={completed_steps:5d}  "
                    f"saved checkpoint -> {ckpt_out}"
                )

        elapsed = time.time() - t0
        steps_run = max(num_steps - start_step, 0)
        rate = steps_run / elapsed if elapsed > 0 else 0.0
        print(f"[train_lewm_heads] done in {elapsed:.1f}s ({rate:.1f} step/s)")

    _save_ckpt(max(start_step, num_steps), ckpt_out, use_best_continuation=True)
    print(
        f"[train_lewm_heads] saved checkpoint -> {ckpt_out} "
        f"(continuation_head swapped to its best-val snapshot, val_loss_c={best_val_loss_c:.4f})"
    )

    return {
        "ckpt_path": str(ckpt_out),
        "final_train": history[-1] if history else {},
        "final_val": val_history[-1] if val_history else {},
        "history": history,
        "val_history": val_history,
    }
