"""DreamerV3 offline training driver.

Reads the canonical replay HDF5 (named-group schema), exports it to a
directory of per-episode ``.npz`` files in the format the vendored
``external/dreamerv3-torch`` trainer expects, then drives that trainer's
``WorldModel`` + ``ImagBehavior`` purely from the offline dataset (no live
env, no rollouts). ``act()`` for online play is deferred until the
``FightingIceEnv`` sync wrapper is built — see plan in
``docs/gemini_research.md`` §7.1.

Mirrors ``train_lewm.train`` shape: same DEFAULTS-merge-with-YAML-then-overrides
config pattern, same logger format, same checkpoint layout.
"""

from __future__ import annotations

import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import torch
import yaml

# Importing the agent module first runs its sys.path + gym alias bootstrap.
from leworldgaming.agents.dreamer.agent import (
    DreamerAgent,
    make_action_space,
    make_obs_space,
)
from leworldgaming.data.dreamer_export import export_episodes_to_npz
from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.state_vector import DREAMER_STATE_DIM
from leworldgaming.utils.device import best_device
from leworldgaming.utils.seed import set_seed

_REPO_ROOT = Path(__file__).resolve().parents[3]
_DREAMER_DIR = _REPO_ROOT / "external" / "dreamerv3-torch"


# Keys exposed at our YAML / CLI layer. Anything else inherits from the
# upstream ``configs.yaml`` ``defaults`` + ``dmc_proprio`` sections.
DEFAULTS: dict[str, Any] = {
    # Data + I/O
    "data_path": "data/replay.h5",
    "episode_dir": "data/dreamer_episodes",
    "ckpt_path": "data/dreamer_checkpoint.pt",
    "logdir": "data/dreamer_logs",
    "state_dim": DREAMER_STATE_DIM,
    "num_actions": NUM_ACTIONS,
    # Observation mode: "vector" (proprio, default) | "image" (full pixels).
    "obs_mode": "vector",
    "image_size": 64,
    # Optimization
    "batch_size": 16,
    "batch_length": 64,
    "model_lr": 1.0e-4,
    "actor_lr": 3.0e-5,
    "critic_lr": 3.0e-5,
    "imag_horizon": 15,
    "log_every": 10,
    "val_every": 0,  # offline pretraining doesn't have a held-out env
    "ckpt_every": 1000,  # save a checkpoint every N steps (0 = only at end)
    "resume": False,  # resume from ckpt_path if it exists; --steps is the TOTAL target
    "seed": 0,
    # Behaviour modifiers
    "compile": False,  # disable torch.compile by default — flaky on MPS
    "precision": 32,   # 16 enables AMP; CPU/MPS prefer 32
    "device": None,    # auto-pick via best_device() if unset
    "actor_dist": "onehot",  # discrete-action default
}


def _load_config(path: str | Path | None, overrides: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(DEFAULTS)
    if path is not None and Path(path).exists():
        with open(path) as fh:
            file_cfg = yaml.safe_load(fh) or {}
        cfg.update({k: v for k, v in file_cfg.items() if k in DEFAULTS})
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def _build_dreamer_config(cfg: dict[str, Any], device: torch.device) -> Namespace:
    """Compose the vendored Dreamer's full config from upstream defaults +
    the mode-specific section + our overrides. Returns an ``argparse.Namespace``.

    ``obs_mode="vector"`` inherits from ``dmc_proprio`` (``encoder.cnn_keys='$^'``,
    ``encoder.mlp_keys='.*'``) so the MLP path handles the ``"vector"``
    observation. The dummy ``"image"`` key is ignored by the encoder regex but
    satisfies the unconditional preprocess step.

    ``obs_mode="image"`` inherits from ``dmc_vision`` (``encoder.cnn_keys='image'``,
    ``encoder.mlp_keys='$^'``) so the ConvEncoder/ConvDecoder process the real
    ``image_size``² frame. ``size`` is set to the export image size.
    """
    import ruamel.yaml as ryaml

    yaml_path = _DREAMER_DIR / "configs.yaml"
    with open(yaml_path) as fh:
        configs = ryaml.YAML(typ="safe").load(fh)

    def deep_update(base: dict, upd: dict) -> None:
        for k, v in upd.items():
            if isinstance(v, dict) and isinstance(base.get(k), dict):
                deep_update(base[k], v)
            else:
                base[k] = v

    obs_mode = cfg.get("obs_mode", "vector")
    base_section = "dmc_vision" if obs_mode == "image" else "dmc_proprio"

    merged: dict[str, Any] = {}
    deep_update(merged, configs["defaults"])
    deep_update(merged, configs[base_section])

    if obs_mode == "image":
        size = int(cfg["image_size"])
        merged["size"] = [size, size]

    # Discrete-action overrides (mirrors the ``crafter`` config).
    merged["actor"]["dist"] = cfg["actor_dist"]
    merged["actor"]["std"] = "none"

    # Our overrides.
    merged["batch_size"] = int(cfg["batch_size"])
    merged["batch_length"] = int(cfg["batch_length"])
    merged["model_lr"] = float(cfg["model_lr"])
    merged["actor"]["lr"] = float(cfg["actor_lr"])
    merged["critic"]["lr"] = float(cfg["critic_lr"])
    merged["imag_horizon"] = int(cfg["imag_horizon"])
    merged["device"] = str(cfg["device"] or device)
    merged["compile"] = bool(cfg["compile"])
    merged["precision"] = int(cfg["precision"])
    merged["seed"] = int(cfg["seed"])
    merged["num_actions"] = int(cfg["num_actions"])
    merged["log_every"] = int(cfg["log_every"]) * 1000  # upstream's tools.Every is divisor-based
    merged["video_pred_log"] = False
    merged["expl_until"] = 0
    merged["expl_behavior"] = "greedy"
    # Offline mode → no env interaction; prevent simulate() paths from triggering.
    merged["offline_traindir"] = str(Path(cfg["episode_dir"]).resolve())
    merged["offline_evaldir"] = ""

    # Flatten to Namespace (top-level attributes only — nested dicts stay dicts).
    return Namespace(**merged)


def train(
    num_steps: int = 1000,
    config_path: str | Path | None = "configs/dreamer.yaml",
    **overrides: Any,
) -> dict[str, Any]:
    """Run the offline DreamerV3 training loop.

    Steps:
      1. Convert the HDF5 replay into per-episode npz files (cached).
      2. Build the upstream Dreamer with our obs_space / act_space.
      3. Loop ``num_steps`` calls to ``WorldModel._train`` + ``ImagBehavior._train``.
      4. Save the agent state_dict + config.
    """
    cfg = _load_config(config_path, overrides)
    set_seed(int(cfg["seed"]))
    device = torch.device(cfg["device"]) if cfg["device"] else best_device()
    state_dim = int(cfg["state_dim"])
    num_actions = int(cfg["num_actions"])

    print(
        f"[train_dreamer] device={device} steps={num_steps} batch={cfg['batch_size']}x"
        f"{cfg['batch_length']} state_dim={state_dim}"
    )
    print(f"[train_dreamer] data={cfg['data_path']} eps={cfg['episode_dir']} "
          f"ckpt={cfg['ckpt_path']}")

    # 1. Export episodes if needed.
    if not Path(cfg["data_path"]).exists():
        raise FileNotFoundError(
            f"{cfg['data_path']} not found — run scripts/collect_data.py first, "
            "or point --data-path at a directory of .h5 files"
        )
    n_episodes = export_episodes_to_npz(
        cfg["data_path"], cfg["episode_dir"],
        action_dim=num_actions,
        obs_mode=cfg.get("obs_mode", "vector"),
        image_size=int(cfg.get("image_size", 64)),
    )
    if n_episodes == 0:
        raise RuntimeError(
            f"No episodes exported from {cfg['data_path']}. Collect more games first."
        )

    # 2. Bring up the upstream stack.
    import dreamer as dreamer_mod  # noqa: E402  — sys.path was extended above
    import networks as dreamer_networks  # noqa: E402
    import tools as dreamer_tools  # noqa: E402

    # Fix upstream MLP default device. ``MultiEncoder`` constructs its MLP
    # without passing ``device``, so MLP falls back to its hardcoded
    # ``device="cuda"`` (networks.py:606) and crashes on non-CUDA hosts at
    # ``torch.tensor((std,), device=device)`` (line 615). In pixel mode this
    # was never hit because mlp_keys='$^' skipped MLP construction; proprio
    # mode hits it on every run. Substitute our actual device for the
    # "cuda" default; explicit device= calls pass through unchanged.
    if not getattr(dreamer_networks.MLP, "_lwg_patched", False):
        _orig_mlp_init = dreamer_networks.MLP.__init__
        _target_device = str(device)

        def _patched_mlp_init(self, *args, device="cuda", **kwargs):  # noqa: ANN001
            if device == "cuda" and _target_device != "cuda":
                device = _target_device
            return _orig_mlp_init(self, *args, device=device, **kwargs)

        dreamer_networks.MLP.__init__ = _patched_mlp_init
        dreamer_networks.MLP._lwg_patched = True

    obs_space = make_obs_space(
        state_dim,
        obs_mode=cfg.get("obs_mode", "vector"),
        image_size=int(cfg.get("image_size", 64)),
    )
    act_space = make_action_space(num_actions)

    dreamer_cfg = _build_dreamer_config(cfg, device)

    logdir = Path(cfg["logdir"])
    logdir.mkdir(parents=True, exist_ok=True)
    logger = dreamer_tools.Logger(logdir, step=0)

    train_eps = dreamer_tools.load_episodes(Path(cfg["episode_dir"]))
    if not train_eps:
        raise RuntimeError(f"load_episodes() returned empty for {cfg['episode_dir']}")
    train_dataset = dreamer_mod.make_dataset(train_eps, dreamer_cfg)

    agent_module = dreamer_mod.Dreamer(
        obs_space, act_space, dreamer_cfg, logger, train_dataset,
    ).to(device)
    agent_module.requires_grad_(requires_grad=False)

    n_params = sum(p.numel() for p in agent_module.parameters()) / 1e6
    print(f"[train_dreamer] params: {n_params:.2f}M, episodes={len(train_eps)}")

    agent = DreamerAgent(agent_module, dreamer_cfg, device)

    # Optional resume: load weights + optimizer state and continue from the
    # saved step. num_steps is the TOTAL target, so re-running the same command
    # with a higher --steps extends training; an equal --steps is a no-op.
    ckpt_path = Path(cfg["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    ckpt_every = int(cfg.get("ckpt_every", 0) or 0)
    start_step = 0
    if bool(cfg.get("resume", False)) and ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        agent_module.load_state_dict(ckpt["agent_state_dict"])
        if "optims_state_dict" in ckpt:
            dreamer_tools.recursively_load_optim_state_dict(
                agent_module, ckpt["optims_state_dict"]
            )
        start_step = int(ckpt.get("num_steps", 0))
        print(
            f"[train_dreamer] resumed from {ckpt_path} at step={start_step} "
            f"-> training to {num_steps}"
        )
        if start_step >= num_steps:
            print(
                f"[train_dreamer] nothing to do: start_step ({start_step}) >= "
                f"num_steps ({num_steps}). Raise --steps to continue."
            )
    elif bool(cfg.get("resume", False)):
        print(f"[train_dreamer] --resume set but no checkpoint at {ckpt_path} — starting fresh")

    def _save_ckpt(step_done: int, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_suffix(dest.suffix + ".tmp")
        torch.save(
            {
                "agent_state_dict": agent_module.state_dict(),
                "optims_state_dict": dreamer_tools.recursively_collect_optim_state_dict(
                    agent_module
                ),
                "config": cfg,
                "num_steps": step_done,
            },
            tmp,
        )
        tmp.replace(dest)

    # 3. Train.
    log_every = int(cfg["log_every"])
    history: list[dict[str, float]] = []
    t0 = time.time()
    for step in range(start_step, num_steps):
        batch = next(train_dataset)
        metrics = agent.learn(batch)
        metrics["step"] = step
        history.append(metrics)
        if step % log_every == 0 or step == num_steps - 1:
            shown = {k: metrics[k] for k in (
                "kl", "image_loss", "reward_loss", "cont_loss",
                "actor_ent", "beh_critic_loss",
            ) if k in metrics}
            shown_str = " ".join(f"{k}={v:.4f}" for k, v in shown.items())
            print(f"[train_dreamer] step={step:5d} {shown_str}")
        if ckpt_every > 0 and step > 0 and step % ckpt_every == 0:
            _save_ckpt(step, ckpt_path)
            print(f"[train_dreamer] checkpoint saved -> {ckpt_path} (step={step})")

    elapsed = time.time() - t0
    print(f"[train_dreamer] done in {elapsed:.1f}s ({num_steps / max(elapsed, 1e-9):.1f} step/s)")

    # 4. Save.
    _save_ckpt(num_steps, ckpt_path)
    print(f"[train_dreamer] saved checkpoint -> {ckpt_path}")

    return {
        "ckpt_path": str(ckpt_path),
        "final": history[-1] if history else {},
        "history": history,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--config", type=str, default="configs/dreamer.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--episode-dir", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--batch-length", type=int, default=None)
    parser.add_argument("--state-dim", type=int, default=None)
    parser.add_argument("--obs-mode", type=str, default=None, choices=["vector", "image"])
    parser.add_argument("--image-size", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--compile", action="store_true", default=None)
    args = parser.parse_args()

    overrides = {
        "data_path": args.data_path,
        "episode_dir": args.episode_dir,
        "ckpt_path": args.ckpt_path,
        "batch_size": args.batch_size,
        "batch_length": args.batch_length,
        "state_dim": args.state_dim,
        "obs_mode": args.obs_mode,
        "image_size": args.image_size,
        "seed": args.seed,
        "device": args.device,
        "compile": args.compile,
    }
    train(num_steps=args.num_steps, config_path=args.config, **overrides)


if __name__ == "__main__":
    main()
