"""PETS offline training driver — ensemble dynamics NLL on transition batches.

Mirrors ``train_lewm.train`` shape: same DEFAULTS-merge-with-YAML-then-overrides
config pattern, same logger format, same checkpoint layout.

The ensemble learns ``Δs`` from ``(s, a, s')`` triples drawn from the shared
HDF5 replay via ``view_pets``. The CEM planner is *inference-side only* —
it isn't exercised in this training loop.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from leworldgaming.agents.pets.agent import PETSAgent
from leworldgaming.data.replay_buffer import open_for_read, sample_window, valid_seq_starts
from leworldgaming.data.views import view_pets
from leworldgaming.env.action_space import NUM_ACTIONS
from leworldgaming.env.state_vector import PETS_STATE_DIM
from leworldgaming.utils.device import best_device
from leworldgaming.utils.seed import set_seed

DEFAULTS: dict[str, Any] = {
    # Architecture
    "state_dim": PETS_STATE_DIM,
    "action_dim": NUM_ACTIONS,
    "ensemble_size": 5,
    "hidden": 200,
    "num_layers": 3,
    "action_emb_dim": 16,
    "max_hp": 400.0,
    # Optimization
    "batch_size": 256,
    "lr": 1.0e-3,
    "weight_decay": 1.0e-5,
    "grad_clip": 1.0,
    # Planner (inference-only; stored on the agent for `act()`)
    "planner_horizon": 15,
    "planner_num_candidates": 200,
    "planner_num_elites": 20,
    "planner_num_iters": 4,
    "planner_gamma": 0.99,
    "planner_sample_dynamics": True,
    # Data + I/O
    "data_path": "data/replay.h5",
    "ckpt_path": "data/pets_checkpoint.pt",
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


def _sample_transition_batch(
    f,
    valid_starts: np.ndarray,
    batch_size: int,
    rng: np.random.Generator,
    max_hp: float,
) -> dict[str, np.ndarray]:
    """One PETS-shaped batch: ``(s, a, s_next, r)``."""
    sample = sample_window(f, valid_starts, batch_size, seq_len=2, rng=rng)
    return view_pets(sample)


def train(
    num_steps: int = 1000,
    config_path: str | Path | None = "configs/pets.yaml",
    **overrides: Any,
) -> dict[str, Any]:
    cfg = _load_config(config_path, overrides)
    set_seed(int(cfg["seed"]))
    device = best_device()
    print(f"[train_pets] device={device} steps={num_steps} batch={cfg['batch_size']} "
          f"ensemble={cfg['ensemble_size']}")
    print(f"[train_pets] data={cfg['data_path']} ckpt={cfg['ckpt_path']}")

    if not Path(cfg["data_path"]).exists():
        raise FileNotFoundError(
            f"{cfg['data_path']} not found — run scripts/collect_data.py first"
        )

    agent = PETSAgent(cfg=cfg, device=device)
    n_params = sum(p.numel() for p in agent.dynamics.parameters()) / 1e6
    print(f"[train_pets] params: {n_params:.2f}M (state_dim={cfg['state_dim']})")

    optim = torch.optim.AdamW(
        agent.dynamics.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    rng = np.random.default_rng(int(cfg["seed"]))
    grad_clip = float(cfg["grad_clip"])
    log_every = int(cfg["log_every"])
    val_every = int(cfg["val_every"])
    batch_size = int(cfg["batch_size"])
    max_hp = float(cfg["max_hp"])

    history: list[dict[str, float]] = []
    val_history: list[dict[str, float]] = []

    f = open_for_read(cfg["data_path"])
    try:
        starts = valid_seq_starts(f, seq_len=2)
        if starts.size == 0:
            raise RuntimeError("No valid transitions in replay buffer.")
        n = starts.size
        n_val = max(int(n * float(cfg["val_split"])), batch_size)
        n_val = min(n_val, max(n - batch_size, 0))
        train_starts = starts[: n - n_val]
        val_starts = starts[n - n_val :] if n_val > 0 else starts
        print(f"[train_pets] frames={f['action'].shape[0]} valid_transitions={n} "
              f"train={train_starts.size} val={val_starts.size} "
              f"episodes={f['episode_starts'].shape[0]}")

        @torch.no_grad()
        def evaluate() -> dict[str, float]:
            agent.dynamics.eval()
            mses: list[float] = []
            n_batches = max(1, val_starts.size // batch_size)
            val_rng = np.random.default_rng(12345)
            for _ in range(n_batches):
                batch = _sample_transition_batch(
                    f, val_starts, batch_size, val_rng, max_hp
                )
                s = torch.as_tensor(batch["s"], device=device)
                a = torch.as_tensor(batch["a"], device=device, dtype=torch.long)
                s_next = torch.as_tensor(batch["s_next"], device=device)
                target = s_next - s
                _, info = agent.dynamics.nll(s, a, target)
                mses.append(info["delta_mse"])
            return {"val_delta_mse": float(np.mean(mses))}

        t0 = time.time()
        for step in range(num_steps):
            batch = _sample_transition_batch(f, train_starts, batch_size, rng, max_hp)
            metrics = agent.learn(batch)
            loss = metrics["loss"]

            optim.zero_grad(set_to_none=True)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(agent.dynamics.parameters(), grad_clip)
            optim.step()

            scalar_metrics = {
                "step": step,
                "loss": float(loss.item()),
                "nll": float(metrics["nll"]),
                "delta_mse": float(metrics["delta_mse"]),
                "grad_norm": float(grad_norm.item()),
                "max_lv": float(metrics["max_logvar_mean"]),
                "min_lv": float(metrics["min_logvar_mean"]),
            }
            history.append(scalar_metrics)
            if step % log_every == 0 or step == num_steps - 1:
                print(
                    f"[train_pets] step={step:5d} train "
                    f"nll={scalar_metrics['nll']:.4f} "
                    f"mse={scalar_metrics['delta_mse']:.4f} "
                    f"grad={scalar_metrics['grad_norm']:.2f} "
                    f"lv=[{scalar_metrics['min_lv']:.2f},{scalar_metrics['max_lv']:.2f}]"
                )

            if val_every > 0 and (step % val_every == 0 or step == num_steps - 1):
                vm = evaluate()
                vm["step"] = step
                val_history.append(vm)
                print(f"[train_pets] step={step:5d}  val  mse={vm['val_delta_mse']:.4f}")

        elapsed = time.time() - t0
        print(f"[train_pets] done in {elapsed:.1f}s ({num_steps / max(elapsed, 1e-9):.1f} step/s)")

    finally:
        f.close()

    ckpt_path = Path(cfg["ckpt_path"])
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "dynamics": agent.dynamics.state_dict(),
            "config": cfg,
            "num_steps": num_steps,
        },
        ckpt_path,
    )
    print(f"[train_pets] saved checkpoint -> {ckpt_path}")

    return {
        "ckpt_path": str(ckpt_path),
        "final_train": history[-1] if history else {},
        "final_val": val_history[-1] if val_history else {},
        "history": history,
        "val_history": val_history,
    }


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--num-steps", type=int, default=1000)
    parser.add_argument("--config", type=str, default="configs/pets.yaml")
    parser.add_argument("--data-path", type=str, default=None)
    parser.add_argument("--ckpt-path", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    overrides = {
        "data_path": args.data_path,
        "ckpt_path": args.ckpt_path,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
    }
    train(num_steps=args.num_steps, config_path=args.config, **overrides)


if __name__ == "__main__":
    main()
