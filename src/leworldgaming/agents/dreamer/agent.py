"""DreamerV3 agent — thin wrapper over ``external/dreamerv3-torch``.

The vendored repo's ``Dreamer`` class is constructed directly (we bypass
its ``main()`` so we don't need a live ``gym.Env``). For the offline-only
training path see ``leworldgaming.training.train_dreamer.train``. Online
play through ``act()`` is gated on a working ``FightingIceEnv`` — see the
plan at ``docs/gemini_research.md`` §7.1.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import torch

from leworldgaming.agents.base import AgentBase
from leworldgaming.env.action_space import NUM_ACTIONS


def _bootstrap_dreamer_imports() -> Path:
    """Make the vendored dreamerv3-torch importable.

    Vendored modules use bare imports (``import models``, ``import tools``,
    ``import envs.wrappers``) and the wrappers module unconditionally
    ``import gym``. We don't ship the legacy ``gym`` package — alias it to
    ``gymnasium`` in ``sys.modules`` so ``gym.Wrapper``/``gym.spaces.*``
    resolve. ``envs/wrappers.py`` only uses class-level references; the
    actual env wrappers (TimeLimit / NormalizeActions / OneHotAction / ...)
    aren't instantiated on the offline-training path.
    """
    if "gym" not in sys.modules:
        import gymnasium as gymnasium_mod

        sys.modules["gym"] = gymnasium_mod
        sys.modules["gym.spaces"] = gymnasium_mod.spaces

    repo_root = Path(__file__).resolve().parents[4]
    dreamer_dir = repo_root / "external" / "dreamerv3-torch"
    if str(dreamer_dir) not in sys.path:
        sys.path.insert(0, str(dreamer_dir))
    return dreamer_dir


_DREAMER_DIR = _bootstrap_dreamer_imports()


def make_obs_space(state_dim: int) -> Any:
    """Build a ``gymnasium`` Dict obs space matching the vendored proprio encoder.

    Vector-mode Dreamer uses ``encoder.cnn_keys='$^'`` and
    ``encoder.mlp_keys='.*'`` — the MLP path picks up the ``"vector"`` key.
    A 1×1×3 dummy ``"image"`` exists only because
    ``WorldModel.preprocess`` at ``external/dreamerv3-torch/models.py:182``
    unconditionally executes ``obs["image"] = obs["image"] / 255.0``; the
    encoder regex ensures it's never actually processed.
    """
    import gymnasium as gym
    import numpy as np

    return gym.spaces.Dict({
        "vector": gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(state_dim,), dtype="float32",
        ),
        "image": gym.spaces.Box(0, 255, (1, 1, 3), dtype="uint8"),
    })


def make_action_space(num_actions: int = NUM_ACTIONS) -> Any:
    import gymnasium as gym

    return gym.spaces.Discrete(num_actions)


class DreamerAgent(AgentBase):
    """Wraps the upstream Dreamer for offline pretraining.

    Construction is deferred to ``train_dreamer.train``: building the
    upstream class needs an obs_space / act_space / config / logger /
    dataset all at once, and reusing those exact objects keeps the
    training loop simple.
    """

    def __init__(self, dreamer_module: Any, config: Any, device: torch.device) -> None:
        self._dreamer = dreamer_module
        self._config = config
        self.device = device

    def act(self, obs: dict[str, Any]) -> int:
        raise NotImplementedError(
            "Dreamer online play needs a working FightingIceEnv "
            "(see docs/gemini_research.md §7.1). Offline pretraining via "
            "leworldgaming.training.train_dreamer.train works today."
        )

    def learn(self, batch: dict[str, Any]) -> dict[str, float]:
        """One world-model + imagined-behavior gradient step."""
        post, context, metrics = self._dreamer._wm._train(batch)
        # Train the actor/critic on imagined rollouts from learned dynamics.
        reward_fn = lambda f, s, a: self._dreamer._wm.heads["reward"](
            self._dreamer._wm.dynamics.get_feat(s)
        ).mode()
        beh_metrics = self._dreamer._task_behavior._train(post, reward_fn)[-1]
        merged: dict[str, float] = {}
        for k, v in metrics.items():
            try:
                merged[k] = float(v)
            except (TypeError, ValueError):
                continue
        for k, v in beh_metrics.items():
            try:
                merged[f"beh_{k}"] = float(v)
            except (TypeError, ValueError):
                continue
        return merged

    def save(self, path: str) -> None:
        torch.save(
            {
                "agent_state_dict": self._dreamer.state_dict(),
                "config": vars(self._config) if hasattr(self._config, "__dict__") else dict(self._config),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self._dreamer.load_state_dict(ckpt["agent_state_dict"])
