"""LeWM agent — encoder + projector + AR predictor + pred_proj + action encoder + probe + planner.

After Stage-B head training (``train_lewm_heads.py``) the checkpoint also
carries reward / continuation / value heads. They are optional at load
time: if the ckpt has only Stage-A keys the heads stay at random init and
``act()`` falls back to the legacy random-shooting planner.
"""

from __future__ import annotations

import warnings
from typing import Any

import torch

from leworldgaming.agents.base import AgentBase
from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.continuation_head import ContinuationHead
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.planner import random_shooting
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.probe import LinearProbe
from leworldgaming.agents.lewm.projector import Projector
from leworldgaming.agents.lewm.reward_head import RewardHead
from leworldgaming.agents.lewm.twohot import make_bins
from leworldgaming.agents.lewm.value_head import ValueHead


class LewmAgent(AgentBase):
    """Inference-time wrapper around the trained modules.

    Architecture is configured by a flat ``cfg`` dict matching the keys in
    ``configs/lewm.yaml`` / ``train_lewm.DEFAULTS``. ``load()`` rebuilds
    modules from the checkpoint's stored config so the agent's architecture
    always matches the saved weights — no need to remember training flags.
    """

    def __init__(self, cfg: dict[str, Any] | None = None, device: str | torch.device = "cpu") -> None:
        self.device = torch.device(device)
        self._build_modules(cfg or {})

    def _build_modules(self, cfg: dict[str, Any]) -> None:
        self.model_cfg = dict(cfg)
        self.action_dim = int(cfg.get("action_dim", 56))
        self.temporal_stride = int(cfg.get("temporal_stride", 1) or 1)
        latent_dim = int(cfg.get("latent_dim", 192))
        self.latent_dim = latent_dim
        self.history_size = int(cfg.get("history_size", 3))
        self.image_size = int(cfg.get("encoder_image_size", 224))
        projector_hidden = int(cfg.get("projector_hidden", 2048))
        self.heads_cfg: dict[str, Any] = dict(cfg.get("heads", {}))
        self.heads_loaded = False  # set True only after Stage-B weights load.

        self.encoder = Encoder(
            latent_dim=latent_dim,
            image_size=int(cfg.get("encoder_image_size", 224)),
            patch_size=int(cfg.get("encoder_patch_size", 14)),
            embed_dim=int(cfg.get("encoder_embed_dim", 192)),
            depth=int(cfg.get("encoder_depth", 12)),
            num_heads=int(cfg.get("encoder_heads", 3)),
            mlp_ratio=float(cfg.get("encoder_mlp_ratio", 4.0)),
            dropout=float(cfg.get("encoder_dropout", 0.0)),
        ).to(self.device)
        self.projector = Projector(latent_dim=latent_dim, hidden_dim=projector_hidden).to(
            self.device
        )
        self.action_encoder = ActionEncoder(
            action_dim=self.action_dim * self.temporal_stride, emb_dim=latent_dim
        ).to(self.device)
        self.predictor = Predictor(
            latent_dim=latent_dim,
            action_dim=latent_dim,
            history_size=self.history_size,
            depth=int(cfg.get("predictor_depth", 6)),
            num_heads=int(cfg.get("predictor_heads", 16)),
            dim_head=int(cfg.get("predictor_dim_head", 64)),
            mlp_dim=int(cfg.get("predictor_mlp_dim", 2048)),
            dropout=float(cfg.get("predictor_dropout", 0.1)),
        ).to(self.device)
        self.pred_proj = Projector(latent_dim=latent_dim, hidden_dim=projector_hidden).to(
            self.device
        )
        probe_targets = self.heads_cfg.get("probe_targets", [47, 0, 22, 46])
        self.probe = LinearProbe(latent_dim=latent_dim, target_dim=len(probe_targets)).to(
            self.device
        )

        head_hidden = int(self.heads_cfg.get("hidden_dim", 512))
        reward_bins = int(self.heads_cfg.get("reward_bins", 41))
        value_bins = int(self.heads_cfg.get("value_bins", 41))
        self.reward_head = RewardHead(
            latent_dim=latent_dim, hidden_dim=head_hidden, num_bins=reward_bins
        ).to(self.device)
        self.continuation_head = ContinuationHead(
            latent_dim=latent_dim, hidden_dim=head_hidden
        ).to(self.device)
        self.value_head = ValueHead(
            latent_dim=latent_dim, hidden_dim=head_hidden, num_bins=value_bins
        ).to(self.device)
        self.reward_bins = make_bins(
            reward_bins,
            float(self.heads_cfg.get("reward_low", -1.0)),
            float(self.heads_cfg.get("reward_high", 1.0)),
            self.device,
        )
        self.value_bins = make_bins(
            value_bins,
            float(self.heads_cfg.get("value_low", -10.0)),
            float(self.heads_cfg.get("value_high", 10.0)),
            self.device,
        )
        self._z_history: list[torch.Tensor] = []
        self._action_history: list[int] = []
        self._set_eval()

    def _set_eval(self) -> None:
        for m in (
            self.encoder,
            self.projector,
            self.action_encoder,
            self.predictor,
            self.pred_proj,
            self.probe,
            self.reward_head,
            self.continuation_head,
            self.value_head,
        ):
            m.eval()

    @torch.no_grad()
    def act(self, obs: dict[str, Any]) -> int:
        x = obs["pixels"].to(self.device)
        z = self.projector(self.encoder(x.unsqueeze(0))).squeeze(0)
        z_context = torch.stack([*self._z_history, z], dim=0)
        past_actions = torch.as_tensor(
            self._action_history, dtype=torch.long, device=self.device
        )
        use_reward = self.heads_loaded and float(
            self.heads_cfg.get("reward_loss_weight", 0.0)
        ) > 0.0
        use_cont = self.heads_loaded and float(
            self.heads_cfg.get("cont_loss_weight", 0.0)
        ) > 0.0
        use_value = self.heads_loaded and float(
            self.heads_cfg.get("value_loss_weight", 0.0)
        ) > 0.0
        action = random_shooting(
            z_context,
            self.predictor,
            self.pred_proj,
            self.action_encoder,
            self.probe,
            self.action_dim,
            history_size=self.history_size,
            temporal_stride=self.temporal_stride,
            past_actions=past_actions,
            reward_head=self.reward_head if use_reward else None,
            continuation_head=self.continuation_head if use_cont else None,
            value_head=self.value_head if use_value else None,
            reward_bins=self.reward_bins,
            value_bins=self.value_bins,
            gamma=float(self.heads_cfg.get("gamma", 0.997)),
        )
        self._z_history.append(z.detach())
        self._action_history.append(action)
        keep = max(self.history_size - 1, 0)
        if keep:
            self._z_history = self._z_history[-keep:]
            self._action_history = self._action_history[-keep:]
        else:
            self._z_history.clear()
            self._action_history.clear()
        return action

    def reset_episode(self) -> None:
        self._z_history.clear()
        self._action_history.clear()

    @torch.no_grad()
    def warmup(self, n_iters: int = 2) -> None:
        """Run dummy ``act()`` calls to JIT-compile MPS/CUDA kernels.

        First-time PyTorch forward passes on a fresh shape can take 1–3 s
        while the backend compiles kernels — long enough to stall the JVM
        and look like a frozen game. Call ``warmup()`` once after ``load()``
        and before ``gateway.run_game()`` to pay that cost up front.
        """
        dummy = torch.zeros(
            (3, self.image_size, self.image_size),
            device=self.device,
            dtype=torch.float32,
        )
        for _ in range(int(n_iters)):
            self.act({"pixels": dummy})

    def learn(self, batch: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError(
            "Use leworldgaming.training.train_lewm.train() for the offline JEPA loop."
        )

    def save(self, path: str) -> None:
        save_dict = {
            "encoder": self.encoder.state_dict(),
            "projector": self.projector.state_dict(),
            "action_encoder": self.action_encoder.state_dict(),
            "predictor": self.predictor.state_dict(),
            "pred_proj": self.pred_proj.state_dict(),
            "probe": self.probe.state_dict(),
            "reward_head": self.reward_head.state_dict(),
            "continuation_head": self.continuation_head.state_dict(),
            "value_head": self.value_head.state_dict(),
            "heads_config": self.heads_cfg,
            "config": self.model_cfg,
        }
        torch.save(save_dict, path)

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        cfg = ckpt.get("config")
        # Stage-B checkpoints carry heads_config; fold it into cfg so
        # _build_modules sizes the heads correctly before loading weights.
        if cfg is not None and "heads_config" in ckpt:
            cfg = dict(cfg)
            cfg["heads"] = ckpt["heads_config"]
        if cfg:
            self._build_modules(cfg)
        self.encoder.load_state_dict(ckpt["encoder"])
        if "projector" in ckpt:
            self.projector.load_state_dict(ckpt["projector"])
        if "action_encoder" in ckpt:
            self.action_encoder.load_state_dict(ckpt["action_encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        if "pred_proj" in ckpt:
            self.pred_proj.load_state_dict(ckpt["pred_proj"])

        stage_b_keys = ("reward_head", "continuation_head", "value_head")
        has_stage_b = all(k in ckpt for k in stage_b_keys)
        if "probe" in ckpt:
            self.probe.load_state_dict(ckpt["probe"])
        if has_stage_b:
            self.reward_head.load_state_dict(ckpt["reward_head"])
            self.continuation_head.load_state_dict(ckpt["continuation_head"])
            self.value_head.load_state_dict(ckpt["value_head"])
            self.heads_loaded = True
        else:
            warnings.warn(
                "LewmAgent.load: Stage-B heads (reward/continuation/value) not found in "
                f"{path}; reward/value heads are at random init. MCTS planning will not "
                "work — train Stage B via train_lewm_heads.py.",
                stacklevel=2,
            )
        self._set_eval()
