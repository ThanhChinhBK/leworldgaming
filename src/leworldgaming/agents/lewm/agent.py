"""LeWM agent — encoder + projector + AR predictor + pred_proj + action encoder + probe + planner."""

from __future__ import annotations

from typing import Any

import torch

from leworldgaming.agents.base import AgentBase
from leworldgaming.agents.lewm.action_encoder import ActionEncoder
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.planner import random_shooting
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.probe import LinearProbe
from leworldgaming.agents.lewm.projector import Projector


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
        self.action_dim = int(cfg.get("action_dim", 56))
        latent_dim = int(cfg.get("latent_dim", 256))
        self.latent_dim = latent_dim
        self.history_size = int(cfg.get("history_size", 3))
        projector_hidden = int(cfg.get("projector_hidden", 2048))

        self.encoder = Encoder(
            latent_dim=latent_dim,
            image_size=int(cfg.get("encoder_image_size", 224)),
            patch_size=int(cfg.get("encoder_patch_size", 16)),
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
            action_dim=self.action_dim, emb_dim=latent_dim
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
        self.probe = LinearProbe(latent_dim=latent_dim).to(self.device)
        self._set_eval()

    def _set_eval(self) -> None:
        for m in (
            self.encoder,
            self.projector,
            self.action_encoder,
            self.predictor,
            self.pred_proj,
            self.probe,
        ):
            m.eval()

    @torch.no_grad()
    def act(self, obs: dict[str, Any]) -> int:
        x = obs["pixels"].to(self.device)
        z = self.projector(self.encoder(x.unsqueeze(0))).squeeze(0)
        return random_shooting(
            z,
            self.predictor,
            self.pred_proj,
            self.action_encoder,
            self.probe,
            self.action_dim,
            history_size=self.history_size,
        )

    def learn(self, batch: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError(
            "Use leworldgaming.training.train_lewm.train() for the offline JEPA loop."
        )

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "projector": self.projector.state_dict(),
                "action_encoder": self.action_encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "pred_proj": self.pred_proj.state_dict(),
                "probe": self.probe.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        cfg = ckpt.get("config")
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
        if "probe" in ckpt:
            self.probe.load_state_dict(ckpt["probe"])
        self._set_eval()
