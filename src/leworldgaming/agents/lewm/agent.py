"""LeWM agent — encoder + predictor + probe + latent-space planner."""

from __future__ import annotations

from typing import Any

import torch

from leworldgaming.agents.base import AgentBase
from leworldgaming.agents.lewm.encoder import Encoder
from leworldgaming.agents.lewm.planner import random_shooting
from leworldgaming.agents.lewm.predictor import Predictor
from leworldgaming.agents.lewm.probe import LinearProbe


class LewmAgent(AgentBase):
    def __init__(
        self,
        action_dim: int = 56,
        latent_dim: int = 256,
        device: str | torch.device = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.encoder = Encoder(latent_dim=latent_dim).to(self.device)
        self.predictor = Predictor(latent_dim=latent_dim, action_dim=action_dim).to(self.device)
        self.probe = LinearProbe(latent_dim=latent_dim).to(self.device)

    def act(self, obs: dict[str, Any]) -> int:
        x = obs["pixels"].to(self.device)
        z = self.encoder(x.unsqueeze(0)).squeeze(0)
        return random_shooting(z, self.predictor, self.probe, self.action_dim)

    def learn(self, batch: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError("Implement JEPA loss + SIGReg during weekend dev")

    def save(self, path: str) -> None:
        torch.save(
            {
                "encoder": self.encoder.state_dict(),
                "predictor": self.predictor.state_dict(),
                "probe": self.probe.state_dict(),
            },
            path,
        )

    def load(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device)
        self.encoder.load_state_dict(ckpt["encoder"])
        self.predictor.load_state_dict(ckpt["predictor"])
        self.probe.load_state_dict(ckpt["probe"])
