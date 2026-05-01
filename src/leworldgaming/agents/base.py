"""Common agent interface implemented by both LeWM and Dreamer."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class AgentBase(ABC):
    @abstractmethod
    def act(self, obs: dict[str, Any]) -> int:
        """Pick a discrete action given the current observation dict."""

    @abstractmethod
    def learn(self, batch: dict[str, Any]) -> dict[str, float]:
        """One optimizer step. Returns a dict of scalar metrics for logging."""

    @abstractmethod
    def save(self, path: str) -> None: ...

    @abstractmethod
    def load(self, path: str) -> None: ...
