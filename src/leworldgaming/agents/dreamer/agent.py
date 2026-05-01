"""DreamerV3 agent — thin wrapper over external/dreamerv3-torch.

The vendored repo lives at external/dreamerv3-torch and is added to sys.path
by training/train_dreamer.py. Implementation is deferred to the weekend
integration phase; this file pins the public interface.
"""

from __future__ import annotations

from typing import Any

from leworldgaming.agents.base import AgentBase


class DreamerAgent(AgentBase):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        raise NotImplementedError(
            "Wire to external/dreamerv3-torch during weekend dev — see plan §4"
        )

    def act(self, obs: dict[str, Any]) -> int:
        raise NotImplementedError

    def learn(self, batch: dict[str, Any]) -> dict[str, float]:
        raise NotImplementedError

    def save(self, path: str) -> None:
        raise NotImplementedError

    def load(self, path: str) -> None:
        raise NotImplementedError
