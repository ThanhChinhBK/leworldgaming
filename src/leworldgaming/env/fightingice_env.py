"""Gymnasium-style wrapper around pyftg.

Bridges the asyncio socket protocol used by pyftg to a synchronous
step()/reset() API so MBRL replay buffers and Gym-style code can consume it.
See gemini_research.md §2 and §7.2 (Buffer Queue mitigation).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class EnvConfig:
    host: str = "127.0.0.1"
    port: int = 31415
    character: str = "ZEN"
    obs_mode: str = "pixel"
    headless: bool = True


class FightingIceEnv:
    """Stub. Implement with pyftg.Gateway + asyncio buffer queue during weekend dev."""

    def __init__(self, cfg: EnvConfig | None = None) -> None:
        self.cfg = cfg or EnvConfig()

    def reset(self) -> tuple[dict[str, Any], dict[str, Any]]:
        raise NotImplementedError

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        raise NotImplementedError

    def close(self) -> None:
        pass
