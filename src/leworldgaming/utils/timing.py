"""Frame-budget profiler for the 16.67 ms / 60 FPS deadline.

Wrap inference calls in `FrameBudget(...)`; it logs frame drops to stdout/tensorboard.
See gemini_research.md §3 and §8 (Frame Drop Rate).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

FRAME_BUDGET_MS = 1000.0 / 60.0  # 16.6667 ms


@dataclass
class FrameBudget:
    budget_ms: float = FRAME_BUDGET_MS
    drops: int = 0
    total: int = 0
    last_ms: float = 0.0
    history_ms: list[float] = field(default_factory=list)

    def __enter__(self) -> FrameBudget:
        self._t0 = time.perf_counter()
        return self

    def __exit__(self, *exc: object) -> None:
        self.last_ms = (time.perf_counter() - self._t0) * 1000.0
        self.total += 1
        self.history_ms.append(self.last_ms)
        if self.last_ms > self.budget_ms:
            self.drops += 1

    @property
    def drop_rate(self) -> float:
        return self.drops / self.total if self.total else 0.0
