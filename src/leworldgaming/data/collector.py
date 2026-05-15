"""Data collector — drives `FightingIceEnv` and writes transitions to a `ReplayBuffer`.

Designed for the mid-week pipeline (gemini_research.md §7.2): runs headlessly
against rule-based opponents (KickAI, BlindAI, Thunder, ERHEA_PI) and logs
every transition with downsampled pixels + state vectors.
"""

from __future__ import annotations


def run_collection(num_episodes: int = 100) -> None:
    raise NotImplementedError("Wire pyftg + ReplayBuffer during weekend dev")
