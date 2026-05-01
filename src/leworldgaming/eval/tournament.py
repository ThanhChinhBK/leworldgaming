"""Round-robin tournament harness for the evaluation framework (gemini_research.md §8).

Pits LeWM and DreamerV3 against the open-source champions — Thunder (MCTS, 2018)
and ERHEA_PI (RHEA, 2020) — plus baselines (KickAI, BlindAI). Reports
win-rate, HP differential, frame-drop rate, and (for LeWM) probe R^2.
"""

from __future__ import annotations


def run_tournament() -> None:
    raise NotImplementedError("Implement during evaluation phase")
