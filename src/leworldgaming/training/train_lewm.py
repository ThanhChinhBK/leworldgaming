"""LeWM training loop — JEPA prediction loss + SIGReg.

Iteratively reads chunks from the replay buffer, encodes pairs (o_t, o_{t+1}),
predicts \\hat z_{t+1} = f(z_t, a_t), and minimises:
    L = ||\\hat z_{t+1} - z_{t+1}||^2 + lambda * sigreg(z)
See gemini_research.md §6, §7.2.
"""

from __future__ import annotations


def train(num_steps: int = 10_000) -> None:
    raise NotImplementedError("Implement during weekend dev — see plan §4")
