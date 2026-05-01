"""Observation conversion: pyftg payloads -> torch tensors.

Two modes (per gemini_research.md §2):
  - 'pixel': RGB ScreenData (uint8 tensors, downsampled to 224x224 by default)
  - 'state': FrameData scalars (HP, energy, coords, frame counters)
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ObsConfig:
    mode: str = "pixel"  # 'pixel' | 'state'
    image_size: int = 224
    grayscale: bool = False


def frame_to_obs(frame_data: object, screen_data: object, cfg: ObsConfig) -> dict:
    """Convert a pyftg FrameData/ScreenData pair into a dict obs.

    Stub: real impl depends on pyftg API surface — port during weekend.
    """
    raise NotImplementedError("Implement once pyftg AIInterface is wired up")
