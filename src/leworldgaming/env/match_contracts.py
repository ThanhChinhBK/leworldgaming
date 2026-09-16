"""Shared validation and fail-fast lifecycle for live matches."""

from __future__ import annotations

import asyncio
import contextlib
import math
import time
from collections.abc import Sequence
from numbers import Integral
from typing import Any


def positive_integer(name: str, value: int) -> None:
    if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
        raise ValueError(f"{name} must be a positive integer, got {value!r}")


def validate_obs_mode(mode: str) -> None:
    if mode not in ("state", "pixel"):
        raise ValueError(f"obs_mode must be 'state' or 'pixel', got {mode!r}")


def round_hps(round_result: Any) -> tuple[float, float]:
    remaining = getattr(round_result, "remaining_hps", None)
    if remaining is None or len(remaining) < 2:
        raise RuntimeError("Round ended without both players' HP; outcome is unknown")
    hp_p1, hp_p2 = float(remaining[0]), float(remaining[1])
    if not math.isfinite(hp_p1) or not math.isfinite(hp_p2):
        raise RuntimeError("Round ended with non-finite HP; outcome is unknown")
    return hp_p1, hp_p2


class PixelReader:
    """Wait only for the first decoded frame, on the AI executor thread."""

    def __init__(self, source: Any, timeout: float = 2.0) -> None:
        self.source = source
        self.timeout = timeout
        self._ready = False

    def read(self):
        if self.source is None:
            raise RuntimeError("Pixel observations require a spectator source")
        pixels = self.source.latest_pixels()
        if pixels is None and not self._ready:
            deadline = time.monotonic() + self.timeout
            while pixels is None and time.monotonic() < deadline:
                time.sleep(0.01)
                pixels = self.source.latest_pixels()
        if pixels is None:
            raise RuntimeError("No decoded spectator pixels available")
        self._ready = True
        return pixels


async def close_writer(writer: Any) -> None:
    if writer is not None:
        with contextlib.suppress(Exception):
            writer.close()
            await writer.wait_closed()


async def _run_controller(controller: Any) -> None:
    # pyftg only closes sockets on normal return, not on errors/cancellation.
    try:
        await controller.run()
    finally:
        await close_writer(getattr(controller, "writer", None))


async def run_controllers(
    controllers: Sequence[Any], spectator: Any | None = None,
) -> None:
    """Propagate any controller failure and cancel/retrieve all sibling tasks.

    A spectator may end normally before the AI sockets. Its failure still
    invalidates the match, but successful AI completion need not wait for it.
    """
    if not controllers:
        raise RuntimeError("A match requires at least one AI controller")
    ai_tasks = {asyncio.create_task(_run_controller(ctrl)) for ctrl in controllers}
    tasks = set(ai_tasks)
    if spectator is not None:
        tasks.add(asyncio.create_task(_run_controller(spectator)))
    pending = set(tasks)
    try:
        while ai_tasks:
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.cancelled():
                    raise RuntimeError("Controller cancelled before match completed")
                task.result()
            ai_tasks.difference_update(done)
        # Include a spectator that failed concurrently with the last AI.
        for task in tasks:
            if task.done():
                if task.cancelled():
                    raise RuntimeError("Controller cancelled before match completed")
                task.result()
    finally:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
