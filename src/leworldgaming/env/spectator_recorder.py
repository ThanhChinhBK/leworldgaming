"""Spectator-stream client that captures pixels alongside the AI socket.

FightingICE 7.x's per-AI socket doesn't ship `screen_data`; only the
spectator handler (`SocketStream`) does. This client subscribes via pyftg's
`StreamController` and caches the latest decoded frame for AI recorders to
consume.
"""

from __future__ import annotations

import logging
import threading

import numpy as np
from pyftg.aiinterface.stream_interface import StreamInterface
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.screen_data import ScreenData

from leworldgaming.env.pixels import decode_display_bytes

logger = logging.getLogger(__name__)


class SpectatorRecorder(StreamInterface):
    def __init__(self, image_size: int = 224) -> None:
        self._image_size = image_size
        self._lock = threading.Lock()
        self._latest_pixels: np.ndarray | None = None
        self._latest_frame: int = -1
        self._frames_seen = 0
        self._warned = False

    def get_frame_data_flag(self) -> bool:
        return True

    def get_screen_data_flag(self) -> bool:
        return True

    def get_audio_data_flag(self) -> bool:
        return False

    def initialize(self, game_data: GameData) -> None:
        logger.info(
            "[spectator] initialized; capturing pixels at %dx%d",
            self._image_size, self._image_size,
        )

    def get_information(self, frame_data: FrameData) -> None:
        if not frame_data.empty_flag:
            with self._lock:
                self._latest_frame = frame_data.current_frame_number

    def get_screen_data(self, screen_data: ScreenData) -> None:
        arr = decode_display_bytes(screen_data.display_bytes, self._image_size)
        if arr is None:
            if not self._warned:
                logger.warning(
                    "[spectator] screen decode failed (size=%d) — pixels will be zero-filled",
                    len(screen_data.display_bytes),
                )
                self._warned = True
            return
        with self._lock:
            self._latest_pixels = arr
        self._frames_seen += 1

    def latest_pixels(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_pixels

    @property
    def frames_seen(self) -> int:
        return self._frames_seen
