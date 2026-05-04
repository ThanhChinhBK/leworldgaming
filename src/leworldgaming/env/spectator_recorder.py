"""Spectator-stream client that captures pixels alongside the AI socket.

FightingICE 7.x's per-AI socket doesn't ship `screen_data`; only the
spectator handler (`SocketStream`) does. This client subscribes via pyftg's
`StreamController`, hands raw display bytes to a background decoder thread,
and caches the latest decoded frame for AI recorders to consume.

Threading: ``get_screen_data`` runs on the asyncio loop (called by pyftg
when a frame arrives) and only enqueues the raw bytes — the heavy PIL
decode + bilinear resize happens on a dedicated worker thread. The decoder
queue is bounded with drop-oldest semantics: we only ever care about the
most recent frame, and on a busy loop letting old frames pile up would
trade latency for nothing.
"""

from __future__ import annotations

import contextlib
import logging
import queue
import threading

import numpy as np
from pyftg.aiinterface.stream_interface import StreamInterface
from pyftg.models.frame_data import FrameData
from pyftg.models.game_data import GameData
from pyftg.models.screen_data import ScreenData

from leworldgaming.env.pixels import decode_display_bytes

logger = logging.getLogger(__name__)


_SHUTDOWN = object()


class SpectatorRecorder(StreamInterface):
    def __init__(self, image_size: int = 224, queue_size: int = 8) -> None:
        self._image_size = image_size
        self._lock = threading.Lock()
        self._latest_pixels: np.ndarray | None = None
        self._latest_frame: int = -1
        self._frames_seen = 0
        self._warned = False

        self._decode_queue: queue.Queue = queue.Queue(maxsize=int(queue_size))
        self._decoder_thread = threading.Thread(
            target=self._decoder_loop, name="spectator-decoder", daemon=True,
        )
        self._decoder_thread.start()

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
        # Drop oldest if full — only the latest frame matters for inference,
        # and for collection any in-flight stale frame is no worse than a
        # decode-skipped one.
        payload = screen_data.display_bytes
        try:
            self._decode_queue.put_nowait(payload)
        except queue.Full:
            with contextlib.suppress(queue.Empty):
                self._decode_queue.get_nowait()
            with contextlib.suppress(queue.Full):
                self._decode_queue.put_nowait(payload)

    def latest_pixels(self) -> np.ndarray | None:
        with self._lock:
            return self._latest_pixels

    @property
    def frames_seen(self) -> int:
        return self._frames_seen

    def close(self) -> None:
        # Blocking `put` with a timeout is fine here — the decoder is still
        # draining, so a slot will free up shortly.
        with contextlib.suppress(queue.Full):
            self._decode_queue.put(_SHUTDOWN, timeout=2.0)
        if self._decoder_thread.is_alive():
            self._decoder_thread.join(timeout=2.0)

    def _decoder_loop(self) -> None:
        while True:
            try:
                payload = self._decode_queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if payload is _SHUTDOWN:
                return
            arr = decode_display_bytes(payload, self._image_size)
            if arr is None:
                if not self._warned:
                    logger.warning(
                        "[spectator] screen decode failed (size=%d) — pixels will be zero-filled",
                        len(payload),
                    )
                    self._warned = True
                continue
            with self._lock:
                self._latest_pixels = arr
                self._frames_seen += 1
