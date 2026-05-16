"""Pixel decoding helper shared by the spectator stream + any future readers."""

from __future__ import annotations

import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Native FightingICE play area.
SCREEN_W, SCREEN_H = 960, 640


def decode_display_bytes(display_bytes: bytes, image_size: int) -> np.ndarray | None:
    """Decode pyftg ScreenData display_bytes into a (3, H, W) uint8 array.

    The JVM ships pixels through the spectator stream (`SocketStream`); pyftg
    gzip-decompresses for us in `ScreenData.from_proto`. The decompressed
    buffer is typically raw RGB at 960x640, but RGBA and PNG-encoded payloads
    are accepted as fallbacks. Returns None if the layout is unrecognizable.

    Uses OpenCV for resize (~5-10× faster than PIL BILINEAR).
    """
    n = len(display_bytes)

    if n == SCREEN_W * SCREEN_H * 3:
        arr = np.frombuffer(display_bytes, dtype=np.uint8).reshape(SCREEN_H, SCREEN_W, 3)
    elif n == SCREEN_W * SCREEN_H * 4:
        arr = np.frombuffer(display_bytes, dtype=np.uint8).reshape(SCREEN_H, SCREEN_W, 4)[..., :3]
    else:
        try:
            buf = np.frombuffer(display_bytes, dtype=np.uint8)
            decoded = cv2.imdecode(buf, cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("imdecode returned None")
            arr = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
        except Exception:  # noqa: BLE001
            logger.warning("could not decode display_bytes (size=%d)", n)
            return None

    if arr.shape[0] != image_size or arr.shape[1] != image_size:
        arr = cv2.resize(arr, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # HWC -> CHW
