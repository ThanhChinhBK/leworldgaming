"""Pixel decoding helper shared by the spectator stream + any future readers."""

from __future__ import annotations

import io
import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Native FightingICE play area.
SCREEN_W, SCREEN_H = 960, 640


def decode_display_bytes(display_bytes: bytes, image_size: int) -> np.ndarray | None:
    """Decode pyftg ScreenData display_bytes into a (3, H, W) uint8 array.

    The JVM ships pixels through the spectator stream (`SocketStream`); pyftg
    gzip-decompresses for us in `ScreenData.from_proto`. The decompressed
    buffer is typically raw RGB at 960x640, but RGBA and PNG-encoded payloads
    are accepted as fallbacks. Returns None if the layout is unrecognizable.
    """
    n = len(display_bytes)

    if n == SCREEN_W * SCREEN_H * 3:
        arr = np.frombuffer(display_bytes, dtype=np.uint8).reshape(SCREEN_H, SCREEN_W, 3)
    elif n == SCREEN_W * SCREEN_H * 4:
        arr = np.frombuffer(display_bytes, dtype=np.uint8).reshape(SCREEN_H, SCREEN_W, 4)[..., :3]
    else:
        try:
            with Image.open(io.BytesIO(display_bytes)) as im:
                arr = np.asarray(im.convert("RGB"))
        except Exception:  # noqa: BLE001
            logger.warning("could not decode display_bytes (size=%d)", n)
            return None

    if arr.shape[0] != image_size or arr.shape[1] != image_size:
        arr = np.asarray(Image.fromarray(arr).resize((image_size, image_size), Image.BILINEAR))
    return np.ascontiguousarray(arr.transpose(2, 0, 1))  # HWC -> CHW
