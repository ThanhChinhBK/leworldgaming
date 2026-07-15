"""Image preprocessing shared by offline training and live inference."""

from __future__ import annotations

import torch

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def normalize_imagenet_pixels(pixels: torch.Tensor) -> torch.Tensor:
    """Convert channel-first uint8-range pixels to ImageNet-normalized float32."""
    pixels = pixels.to(dtype=torch.float32).div(255.0)
    shape = (1,) * (pixels.ndim - 3) + (3, 1, 1)
    mean = pixels.new_tensor(IMAGENET_MEAN).view(shape)
    std = pixels.new_tensor(IMAGENET_STD).view(shape)
    return (pixels - mean) / std
