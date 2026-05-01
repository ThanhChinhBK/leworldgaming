"""Device selection and AMP autocast helpers.

Auto-picks CUDA on the Linux+RTX 3080 box, MPS on Apple Silicon, CPU otherwise.
AMP is bf16 on CUDA (3rd-gen Tensor Cores on Ampere); no-op elsewhere.
See gemini_research.md §3 (VRAM budgeting).
"""

from collections.abc import Iterator
from contextlib import contextmanager

import torch


def best_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@contextmanager
def amp_autocast(device: torch.device) -> Iterator[None]:
    if device.type == "cuda":
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            yield
    else:
        yield
