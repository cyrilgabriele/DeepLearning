"""Runtime helpers for global seeding and device detection."""

from __future__ import annotations

import os
import random
import torch

from typing import Final

import numpy as np


GLOBAL_RANDOM_SEED: Final[int] = 42


def set_global_seed(seed: int = GLOBAL_RANDOM_SEED) -> int:
    """Seed Python, NumPy, and (optionally) PyTorch for deterministic runs."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)

    return seed


def detect_device() -> str:
    """Return the best available compute device."""

    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:   
        return "cpu"


__all__ = ["GLOBAL_RANDOM_SEED", "set_global_seed", "detect_device"]

