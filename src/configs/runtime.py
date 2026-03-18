"""Runtime helpers for global seeding and device detection."""

from __future__ import annotations

import os
import random

import numpy as np

try:  # Optional dependency for CPU-only environments
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def set_global_seed(seed: int) -> int:
    """Seed Python, NumPy, and (optionally) PyTorch for deterministic runs."""

    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    if torch is not None:
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            torch.mps.manual_seed(seed)

    return seed


def detect_device() -> str:
    """Return the best available compute device."""

    if torch is not None and torch.cuda.is_available():
        return "cuda"
    if torch is not None and hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    else:   
        return "cpu"


__all__ = ["set_global_seed", "detect_device"]
