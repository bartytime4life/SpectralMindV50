# src/spectramind/data/synthetic.py
from __future__ import annotations

"""
SpectraMind V50 — Synthetic Data Utilities
==========================================

This module provides lightweight synthetic generators for FGS1 (time-series)
and AIRS (spectral channels). They are designed for:
    • Sanity checks of pipeline stages
    • Unit tests (CI / Kaggle-safe, deterministic)
    • Quick debugging (no heavy I/O required)

Notes
-----
- Uses deterministic seeds for reproducibility.
- Returns NumPy arrays by default; can be cast to torch tensors in downstream
  code if needed.
- Shapes:
    tiny_fgs1(T) -> (T,)
    tiny_airs(B) -> (B,)
"""

import numpy as np
from typing import Optional


__all__ = ["tiny_fgs1", "tiny_airs"]


def tiny_fgs1(T: int = 64, seed: Optional[int] = 0) -> np.ndarray:
    """
    Generate a toy FGS1 photometric light curve.

    Args:
        T: Number of timesteps.
        seed: Random seed for reproducibility. Defaults to 0.

    Returns:
        Array of shape [T], representing a sinusoidal transit-like curve
        with small Gaussian noise.
    """
    rng = np.random.RandomState(seed)
    t = np.linspace(0, 2 * np.pi, T, endpoint=False)
    signal = np.sin(t)
    noise = 0.01 * rng.randn(T)
    return signal + noise


def tiny_airs(B: int = 283) -> np.ndarray:
    """
    Generate a toy AIRS spectrum.

    Args:
        B: Number of spectral bins (default 283 to match challenge).

    Returns:
        Array of shape [B], monotonically increasing (linear ramp).
    """
    return np.linspace(0.1, 1.0, B, dtype=np.float32)
