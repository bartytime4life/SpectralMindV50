# src/spectramind/data/transforms.py
from __future__ import annotations

"""
SpectraMind V50 — Mask-aware Tensor Transforms
===============================================

Utility transforms for FGS1 (photometric) and AIRS (spectral) time-series.

Design goals
------------
- **Mask-aware**: every op can ignore padded timesteps via `mask`.
- **Shape-stable**: works with [B, T] (FGS1) and [B, T, C] (AIRS).
- **Deterministic & Kaggle-safe**: pure PyTorch / NumPy ops.
- **Production-ready**: type hints, docstrings, no side effects.

Conventions
-----------
`x`  : torch.Tensor, float32 recommended
       FGS1: [B, T]
       AIRS: [B, T, C]

`mask`: torch.Tensor, float32/bool, shape:
       FGS1: [B, T]
       AIRS: [B, T]   (same mask broadcast across channels)

All functions preserve `x.dtype` and `device`.
"""

from typing import Callable, Iterable, Optional, Sequence, Tuple, Union

import torch

__all__ = [
    "Compose",
    "ensure_tensor",
    "zscore_time",
    "minmax_time",
    "detrend_time_linear",
    "clip",
]


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def ensure_tensor(x: Union[torch.Tensor, "np.ndarray", float, int], device: Optional[torch.device] = None) -> torch.Tensor:
    """
    Convert inputs to a torch.Tensor with no copy where possible.

    Args:
        x: tensor-like input
        device: optional device to move tensor to

    Returns:
        torch.Tensor
    """
    if not isinstance(x, torch.Tensor):
        x = torch.as_tensor(x)
    if device is not None and x.device != device:
        x = x.to(device)
    return x


def _infer_shapes(x: torch.Tensor) -> Tuple[bool, int, int, int]:
    """
    Infer layout for FGS1 ([B, T]) vs AIRS ([B, T, C]).

    Returns:
        (is_airs, B, T, C)
    """
    if x.dim() == 2:
        B, T = x.shape
        return False, B, T, 1
    if x.dim() == 3:
        B, T, C = x.shape
        return True, B, T, C
    raise ValueError(f"Expected x.ndim in {{2,3}}; got {x.dim()} with shape {tuple(x.shape)}")


def _broadcast_mask(mask: Optional[torch.Tensor], B: int, T: int, device, dtype) -> torch.Tensor:
    """
    Normalize mask: None -> all ones; ensure shape [B, T] and dtype float32.
    """
    if mask is None:
        return torch.ones(B, T, device=device, dtype=dtype)
    m = mask
    if not isinstance(m, torch.Tensor):
        m = torch.as_tensor(m, device=device, dtype=dtype)
    else:
        m = m.to(device=device, dtype=dtype)
    if m.shape != (B, T):
        raise ValueError(f"mask expected shape [{B}, {T}] but got {tuple(m.shape)}")
    return m


# ---------------------------------------------------------------------------
# Public transforms (mask-aware)
# ---------------------------------------------------------------------------

def zscore_time(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-6) -> torch.Tensor:
    """
    Standardize along the **time** axis (per-sequence z-score), ignoring padded steps.

    For FGS1 [B, T]: per sequence mean/std over T.
    For AIRS [B, T, C]: per sequence & per channel mean/std over T (channels independent).

    Args:
        x: [B, T] or [B, T, C]
        mask: [B, T] (1=valid, 0=padded), shared across channels for AIRS.
        eps: numerical stability term

    Returns:
        z-scored tensor with same shape as `x`.
    """
    is_airs, B, T, C = _infer_shapes(x)
    device, dtype = x.device, x.dtype
    m = _broadcast_mask(mask, B, T, device, torch.float32)

    if not is_airs:
        # [B, T]
        denom = m.sum(dim=1, keepdim=True).clamp_min(1.0)
        mean = (x * m).sum(dim=1, keepdim=True) / denom
        var = ((x - mean) ** 2 * m).sum(dim=1, keepdim=True) / denom
        std = (var + eps).sqrt()
        return (x - mean) / std

    # [B, T, C]
    m3 = m.unsqueeze(-1)  # [B, T, 1] broadcast across C
    denom = m3.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1, 1]
    mean = (x * m3).sum(dim=1, keepdim=True) / denom
    var = ((x - mean) ** 2 * m3).sum(dim=1, keepdim=True) / denom
    std = (var + eps).sqrt()
    return (x - mean) / std


def minmax_time(
    x: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    min_val: float = 0.0,
    max_val: float = 1.0,
    eps: float = 1e-9,
) -> torch.Tensor:
    """
    Min-Max scale along **time** per sequence (and per channel for AIRS).

    For padded steps (mask=0), the min/max are computed ignoring padding.

    Args:
        x: [B, T] or [B, T, C]
        mask: [B, T]
        min_val: target min (default 0.0)
        max_val: target max (default 1.0)
        eps: numerical floor for range

    Returns:
        Scaled tensor with same shape as `x`.
    """
    if max_val <= min_val:
        raise ValueError("max_val must be > min_val")

    is_airs, B, T, C = _infer_shapes(x)
    device, dtype = x.device, x.dtype
    m = _broadcast_mask(mask, B, T, device, torch.float32)

    if not is_airs:
        # [B, T]
        # masked min/max: replace pads with +inf/-inf so they don't affect extrema
        pos_inf = torch.tensor(float("inf"), device=device, dtype=dtype)
        neg_inf = -pos_inf
        x_min = torch.where(m.bool(), x, pos_inf).amin(dim=1, keepdim=True)
        x_max = torch.where(m.bool(), x, neg_inf).amax(dim=1, keepdim=True)
        rng = (x_max - x_min).clamp_min(eps)
        y = (x - x_min) / rng
        return y * (max_val - min_val) + min_val

    # [B, T, C]
    pos_inf = torch.tensor(float("inf"), device=device, dtype=dtype)
    neg_inf = -pos_inf
    m3 = m.unsqueeze(-1)
    x_min = torch.where(m3.bool(), x, pos_inf).amin(dim=1, keepdim=True)
    x_max = torch.where(m3.bool(), x, neg_inf).amax(dim=1, keepdim=True)
    rng = (x_max - x_min).clamp_min(eps)
    y = (x - x_min) / rng
    return y * (max_val - min_val) + min_val


def detrend_time_linear(x: torch.Tensor, mask: Optional[torch.Tensor] = None, eps: float = 1e-9) -> torch.Tensor:
    """
    Remove linear trend along **time** via masked least squares.

    Model per sequence (and per channel for AIRS):
        x_t ≈ a * t + b
    and return residuals: x - (a*t + b)

    Args:
        x: [B, T] or [B, T, C]
        mask: [B, T] (1=valid, 0=padded)
        eps: numerical stability in normal equations

    Returns:
        Detrended tensor with same shape as `x`.
    """
    is_airs, B, T, C = _infer_shapes(x)
    device, dtype = x.device, x.dtype
    m = _broadcast_mask(mask, B, T, device, torch.float32)

    # Design matrix columns: [t, 1]  (t normalized to [-1, 1] for numeric stability)
    t = torch.linspace(-1.0, 1.0, steps=T, device=device, dtype=dtype)  # [T]
    X1 = t.unsqueeze(0).expand(B, T)  # [B, T]
    X0 = torch.ones(B, T, device=device, dtype=dtype)
    W = m  # [B, T] weights 0/1

    # Precompute normal equation components (weighted)
    # A = X^T W X,   b = X^T W y
    XtW = torch.stack([
        (X1 * W).sum(dim=1),  # sum(t * w)
        (X0 * W).sum(dim=1),  # sum(1 * w) = sum(w)
    ], dim=1)  # [B, 2], not yet full A

    # Full A per batch:
    # A = [[sum(w*t^2), sum(w*t)],
    #      [sum(w*t),   sum(w)   ]]
    A00 = (X1 * X1 * W).sum(dim=1)           # [B]
    A01 = (X1 * W).sum(dim=1)                # [B]
    A11 = (X0 * W).sum(dim=1)                # [B]
    # A10 == A01 (symmetric)

    if not is_airs:
        # y: [B, T]
        y = x
        b0 = (X1 * y * W).sum(dim=1)         # [B]
        b1 = (X0 * y * W).sum(dim=1)         # [B]

        det = A00 * A11 - A01 * A01
        det = det + (det.abs() < eps).float() * eps  # guard

        a = ( A11 * b0 - A01 * b1) / det  # slope
        b = (-A01 * b0 + A00 * b1) / det  # intercept

        trend = a.unsqueeze(1) * X1 + b.unsqueeze(1) * X0
        return y - trend

    # AIRS: solve per channel using batch broadcasting
    # reshape to [B, 1] for broadcasting with C
    A00 = A00.unsqueeze(-1)  # [B, 1]
    A01 = A01.unsqueeze(-1)  # [B, 1]
    A11 = A11.unsqueeze(-1)  # [B, 1]

    X1b = X1.unsqueeze(-1)  # [B, T, 1]
    X0b = X0.unsqueeze(-1)  # [B, T, 1]
    Wb  = W.unsqueeze(-1)   # [B, T, 1]
    y = x  # [B, T, C]

    b0 = (X1b * y * Wb).sum(dim=1)  # [B, C]
    b1 = (X0b * y * Wb).sum(dim=1)  # [B, C]

    det = A00 * A11 - A01 * A01
    det = det + (det.abs() < eps).float() * eps  # [B, 1]

    a = (A11 * b0 - A01 * b1) / det  # [B, C]
    b = (-A01 * b0 + A00 * b1) / det  # [B, C]

    trend = a.unsqueeze(1) * X1b + b.unsqueeze(1) * X0b  # [B, T, C]
    return x - trend


def clip(x: torch.Tensor, min_val: float = -5.0, max_val: float = 5.0) -> torch.Tensor:
    """
    Value clipping with stable order: clamp to [min_val, max_val].

    Args:
        x: any shape tensor
        min_val: lower bound
        max_val: upper bound

    Returns:
        clamped tensor
    """
    if max_val < min_val:
        raise ValueError("clip: max_val must be >= min_val")
    return x.clamp(min=min_val, max=max_val)


# ---------------------------------------------------------------------------
# Simple transform composer
# ---------------------------------------------------------------------------

class Compose:
    """
    Compose a list of callables into a single transform.

    Each callable must have signature `fn(x, **kwargs)` and return a tensor.
    Additional static kwargs per transform can be supplied via lambdas or partials.

    Example
    -------
    >>> pipeline = Compose([
    ...     lambda x, mask=None: detrend_time_linear(x, mask=mask),
    ...     lambda x, mask=None: zscore_time(x, mask=mask),
    ...     lambda x, **k: clip(x, -3, 3),
    ... ])
    >>> y = pipeline(x, mask=mask)
    """

    def __init__(self, transforms: Sequence[Callable[..., torch.Tensor]]):
        self.transforms = list(transforms)

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        for t in self.transforms:
            x = t(x, **kwargs)
        return x
