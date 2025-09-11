from __future__ import annotations

from typing import Literal, Optional, Tuple

import numpy as np

__all__ = ["apply", "cds"]


def _split_halves(T: int) -> Tuple[slice, slice]:
    """
    Split the time axis [0..T) into two contiguous halves (first, last).
    For odd T, the extra frame goes to the last half.
    """
    mid = T // 2
    if T % 2 == 0:
        first = slice(0, mid)
        last = slice(mid, T)
    else:
        first = slice(0, mid)
        last = slice(mid, T)  # last half has one more frame
    return first, last


def cds(
    frames: np.ndarray,
    *,
    mode: Literal["auto", "pair", "halves"] = "auto",
    dtype: np.dtype = np.float32,
    nan_policy: Literal["propagate", "ignore"] = "ignore",
) -> np.ndarray:
    """
    Correlated Double Sampling (CDS) for imaging time series.

    Parameters
    ----------
    frames : np.ndarray
        Array of shape [T, H, W] (T >= 2). Integer/float accepted.
    mode : {"auto","pair","halves"}, default "auto"
        "pair"   -> last - first
        "halves" -> mean(last half) - mean(first half)
        "auto"   -> "halves" if T >= 4 else "pair"
    dtype : np.dtype, default np.float32
        Output dtype.
    nan_policy : {"propagate","ignore"}, default "ignore"
        If "ignore", uses nanmean for half-averages so NaNs don't poison
        the estimate; if "propagate", regular mean is used.

    Returns
    -------
    out : np.ndarray
        CDS image of shape [H, W], dtype `dtype`.

    Notes
    -----
    • CDS suppresses static backgrounds and per-pixel offsets, increasing contrast.
    • For short stacks, last-first is often sufficient and higher SNR than
      averaging; for longer stacks, halves averaging reduces read noise.
    """
    f = np.asarray(frames)
    if f.ndim != 3:
        raise ValueError(f"frames must have shape [T,H,W]; got {f.shape}")
    T, H, W = f.shape
    if T < 2:
        raise ValueError("frames must have T >= 2")

    # Work in float for math
    f = f.astype(np.float32, copy=False)

    # Select strategy
    if mode == "auto":
        mode_eff = "halves" if T >= 4 else "pair"
    else:
        mode_eff = mode

    if mode_eff == "pair":
        # last - first
        first_img = f[0]
        last_img = f[-1]
        out = last_img - first_img
    elif mode_eff == "halves":
        # mean of last half minus mean of first half
        sl_first, sl_last = _split_halves(T)
        if nan_policy == "ignore":
            first_img = np.nanmean(f[sl_first], axis=0)
            last_img = np.nanmean(f[sl_last], axis=0)
        else:
            first_img = f[sl_first].mean(axis=0)
            last_img = f[sl_last].mean(axis=0)
        out = last_img - first_img
    else:
        raise ValueError(f"Unknown mode: {mode!r}")

    return out.astype(dtype, copy=False)


# Back-compat friendly alias expected by tests
def apply(frames: np.ndarray, **kwargs) -> np.ndarray:
    """
    Apply correlated double sampling (CDS) to a [T,H,W] stack.

    This is a thin wrapper over `cds()` for compatibility.
    """
    return cds(frames, **kwargs)