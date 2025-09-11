from __future__ import annotations

from typing import Literal, Optional

import numpy as np

__all__ = ["apply", "correct"]


def _match_level(
    img: np.ndarray,
    dark: np.ndarray,
    how: Literal["none", "mean", "median"] = "median",
    nan_policy: Literal["propagate", "ignore"] = "ignore",
) -> float:
    """
    Compute an additive delta to align the dark frame's DC level to the image.
    Returns the amount to ADD to 'dark' before subtraction.
    """
    if how == "none":
        return 0.0

    red = np.nanmedian if nan_policy == "ignore" else np.median
    if how == "mean":
        red = np.nanmean if nan_policy == "ignore" else np.mean

    try:
        d_img = float(red(img))
        d_dark = float(red(dark))
    except Exception:
        # Fallback: treat as no adjustment if reduction fails
        return 0.0
    return d_img - d_dark


def apply(
    img: np.ndarray,
    dark: np.ndarray,
    *,
    dtype: np.dtype = np.float32,
    level_match: Literal["none", "mean", "median"] = "median",
    scale: Optional[float] = None,
    clip_min: Optional[float] = None,
    clip_max: Optional[float] = None,
    nan_policy: Literal["propagate", "ignore"] = "ignore",
) -> np.ndarray:
    """
    Dark-frame correction.

    out = img - (scale * (dark + delta)), where delta aligns DC level
          (if level_match != "none").

    Parameters
    ----------
    img : ndarray
        Input image (H,W) or stack (...,H,W). Integer/float accepted.
    dark : ndarray
        Master dark frame, broadcastable to img's shape.
    dtype : numpy dtype, default float32
        Output dtype.
    level_match : {"none","mean","median"}, default "median"
        If not "none", shift the dark by delta so its global level matches img.
        This robustly handles small bias drifts between acquisition sets.
    scale : float or None
        Optional multiplicative scale applied to dark (e.g., exposure/gain ratio).
        If None, scale=1.0.
    clip_min, clip_max : float or None
        Optional clipping of the result after subtraction.
    nan_policy : {"propagate","ignore"}, default "ignore"
        If "ignore", reductions for level matching use nan-* reducers and subtraction
        operates with NaNs preserved.

    Returns
    -------
    out : ndarray
        Corrected image, same shape as img, dtype `dtype`.
    """
    x = np.asarray(img)
    d = np.asarray(dark)

    # Promote to float for safe math
    xf = x.astype(np.float32, copy=False)
    df = d.astype(np.float32, copy=False)

    # Broadcast check (will raise naturally on shape mismatch)
    try:
        _ = xf - df  # dry run broadcast
    except ValueError as e:
        raise ValueError(f"dark frame is not broadcastable to image shape: img={x.shape}, dark={d.shape}") from e

    # Optional scale
    s = 1.0 if scale is None else float(scale)

    # Compute DC alignment delta to add to dark before subtraction
    delta = _match_level(xf, df, how=level_match, nan_policy=nan_policy)

    # Core subtraction
    dark_adj = s * (df + delta)
    out = xf - dark_adj

    # Optional clipping
    if clip_min is not None or clip_max is not None:
        lo = -np.inf if clip_min is None else float(clip_min)
        hi = np.inf if clip_max is None else float(clip_max)
        out = np.clip(out, lo, hi, out=out)

    return out.astype(dtype, copy=False)


# Back-compat alias
def correct(img: np.ndarray, dark: np.ndarray, **kwargs) -> np.ndarray:
    """Alias for `apply`."""
    return apply(img, dark, **kwargs)