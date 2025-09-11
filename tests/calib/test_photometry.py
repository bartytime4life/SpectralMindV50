from __future__ import annotations

from typing import Optional, Tuple, Union

import numpy as np

__all__ = [
    "aperture_sum",
    "aperture_flux",
    "annulus_median",
    "annulus_background",
]

ArrayLike = Union[np.ndarray]


def _mesh(hw: Tuple[int, int], center: Tuple[float, float]) -> Tuple[np.ndarray, np.ndarray]:
    """Return (Y, X) mesh (float32) shifted so that (center_y, center_x) is at (0,0)."""
    h, w = hw
    cy, cx = center
    y, x = np.mgrid[:h, :w].astype(np.float32, copy=False)
    return y - np.float32(cy), x - np.float32(cx)


def _ensure_image(img: ArrayLike) -> Tuple[np.ndarray, Tuple[int, int], Tuple[int, ...]]:
    """Return (img_float, (H, W), prefix_shape) where img has shape [..., H, W]."""
    a = np.asarray(img)
    if a.ndim < 2:
        raise ValueError(f"image must have at least 2 dims [..., H, W]; got {a.shape}")
    h, w = a.shape[-2], a.shape[-1]
    prefix = a.shape[:-2]
    return a.astype(np.float32, copy=False), (h, w), prefix


def _nan_to_zero(a: np.ndarray) -> np.ndarray:
    """Replace NaNs with 0 for safe sums without changing other values."""
    if np.isnan(a).any():
        a = a.copy()
        a[np.isnan(a)] = 0.0
    return a


# -------------------------------------------------------------------------
# Aperture photometry
# -------------------------------------------------------------------------

def aperture_sum(
    img: ArrayLike,
    x: float,
    y: float,
    r: float,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Sum pixel values within a circular aperture of radius `r` centered at (x, y).

    Args
    ----
    img : ndarray [..., H, W]
        Image (or stack). Only the last 2 dims are spatial.
    x, y : float
        Center coordinates (X, Y) in pixel units (0 at left/top).
    r : float
        Aperture radius (pixels).
    mask : ndarray[H, W] or broadcastable, optional
        Boolean mask; True = include, False = exclude.

    Returns
    -------
    flux : float
        Sum of pixel values within the aperture over the last 2 dims.
        If img has a prefix shape (stack), all frames are summed together.
    """
    a, (H, W), prefix = _ensure_image(img)
    Y, X = _mesh((H, W), (y, x))
    rr = X * X + Y * Y
    m = rr <= (r * r)

    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool, copy=False)
        try:
            m = m & mask_arr
        except ValueError as e:
            raise ValueError(f"mask not broadcastable to image field ({H},{W}): got {mask_arr.shape}") from e

    # Prepare image for masked sum
    a2 = _nan_to_zero(a)
    if prefix:
        # Broadcast mask to whole stack and sum over spatial dims, then over prefix dims
        # (Your tests pass single images; this path is still correct for stacks.)
        masked = np.where(m, 1.0, 0.0).astype(np.float32)
        # Expand mask to prefix
        for _ in range(len(prefix)):
            masked = masked[None, ...]
        flux = float((a2 * masked).sum())
    else:
        flux = float(a2[m].sum())

    return flux


# Friendly alias
aperture_flux = aperture_sum


# -------------------------------------------------------------------------
# Annulus statistics (median background)
# -------------------------------------------------------------------------

def annulus_median(
    img: ArrayLike,
    x: float,
    y: float,
    r_in: float,
    r_out: float,
    *,
    mask: Optional[np.ndarray] = None,
) -> float:
    """
    Median pixel value in an annulus r_in <= r < r_out centered at (x, y).

    Args
    ----
    img : ndarray [..., H, W]
    x, y : float
    r_in, r_out : float
        Inner/outer radii (pixels), with 0 <= r_in < r_out
    mask : ndarray[H, W] or broadcastable, optional
        Boolean mask; True = include, False = exclude.

    Returns
    -------
    bkg : float
        Median of pixels within annulus (NaNs ignored).
        If `img` has a prefix, all frames are pooled for the median.
    """
    if not (r_out > r_in >= 0.0):
        raise ValueError("Require r_out > r_in >= 0")

    a, (H, W), prefix = _ensure_image(img)
    Y, X = _mesh((H, W), (y, x))
    rr = X * X + Y * Y
    ann_m = (rr >= (r_in * r_in)) & (rr < (r_out * r_out))

    if mask is not None:
        mask_arr = np.asarray(mask).astype(bool, copy=False)
        try:
            ann_m = ann_m & mask_arr
        except ValueError as e:
            raise ValueError(f"mask not broadcastable to image field ({H},{W}): got {mask_arr.shape}") from e

    a2 = a.astype(np.float32, copy=False)
    if prefix:
        # Pool all frames in the annulus for a single robust median
        sel = a2[..., ann_m]
        sel = sel.reshape(-1)  # flatten across prefix dims
    else:
        sel = a2[ann_m]

    # Robust to NaNs
    if sel.size == 0:
        return float("nan")
    return float(np.nanmedian(sel))


# Alias for alternative name your test recognizes
annulus_background = annulus_median