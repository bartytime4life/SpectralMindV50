from __future__ import annotations

from typing import Iterable, Optional, Tuple, Union

import numpy as np

__all__ = ["fit_polynomial", "fit", "evaluate", "predict"]


def _ensure_points(points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Validate and normalize input points.

    Returns
    -------
    x : (N,) float64
    y : (N,) float64
    """
    pts = np.asarray(points)
    if pts.ndim != 2 or pts.shape[1] != 2:
        raise ValueError(f"`points` must be shape (N, 2); got {pts.shape}")
    if not np.issubdtype(pts.dtype, np.floating):
        pts = pts.astype(np.float64)
    if np.isnan(pts).any() or np.isinf(pts).any():
        raise ValueError("points contain NaN/Inf; clean or mask before fitting")
    x = pts[:, 0].astype(np.float64, copy=False)
    y = pts[:, 1].astype(np.float64, copy=False)

    # Sort by x for better conditioning (duplicates allowed)
    order = np.argsort(x, kind="mergesort")
    x = x[order]
    y = y[order]
    return x, y


def fit_polynomial(points: np.ndarray, order: int = 2, rcond: Optional[float] = None) -> np.ndarray:
    """
    Fit y(x) with a polynomial of given `order` using least squares.

    Parameters
    ----------
    points : (N, 2) array_like
        Stacked [x, y] pairs. NaN/Inf not allowed.
    order : int
        Polynomial degree (>= 1).
    rcond : float or None
        Cutoff for small singular values in least squares (passed to numpy.polyfit).

    Returns
    -------
    coeffs : (order+1,) np.ndarray float64
        Polynomial coefficients in decreasing powers (compatible with numpy.polyval).
    """
    if order < 1:
        raise ValueError(f"order must be >= 1; got {order}")
    x, y = _ensure_points(points)
    if x.size < order + 1:
        raise ValueError(f"need at least {order+1} points; got {x.size}")

    # Use numpy.polyfit (float64) for robust LS; no weights by default
    coeffs = np.polyfit(x, y, deg=order, rcond=rcond)
    # Ensure ndarray float64
    coeffs = np.asarray(coeffs, dtype=np.float64)
    return coeffs


# Back-compat alias
def fit(points: np.ndarray, order: int = 2, rcond: Optional[float] = None) -> np.ndarray:
    return fit_polynomial(points, order=order, rcond=rcond)


def evaluate(coeffs: Union[np.ndarray, Iterable[float]], x: Union[np.ndarray, float]) -> np.ndarray:
    """
    Evaluate polynomial with `coeffs` at `x`.

    Parameters
    ----------
    coeffs : iterable of float
        High-to-low power coefficients as returned by `fit_polynomial`.
    x : array_like or float
        Points where to evaluate.

    Returns
    -------
    y_hat : np.ndarray
        Predicted values with dtype float64 (scalar input returns scalar ndarray).
    """
    c = np.asarray(coeffs, dtype=np.float64)
    xv = np.asarray(x, dtype=np.float64)
    y_hat = np.polyval(c, xv)
    return y_hat


# Back-compat alias
def predict(coeffs: Union[np.ndarray, Iterable[float]], x: Union[np.ndarray, float]) -> np.ndarray:
    return evaluate(coeffs, x)