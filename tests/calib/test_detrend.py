from __future__ import annotations

from typing import Tuple, Optional

import numpy as np

__all__ = ["remove_linear_trend", "linear_detrend"]


def _as_2d_series(y: np.ndarray) -> Tuple[np.ndarray, bool]:
    """
    Ensure y is (N, K). Returns (y2d, was_1d).
    """
    y = np.asarray(y)
    if y.ndim == 1:
        return y[:, None], True
    if y.ndim == 2:
        return y, False
    raise ValueError(f"y must be 1D or 2D, got shape {y.shape}")


def _check_design(X: np.ndarray, n: int) -> np.ndarray:
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError(f"X must be 2D design matrix (N,P); got shape {X.shape}")
    if X.shape[0] != n:
        raise ValueError(f"X.shape[0] must equal len(y)={n}; got X.shape[0]={X.shape[0]}")
    return X


def remove_linear_trend(
    y: np.ndarray,
    X: np.ndarray,
    *,
    rcond: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remove a linear trend given a design matrix.

    Solves (per series):
        minimize || X @ beta - y ||_2
      => beta = argmin; residual = y - X @ beta

    Args
    ----
    y : array_like, shape (N,) or (N,K)
        Observations. If 2D, detrends each column independently using the same X.
    X : array_like, shape (N,P)
        Design matrix (e.g., [1, t] for intercept + linear trend).
    rcond : float or None
        Passed to numpy.linalg.lstsq (cutoff for small singular values).
        Defaults to NumPy’s internal heuristic if None.

    Returns
    -------
    residual : np.ndarray, shape (N,) or (N,K)
        Detrended series (same shape as input y).
    beta : np.ndarray, shape (P,) or (P,K)
        Estimated coefficients. If y was 1D -> shape (P,); if (N,K) -> (P,K).

    Notes
    -----
    • Internally solves in float64 for stability, then casts back to y.dtype.
    • Raises on NaNs/Infs in X or y (explicitly check/clean upstream if needed).
    """
    y_in = np.asarray(y)
    y2d, was_1d = _as_2d_series(y_in)
    N, K = y2d.shape

    X = _check_design(X, N)

    # Promote to float64 for numerical robustness (esp. long N or collinearity)
    X64 = X.astype(np.float64, copy=False)
    y64 = y2d.astype(np.float64, copy=False)

    # Solve per-series (vectorized over columns using lstsq on stacked RHS)
    # numpy.linalg.lstsq supports multiple RHS: shape (N, K)
    beta64, *_rest = np.linalg.lstsq(X64, y64, rcond=rcond)

    # residuals: y - X @ beta
    fit64 = X64 @ beta64
    resid64 = y64 - fit64

    # Cast back to original dtype
    out_dtype = y_in.dtype if np.issubdtype(y_in.dtype, np.floating) else np.float32
    resid = resid64.astype(out_dtype, copy=False)
    beta = beta64.astype(np.float64 if out_dtype == np.float32 else out_dtype, copy=False)

    if was_1d:
        return resid[:, 0], beta[:, 0]
    return resid, beta


# Friendly alias
def linear_detrend(
    y: np.ndarray,
    X: np.ndarray,
    *,
    rcond: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Alias for `remove_linear_trend`. See that docstring for details.
    """
    return remove_linear_trend(y, X, rcond=rcond)