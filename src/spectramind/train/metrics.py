# src/spectramind/train/metrics.py
"""
SpectraMind V50 — Training Metrics
----------------------------------

Implements metrics for the NeurIPS Ariel Data Challenge:
- Gaussian Negative Log-Likelihood (NLL/GLL) with dominant FGS1 weight on bin 0 (~58×)
- MAE / RMSE
- Uncertainty diagnostics: empirical coverage, sharpness (avg sigma)

Design goals
------------
- Kaggle/CI-safe: NumPy required; PyTorch is optional (auto-detected).
- Batched + broadcast-friendly; robust masking that never divides by zero.
- Deterministic, numerically-stable; eps clamps for log/variance.
- Flexible per-bin weighting (default: w[0]≈58 for FGS1; others 1.0).

Shapes
------
- Elementwise inputs broadcast to a common shape.
- Reductions: {"mean","sum","none"}.
- For (N, D) predictions: reductions aggregate across all elements.

"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:  # optional torch support
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------
__all__ = [
    "gaussian_nll",
    "challenge_gll",
    "weighted_gll",
    "mae",
    "rmse",
    "coverage",
    "sharpness",
    "compute_all",
]


# ---------------------------------------------------------------------
# Backend helpers
# ---------------------------------------------------------------------
def _is_tensor(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _to_numpy(x: Any) -> np.ndarray:
    if _HAS_TORCH and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _nan_like(x: Any) -> Any:
    if _is_tensor(x):
        return torch.tensor(float("nan"), device=x.device, dtype=x.dtype)
    return float("nan")


def _finite_mask(x: Any) -> Any:
    if _is_tensor(x):
        return torch.isfinite(x)
    return np.isfinite(x)


def _logical_and(a: Any, b: Any) -> Any:
    if _is_tensor(a) or _is_tensor(b):
        a_t = a if _is_tensor(a) else torch.as_tensor(a, dtype=torch.bool, device=(b.device if _is_tensor(b) else None))
        b_t = b if _is_tensor(b) else torch.as_tensor(b, dtype=torch.bool, device=(a.device if _is_tensor(a) else None))
        return a_t & b_t
    return np.logical_and(np.asarray(a, dtype=bool), np.asarray(b, dtype=bool))


def _reduce(values: Any, reduction: str = "mean") -> Any:
    """
    Reduce with NaN-safety. 'none' returns input; 'sum' sums ignoring NaNs; 'mean' mean over finite.
    """
    if _is_tensor(values):
        if reduction == "none":
            return values
        if reduction == "sum":
            return torch.nansum(values)
        # mean
        finite = torch.isfinite(values)
        denom = finite.sum()
        return torch.nansum(torch.where(finite, values, torch.tensor(0.0, device=values.device, dtype=values.dtype))) / torch.clamp(denom, min=1)
    # NumPy
    v = np.asarray(values)
    if reduction == "none":
        return v
    if reduction == "sum":
        return float(np.nansum(v))
    # mean
    finite = np.isfinite(v)
    denom = int(finite.sum())
    if denom == 0:
        return float("nan")
    return float(np.nansum(np.where(finite, v, 0.0)) / denom)


def _weighted_reduce(values: Any, weights: Any, reduction: str = "mean") -> Any:
    """
    Weighted reduction; supports broadcasting. For 'mean', computes sum(w*v)/sum(w) over valid elements.
    NaN elements in values are ignored (not counted in denominator); weights at those positions are ignored too.
    """
    if reduction == "none":
        # Return elementwise weight * value to allow caller to post-process
        return values * weights

    if _is_tensor(values) or _is_tensor(weights):
        v = values if _is_tensor(values) else torch.as_tensor(values, dtype=torch.float32)
        w = weights if _is_tensor(weights) else torch.as_tensor(weights, dtype=v.dtype, device=v.device)
        v = v.to(dtype=w.dtype, device=w.device)
        mask = torch.isfinite(v) & torch.isfinite(w)
        v_masked = torch.where(mask, v, torch.tensor(0.0, device=v.device, dtype=v.dtype))
        w_masked = torch.where(mask, w, torch.tensor(0.0, device=w.device, dtype=w.dtype))
        if reduction == "sum":
            return torch.sum(v_masked * w_masked)
        # mean
        denom = torch.sum(w_masked)
        return torch.sum(v_masked * w_masked) / torch.clamp(denom, min=torch.finfo(v.dtype).eps)
    # NumPy
    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    mask = np.isfinite(v) & np.isfinite(w)
    v_masked = np.where(mask, v, 0.0)
    w_masked = np.where(mask, w, 0.0)
    if reduction == "sum":
        return float(np.sum(v_masked * w_masked))
    denom = np.sum(w_masked)
    if denom <= 0:
        return float("nan")
    return float(np.sum(v_masked * w_masked) / denom)


# ---------------------------------------------------------------------
# Core Gaussian NLL (negative log-likelihood)
# ---------------------------------------------------------------------
def gaussian_nll(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> Union[float, np.ndarray, "torch.Tensor"]:
    """
    Gaussian Negative Log-Likelihood:
        nll = 0.5 * [ log(2πσ²) + (y - μ)² / σ² ]

    - Works with NumPy arrays or PyTorch tensors.
    - 'mask' (bool) selects valid elements; invalids are ignored in reductions.
    - 'eps' clamps σ to avoid log(0) and div-by-zero.
    """
    if _HAS_TORCH and any(_is_tensor(t) for t in (y, mu, sigma, mask)):
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        mu_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=y_t.device, dtype=y_t.dtype)
        sg_t = sigma if _is_tensor(sigma) else torch.as_tensor(sigma, device=y_t.device, dtype=y_t.dtype)

        var = torch.clamp(sg_t, min=eps) ** 2
        nll = 0.5 * (torch.log(2.0 * math.pi * var) + (y_t - mu_t) ** 2 / var)

        if mask is not None:
            m = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            nll = torch.where(m, nll, torch.tensor(float("nan"), device=y_t.device, dtype=y_t.dtype))

        # Drop NaNs in reduction paths; keep as-is for 'none'
        return _reduce(nll, reduction=reduction)

    # NumPy path
    y_n = np.asarray(y)
    mu_n = np.asarray(mu)
    sg_n = np.asarray(sigma)
    var = np.maximum(sg_n, eps) ** 2
    nll = 0.5 * (np.log(2.0 * math.pi * var) + (y_n - mu_n) ** 2 / var)

    if mask is not None:
        nll = np.where(np.asarray(mask, dtype=bool), nll, np.nan)

    return _reduce(nll, reduction=reduction)


# ---------------------------------------------------------------------
# Weighted challenge GLL
# ---------------------------------------------------------------------
def _build_weights(
    D: int,
    *,
    fgs1_weight: float = 58.0,
    custom: Optional[Union[np.ndarray, "torch.Tensor", Tuple[float, ...]]] = None,
    like: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> Union[np.ndarray, "torch.Tensor"]:
    """
    Build per-bin weights of length D. By default, sets w[0]=fgs1_weight and w[1:]=1.
    If 'custom' is provided, it overrides the entire vector.
    """
    if custom is not None:
        if _is_tensor(like) or _is_tensor(custom):
            c = custom if _is_tensor(custom) else torch.as_tensor(custom, dtype=(like.dtype if _is_tensor(like) else torch.float32),
                                                                  device=(like.device if _is_tensor(like) else None))
            if c.numel() != D:
                raise ValueError(f"custom weights length {c.numel()} != D={D}")
            return c
        c = np.asarray(custom, dtype=np.float64)
        if c.size != D:
            raise ValueError(f"custom weights length {c.size} != D={D}")
        return c

    if _is_tensor(like):
        w = torch.ones(D, dtype=like.dtype, device=like.device)
        w[0] = float(fgs1_weight)
        return w
    w = np.ones(D, dtype=np.float64)
    w[0] = float(fgs1_weight)
    return w


def weighted_gll(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    weights: Optional[Union[np.ndarray, "torch.Tensor", Tuple[float, ...]]] = None,
    fgs1_weight: float = 58.0,
    reduction: str = "mean",
    eps: float = 1e-8,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> Union[float, "torch.Tensor", np.ndarray]:
    """
    Gaussian NLL with per-bin weights.

    - If 'weights' is None, uses default challenge weights (w[0]=~58, others=1).
    - Supports broadcasting of weights over batch dims (e.g., (D,) applied to (N,D)).
    - Reductions: 'mean' computes weighted mean (sum(w*nll)/sum(w)).
    """
    # Elementwise NLL (no reduction)
    nll = gaussian_nll(y, mu, sigma, mask=mask, reduction="none", eps=eps)

    # Infer D from last dimension (required)
    if _is_tensor(nll):
        if nll.dim() == 0:
            raise ValueError("Expected at least 1D for NLL; got scalar.")
        D = nll.shape[-1]
        w = _build_weights(D, fgs1_weight=fgs1_weight, custom=weights, like=nll)
        # Align weights for broadcasting against nll
        while w.dim() < nll.dim():
            w = w.unsqueeze(0)
        # Weighted reduction
        return _weighted_reduce(nll, w.expand_as(nll), reduction=reduction)
    # NumPy
    nll_n = np.asarray(nll)
    if nll_n.ndim == 0:
        raise ValueError("Expected at least 1D for NLL; got scalar.")
    D = nll_n.shape[-1]
    w = _build_weights(D, fgs1_weight=fgs1_weight, custom=weights, like=None)
    # Broadcast weights over batch dimensions
    w_shape = (1,) * (nll_n.ndim - 1) + (D,)
    w_b = np.reshape(w, w_shape)
    return _weighted_reduce(nll_n, np.broadcast_to(w_b, nll_n.shape), reduction=reduction)


def challenge_gll(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    fgs1_weight: float = 58.0,   # default ~58× for bin 0 (FGS1)
    reduction: str = "mean",
    eps: float = 1e-8,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> Union[float, "torch.Tensor", np.ndarray]:
    """
    Convenience wrapper for weighted GLL using the Ariel challenge default:
    w = [fgs1_weight, 1, 1, ..., 1].
    """
    return weighted_gll(
        y, mu, sigma,
        weights=None,
        fgs1_weight=fgs1_weight,
        reduction=reduction,
        eps=eps,
        mask=mask,
    )


# ---------------------------------------------------------------------
# Pointwise errors
# ---------------------------------------------------------------------
def mae(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Mean Absolute Error with NaN/Mask safety."""
    if _HAS_TORCH and any(_is_tensor(t) for t in (y, mu, mask)):
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        m_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=y_t.device, dtype=y_t.dtype)
        diff = torch.abs(y_t - m_t)
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            diff = torch.where(mm, diff, torch.tensor(float("nan"), device=y_t.device, dtype=diff.dtype))
        return _reduce(diff, reduction=reduction)
    # NumPy
    diff = np.abs(np.asarray(y) - np.asarray(mu))
    if mask is not None:
        diff = np.where(np.asarray(mask, dtype=bool), diff, np.nan)
    return _reduce(diff, reduction=reduction)


def rmse(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Root Mean Squared Error with NaN/Mask safety."""
    if _HAS_TORCH and any(_is_tensor(t) for t in (y, mu, mask)):
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        m_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=y_t.device, dtype=y_t.dtype)
        sq = (y_t - m_t) ** 2
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            sq = torch.where(mm, sq, torch.tensor(float("nan"), device=y_t.device, dtype=sq.dtype))
        # Reduce after sqrt for correct RMSE
        if reduction == "none":
            return torch.sqrt(sq)
        return _reduce(torch.sqrt(sq), reduction=reduction)
    # NumPy
    sq = (np.asarray(y) - np.asarray(mu)) ** 2
    if mask is not None:
        sq = np.where(np.asarray(mask, dtype=bool), sq, np.nan)
    if reduction == "none":
        return np.sqrt(sq)
    return _reduce(np.sqrt(sq), reduction=reduction)


# ---------------------------------------------------------------------
# Uncertainty diagnostics
# ---------------------------------------------------------------------
def coverage(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    alpha: float = 0.95,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> Union[float, "torch.Tensor"]:
    """
    Empirical coverage of the central interval with nominal mass=alpha under Normal assumption:
        interval = μ ± z * σ, where z = Φ^{-1}((1+alpha)/2).
    Returns the fraction of VALID elements falling inside the interval.
    """
    z = float(_z_from_alpha(alpha))

    if _HAS_TORCH and any(_is_tensor(t) for t in (y, mu, sigma, mask)):
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        m_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=y_t.device, dtype=y_t.dtype)
        s_t = sigma if _is_tensor(sigma) else torch.as_tensor(sigma, device=y_t.device, dtype=y_t.dtype)

        lo, hi = m_t - z * s_t, m_t + z * s_t
        inside = (y_t >= lo) & (y_t <= hi)

        valid = torch.ones_like(inside, dtype=torch.bool)
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            valid = valid & mm
        # Also exclude non-finite inputs from the denominator
        valid = valid & _finite_mask(y_t) & _finite_mask(m_t) & _finite_mask(s_t)

        numer = (inside & valid).sum()
        denom = torch.clamp(valid.sum(), min=1)
        return numer.to(dtype=y_t.dtype) / denom.to(dtype=y_t.dtype)

    # NumPy
    y_n, m_n, s_n = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    lo, hi = m_n - z * s_n, m_n + z * s_n
    inside = (y_n >= lo) & (y_n <= hi)

    valid = np.ones_like(inside, dtype=bool)
    if mask is not None:
        valid = valid & np.asarray(mask, dtype=bool)
    valid = valid & np.isfinite(y_n) & np.isfinite(m_n) & np.isfinite(s_n)

    numer = int(np.sum(inside & valid))
    denom = int(np.sum(valid))
    denom = max(denom, 1)
    return float(numer / denom)


def sharpness(
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Sharpness = average predictive σ (lower is sharper)."""
    if _HAS_TORCH and _is_tensor(sigma):
        s = sigma
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=s.device, dtype=torch.bool)
            s = torch.where(mm, s, torch.tensor(float("nan"), device=s.device, dtype=s.dtype))
        return _reduce(s, reduction=reduction)
    # NumPy
    s = np.asarray(sigma)
    if mask is not None:
        s = np.where(np.asarray(mask, dtype=bool), s, np.nan)
    return _reduce(s, reduction=reduction)


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _z_from_alpha(alpha: float) -> float:
    """
    Two-sided z-score for central coverage alpha (Acklam's inverse-normal approximation).
    Accurate for typical alpha in (0.5, 0.9999).
    """
    p = 0.5 * (1.0 + alpha)  # central mass -> upper-tail CDF point

    a1, a2, a3 = -39.69683028665376, 220.9460984245205, -275.9285104469687
    a4, a5, a6 = 138.3577518672690, -30.66479806614716, 2.506628277459239
    b1, b2, b3 = -54.47609879822406, 161.5858368580409, -155.6989798598866
    b4, b5, b6 = 66.80131188771972, -13.28068155288572, 1.0
    c1, c2, c3 = -0.007784894002430293, -0.3223964580411365, -2.400758277161838
    c4, c5, c6 = -2.549732539343734, 4.374664141464968, 2.938163982698783
    d1, d2, d3 = 0.007784695709041462, 0.3224671290700398, 2.445134137142996
    d4 = 3.754408661907416

    plow = 0.02425
    phigh = 1.0 - plow

    if p < plow:
        q = math.sqrt(-2.0 * math.log(p))
        x = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        return float(x)
    if p > phigh:
        q = math.sqrt(-2.0 * math.log(1.0 - p))
        x = -(((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) / ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0)
        return float(x)
    q = p - 0.5
    r = q * q
    x = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q / (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + b6)
    return float(x)


def compute_all(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    alpha: float = 0.95,
    fgs1_weight: float = 58.0,
) -> Dict[str, Union[float, "torch.Tensor"]]:
    """
    Convenience bundle for training/validation loops.
    Returns a dict of scalar metrics (NumPy floats or 0-dim torch tensors).
    """
    results: Dict[str, Union[float, "torch.Tensor"]] = {}
    results["gll"] = gaussian_nll(y, mu, sigma, mask=mask, reduction="mean")
    results["challenge_gll"] = challenge_gll(y, mu, sigma, mask=mask, fgs1_weight=fgs1_weight, reduction="mean")
    results["mae"] = mae(y, mu, mask=mask, reduction="mean")
    results["rmse"] = rmse(y, mu, mask=mask, reduction="mean")
    results["coverage"] = coverage(y, mu, sigma, alpha=alpha, mask=mask)
    results["sharpness"] = sharpness(sigma, mask=mask, reduction="mean")
    return results
