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
- Consistent return types: NumPy float or 0-D torch.Tensor (matching backend).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

try:  # optional torch support
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


ArrayLike = Union[np.ndarray, "torch.Tensor"]
MaybeMask = Optional[ArrayLike]

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


# ──────────────────────────────────────────────────────────────────────────────
# Backend helpers
# ──────────────────────────────────────────────────────────────────────────────

def _is_tensor(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, torch.Tensor)


def _same_backend(*xs: Any) -> str:
    """Infer backend: 'torch' if any tensor present, else 'numpy'."""
    return "torch" if _HAS_TORCH and any(_is_tensor(x) for x in xs) else "numpy"


def _device_dtype_like(a: ArrayLike, b: ArrayLike) -> Tuple[Any, Any]:
    """Return (device, dtype) to use for torch tensors, mirrored on 'a' then 'b'."""
    if not _HAS_TORCH:
        return None, None  # type: ignore[return-value]
    if _is_tensor(a):
        return a.device, a.dtype
    if _is_tensor(b):
        return b.device, b.dtype
    return None, None  # numpy path


def _to_numpy(x: Any) -> np.ndarray:
    if _is_tensor(x):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _finite_mask(x: ArrayLike) -> ArrayLike:
    if _is_tensor(x):
        return torch.isfinite(x)
    return np.isfinite(np.asarray(x))


def _broadcast_np(*xs: np.ndarray) -> Tuple[np.ndarray, ...]:
    """Strict broadcast using NumPy; raises on failure."""
    return np.broadcast_arrays(*xs)


def _broadcast_torch(*xs: "torch.Tensor") -> Tuple["torch.Tensor", ...]:
    """Strict broadcast using torch; raises on failure."""
    return torch.broadcast_tensors(*xs)


def _reduce(values: ArrayLike, reduction: str = "mean") -> Union[float, "torch.Tensor", np.ndarray]:
    """
    Reduce with NaN-safety.
    - 'none' returns input
    - 'sum' sums ignoring NaNs
    - 'mean' mean over finite elements
    """
    if _is_tensor(values):
        v = values
        if reduction == "none":
            return v
        if reduction == "sum":
            return torch.nansum(v)
        finite = torch.isfinite(v)
        denom = finite.sum()
        # Avoid dtype mismatch creating new scalar on CPU by mirroring device/dtype
        zero = torch.zeros((), device=v.device, dtype=v.dtype)
        return torch.nansum(torch.where(finite, v, zero)) / torch.clamp(denom, min=1)
    v = np.asarray(values)
    if reduction == "none":
        return v
    if reduction == "sum":
        return float(np.nansum(v))
    finite = np.isfinite(v)
    denom = int(finite.sum())
    if denom == 0:
        return float("nan")
    return float(np.nansum(np.where(finite, v, 0.0)) / denom)


def _weighted_reduce(values: ArrayLike, weights: ArrayLike, reduction: str = "mean") -> Union[float, "torch.Tensor", np.ndarray]:
    """
    Weighted reduction with NaN-safety; supports broadcasting.
    For 'mean', computes sum(w*v)/sum(w) over finite v and w.
    """
    if reduction == "none":
        return values if not _is_tensor(values) else values  # pass-through

    if _same_backend(values, weights) == "torch":
        v = values if _is_tensor(values) else torch.as_tensor(values)
        w = weights if _is_tensor(weights) else torch.as_tensor(weights, device=v.device, dtype=v.dtype)
        v, w = _broadcast_torch(v, w)
        mask = torch.isfinite(v) & torch.isfinite(w)
        v_mask = torch.where(mask, v, torch.zeros((), device=v.device, dtype=v.dtype))
        w_mask = torch.where(mask, w, torch.zeros((), device=v.device, dtype=v.dtype))
        if reduction == "sum":
            return torch.sum(v_mask * w_mask)
        denom = torch.sum(w_mask)
        eps = torch.finfo(v.dtype).eps
        return torch.sum(v_mask * w_mask) / torch.clamp(denom, min=eps)

    v = np.asarray(values, dtype=np.float64)
    w = np.asarray(weights, dtype=np.float64)
    v, w = _broadcast_np(v, w)
    mask = np.isfinite(v) & np.isfinite(w)
    v_mask = np.where(mask, v, 0.0)
    w_mask = np.where(mask, w, 0.0)
    if reduction == "sum":
        return float(np.sum(v_mask * w_mask))
    denom = np.sum(w_mask)
    if denom <= 0:
        return float("nan")
    return float(np.sum(v_mask * w_mask) / denom)


# ──────────────────────────────────────────────────────────────────────────────
# Gaussian NLL (negative log-likelihood)
# ──────────────────────────────────────────────────────────────────────────────

def gaussian_nll(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    mask: MaybeMask = None,
    reduction: str = "mean",
    eps: float = 1e-8,
    min_sigma: Optional[float] = None,
    max_sigma: Optional[float] = None,
) -> Union[float, np.ndarray, "torch.Tensor"]:
    """
    Gaussian Negative Log-Likelihood:
        nll = 0.5 * [ log(2πσ²) + (y - μ)² / σ² ]

    Args:
        mask: bool mask of valid elements (broadcastable to inputs).
        reduction: {"mean","sum","none"}.
        eps: small stabilization added as σ ← max(σ, eps).
        min_sigma / max_sigma: optional clamps on σ after eps.
    """
    backend = _same_backend(y, mu, sigma, mask)

    if backend == "torch":
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        device, dtype = _device_dtype_like(y_t, mu if _is_tensor(mu) else y_t)
        mu_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=device, dtype=y_t.dtype)
        sg_t = sigma if _is_tensor(sigma) else torch.as_tensor(sigma, device=device, dtype=y_t.dtype)

        sg_t = torch.clamp(sg_t, min=max(eps, 0.0))
        if min_sigma is not None:
            sg_t = torch.clamp(sg_t, min=float(min_sigma))
        if max_sigma is not None:
            sg_t = torch.clamp(sg_t, max=float(max_sigma))

        var = sg_t * sg_t
        # log(2πσ²) is numerically more stable than 2*log(σ) at tiny σ
        nll = 0.5 * (torch.log(2.0 * math.pi * var) + (y_t - mu_t) ** 2 / var)

        if mask is not None:
            m = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            nll = torch.where(m, nll, torch.tensor(float("nan"), device=y_t.device, dtype=y_t.dtype))
        return _reduce(nll, reduction=reduction)

    # NumPy
    y_n = np.asarray(y, dtype=np.float64)
    mu_n = np.asarray(mu, dtype=np.float64)
    sg_n = np.asarray(sigma, dtype=np.float64)

    sg_n = np.maximum(sg_n, max(eps, 0.0))
    if min_sigma is not None:
        sg_n = np.maximum(sg_n, float(min_sigma))
    if max_sigma is not None:
        sg_n = np.minimum(sg_n, float(max_sigma))

    var = sg_n * sg_n
    nll = 0.5 * (np.log(2.0 * math.pi * var) + (y_n - mu_n) ** 2 / var)
    if mask is not None:
        nll = np.where(np.asarray(mask, dtype=bool), nll, np.nan)
    return _reduce(nll, reduction=reduction)


# ──────────────────────────────────────────────────────────────────────────────
# Weighted challenge GLL
# ──────────────────────────────────────────────────────────────────────────────

def _build_weights(
    D: int,
    *,
    fgs1_weight: float = 58.0,
    custom: Optional[ArrayLike] = None,
    like: Optional[ArrayLike] = None,
) -> ArrayLike:
    """
    Build per-bin weights of length D. By default, sets w[0]=fgs1_weight and w[1:]=1.
    If 'custom' is provided, it overrides the entire vector (must be length D).
    """
    if custom is not None:
        if _is_tensor(custom) or _is_tensor(like):
            c = custom if _is_tensor(custom) else torch.as_tensor(
                custom, dtype=(like.dtype if _is_tensor(like) else torch.float32),
                device=(like.device if _is_tensor(like) else None)
            )
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
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    weights: Optional[ArrayLike] = None,
    fgs1_weight: float = 58.0,
    reduction: str = "mean",
    eps: float = 1e-8,
    min_sigma: Optional[float] = None,
    max_sigma: Optional[float] = None,
    mask: MaybeMask = None,
    sample_weights: Optional[ArrayLike] = None,  # optional per-sample weighting over leading dims
) -> Union[float, "torch.Tensor", np.ndarray]:
    """
    Gaussian NLL with per-bin weights (applied on the last dimension D).

    - If 'weights' is None, uses default challenge weights (w[0]=~58, others=1).
    - Supports broadcasting of weights over batch dims (e.g., (D,) → (N,D)).
    - 'sample_weights' may weight each element (broadcastable to nll shape without last dim).
    - Reductions:
        * 'mean' → weighted mean: sum(w * nll)/sum(w) (optionally with sample_weights)
        * 'sum'  → sum of weighted nll
        * 'none' → elementwise nll (unreduced)
    """
    # Elementwise NLL (no reduction)
    nll = gaussian_nll(y, mu, sigma, mask=mask, reduction="none", eps=eps, min_sigma=min_sigma, max_sigma=max_sigma)

    if _is_tensor(nll):
        if nll.dim() == 0:
            raise ValueError("Expected at least 1D for NLL; got scalar.")
        D = nll.shape[-1]
        w = _build_weights(D, fgs1_weight=fgs1_weight, custom=weights, like=nll)
        # expand per-bin weights to nll shape
        w_exp = w
        while w_exp.dim() < nll.dim():
            w_exp = w_exp.unsqueeze(0)
        w_exp = w_exp.expand_as(nll)

        # optional sample weights (must NOT include last dim)
        if sample_weights is not None:
            sw = sample_weights if _is_tensor(sample_weights) else torch.as_tensor(sample_weights, device=nll.device, dtype=nll.dtype)
            # expand to all dims except last; then tile last dim to D
            while sw.dim() < nll.dim() - 1:
                sw = sw.unsqueeze(0)
            sw = sw.expand(*nll.shape[:-1]).unsqueeze(-1).expand_as(nll)
            w_exp = w_exp * sw

        return _weighted_reduce(nll, w_exp, reduction=reduction)

    # NumPy path
    nll_n = np.asarray(nll, dtype=np.float64)
    if nll_n.ndim == 0:
        raise ValueError("Expected at least 1D for NLL; got scalar.")
    D = nll_n.shape[-1]
    w = _build_weights(D, fgs1_weight=fgs1_weight, custom=weights, like=None)
    w_shape = (1,) * (nll_n.ndim - 1) + (D,)
    w_b = np.broadcast_to(np.reshape(w, w_shape), nll_n.shape)

    if sample_weights is not None:
        sw = np.asarray(sample_weights, dtype=np.float64)
        sw_shape = nll_n.shape[:-1] + (1,)
        sw_b = np.broadcast_to(np.reshape(sw, sw_shape), nll_n.shape)
        w_b = w_b * sw_b

    return _weighted_reduce(nll_n, w_b, reduction=reduction)


def challenge_gll(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    fgs1_weight: float = 58.0,   # default ~58× for bin 0 (FGS1)
    reduction: str = "mean",
    eps: float = 1e-8,
    min_sigma: Optional[float] = None,
    max_sigma: Optional[float] = None,
    mask: MaybeMask = None,
    sample_weights: Optional[ArrayLike] = None,
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
        min_sigma=min_sigma,
        max_sigma=max_sigma,
        mask=mask,
        sample_weights=sample_weights,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Pointwise errors
# ──────────────────────────────────────────────────────────────────────────────

def mae(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    mask: MaybeMask = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Mean Absolute Error with NaN/Mask safety."""
    if _same_backend(y, mu, mask) == "torch":
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        m_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=y_t.device, dtype=y_t.dtype)
        diff = torch.abs(y_t - m_t)
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            diff = torch.where(mm, diff, torch.tensor(float("nan"), device=y_t.device, dtype=diff.dtype))
        return _reduce(diff, reduction=reduction)
    diff = np.abs(np.asarray(y) - np.asarray(mu))
    if mask is not None:
        diff = np.where(np.asarray(mask, dtype=bool), diff, np.nan)
    return _reduce(diff, reduction=reduction)


def rmse(
    y: ArrayLike,
    mu: ArrayLike,
    *,
    mask: MaybeMask = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Root Mean Squared Error with NaN/Mask safety."""
    if _same_backend(y, mu, mask) == "torch":
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        m_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=y_t.device, dtype=y_t.dtype)
        sq = (y_t - m_t) ** 2
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            sq = torch.where(mm, sq, torch.tensor(float("nan"), device=y_t.device, dtype=sq.dtype))
        if reduction == "none":
            return torch.sqrt(sq)
        return _reduce(torch.sqrt(sq), reduction=reduction)
    sq = (np.asarray(y) - np.asarray(mu)) ** 2
    if mask is not None:
        sq = np.where(np.asarray(mask, dtype=bool), sq, np.nan)
    if reduction == "none":
        return np.sqrt(sq)
    return _reduce(np.sqrt(sq), reduction=reduction)


# ──────────────────────────────────────────────────────────────────────────────
# Uncertainty diagnostics
# ──────────────────────────────────────────────────────────────────────────────

def coverage(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    alpha: float = 0.95,
    mask: MaybeMask = None,
) -> Union[float, "torch.Tensor"]:
    """
    Empirical coverage of the central interval with nominal mass=alpha under Normal assumption:
        interval = μ ± z * σ, where z = Φ^{-1}((1+alpha)/2).
    Returns the fraction of VALID elements falling inside the interval.
    """
    z = float(_z_from_alpha(alpha))

    if _same_backend(y, mu, sigma, mask) == "torch":
        y_t = y if _is_tensor(y) else torch.as_tensor(y)
        m_t = mu if _is_tensor(mu) else torch.as_tensor(mu, device=y_t.device, dtype=y_t.dtype)
        s_t = sigma if _is_tensor(sigma) else torch.as_tensor(sigma, device=y_t.device, dtype=y_t.dtype)

        lo, hi = m_t - z * s_t, m_t + z * s_t
        inside = (y_t >= lo) & (y_t <= hi)

        valid = torch.ones_like(inside, dtype=torch.bool)
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=y_t.device, dtype=torch.bool)
            valid = valid & mm
        valid = valid & _finite_mask(y_t) & _finite_mask(m_t) & _finite_mask(s_t)

        numer = (inside & valid).sum()
        denom = torch.clamp(valid.sum(), min=1)
        return numer.to(dtype=y_t.dtype) / denom.to(dtype=y_t.dtype)

    y_n, m_n, s_n = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    lo, hi = m_n - z * s_n, m_n + z * s_n
    inside = (y_n >= lo) & (y_n <= hi)

    valid = np.ones_like(inside, dtype=bool)
    if mask is not None:
        valid &= np.asarray(mask, dtype=bool)
    valid &= np.isfinite(y_n) & np.isfinite(m_n) & np.isfinite(s_n)

    numer = int(np.sum(inside & valid))
    denom = int(np.sum(valid))
    denom = max(denom, 1)
    return float(numer / denom)


def sharpness(
    sigma: ArrayLike,
    *,
    mask: MaybeMask = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Sharpness = average predictive σ (lower is sharper)."""
    if _is_tensor(sigma):
        s = sigma
        if mask is not None:
            mm = mask if _is_tensor(mask) else torch.as_tensor(mask, device=s.device, dtype=torch.bool)
            s = torch.where(mm, s, torch.tensor(float("nan"), device=s.device, dtype=s.dtype))
        return _reduce(s, reduction=reduction)
    s = np.asarray(sigma)
    if mask is not None:
        s = np.where(np.asarray(mask, dtype=bool), s, np.nan)
    return _reduce(s, reduction=reduction)


# ──────────────────────────────────────────────────────────────────────────────
# Utilities
# ──────────────────────────────────────────────────────────────────────────────

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


# ──────────────────────────────────────────────────────────────────────────────
# Bundle
# ──────────────────────────────────────────────────────────────────────────────

def compute_all(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    mask: MaybeMask = None,
    alpha: float = 0.95,
    fgs1_weight: float = 58.0,
    min_sigma: Optional[float] = None,
    max_sigma: Optional[float] = None,
    sample_weights: Optional[ArrayLike] = None,
) -> Dict[str, Union[float, "torch.Tensor"]]:
    """
    Convenience bundle for training/validation loops.
    Returns a dict of scalar metrics (NumPy floats or 0-dim torch tensors).
    """
    results: Dict[str, Union[float, "torch.Tensor"]] = {}
    results["gll"] = gaussian_nll(
        y, mu, sigma, mask=mask, reduction="mean",
        min_sigma=min_sigma, max_sigma=max_sigma,
    )
    results["challenge_gll"] = challenge_gll(
        y, mu, sigma, mask=mask, fgs1_weight=fgs1_weight, reduction="mean",
        min_sigma=min_sigma, max_sigma=max_sigma, sample_weights=sample_weights,
    )
    results["mae"] = mae(y, mu, mask=mask, reduction="mean")
    results["rmse"] = rmse(y, mu, mask=mask, reduction="mean")
    results["coverage"] = coverage(y, mu, sigma, alpha=alpha, mask=mask)
    results["sharpness"] = sharpness(sigma, mask=mask, reduction="mean")
    return results
