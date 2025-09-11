from __future__ import annotations

"""
SpectraMind V50 — Metrics Suite (NumPy/Torch interop)

Implements:
  • gaussian_nll(y, mu, sigma, reduction="mean")
  • challenge_gll(y, mu, sigma, fgs1_weight=58.0, reduction="mean", mask=None)
  • mae(y, mu, mask=None)
  • rmse(y, mu, mask=None)
  • coverage(y, mu, sigma, alpha=0.95)
  • sharpness(sigma, reduction="mean")

Design goals:
  • Accept NumPy arrays or Torch tensors (Torch optional).
  • Broadcasting along last dimension (spectral bins) is supported.
  • Numerical stability: clamp σ with eps; handle tiny σ gracefully.
  • Reductions: "none" → elementwise (B,K); "mean"/"sum" → scalar.
  • Optional boolean mask (B,K) for metrics that aggregate (challenge_gll, mae/rmse).

Notes:
  • gaussian_nll(reduction="none") always returns per-element NLLs.
  • challenge_gll(reduction="none") returns per-element NLLs (unweighted).
    Weighting is applied only at aggregation time for "mean"/"sum".
"""

from typing import Any, Optional, Tuple, Union

import math
import numpy as _np

try:  # Torch is optional
    import torch as _torch  # type: ignore
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False

ArrayLike = Union[_np.ndarray, "._torch.Tensor"]  # type: ignore[name-defined]


# ---------------------------------------------------------------------
# Backend helpers (NumPy <-> Torch)
# ---------------------------------------------------------------------
def _is_torch(x: Any) -> bool:
    return _HAS_TORCH and isinstance(x, _torch.Tensor)


def _to_backend(*xs: ArrayLike):
    """
    Return ('np'|'torch', ops) where ops is a tiny namespace of math fns for that backend.
    Derives backend from the first array-like argument.
    """
    x0 = next((x for x in xs if x is not None), None)
    if _is_torch(x0):
        t = _torch

        class Ops:
            asarray = lambda v: v if _is_torch(v) else t.as_tensor(v)  # noqa: E731
            abs = t.abs
            sqrt = t.sqrt
            log = t.log
            sum = lambda a, axis=None: t.sum(a, dim=axis)  # noqa: E731
            mean = lambda a, axis=None: t.mean(a, dim=axis)  # noqa: E731
            where = t.where
            maximum = t.maximum
            square = lambda a: a * a  # noqa: E731
            isfinite = t.isfinite
            astype = lambda a, dt: a.to(dt)  # noqa: E731
            float_dtype = t.float32
            asfloat = lambda a: a  # already tensor
            dtype = lambda a: a.dtype  # noqa: E731
            zeros_like = t.zeros_like
            ones_like = t.ones_like

        return "torch", Ops()
    else:
        n = _np

        class Ops:
            asarray = n.asarray
            abs = n.abs
            sqrt = n.sqrt
            log = n.log
            sum = lambda a, axis=None: n.sum(a, axis=axis)  # noqa: E731
            mean = lambda a, axis=None: n.mean(a, axis=axis)  # noqa: E731
            where = n.where
            maximum = n.maximum
            square = n.square
            isfinite = n.isfinite
            astype = lambda a, dt: a.astype(dt, copy=False)  # noqa: E731
            float_dtype = n.float32
            asfloat = n.asarray
            dtype = lambda a: a.dtype  # noqa: E731
            zeros_like = n.zeros_like
            ones_like = n.ones_like

        return "np", Ops()


def _reduce(arr: ArrayLike, reduction: str, backend: str, ops) -> Union[ArrayLike, float]:
    if reduction == "none":
        return arr
    if reduction == "sum":
        # Sum over all axes
        if backend == "torch":
            return ops.sum(arr, axis=None)
        return ops.sum(arr, axis=None).item() if arr.ndim == 0 else ops.sum(arr, axis=None)
    if reduction == "mean":
        if backend == "torch":
            # count of elements
            denom = 1
            for s in arr.shape:
                denom *= int(s)
            s = ops.sum(arr, axis=None) / denom
            return s
        # NumPy scalar ok
        total = ops.sum(arr, axis=None)
        denom = 1
        for s in _np.shape(arr):
            denom *= int(s)
        return (total / denom).item() if _np.ndim(total) == 0 else total / denom
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def _safe_sigma(sigma: ArrayLike, eps: float, ops):
    # clamp σ to at least eps; keep dtype
    if _is_torch(sigma):
        eps_t = _torch.tensor(eps, dtype=sigma.dtype, device=sigma.device)
        return ops.maximum(sigma, eps_t)
    return ops.maximum(sigma, _np.asarray(eps, dtype=sigma.dtype))


# ---------------------------------------------------------------------
# Core metrics
# ---------------------------------------------------------------------
def gaussian_nll(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    reduction: str = "mean",
    eps: float = 1e-12,
) -> Union[ArrayLike, float]:
    """
    Elementwise Gaussian negative log-likelihood:
      0.5 * [ log(2πσ^2) + (y-μ)^2 / σ^2 ]

    Supports NumPy or Torch inputs; broadcasting on y, mu, sigma is allowed.
    reduction: "none" → elementwise, "mean" → scalar average, "sum" → scalar sum.
    eps: small floor for σ to avoid division by zero.
    """
    backend, ops = _to_backend(y, mu, sigma)
    y = ops.asarray(y)
    mu = ops.asarray(mu)
    sigma = ops.asarray(sigma)

    sigma = _safe_sigma(sigma, eps, ops)
    var = sigma * sigma
    two_pi = 2.0 * math.pi

    nll = 0.5 * (ops.log(two_pi * var) + ops.square(y - mu) / var)

    if reduction == "none":
        return nll
    return _reduce(nll, reduction, backend, ops)  # type: ignore[return-value]


def challenge_gll(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    fgs1_weight: float = 58.0,
    reduction: str = "mean",
    mask: Optional[ArrayLike] = None,
    eps: float = 1e-12,
) -> Union[ArrayLike, float]:
    """
    Kaggle challenge Gaussian log-loss with FGS1 upweighting on bin 0.

    Behavior:
      • For reduction="none", returns the per-element NLL (same as gaussian_nll(..., "none")).
      • For "mean"/"sum", aggregates with bin-0 weight = fgs1_weight, all other bins weight=1.
        If `mask` is provided (bool [B,K]), only True entries contribute; weights apply
        only to included bins.

    Args:
      y, mu, sigma: arrays/tensors broadcastable to (B,K)
      fgs1_weight: weight on column 0
      mask: optional boolean mask (B,K) or (K,) where True = include
      reduction: "none" | "mean" | "sum"
    """
    backend, ops = _to_backend(y, mu, sigma)
    nll = gaussian_nll(y, mu, sigma, reduction="none", eps=eps)  # (B,K) or broadcasted

    if reduction == "none":
        return nll

    # Build weights per-bin: w[0]=fgs1_weight, others 1.0
    # Shape to (1,K) then broadcast to nll
    if _is_torch(nll):
        K = int(nll.shape[-1])
        w = _torch.ones((1, K), dtype=nll.dtype, device=nll.device)
        w[..., 0] = float(fgs1_weight)
    else:
        K = int(_np.shape(nll)[-1])
        w = _np.ones((1, K), dtype=getattr(nll, "dtype", _np.float64))
        w[..., 0] = float(fgs1_weight)

    # Broadcast to nll shape
    while w.ndim < nll.ndim:
        if _is_torch(nll):
            w = w.expand(nll.shape[:-1] + (w.shape[-1],))
        else:
            w = _np.broadcast_to(w, nll.shape)

    # Optional mask: include only True entries
    if mask is not None:
        m = mask
        if _is_torch(nll):
            m = _torch.as_tensor(mask, dtype=_torch.bool, device=nll.device)
            # broadcast to nll shape
            while m.ndim < nll.ndim:
                m = m.unsqueeze(0)
            nll_eff = _torch.where(m, nll, _torch.zeros_like(nll))
            w_eff = _torch.where(m, w, _torch.zeros_like(w))
        else:
            m = _np.asarray(mask, dtype=bool)
            # broadcast
            if m.ndim < nll.ndim:
                m = _np.broadcast_to(m, nll.shape)
            nll_eff = _np.where(m, nll, 0.0)
            w_eff = _np.where(m, w, 0.0)
    else:
        nll_eff, w_eff = nll, w

    # Aggregations
    if reduction == "sum":
        num = _np.sum(nll_eff * w_eff) if not _is_torch(nll) else _torch.sum(nll_eff * w_eff)
        return num
    if reduction == "mean":
        # Weighted mean over last axis & batch together
        if _is_torch(nll):
            num = _torch.sum(nll_eff * w_eff)
            den = _torch.sum(w_eff)
            # guard denominator (in case mask=all False)
            den = _torch.clamp(den, min=_torch.tensor(1.0, dtype=den.dtype, device=den.device))
            return num / den
        else:
            num = _np.sum(nll_eff * w_eff)
            den = _np.sum(w_eff)
            den = den if den > 0 else 1.0
            return float(num / den)
    raise ValueError(f"Unsupported reduction: {reduction!r}")


def mae(y: ArrayLike, mu: ArrayLike, *, mask: Optional[ArrayLike] = None) -> float:
    """
    Mean Absolute Error with optional boolean mask (B,K).
    """
    backend, ops = _to_backend(y, mu)
    y = ops.asarray(y)
    mu = ops.asarray(mu)
    diff = ops.abs(y - mu)

    if mask is not None:
        if _is_torch(diff):
            m = _torch.as_tensor(mask, dtype=_torch.bool, device=diff.device)
            while m.ndim < diff.ndim:
                m = m.unsqueeze(0)
            diff = _torch.where(m, diff, _torch.zeros_like(diff))
            denom = _torch.sum(m.to(diff.dtype))
            denom = _torch.clamp(denom, min=_torch.tensor(1.0, dtype=diff.dtype, device=diff.device))
            return float(_torch.sum(diff) / denom)
        else:
            m = _np.asarray(mask, dtype=bool)
            if m.ndim < diff.ndim:
                m = _np.broadcast_to(m, diff.shape)
            denom = m.sum()
            denom = float(denom if denom > 0 else 1.0)
            return float(_np.sum(_np.where(m, diff, 0.0)) / denom)

    # no mask → plain mean over all elements
    if _is_torch(diff):
        return float(_torch.mean(diff))
    return float(_np.mean(diff))


def rmse(y: ArrayLike, mu: ArrayLike, *, mask: Optional[ArrayLike] = None) -> float:
    """
    Root Mean Squared Error with optional boolean mask (B,K).
    """
    backend, ops = _to_backend(y, mu)
    y = ops.asarray(y)
    mu = ops.asarray(mu)
    sq = (y - mu) * (y - mu)

    if mask is not None:
        if _is_torch(sq):
            m = _torch.as_tensor(mask, dtype=_torch.bool, device=sq.device)
            while m.ndim < sq.ndim:
                m = m.unsqueeze(0)
            sq = _torch.where(m, sq, _torch.zeros_like(sq))
            denom = _torch.sum(m.to(sq.dtype))
            denom = _torch.clamp(denom, min=_torch.tensor(1.0, dtype=sq.dtype, device=sq.device))
            return float(_torch.sqrt(_torch.sum(sq) / denom))
        else:
            m = _np.asarray(mask, dtype=bool)
            if m.ndim < sq.ndim:
                m = _np.broadcast_to(m, sq.shape)
            denom = m.sum()
            denom = float(denom if denom > 0 else 1.0)
            return float(_np.sqrt(_np.sum(_np.where(m, sq, 0.0)) / denom))

    if _is_torch(sq):
        return float(_torch.sqrt(_torch.mean(sq)))
    return float(_np.sqrt(_np.mean(sq)))


# ---------------------------------------------------------------------
# Uncertainty diagnostics
# ---------------------------------------------------------------------
def _norm_ppf(p: float) -> float:
    """
    High-accuracy approximation to Φ^{-1}(p) (Acklam's method).
    Valid for p in (0,1). See: https://web.archive.org/web/20150910044730/http://home.online.no/~pjacklam/notes/invnorm/
    """
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return -_np.inf
        if p == 1.0:
            return _np.inf
        raise ValueError("p must be in (0,1)")
    # Coefficients for Acklam's approximation
    a = [-3.969683028665376e+01,  2.209460984245205e+02,
         -2.759285104469687e+02,  1.383577518672690e+02,
         -3.066479806614716e+01,  2.506628277459239e+00]
    b = [-5.447609879822406e+01,  1.615858368580409e+02,
         -1.556989798598866e+02,  6.680131188771972e+01,
         -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01,
         -2.400758277161838e+00, -2.549732539343734e+00,
          4.374664141464968e+00,  2.938163982698783e+00]
    d = [ 7.784695709041462e-03,  3.224671290700398e-01,
          2.445134137142996e+00,  3.754408661907416e+00]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
               ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    if phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q + c[1])*q + c[2])*q + c[3])*q + c[4])*q + c[5]) / \
                 ((((d[0]*q + d[1])*q + d[2])*q + d[3])*q + 1)
    q = p - 0.5
    r = q*q
    return (((((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4])*r + a[5]) * q / \
           (((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4])*r + 1)


def coverage(
    y: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
    *,
    alpha: float = 0.95,
    eps: float = 1e-12,
) -> float:
    """
    Fraction of elements with y inside the symmetric (alpha)-level Gaussian interval:
      μ ± z * σ, where z = Φ^{-1}((1+alpha)/2).
    """
    backend, ops = _to_backend(y, mu, sigma)
    y = ops.asarray(y)
    mu = ops.asarray(mu)
    sigma = ops.asarray(sigma)
    sigma = _safe_sigma(sigma, eps, ops)

    p = (1.0 + float(alpha)) / 2.0
    z = _norm_ppf(p)

    lower = mu - z * sigma
    upper = mu + z * sigma
    inside = (y >= lower) & (y <= upper)

    if _is_torch(inside):
        total = 1
        for s in inside.shape:
            total *= int(s)
        return float(_torch.sum(inside.to(dtype=_torch.float64)) / total)
    return float(_np.sum(inside) / inside.size)


def sharpness(sigma: ArrayLike, *, reduction: str = "mean", eps: float = 0.0) -> Union[ArrayLike, float]:
    """
    Mean/ Sum / Elementwise sharpness ≡ σ (optionally clamped by eps).
    """
    backend, ops = _to_backend(sigma)
    s = ops.asarray(sigma)
    if eps > 0:
        s = _safe_sigma(s, eps, ops)
    if reduction == "none":
        return s
    if reduction == "sum":
        if _is_torch(s):
            return _torch.sum(s)
        return float(_np.sum(s))
    if reduction == "mean":
        if _is_torch(s):
            return _torch.mean(s)
        return float(_np.mean(s))
    raise ValueError(f"Unsupported reduction: {reduction!r}")
