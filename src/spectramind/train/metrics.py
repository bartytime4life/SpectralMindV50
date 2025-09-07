# src/spectramind/train/metrics.py
"""
SpectraMind V50 — Training Metrics
----------------------------------

Implements metrics for the NeurIPS Ariel Data Challenge:
- Gaussian Log-Likelihood (GLL) with dominant FGS1 weight on bin 0 (~58×)
- MAE / RMSE
- Uncertainty diagnostics: empirical coverage, sharpness (avg sigma)

Supports both NumPy and (optional) PyTorch tensors and handles batched inputs.

Notes
-----
- Challenge metric: Gaussian negative log-likelihood; bin 0 (FGS1) is up-weighted
  around ~58× per problem statement [oai_citation:0‡NeurIPS 2025 Ariel Data Challenge_ Dual-Channel End-to-End Solution.pdf](file-service://file-SJeTx7mG2ppmDvLxR2cKzf).
- This module is dependency-tolerant: NumPy is required; PyTorch functions are
  available if `torch` is installed.

Shapes
------
- Unbatched: (B,) for y, mu, sigma → scalar metric
- Batched: (N, D) or broadcastable shapes; reductions default to mean over all items.

"""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple, Union

import numpy as np

try:  # optional: torch support
    import torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover
    _HAS_TORCH = False


# ---------------------------------------------------------------------
# Core Gaussian NLL (negative log-likelihood)
# ---------------------------------------------------------------------

def _np_gaussian_nll(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    reduction: str = "mean",
    eps: float = 1e-8,
) -> float:
    y, mu, sigma = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    var = np.maximum(sigma, eps) ** 2
    nll = 0.5 * (np.log(2.0 * math.pi * var) + (y - mu) ** 2 / var)

    if mask is not None:
        nll = np.where(mask, nll, np.nan)

    # drop NaNs
    nll = nll[np.isfinite(nll)]
    if nll.size == 0:
        return float("nan")

    if reduction == "sum":
        return float(np.sum(nll))
    elif reduction == "none":
        return nll  # type: ignore[return-value]
    else:
        return float(np.mean(nll))


def _np_weighted(
    values: np.ndarray,
    weights: Optional[np.ndarray],
    *,
    reduction: str = "mean",
) -> float:
    v = np.asarray(values)
    if weights is not None:
        w = np.asarray(weights, dtype=np.float64)
        v = v * w
    if reduction == "sum":
        return float(np.sum(v))
    elif reduction == "none":
        return v  # type: ignore[return-value]
    else:
        if weights is not None:
            denom = np.sum(np.asarray(weights))
            if denom <= 0:
                return float("nan")
            return float(np.sum(v) / denom)
        return float(np.mean(v))


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

    Handles NumPy arrays or PyTorch tensors; reduction in {"mean","sum","none"}.
    """
    # PyTorch path
    if _HAS_TORCH and isinstance(y, torch.Tensor):
        mu_t = mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu, device=y.device, dtype=y.dtype)
        sig_t = sigma if isinstance(sigma, torch.Tensor) else torch.as_tensor(sigma, device=y.device, dtype=y.dtype)
        var = torch.clamp(sig_t, min=eps) ** 2
        nll = 0.5 * (torch.log(2.0 * math.pi * var) + (y - mu_t) ** 2 / var)
        if mask is not None:
            m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=y.device, dtype=torch.bool)
            nll = torch.where(m, nll, torch.tensor(float("nan"), device=y.device, dtype=y.dtype))
        # drop NaNs
        nll = nll[torch.isfinite(nll)]
        if nll.numel() == 0:
            return torch.tensor(float("nan"), device=y.device, dtype=y.dtype)
        if reduction == "sum":
            return nll.sum()
        elif reduction == "none":
            return nll
        else:
            return nll.mean()

    # NumPy path
    return _np_gaussian_nll(
        np.asarray(y),
        np.asarray(mu),
        np.asarray(sigma),
        mask=np.asarray(mask) if mask is not None else None,
        reduction=reduction,
        eps=eps,
    )


# ---------------------------------------------------------------------
# Challenge GLL with FGS1 weight
# ---------------------------------------------------------------------

def challenge_gll(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    fgs1_weight: float = 58.0,   # ~58× weighting for FGS1 bin 0 [oai_citation:1‡NeurIPS 2025 Ariel Data Challenge_ Dual-Channel End-to-End Solution.pdf](file-service://file-SJeTx7mG2ppmDvLxR2cKzf)
    reduction: str = "mean",
    eps: float = 1e-8,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
) -> Union[float, "torch.Tensor"]:
    """
    Gaussian NLL with per-bin weighting for the Ariel challenge.
    Bin 0 (FGS1) is up-weighted (default ~58×); other bins weight=1.

    Accepts (N, D) or (D,) shapes; broadcasting is supported.
    """
    D = y.shape[-1] if hasattr(y, "shape") else None

    # Build weights vector [w0, 1, 1, ..., 1]
    if _HAS_TORCH and isinstance(y, torch.Tensor):
        # Attempt to infer D; fallback to vectorized approach
        if D is None:
            raise ValueError("Cannot infer last-dimension size for y.")
        w = torch.ones(D, dtype=y.dtype, device=y.device)
        w[0] = float(fgs1_weight)

        # Compute per-element NLL without reduction
        nll = gaussian_nll(y, mu, sigma, mask=mask, reduction="none", eps=eps)  # type: ignore
        # If reduced "none", may still be flattened; ensure last dim alignment:
        # We assume broadcasting kept shape; reshape weights for broadcast
        while w.dim() < nll.dim():
            w = w.unsqueeze(0)
        nll_w = nll * w

        if reduction == "sum":
            return torch.nansum(nll_w)
        elif reduction == "none":
            return nll_w
        else:
            # weighted mean: sum(w*nll)/sum(w) accounting for mask/NaNs
            denom = torch.nansum(w.expand_as(nll_w))
            return torch.nansum(nll_w) / denom.clamp_min(1e-12)

    # NumPy path
    y_np, mu_np, sig_np = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    if D is None:
        D = y_np.shape[-1]
    w = np.ones((D,), dtype=np.float64)
    w[0] = float(fgs1_weight)

    # Raw per-element NLL
    nll = gaussian_nll(y_np, mu_np, sig_np, mask=mask, reduction="none", eps=eps)  # type: ignore
    # Align weights to nll's last dimension
    w_shape = [1] * nll.ndim
    w_shape[-1] = D
    w_ = w.reshape(w_shape)
    nll_w = nll * w_

    if reduction == "sum":
        return float(np.nansum(nll_w))
    elif reduction == "none":
        return nll_w  # type: ignore[return-value]
    else:
        denom = np.nansum(np.broadcast_to(w_, nll_w.shape))
        if denom <= 0:
            return float("nan")
        return float(np.nansum(nll_w) / denom)


# ---------------------------------------------------------------------
# Pointwise metrics
# ---------------------------------------------------------------------

def mae(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Mean absolute error."""
    if _HAS_TORCH and isinstance(y, torch.Tensor):
        diff = torch.abs(y - (mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu, device=y.device, dtype=y.dtype)))
        if mask is not None:
            m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=y.device, dtype=torch.bool)
            diff = torch.where(m, diff, torch.tensor(float("nan"), device=y.device, dtype=y.dtype))
        if reduction == "sum":
            return torch.nansum(diff)
        elif reduction == "none":
            return diff
        else:
            return torch.nanmean(diff)
    # NumPy
    diff = np.abs(np.asarray(y) - np.asarray(mu))
    if mask is not None:
        diff = np.where(np.asarray(mask), diff, np.nan)
    if reduction == "sum":
        return float(np.nansum(diff))
    elif reduction == "none":
        return diff  # type: ignore[return-value]
    else:
        return float(np.nanmean(diff))


def rmse(
    y: Union[np.ndarray, "torch.Tensor"],
    mu: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """Root mean squared error."""
    if _HAS_TORCH and isinstance(y, torch.Tensor):
        sq = (y - (mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu, device=y.device, dtype=y.dtype))) ** 2
        if mask is not None:
            m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=y.device, dtype=torch.bool)
            sq = torch.where(m, sq, torch.tensor(float("nan"), device=y.device, dtype=y.dtype))
        if reduction == "sum":
            return torch.sqrt(torch.nansum(sq))
        elif reduction == "none":
            return torch.sqrt(sq)
        else:
            return torch.sqrt(torch.nanmean(sq))
    # NumPy
    sq = (np.asarray(y) - np.asarray(mu)) ** 2
    if mask is not None:
        sq = np.where(np.asarray(mask), sq, np.nan)
    if reduction == "sum":
        return float(np.sqrt(np.nansum(sq)))
    elif reduction == "none":
        return np.sqrt(sq)  # type: ignore[return-value]
    else:
        return float(np.sqrt(np.nanmean(sq)))


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
    Empirical coverage of the nominal (1-alpha) interval under Normal assumption:
    interval = μ ± z * σ, where z = Φ^{-1}((1+alpha)/2).

    Returns fraction of points within the interval.
    """
    z = float(_z_from_alpha(alpha))
    if _HAS_TORCH and isinstance(y, torch.Tensor):
        mu_t = mu if isinstance(mu, torch.Tensor) else torch.as_tensor(mu, device=y.device, dtype=y.dtype)
        sig_t = sigma if isinstance(sigma, torch.Tensor) else torch.as_tensor(sigma, device=y.device, dtype=y.dtype)
        lo, hi = mu_t - z * sig_t, mu_t + z * sig_t
        inside = (y >= lo) & (y <= hi)
        if mask is not None:
            m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=y.device, dtype=torch.bool)
            inside = torch.where(m, inside, torch.tensor(False, device=y.device))
        numer = inside.sum()
        denom = inside.numel() if mask is None else m.sum()  # type: ignore[name-defined]
        denom = torch.clamp(denom, min=1)
        return (numer.to(dtype=y.dtype) / denom.to(dtype=y.dtype))
    # NumPy
    y_np, mu_np, sig_np = np.asarray(y), np.asarray(mu), np.asarray(sigma)
    lo, hi = mu_np - z * sig_np, mu_np + z * sig_np
    inside = (y_np >= lo) & (y_np <= hi)
    if mask is not None:
        inside = inside & np.asarray(mask)
    denom = np.sum(inside | ~inside) if mask is None else np.sum(np.asarray(mask))
    denom = max(int(denom), 1)
    return float(np.sum(inside) / denom)


def sharpness(
    sigma: Union[np.ndarray, "torch.Tensor"],
    *,
    mask: Optional[Union[np.ndarray, "torch.Tensor"]] = None,
    reduction: str = "mean",
) -> Union[float, "torch.Tensor", np.ndarray]:
    """
    Sharpness = average predictive sigma (lower is sharper).
    """
    if _HAS_TORCH and isinstance(sigma, torch.Tensor):
        s = sigma
        if mask is not None:
            m = mask if isinstance(mask, torch.Tensor) else torch.as_tensor(mask, device=s.device, dtype=torch.bool)
            s = torch.where(m, s, torch.tensor(float("nan"), device=s.device, dtype=s.dtype))
        if reduction == "sum":
            return torch.nansum(s)
        elif reduction == "none":
            return s
        else:
            return torch.nanmean(s)
    # NumPy
    s = np.asarray(sigma)
    if mask is not None:
        s = np.where(np.asarray(mask), s, np.nan)
    if reduction == "sum":
        return float(np.nansum(s))
    elif reduction == "none":
        return s  # type: ignore[return-value]
    else:
        return float(np.nanmean(s))


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _z_from_alpha(alpha: float) -> float:
    """
    Two-sided z-score for central coverage alpha.

    Uses an approximation (Acklam's inverse CDF); accurate for typical alpha.
    """
    # Convert alpha central coverage -> tail probability
    p = 0.5 * (1.0 + alpha)
    # Acklam's approximation of inverse normal CDF
    # https://web.archive.org/web/20150910044719/http://home.online.no/~pjacklam/notes/invnorm/
    a1, a2, a3 = -3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02
    a4, a5, a6 = 1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00
    b1, b2, b3 = -5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02
    b4, b5, b6 = 6.680131188771972e+01, -1.328068155288572e+01, 1.0
    c1, c2, c3 = -7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00
    c4, c5, c6 = -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00
    d1, d2, d3 = 7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00
    d4 = 3.754408661907416e+00

    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        x = (((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    elif phigh < p:
        q = math.sqrt(-2 * math.log(1 - p))
        x = -(((((c1 * q + c2) * q + c3) * q + c4) * q + c5) * q + c6) / \
            ((((d1 * q + d2) * q + d3) * q + d4) * q + 1)
    else:
        q = p - 0.5
        r = q * q
        x = (((((a1 * r + a2) * r + a3) * r + a4) * r + a5) * r + a6) * q / \
            (((((b1 * r + b2) * r + b3) * r + b4) * r + b5) * r + b6)
    return float(x)