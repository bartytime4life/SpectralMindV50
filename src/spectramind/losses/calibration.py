# src/spectramind/losses/calibration.py
from __future__ import annotations

"""
SpectraMind V50 — Uncertainty Calibration Losses
================================================

This module provides GPU-friendly PyTorch utilities to *calibrate*
heteroscedastic Gaussian predictions:

  • Variance calibration (match σ^2 to empirical squared error)
  • Interval coverage calibration (target coverage for Gaussian CI)
  • ECE-style calibration for regression (smooth surrogate)

These penalties are complementary to Gaussian NLL. NLL encourages sharp
and accurate predictions; calibration terms steer *uncertainty* (σ) to
reflect empirical residuals and coverage.

Common usage
------------
>>> import torch
>>> from spectramind.losses.calibration import CalibrationLoss
>>> crit = CalibrationLoss(var_lambda=1e-3, cover_lambda=0.0, ece_lambda=0.0)
>>> mu = torch.randn(8, 283)
>>> sigma = torch.rand(8, 283) * 0.05 + 1e-3
>>> target = torch.randn(8, 283)
>>> mask = torch.ones_like(mu)
>>> out = crit(mu=mu, sigma=sigma, target=target, mask=mask)
>>> out["loss"], out["var_calib"]

Notes
-----
- All tensors are `[B, n_bins]`.
- `mask` is broadcastable to `[B, n_bins]` (1=valid, 0=ignore).
- Per-bin FGS1 up-weighting can be passed via `fgs1_weight` (bin 0).
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Utility helpers (masking, weights)
# ----------------------------------------------------------------------------- #


def _make_weights(
    shape: torch.Size,
    *,
    device: torch.device,
    dtype: torch.dtype,
    fgs1_weight: float = 1.0,
) -> torch.Tensor:
    """
    Construct per-bin weights with optional FGS1 up-weighting (bin 0).
    """
    B, N = shape
    w = torch.ones(N, device=device, dtype=dtype)
    if fgs1_weight > 1.0 and N > 0:
        w[0] = fgs1_weight
    w = w.unsqueeze(0).expand(B, -1)
    return w


def _masked_mean(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Weighted mean over last dimension with stability guards, averaged over batch.
    """
    if w is None:
        return x.mean(dim=-1).mean()
    num = (x * w).sum(dim=-1)
    den = w.sum(dim=-1).clamp_min(1.0)
    return (num / den).mean()


# ----------------------------------------------------------------------------- #
# Calibration penalties
# ----------------------------------------------------------------------------- #


def variance_calibration_penalty(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    fgs1_weight: float = 1.0,
    eps: float = 1e-8,
    normalization: str = "none",
) -> torch.Tensor:
    """
    Penalize mismatch between empirical squared error and predicted variance.

        L_var = mean_b mean_i w[i] * ( (μ - y)^2 - σ^2 )^2

    Args
    ----
    mu, sigma, target : [B, N]
    mask : [B, N], optional
        Binary (1 valid, 0 invalid). If provided, weights are multiplied by mask.
    fgs1_weight : float
        Up-weight bin 0 to reflect competition metric sensitivity.
    eps : float
        Numerical floor for stability.
    normalization : {'none','per_bin','per_batch'}
        Optional normalization of the error-variance residual:
          - 'none'      : raw ((err2 - var)^2)
          - 'per_bin'   : divide residual by (var + eps) to reduce scale bias
          - 'per_batch' : normalize residual across bins within a batch

    Returns
    -------
    Tensor scalar
    """
    B, N = mu.shape
    assert sigma.shape == mu.shape == target.shape, "mu/sigma/target must share shape [B, N]"

    w = _make_weights(mu.shape, device=mu.device, dtype=mu.dtype, fgs1_weight=fgs1_weight)
    if mask is not None:
        w = w * mask.to(mu.dtype)

    err2 = (mu - target).pow(2)         # [B, N]
    var = torch.clamp(sigma, min=eps).pow(2)

    resid = err2 - var                   # residual: empirical vs predicted variance
    if normalization == "per_bin":
        resid = resid / (var + eps)
    elif normalization == "per_batch":
        # Normalize residual scale per batch to reduce bin-scale dominance
        scale = resid.abs().mean(dim=-1, keepdim=True).clamp_min(eps)
        resid = resid / scale

    pen = resid.pow(2)                  # squared mismatch
    return _masked_mean(pen, w)


def coverage_calibration_penalty(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    alpha: float = 0.1,
    fgs1_weight: float = 1.0,
    eps: float = 1e-8,
    surrogate: str = "sigmoid",
    temperature: float = 50.0,
) -> torch.Tensor:
    """
    Target *coverage* calibration for symmetric Gaussian intervals.

    We aim for P(|y - μ| <= z_{1-α/2} * σ) ≈ (1 - α).
    Since the indicator is non-differentiable, we use a smooth surrogate:

        ĉ = mean_b mean_i w[i] * sigmoid( T * ( z*σ - |y - μ| ) )

    and penalize squared error between ĉ and (1-α).

    Args
    ----
    mu, sigma, target : [B, N]
    mask : [B, N], optional
    alpha : float
        Miscoverage; target coverage is (1 - alpha). Example: alpha=0.1 → 90% CI.
    surrogate : {'sigmoid','softplus'}
        Smooth proxy for the indicator. 'sigmoid' defaults to a steeper boundary;
        'softplus' is a softer hinge-like proxy.
    temperature : float
        Controls boundary steepness for the sigmoid surrogate (higher→sharper).

    Returns
    -------
    Tensor scalar
    """
    from math import sqrt
    import math

    B, N = mu.shape
    assert sigma.shape == mu.shape == target.shape
    # z_{1 - α/2} for two-sided CI
    z = torch.tensor(float(_gaussian_quantile(1.0 - alpha / 2.0)), device=mu.device, dtype=mu.dtype)

    w = _make_weights(mu.shape, device=mu.device, dtype=mu.dtype, fgs1_weight=fgs1_weight)
    if mask is not None:
        w = w * mask.to(mu.dtype)

    sigma_safe = torch.clamp(sigma, min=eps)
    margin = z * sigma_safe                             # [B, N]
    dist = (target - mu).abs()                          # [B, N]
    # Smooth indicator: 1 if dist <= margin, 0 otherwise (approx)
    if surrogate == "softplus":
        # soft hinge: max(0, margin - dist) ~ softplus(k*(margin - dist))/k
        k = torch.tensor(temperature, device=mu.device, dtype=mu.dtype)
        soft = F.softplus(k * (margin - dist)) / k
        soft = soft / (soft.max(dim=-1, keepdim=True).values.clamp_min(1.0))  # scale to [0,1]-ish
        s = soft
    else:
        # default sigmoid boundary proxy
        T = torch.tensor(temperature, device=mu.device, dtype=mu.dtype)
        s = torch.sigmoid(T * (margin - dist))

    # Weighted average surrogate coverage per batch
    num = (s * w).sum(dim=-1)
    den = w.sum(dim=-1).clamp_min(1.0)
    cov_hat = num / den                                # [B]
    cov_target = 1.0 - alpha

    pen = (cov_hat - cov_target) ** 2                  # [B]
    return pen.mean()


def ece_regression_penalty(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    fgs1_weight: float = 1.0,
    eps: float = 1e-8,
    bins: int = 10,
) -> torch.Tensor:
    """
    ECE-style calibration for regression via **standardized residuals**.

    We bin residual magnitudes normalized by σ:
       r = |y - μ| / σ   (σ>0)
    Ideal Gaussian predictive intervals imply that proportions falling under
    z-quantiles match nominal levels. We approximate ECE by:

      ECE ≈ mean_k | p_emp(r <= z_k) - p_nom(z_k) |

    where z_k are k/bins quantiles of N(0,1). This is a differentiable surrogate
    if we replace hard indicators by sigmoids; here we use a *soft histogram*
    via sigmoids for a smooth approximation.

    Args
    ----
    bins : int
        Number of quantile bins (e.g., 10 → deciles).

    Returns
    -------
    Tensor scalar
    """
    from math import sqrt

    B, N = mu.shape
    assert sigma.shape == mu.shape == target.shape
    sigma_safe = torch.clamp(sigma, min=eps)

    w = _make_weights(mu.shape, device=mu.device, dtype=mu.dtype, fgs1_weight=fgs1_weight)
    if mask is not None:
        w = w * mask.to(mu.dtype)

    r = (target - mu).abs() / sigma_safe          # [B,N], standardized absolute residual

    # Soft CDF bins using sigmoids around z_k
    # z_k: k/bins two-sided absolute Gaussian quantiles ~ invPhi((1 + p_k)/2)
    # so that P(|Z| <= z) = 2*Phi(z) - 1
    z = torch.stack([
        torch.tensor(float(_gaussian_quantile(0.5 * (1.0 + k / bins))), device=mu.device, dtype=mu.dtype)
        for k in range(1, bins + 1)
    ])  # [bins]

    # Empirical (weighted) CDF at z_k using sigmoid smoothing
    T = torch.tensor(30.0, device=mu.device, dtype=mu.dtype)  # smoothing temperature
    # shape to [1, bins] for broadcasting
    z_b = z.view(1, -1)                                      # [1, bins]
    # soft indicator: 1 if r <= z_k, 0 otherwise
    soft_ind = torch.sigmoid(T * (z_b.unsqueeze(1) - r.unsqueeze(-1)))  # [B, N, bins]
    # weighted mean across bins
    num = (soft_ind * w.unsqueeze(-1)).sum(dim=1)     # [B, bins]
    den = w.sum(dim=1, keepdim=True).clamp_min(1.0)  # [B, 1]
    p_emp = (num / den)                               # [B, bins]
    # Nominal two-sided probabilities: p_nom(z_k) = P(|Z| <= z_k) = 2*Phi(z_k) - 1
    p_nom = (2.0 * _gaussian_cdf(z)) - 1.0            # [bins]
    # ECE-like penalty
    ece = (p_emp - p_nom.view(1, -1)).abs().mean(dim=-1)  # [B]
    return ece.mean()


# ----------------------------------------------------------------------------- #
# Wrapper composite module
# ----------------------------------------------------------------------------- #


@dataclass
class CalibrationConfig:
    """
    Configuration for uncertainty calibration penalties.

    Attributes
    ----------
    var_lambda : float
        Weight for variance calibration penalty.
    cover_lambda : float
        Weight for coverage calibration penalty (Gaussian CI).
    ece_lambda : float
        Weight for ECE-style calibration penalty.
    alpha : float
        Miscoverage for coverage penalty (target = 1 - alpha).
    fgs1_weight : float
        Per-bin weighting for bin 0 (FGS1).
    eps : float
        Numerical floor.
    var_normalization : str
        {'none','per_bin','per_batch'} normalization mode for variance residual.
    ece_bins : int
        Number of quantile bins for ECE surrogate.
    """

    var_lambda: float = 0.0
    cover_lambda: float = 0.0
    ece_lambda: float = 0.0
    alpha: float = 0.1
    fgs1_weight: float = 1.0
    eps: float = 1e-8
    var_normalization: str = "none"
    ece_bins: int = 10


class CalibrationLoss(nn.Module):
    """
    Composite calibration loss.

    Returns a dictionary with individual terms and the total loss:

        {
          "loss": total,
          "var_calib": ...,
          "cover_calib": ...,
          "ece_calib": ...,
        }

    Example
    -------
    >>> crit = CalibrationLoss(var_lambda=1e-3, cover_lambda=0.0, ece_lambda=0.0, fgs1_weight=58.0)
    >>> out = crit(mu=mu, sigma=sigma, target=target, mask=mask)
    >>> out["loss"]
    """

    def __init__(
        self,
        *,
        var_lambda: float = 0.0,
        cover_lambda: float = 0.0,
        ece_lambda: float = 0.0,
        alpha: float = 0.1,
        fgs1_weight: float = 1.0,
        eps: float = 1e-8,
        var_normalization: str = "none",
        ece_bins: int = 10,
    ) -> None:
        super().__init__()
        self.cfg = CalibrationConfig(
            var_lambda=var_lambda,
            cover_lambda=cover_lambda,
            ece_lambda=ece_lambda,
            alpha=alpha,
            fgs1_weight=fgs1_weight,
            eps=eps,
            var_normalization=var_normalization,
            ece_bins=ece_bins,
        )

    def forward(
        self,
        *,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite calibration loss.

        Parameters
        ----------
        mu, sigma, target : Tensor [B, N]
        mask : Tensor [B, N], optional

        Returns
        -------
        dict
            Keys: 'loss', 'var_calib', 'cover_calib', 'ece_calib'
        """
        var_term = (
            variance_calibration_penalty(
                mu, sigma, target,
                mask=mask,
                fgs1_weight=self.cfg.fgs1_weight,
                eps=self.cfg.eps,
                normalization=self.cfg.var_normalization,
            )
            if self.cfg.var_lambda > 0
            else mu.new_tensor(0.0)
        )
        cover_term = (
            coverage_calibration_penalty(
                mu, sigma, target,
                mask=mask,
                alpha=self.cfg.alpha,
                fgs1_weight=self.cfg.fgs1_weight,
                eps=self.cfg.eps,
            )
            if self.cfg.cover_lambda > 0
            else mu.new_tensor(0.0)
        )
        ece_term = (
            ece_regression_penalty(
                mu, sigma, target,
                mask=mask,
                fgs1_weight=self.cfg.fgs1_weight,
                eps=self.cfg.eps,
                bins=self.cfg.ece_bins,
            )
            if self.cfg.ece_lambda > 0
            else mu.new_tensor(0.0)
        )

        total = (
            self.cfg.var_lambda * var_term
            + self.cfg.cover_lambda * cover_term
            + self.cfg.ece_lambda * ece_term
        )

        return {
            "loss": total,
            "var_calib": var_term.detach(),
            "cover_calib": cover_term.detach(),
            "ece_calib": ece_term.detach(),
        }


# ----------------------------------------------------------------------------- #
# Gaussian utilities (quantile / CDF)
# ----------------------------------------------------------------------------- #


def _gaussian_cdf(x: torch.Tensor) -> torch.Tensor:
    """
    Standard normal CDF via error function.
    """
    # CDF = 0.5 * [1 + erf(x / sqrt(2))]
    return 0.5 * (1.0 + torch.erf(x / torch.sqrt(torch.tensor(2.0, device=x.device, dtype=x.dtype))))


def _gaussian_quantile(p: float) -> float:
    """
    Approximate inverse CDF (probit) for N(0,1).

    Uses Acklam's rational approximation.
    """
    # Source of coefficients: Peter J. Acklam, 2003 / 2010 update.
    # Implementation adapted for clarity.
    if not (0.0 < p < 1.0):
        if p == 0.0:
            return float("-inf")
        if p == 1.0:
            return float("inf")
        raise ValueError("p must be in (0,1)")

    # Coefficients
    a = [
        -3.969683028665376e+01,
         2.209460984245205e+02,
        -2.759285104469687e+02,
         1.383577518672690e+02,
        -3.066479806614716e+01,
         2.506628277459239e+00,
    ]
    b = [
        -5.447609879822406e+01,
         1.615858368580409e+02,
        -1.556989798598866e+02,
         6.680131188771972e+01,
        -1.328068155288572e+01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e+00,
        -2.549732539343734e+00,
         4.374664141464968e+00,
         2.938163982698783e+00,
    ]
    d = [
         7.784695709041462e-03,
         3.224671290700398e-01,
         2.445134137142996e+00,
         3.754408661907416e+00,
    ]
    # Define break-points
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = (2 * p) ** 0.5
        # rational approximation for lower region
        q = torch.tensor(p, dtype=torch.float64)
        r = torch.sqrt(-2.0 * torch.log(q))
        num = ((c[0]*r + c[1])*r + c[2])*r + c[3]
        num = (num*r + c[4])*r + c[5]
        den = ((d[0]*r + d[1])*r + d[2])*r + d[3]
        return float(num / den)
    if phigh < p:
        q = torch.tensor(1.0 - p, dtype=torch.float64)
        r = torch.sqrt(-2.0 * torch.log(q))
        num = ((c[0]*r + c[1])*r + c[2])*r + c[3]
        num = (num*r + c[4])*r + c[5]
        den = ((d[0]*r + d[1])*r + d[2])*r + d[3]
        return float(-(num / den))
    # central region
    q = torch.tensor(p - 0.5, dtype=torch.float64)
    r = q * q
    num = (((a[0]*r + a[1])*r + a[2])*r + a[3])*r + a[4]
    num = num*r + a[5]
    den = ((((b[0]*r + b[1])*r + b[2])*r + b[3])*r + b[4]) * r + 1.0
    return float(q * num / den)
