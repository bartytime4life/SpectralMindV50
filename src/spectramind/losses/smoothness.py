# src/spectramind/losses/smoothness.py
from __future__ import annotations

"""
SpectraMind V50 — Spectral Smoothness / Curvature / Coherence Losses
====================================================================

This module provides GPU-friendly PyTorch utilities to penalize *undesired*
rapid variation in wavelength-binned spectra (e.g., the 283-bin Ariel output).

Features
--------
- First- and second-order finite-difference operators (μ' and μ'')
- L1/L2 penalties with optional *mask* for missing/invalid bins
- Windowed *coherence* penalty (difference between local slope and its moving average)
- Stable, weighted reductions (avoid NaNs/div-by-zero)
- Functional API + `SmoothnessLoss` module wrapper

Common usage
------------
>>> import torch
>>> from spectramind.losses.smoothness import SmoothnessLoss
>>> criterion = SmoothnessLoss(smoothness_lambda=1e-3, curvature_lambda=1e-4)
>>> mu = torch.randn(8, 283)  # predicted mean spectra
>>> mask = torch.ones_like(mu)  # or per-bin validity
>>> out = criterion(mu=mu, mask=mask)
>>> out["loss"], out["smooth"], out["curv"]

Notes
-----
- Shapes are `[B, n_bins]` for all inputs/outputs.
- Weights/masks are broadcastable to `[B, n_bins]`.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Finite-difference helpers
# ----------------------------------------------------------------------------- #

def first_derivative(mu: torch.Tensor) -> torch.Tensor:
    """
    First-order finite difference along spectral bins: μ'[i] ≈ μ[i] - μ[i-1].

    Parameters
    ----------
    mu : Tensor [B, N]
        Mean spectrum per batch.

    Returns
    -------
    Tensor [B, N-1]
        First derivative.
    """
    return mu[:, 1:] - mu[:, :-1]


def second_derivative(mu: torch.Tensor) -> torch.Tensor:
    """
    Second-order finite difference (curvature) along spectral bins:
    μ''[i] ≈ μ[i+1] - 2μ[i] + μ[i-1].

    Parameters
    ----------
    mu : Tensor [B, N]

    Returns
    -------
    Tensor [B, N-2]
        Second derivative.
    """
    if mu.shape[1] < 3:
        # Minimal shape guard
        return mu.new_zeros(mu.shape[0], 0)
    return mu[:, 2:] - 2.0 * mu[:, 1:-1] + mu[:, :-2]


# ----------------------------------------------------------------------------- #
# Reduction helpers (masked, stable)
# ----------------------------------------------------------------------------- #

def _masked_mean(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Weighted mean over last dimension with stability guards.

    x: [..., K]
    w: [..., K] or None (default uniform weighting)

    Returns scalar mean (averaged over batch) if input dims are [B, K],
    otherwise reduces last dimension only.
    """
    if w is None:
        return x.mean(dim=-1).mean()

    num = (x * w).sum(dim=-1)
    den = w.sum(dim=-1).clamp_min(1.0)
    return (num / den).mean()


def _pairwise_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Combine adjacent masks for first-order ops: m[i] = mask[i] * mask[i-1].

    Input:  [B, N]
    Output: [B, N-1]
    """
    return (mask[:, 1:] * mask[:, :-1]).to(mask.dtype)


def _triple_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Combine triple-adjacent masks for second-order ops: m[i] = mask[i+1]*mask[i]*mask[i-1].

    Input:  [B, N]
    Output: [B, N-2]
    """
    return (mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]).to(mask.dtype)


# ----------------------------------------------------------------------------- #
# Penalties (smoothness, curvature, coherence)
# ----------------------------------------------------------------------------- #

def smoothness_penalty(
    mu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    norm: str = "l2",
) -> torch.Tensor:
    """
    First-derivative smoothness penalty along spectral bins.

    Encourages μ[:, i] ~ μ[:, i-1].

    Parameters
    ----------
    mu : Tensor [B, N]
    mask : Tensor [B, N], optional
        Per-bin validity mask (1 valid, 0 invalid).
    norm : {'l1','l2'}
        Norm for penalty aggregation.

    Returns
    -------
    Tensor scalar
    """
    d1 = first_derivative(mu)  # [B, N-1]
    pen = d1.abs() if norm.lower() == "l1" else d1.pow(2)

    w = None
    if mask is not None:
        w = _pairwise_mask(mask).to(mu.dtype)
    return _masked_mean(pen, w)


def curvature_penalty(
    mu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    norm: str = "l2",
) -> torch.Tensor:
    """
    Second-derivative (curvature) penalty along spectral bins.

    Encourages μ[:, i] ≈ (μ[:, i-1] + μ[:, i+1]) / 2.

    Parameters
    ----------
    mu : Tensor [B, N]
    mask : Tensor [B, N], optional
        Per-bin validity mask (1 valid, 0 invalid).
    norm : {'l1','l2'}

    Returns
    -------
    Tensor scalar
    """
    d2 = second_derivative(mu)  # [B, N-2]
    if d2.numel() == 0:
        return mu.new_tensor(0.0)

    pen = d2.abs() if norm.lower() == "l1" else d2.pow(2)

    w = None
    if mask is not None:
        w = _triple_mask(mask).to(mu.dtype)

    return _masked_mean(pen, w)


def coherence_penalty(
    mu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    window: int = 5,
) -> torch.Tensor:
    """
    Penalize incoherent drift of local slope by comparing it against a moving average.

    Implements an L2 penalty on Δμ[i] - MA_k(Δμ)[i], where Δμ is the first derivative
    and MA_k is a length-k moving average (replicate padding at edges).

    Parameters
    ----------
    mu : Tensor [B, N]
    mask : Tensor [B, N], optional
        Per-bin validity mask (1 valid, 0 invalid).
    window : int
        Window length for the moving average (odd positive integer recommended).

    Returns
    -------
    Tensor scalar
    """
    B, N = mu.shape
    if N < 3 or window < 2:
        return mu.new_tensor(0.0)

    d1 = first_derivative(mu)  # [B, N-1]
    pad = (window - 1) // 2
    kernel = torch.ones(1, 1, window, device=mu.device, dtype=mu.dtype) / float(window)

    # shape to conv: [B, C=1, L], pad with replication for stable borders
    d1_pad = F.pad(d1.unsqueeze(1), (pad, pad), mode="replicate")
    d1_ma = F.conv1d(d1_pad, kernel).squeeze(1)  # [B, N-1]
    incoh = (d1 - d1_ma).pow(2)                 # [B, N-1]

    w = None
    if mask is not None:
        w = _pairwise_mask(mask).to(mu.dtype)

    return _masked_mean(incoh, w)


# ----------------------------------------------------------------------------- #
# Wrapper module for easy wiring from Hydra/encoders
# ----------------------------------------------------------------------------- #

@dataclass
class SmoothnessConfig:
    """
    Configuration for smoothness-related penalties.

    Attributes
    ----------
    smoothness_lambda : float
        Weight for first-derivative penalty.
    curvature_lambda : float
        Weight for second-derivative penalty.
    coherence_lambda : float
        Weight for moving-average slope coherence penalty.
    smoothness_norm : str
        'l1' or 'l2' for first derivative.
    curvature_norm : str
        'l1' or 'l2' for second derivative.
    window : int
        Moving-average window for coherence (>1).
    """
    smoothness_lambda: float = 0.0
    curvature_lambda: float = 0.0
    coherence_lambda: float = 0.0
    smoothness_norm: str = "l2"
    curvature_norm: str = "l2"
    window: int = 5


class SmoothnessLoss(nn.Module):
    """
    Composite spectral smoothness loss.

    Returns a dictionary with individual terms and the total loss:

        {
          "loss": total,
          "smooth": smooth_term,
          "curv": curvature_term,
          "coh": coherence_term,
        }

    Example
    -------
    >>> crit = SmoothnessLoss(smoothness_lambda=1e-3, curvature_lambda=1e-4, coherence_lambda=0.0)
    >>> out = crit(mu=mu, mask=mask)
    >>> out["loss"]
    """

    def __init__(
        self,
        *,
        smoothness_lambda: float = 0.0,
        curvature_lambda: float = 0.0,
        coherence_lambda: float = 0.0,
        smoothness_norm: str = "l2",
        curvature_norm: str = "l2",
        window: int = 5,
    ) -> None:
        super().__init__()
        self.cfg = SmoothnessConfig(
            smoothness_lambda=smoothness_lambda,
            curvature_lambda=curvature_lambda,
            coherence_lambda=coherence_lambda,
            smoothness_norm=smoothness_norm,
            curvature_norm=curvature_norm,
            window=window,
        )

    def forward(
        self,
        *,
        mu: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute composite loss.

        Parameters
        ----------
        mu : Tensor [B, N]
        mask : Tensor [B, N], optional

        Returns
        -------
        dict
            Keys: 'loss', 'smooth', 'curv', 'coh'
        """
        smooth_term = (
            smoothness_penalty(mu, mask=mask, norm=self.cfg.smoothness_norm)
            if self.cfg.smoothness_lambda > 0
            else mu.new_tensor(0.0)
        )
        curv_term = (
            curvature_penalty(mu, mask=mask, norm=self.cfg.curvature_norm)
            if self.cfg.curvature_lambda > 0
            else mu.new_tensor(0.0)
        )
        coh_term = (
            coherence_penalty(mu, mask=mask, window=self.cfg.window)
            if self.cfg.coherence_lambda > 0
            else mu.new_tensor(0.0)
        )

        total = (
            self.cfg.smoothness_lambda * smooth_term
            + self.cfg.curvature_lambda * curv_term
            + self.cfg.coherence_lambda * coh_term
        )

        return {
            "loss": total,
            "smooth": smooth_term.detach(),
            "curv": curv_term.detach(),
            "coh": coh_term.detach(),
        }
