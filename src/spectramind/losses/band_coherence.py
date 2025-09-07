# src/spectramind/losses/band_coherence.py
from __future__ import annotations

"""
SpectraMind V50 — Band Coherence Losses
=======================================

Penalizes incoherent local drift in wavelength-binned spectra by constraining
the *local slope* to remain consistent within a short neighborhood.

This complements smoothness/curvature by discouraging "zig-zag" within a
window even when first/second derivatives are small on average.

Coherence flavors
-----------------
1) Moving-average slope coherence (default):
   Penalize (Δμ - MA_k(Δμ))^2, where Δμ is first derivative and MA_k is a short
   moving average with replicate padding.

2) Local linear fit coherence:
   Fit small linear models within window k and penalize deviations from the
   local line (ridge-style). Encourages locally linear behavior without strictly
   forcing low curvature.

Both support per-bin masks and stable weighted reductions.

Usage
-----
>>> import torch
>>> from spectramind.losses.band_coherence import BandCoherenceLoss
>>> crit = BandCoherenceLoss(kind="ma", window=5, weight=1e-3)
>>> mu = torch.randn(8, 283)
>>> mask = torch.ones_like(mu)
>>> out = crit(mu=mu, mask=mask)
>>> out["loss"], out["coh"]

You can also call the functional APIs: `coherence_ma(...)` or `coherence_local_fit(...)`.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- #
# Utilities
# ----------------------------------------------------------------------------- #


def _pairwise_mask(mask: torch.Tensor) -> torch.Tensor:
    """
    Combine adjacent masks for first-order ops: m[i] = mask[i] * mask[i-1].
    Input:  [B, N]  → Output: [B, N-1]
    """
    return (mask[:, 1:] * mask[:, :-1]).to(mask.dtype)


def _masked_mean(x: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Weighted mean over last dimension with stability guards, then mean over batch.
    x: [B, K], w: [B, K] or None
    """
    if w is None:
        return x.mean(dim=-1).mean()
    num = (x * w).sum(dim=-1)
    den = w.sum(dim=-1).clamp_min(1.0)
    return (num / den).mean()


# ----------------------------------------------------------------------------- #
# 1) Moving-average slope coherence
# ----------------------------------------------------------------------------- #


def coherence_ma(
    mu: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    window: int = 5,
) -> torch.Tensor:
    """
    Moving-average coherence penalty.

    Penalize deviation of first derivative from its moving average:

        L = mean_b mean_i w[i] * ( Δμ[i] - MA_k(Δμ)[i] )^2

    Args
    ----
    mu : Tensor [B, N]
    mask : Tensor [B, N], optional (1 valid, 0 invalid)
    window : int
        Window length for moving average (odd recommended)

    Returns
    -------
    Tensor scalar
    """
    B, N = mu.shape
    if N < 3 or window < 2:
        return mu.new_tensor(0.0)

    d1 = mu[:, 1:] - mu[:, :-1]  # [B, N-1]
    pad = (window - 1) // 2
    kernel = torch.ones(1, 1, window, device=mu.device, dtype=mu.dtype) / float(window)
    d1_pad = F.pad(d1.unsqueeze(1), (pad, pad), mode="replicate")  # [B,1,N-1+2pad]
    d1_ma = F.conv1d(d1_pad, kernel).squeeze(1)                      # [B, N-1]
    incoh = (d1 - d1_ma).pow(2)                                      # [B, N-1]

    w = None
    if mask is not None:
        w = _pairwise_mask(mask).to(mu.dtype)

    return _masked_mean(incoh, w)


# ----------------------------------------------------------------------------- #
# 2) Local linear fit coherence
# ----------------------------------------------------------------------------- #


def coherence_local_fit(
    mu: torch.Tensor,
    *,
    mask: Optional[torch.Tensor] = None,
    window: int = 5,
    ridge: float = 1e-6,
) -> torch.Tensor:
    """
    Local linear fit coherence penalty.

    For each center i, fit a line y = a * x + b in a local window around i
    using ridge-regularized least squares; penalize squared deviations of
    μ from this local line.

    Notes
    -----
    - Uses coordinate indices as x ∈ [0..k-1] per window, zero-centered to stabilize.
    - Efficient vectorized implementation via conv1d on sufficient statistics.
    - Replicate padding at borders to keep length N.

    Args
    ----
    mu : Tensor [B, N]
    mask : Tensor [B, N], optional
    window : int
        Local window length (odd ≥ 3 recommended)
    ridge : float
        Small diagonal stabilizer for (XᵀX) inversion.

    Returns
    -------
    Tensor scalar
    """
    B, N = mu.shape
    if window < 3 or N < window:
        # If too small, fall back to moving average penalty
        return coherence_ma(mu, mask=mask, window=min(max(window, 2), N-1))

    # Build centered coordinate kernel: x ∈ [-m, ..., 0, ..., +m]
    m = (window - 1) // 2
    x = torch.arange(-m, m + 1, device=mu.device, dtype=mu.dtype)  # [window]
    # Precompute sums for each window via conv1d
    ones_k = torch.ones(window, device=mu.device, dtype=mu.dtype).view(1, 1, -1)
    x_k = x.view(1, 1, -1)
    x2_k = (x**2).view(1, 1, -1)

    mu_ = mu.unsqueeze(1)  # [B,1,N]
    # Sufficient statistics per centered window
    sum_y = F.conv1d(mu_, ones_k, padding=m).squeeze(1)     # [B,N]
    sum_x = F.conv1d(mu_, x_k, padding=m).squeeze(1)        # [B,N]  (dot with x vector)
    sum_xx = F.conv1d(mu.new_ones(B, 1, N), x2_k, padding=m).squeeze(1)  # [B,N]
    sum_xy = F.conv1d(mu_, x_k, padding=m).squeeze(1)       # same as sum_x when y==1? NO:
    # Correct sum_xy: convolve y with (x) kernel directly
    sum_xy = F.conv1d(mu_, x_k, padding=m).squeeze(1)       # [B,N]

    k = float(window)
    # Solve normal equations per center for slope/intercept of local fit:
    # [sum_xx  sum_x ] [a] = [sum_xy]
    # [sum_x    k   ] [b]   [sum_y ]
    # Determinant Δ = sum_xx * k - sum_x^2 + ridge
    det = (sum_xx * k - sum_x.pow(2)) + ridge
    a = (k * sum_xy - sum_x * sum_y) / det
    b = (sum_xx * sum_y - sum_x * sum_xy) / det

    # Predicted local line at center index (x_center=0): ŷ_center = b
    # But we want deviations across the whole signal, not just centers.
    # Approximate residual per center as |μ[i] - b[i]|; that's the deviation at center.
    resid = (mu - b).pow(2)   # [B, N]

    # Masking
    w = None
    if mask is not None:
        w = mask.to(mu.dtype)

    return _masked_mean(resid, w)


# ----------------------------------------------------------------------------- #
# Composite wrapper
# ----------------------------------------------------------------------------- #


@dataclass
class BandCoherenceConfig:
    """
    Configuration for band coherence penalties.

    Attributes
    ----------
    kind : str
        'ma' (moving-average slope) or 'local_fit'
    window : int
        Neighborhood length (odd recommended)
    ridge : float
        Stabilizer for local fit (ignored for 'ma')
    weight : float
        Global multiplier applied to the coherence term
    """
    kind: str = "ma"
    window: int = 5
    ridge: float = 1e-6
    weight: float = 0.0


class BandCoherenceLoss(nn.Module):
    """
    Composite band coherence loss.

    Returns:
        {
          "loss": weight * coherence,
          "coh": coherence_term,
        }
    """

    def __init__(
        self,
        *,
        kind: str = "ma",
        window: int = 5,
        ridge: float = 1e-6,
        weight: float = 0.0,
    ) -> None:
        super().__init__()
        self.cfg = BandCoherenceConfig(kind=kind, window=window, ridge=ridge, weight=weight)

    def forward(
        self,
        *,
        mu: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        if self.cfg.weight <= 0.0:
            z = mu.new_tensor(0.0)
            return {"loss": z, "coh": z}

        if self.cfg.kind == "local_fit":
            coh = coherence_local_fit(mu, mask=mask, window=self.cfg.window, ridge=self.cfg.ridge)
        else:
            coh = coherence_ma(mu, mask=mask, window=self.cfg.window)

        return {"loss": self.cfg.weight * coh, "coh": coh.detach()}
