from __future__ import annotations

"""
Composite physics loss for SpectraMind V50.

Combines:
  • Heteroscedastic Gaussian NLL (μ, σ vs. target y)
  • Non-negativity / bounds penalty on μ
  • Spectral smoothness penalty on μ (1st/2nd order; L1/L2/Charbonnier/Huber)
  • Band-coherence penalty on μ (encourage within-band variance ↓)

All losses support masking and tunable weights; returns a dict with each component and a
"total" field suitable for logging & backprop.

Example (Lightning):
    losses = composite_loss(mu=out["mu"], sigma=out["sigma"], target=batch["y"], mask=batch.get("mask"))
    loss = losses["total"]
    self.log_dict({f"train/{k}": v for k, v in losses.items()}, prog_bar=True)
"""

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from .nonneg import NonNegConfig, bounds_penalty  # reuse nonneg/bounds helpers


__all__ = [
    "SmoothnessConfig",
    "BandCoherenceConfig",
    "CompositeLossConfig",
    "CompositeLoss",
    "gaussian_nll",
]


# ======================================================================================
# Gaussian NLL (heteroscedastic)
# ======================================================================================

def gaussian_nll(mu: Tensor, sigma: Tensor, target: Tensor, mask: Optional[Tensor] = None,
                 reduction: str = "mean", eps: float = 1e-12) -> Tensor:
    """
    Negative log-likelihood of Gaussian with per-element μ, σ.

    L = 0.5 * [ log(2πσ^2) + (y - μ)^2 / σ^2 ]

    Shapes:
      mu, sigma, target, mask: broadcastable to the same shape (e.g., B x Nλ)
    """
    var = sigma.clamp_min(eps) ** 2
    nll = 0.5 * (torch.log(2 * torch.pi * var) + (target - mu) ** 2 / var)
    if mask is not None:
        nll = nll * mask
    if reduction == "sum":
        return nll.sum()
    if reduction == "none":
        return nll
    return nll.mean()


# ======================================================================================
# Smoothness penalty (finite differences over wavelength bins)
# ======================================================================================

@dataclass
class SmoothnessConfig:
    enable: bool = True
    # order: 1 (first diff) or 2 (second diff)
    order: int = 2
    # mode: "l2" | "l1" | "charbonnier" | "huber"
    mode: str = "charbonnier"
    # delta used by Charbonnier / Huber
    delta: float = 1e-3
    # global weight
    weight: float = 1.0
    # reduction: "mean" | "sum" | "none"
    reduction: str = "mean"


def _finite_difference(x: Tensor, order: int = 2) -> Tensor:
    """
    Compute 1st or 2nd order differences along dim=1 (bins).
    x: (B, Nλ)
    returns:
      order=1 -> (B, Nλ-1)
      order=2 -> (B, Nλ-2)
    """
    if order == 1:
        return x[:, 1:] - x[:, :-1]
    if order == 2:
        # second difference: x(i+1) - 2x(i) + x(i-1)
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    raise ValueError("order must be 1 or 2")


def _apply_diff_mask(mask: Optional[Tensor], order: int) -> Optional[Tensor]:
    """
    Produce a mask for differenced array:
      order=1: m[:,1:] * m[:,:-1]
      order=2: m[:,2:] * m[:,1:-1] * m[:,:-2]
    """
    if mask is None:
        return None
    if order == 1:
        return (mask[:, 1:] * mask[:, :-1]).to(mask.dtype)
    if order == 2:
        return (mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]).to(mask.dtype)
    raise ValueError("order must be 1 or 2")


def _penalty_on_diffs(d: Tensor, mode: str, delta: float) -> Tensor:
    """
    Apply penalty to differences:
      - l2:           d^2
      - l1:           |d|
      - charbonnier:  sqrt(d^2 + delta^2) - delta
      - huber:        0.5*d^2 if |d| < delta else delta*(|d| - 0.5*delta)
    """
    mode = mode.lower()
    if mode == "l2":
        return d ** 2
    if mode == "l1":
        return d.abs()
    if mode == "charbonnier":
        return torch.sqrt(d * d + delta * delta) - delta
    if mode == "huber":
        absd = d.abs()
        return torch.where(absd < delta, 0.5 * d * d, delta * (absd - 0.5 * delta))
    raise ValueError(f"unknown smoothness mode '{mode}'")


def smoothness_loss(mu: Tensor, mask: Optional[Tensor], cfg: SmoothnessConfig) -> Tensor:
    """
    Encourage spectral smoothness across bins via finite differences.
    mu: (B, Nλ)
    mask: (B, Nλ) or None
    """
    if not cfg.enable:
        return mu.new_zeros(())
    d = _finite_difference(mu, order=int(cfg.order))
    m = _apply_diff_mask(mask, order=int(cfg.order))
    pen = _penalty_on_diffs(d, cfg.mode, cfg.delta)
    if m is not None:
        pen = pen * m
    if cfg.reduction == "sum":
        val = pen.sum()
    elif cfg.reduction == "none":
        val = pen
    else:
        val = pen.mean()
    return cfg.weight * val


# ======================================================================================
# Band coherence penalty
# ======================================================================================

@dataclass
class BandCoherenceConfig:
    enable: bool = False
    # list of (start_idx, end_idx) inclusive bands, e.g., [(0,20), (21,40), ...]
    bands: List[Tuple[int, int]] = field(default_factory=list)
    # weight per band (optional; broadcast if length 1)
    band_weights: Optional[List[float]] = None
    # "var" = variance cost; "l2_center" = L2 to band-mean
    mode: str = "var"
    weight: float = 1.0
    reduction: str = "mean"


def band_coherence_loss(mu: Tensor, mask: Optional[Tensor], cfg: BandCoherenceConfig) -> Tensor:
    """
    Encourage coherence within specified molecular bands by reducing within-band variance.

    mu:   (B, Nλ)
    mask: (B, Nλ) or None
    """
    if not cfg.enable or not cfg.bands:
        return mu.new_zeros(())
    B, N = mu.shape
    vals = []

    # Prepare per-band weights
    bw: Optional[List[float]] = None
    if cfg.band_weights is not None and len(cfg.band_weights) > 0:
        if len(cfg.band_weights) == 1 and len(cfg.bands) > 1:
            bw = [cfg.band_weights[0]] * len(cfg.bands)
        else:
            assert len(cfg.band_weights) == len(cfg.bands), "band_weights length must match bands"
            bw = cfg.band_weights

    for i, (s, e) in enumerate(cfg.bands):
        s = max(int(s), 0)
        e = min(int(e), N - 1)
        if e < s:
            continue
        seg = mu[:, s : e + 1]  # (B, L)
        if mask is not None:
            m = mask[:, s : e + 1].to(mu.dtype)
            # weighted mean (avoid division by zero)
            denom = m.sum(dim=1, keepdim=True).clamp_min(1e-12)
            mean = (seg * m).sum(dim=1, keepdim=True) / denom
            if cfg.mode == "var":
                band_cost = ((seg - mean) ** 2 * m).sum(dim=1) / denom.squeeze(1)
            else:  # "l2_center"
                band_cost = ((seg - mean).abs() * m).sum(dim=1) / denom.squeeze(1)
        else:
            mean = seg.mean(dim=1, keepdim=True)
            if cfg.mode == "var":
                band_cost = ((seg - mean) ** 2).mean(dim=1)
            else:
                band_cost = (seg - mean).abs().mean(dim=1)
        if bw is not None:
            band_cost = band_cost * float(bw[i])
        vals.append(band_cost)

    if not vals:
        return mu.new_zeros(())
    # (num_bands, B) → (B,) by mean
    stacked = torch.stack(vals, dim=0)  # (K, B)
    per_batch = stacked.mean(dim=0)     # (B,)
    if cfg.reduction == "sum":
        out = per_batch.sum()
    elif cfg.reduction == "none":
        out = per_batch
    else:
        out = per_batch.mean()
    return cfg.weight * out


# ======================================================================================
# Composite config + module
# ======================================================================================

@dataclass
class CompositeLossConfig:
    # Enable/weights for each component
    use_gll: bool = True
    w_gll: float = 1.0

    nonneg: NonNegConfig = field(default_factory=lambda: NonNegConfig(lower=0.0, upper=None, weight=0.0))
    smooth: SmoothnessConfig = field(default_factory=SmoothnessConfig)
    band: BandCoherenceConfig = field(default_factory=BandCoherenceConfig)

    # Reduction for Gaussian NLL
    gll_reduction: str = "mean"
    # epsilon for Gaussian NLL
    gll_eps: float = 1e-12


class CompositeLoss(nn.Module):
    """
    Composite loss computing:
      - Gaussian NLL (μ, σ, y), if enabled
      - Bounds / nonnegativity on μ
      - Smoothness penalty on μ
      - Band coherence penalty on μ

    Returns a dict with individual components and a "total" key.
    """

    def __init__(self, cfg: CompositeLossConfig):
        super().__init__()
        self.cfg = cfg

    def forward(
        self,
        *,
        mu: Tensor,
        sigma: Optional[Tensor] = None,
        target: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        Inputs
        ------
        mu:     (B, Nλ)
        sigma:  (B, Nλ) or None
        target: (B, Nλ) or None
        mask:   (B, Nλ) or None (1=keep, 0=ignore) or per-bin weights
        """
        components: Dict[str, Tensor] = {}

        # Gaussian NLL
        gll = mu.new_zeros(())
        if self.cfg.use_gll:
            if sigma is None or target is None:
                raise ValueError("Gaussian NLL requires both sigma and target when use_gll=True.")
            gll = gaussian_nll(mu, sigma, target, mask=mask, reduction=self.cfg.gll_reduction, eps=self.cfg.gll_eps)
            components["gll"] = self.cfg.w_gll * gll

        # Bounds / nonnegativity on μ
        nonneg_weight = float(self.cfg.nonneg.weight)
        if nonneg_weight != 0.0:
            nn_pen = bounds_penalty(
                mu,
                lower=self.cfg.nonneg.lower,
                upper=self.cfg.nonneg.upper,
                mode=self.cfg.nonneg.mode,
                p=float(self.cfg.nonneg.p),
                beta=float(self.cfg.nonneg.beta),
                eps=float(self.cfg.nonneg.eps),
                mask=mask,
                weight=self.cfg.nonneg.weight,
                reduction=self.cfg.nonneg.reduction,
            )
            components["nonneg"] = nn_pen

        # Smoothness on μ
        if self.cfg.smooth.enable and float(self.cfg.smooth.weight) != 0.0:
            sm = smoothness_loss(mu, mask, self.cfg.smooth)
            components["smoothness"] = sm

        # Band coherence on μ
        if self.cfg.band.enable and float(self.cfg.band.weight) != 0.0:
            bc = band_coherence_loss(mu, mask, self.cfg.band)
            components["band_coherence"] = bc

        # Total
        total = mu.new_zeros(())
        for k, v in components.items():
            total = total + v
        components["total"] = total
        return components


# ======================================================================================
# Quick smoke tests
# ======================================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    B, N = 2, 16
    mu = torch.randn(B, N)
    sigma = torch.rand(B, N) * 0.2 + 0.1
    y = torch.randn(B, N)
    mask = torch.ones(B, N)

    cfg = CompositeLossConfig(
        use_gll=True,
        w_gll=1.0,
        nonneg=NonNegConfig(lower=0.0, upper=1.0, mode="soft_hinge", p=2.0, weight=0.5, reduction="mean"),
        smooth=SmoothnessConfig(enable=True, order=2, mode="charbonnier", delta=1e-3, weight=0.1, reduction="mean"),
        band=BandCoherenceConfig(enable=True, bands=[(0, 3), (4, 7), (8, 11), (12, 15)], band_weights=[1.0], mode="var", weight=0.2),
        gll_reduction="mean",
    )
    comp = CompositeLoss(cfg)
    losses = comp(mu=mu, sigma=sigma, target=y, mask=mask)
    print({k: float(v) for k, v in losses.items()})
