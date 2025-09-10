# src/spectramind/losses/composite_physics.py
from __future__ import annotations
"""
Composite physics loss for SpectraMind V50 (upgraded).

Combines:
  • Heteroscedastic Gaussian NLL (μ, σ vs. target y) with optional per-element weights (e.g., FGS1 bin weight)
  • Non-negativity / bounds penalty on μ
  • Spectral smoothness penalty on μ (1st/2nd order; L1/L2/Charbonnier/Huber)
  • Band-coherence penalty on μ (encourage within-band variance ↓)

All losses support masking/weights and tunable weights; returns a dict with each component
and a "total" field suitable for logging & backprop.

Example (Lightning):
    losses = comp(mu=out["mu"], sigma=out["sigma"], target=batch["y"],
                  mask=batch.get("mask"), gll_weights=batch.get("bin_w"))
    loss = losses["total"]
    self.log_dict({f"train/{k}": v for k, v in losses.items()}, prog_bar=True)
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn

from .nonneg import NonNegConfig, bounds_penalty  # reuse nonneg/bounds helpers

__all__ = [
    "GLLConfig",
    "SmoothnessConfig",
    "BandCoherenceConfig",
    "CompositeLossConfig",
    "CompositeLoss",
    "gaussian_nll",
]

_LOG_TWO_PI = math.log(2.0 * math.pi)


# ======================================================================================
# Gaussian NLL (heteroscedastic) with optional per-element weights
# ======================================================================================

@dataclass
class GLLConfig:
    reduction: str = "mean"            # "mean" | "sum" | "none"
    eps: float = 1e-12                 # numerical floor before squares/divisions/logs
    sigma_floor: float = 1e-6          # clamp σ >= floor after transform
    sigma_ceil: Optional[float] = None # optional clamp σ <= ceil
    # Optional transform: None | "softplus" | "exp"
    sigma_transform: Optional[str] = None
    sigma_softplus_beta: float = 1.0   # β for softplus if used


def _apply_sigma_transform(sigma: Tensor, cfg: GLLConfig) -> Tensor:
    if cfg.sigma_transform is None:
        out = sigma
    else:
        t = cfg.sigma_transform.lower()
        if t == "softplus":
            out = torch.nn.functional.softplus(sigma, beta=cfg.sigma_softplus_beta)
        elif t == "exp":
            out = torch.exp(sigma)
        else:
            raise ValueError(f"unknown sigma_transform '{cfg.sigma_transform}'")
    # Final clamping for numerical stability
    out = out.clamp_min(cfg.sigma_floor)
    if cfg.sigma_ceil is not None:
        out = out.clamp_max(cfg.sigma_ceil)
    return out


def gaussian_nll(
    mu: Tensor,
    sigma: Tensor,
    target: Tensor,
    mask: Optional[Tensor] = None,
    w: Optional[Tensor] = None,
    cfg: Optional[GLLConfig] = None,
) -> Tensor:
    """
    Negative log-likelihood of Gaussian with per-element μ, σ and optional weights.

    Elementwise:
      L = 0.5 * [ log(2πσ^2) + (y - μ)^2 / σ^2 ]

    Shapes:
      mu, sigma, target, mask, w : broadcastable to the same shape (e.g., (B, Nλ))
      - 'mask' and 'w' can be floats (per-element weights). Effective weight = mask * w.

    Returns:
      scalar (mean/sum) or per-element tensor ("none") depending on cfg.reduction.
    """
    cfg = cfg or GLLConfig()
    # Broadcast sanity
    mu, sigma, target = torch.broadcast_tensors(mu, sigma, target)

    # Optional per-element weight (allow both boolean/binary masks and real weights)
    eff_w: Optional[Tensor] = None
    if mask is not None and w is not None:
        m = torch.broadcast_tensors(mask.to(mu.dtype), mu)[0]
        ww = torch.broadcast_tensors(w.to(mu.dtype), mu)[0]
        eff_w = m * ww
    elif mask is not None:
        eff_w = torch.broadcast_tensors(mask.to(mu.dtype), mu)[0]
    elif w is not None:
        eff_w = torch.broadcast_tensors(w.to(mu.dtype), mu)[0]

    # σ transform & clamping
    sigma = _apply_sigma_transform(sigma, cfg)
    var = (sigma.clamp_min(cfg.eps)) ** 2

    # 0.5 * [log(2πσ^2) + (y - μ)^2 / σ^2]
    nll = 0.5 * (torch.log(var) + _LOG_TWO_PI + (target - mu) ** 2 / var)

    if eff_w is not None:
        nll = nll * eff_w

    red = cfg.reduction
    if red == "sum":
        return nll.sum()
    if red == "none":
        return nll
    # Default "mean" (normalize by sum of weights if given, else by numel)
    if eff_w is not None:
        denom = eff_w.sum().clamp_min(cfg.eps)
        return nll.sum() / denom
    return nll.mean()


# ======================================================================================
# Smoothness penalty (finite differences over wavelength bins)
# ======================================================================================

@dataclass
class SmoothnessConfig:
    enable: bool = True
    order: int = 2                      # 1 (first diff) or 2 (second diff)
    mode: str = "charbonnier"           # "l2" | "l1" | "charbonnier" | "huber"
    delta: float = 1e-3                 # delta for Charbonnier / Huber
    weight: float = 1.0                 # global multiplier
    reduction: str = "mean"             # "mean" | "sum" | "none"


def _finite_difference(x: Tensor, order: int = 2) -> Tensor:
    """
    Compute 1st/2nd order differences along dim=1 (bins).
      x: (B, Nλ)
      order=1 -> (B, Nλ-1)
      order=2 -> (B, Nλ-2)
    """
    if order == 1:
        return x[:, 1:] - x[:, :-1]
    if order == 2:
        return x[:, 2:] - 2 * x[:, 1:-1] + x[:, :-2]
    raise ValueError("order must be 1 or 2")


def _apply_diff_mask(mask: Optional[Tensor], order: int) -> Optional[Tensor]:
    """
    Propagate mask to differenced array:
      order=1: m[:,1:] * m[:,:-1]
      order=2: m[:,2:] * m[:,1:-1] * m[:,:-2]
    """
    if mask is None:
        return None
    dtype = mask.dtype
    if order == 1:
        return (mask[:, 1:] * mask[:, :-1]).to(dtype)
    if order == 2:
        return (mask[:, 2:] * mask[:, 1:-1] * mask[:, :-2]).to(dtype)
    raise ValueError("order must be 1 or 2")


def _penalty_on_diffs(d: Tensor, mode: str, delta: float) -> Tensor:
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
    if not cfg.enable or float(cfg.weight) == 0.0:
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
        # weighted mean if m provided
        if m is not None:
            denom = m.sum().clamp_min(1e-12)
            val = pen.sum() / denom
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
    # "var" = variance cost; "l1_center" = L1 distance to band-mean
    mode: str = "var"
    weight: float = 1.0
    reduction: str = "mean"


def band_coherence_loss(mu: Tensor, mask: Optional[Tensor], cfg: BandCoherenceConfig) -> Tensor:
    if not cfg.enable or not cfg.bands or float(cfg.weight) == 0.0:
        return mu.new_zeros(())
    B, N = mu.shape
    vals: List[Tensor] = []

    # normalize band weights
    bw: Optional[List[float]] = None
    if cfg.band_weights:
        if len(cfg.band_weights) == 1 and len(cfg.bands) > 1:
            bw = [cfg.band_weights[0]] * len(cfg.bands)
        else:
            if len(cfg.band_weights) != len(cfg.bands):
                raise ValueError("band_weights length must match bands")
            bw = cfg.band_weights

    for i, (s, e) in enumerate(cfg.bands):
        s = max(int(s), 0)
        e = min(int(e), N - 1)
        if e < s:
            continue
        seg = mu[:, s : e + 1]  # (B, L)
        if mask is not None:
            m = mask[:, s : e + 1].to(mu.dtype)
            denom = m.sum(dim=1, keepdim=True).clamp_min(1e-12)
            mean = (seg * m).sum(dim=1, keepdim=True) / denom
            if cfg.mode == "var":
                band_cost = ((seg - mean) ** 2 * m).sum(dim=1) / denom.squeeze(1)
            elif cfg.mode == "l1_center":
                band_cost = ((seg - mean).abs() * m).sum(dim=1) / denom.squeeze(1)
            else:
                raise ValueError(f"unknown band mode '{cfg.mode}'")
        else:
            mean = seg.mean(dim=1, keepdim=True)
            if cfg.mode == "var":
                band_cost = ((seg - mean) ** 2).mean(dim=1)
            elif cfg.mode == "l1_center":
                band_cost = (seg - mean).abs().mean(dim=1)
            else:
                raise ValueError(f"unknown band mode '{cfg.mode}'")
        if bw is not None:
            band_cost = band_cost * float(bw[i])
        vals.append(band_cost)

    if not vals:
        return mu.new_zeros(())
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
    # Gaussian NLL
    use_gll: bool = True
    w_gll: float = 1.0
    gll: GLLConfig = field(default_factory=GLLConfig)

    # Bounds / nonnegativity on μ
    nonneg: NonNegConfig = field(default_factory=lambda: NonNegConfig(lower=0.0, upper=None, weight=0.0))

    # Smoothness on μ
    smooth: SmoothnessConfig = field(default_factory=SmoothnessConfig)

    # Band coherence on μ
    band: BandCoherenceConfig = field(default_factory=BandCoherenceConfig)


class CompositeLoss(nn.Module):
    """
    Composite loss computing:
      - Gaussian NLL (μ, σ, y, [optional weights]), if enabled
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
        mu: Tensor,                       # (B, Nλ)
        sigma: Optional[Tensor] = None,   # (B, Nλ)
        target: Optional[Tensor] = None,  # (B, Nλ)
        mask: Optional[Tensor] = None,    # (B, Nλ) or per-element weights
        gll_weights: Optional[Tensor] = None,  # (B, Nλ) or (Nλ,) e.g., FGS1 weighting
    ) -> Dict[str, Tensor]:
        components: Dict[str, Tensor] = {}
        mu = mu  # alias

        # Gaussian NLL
        if self.cfg.use_gll:
            if sigma is None or target is None:
                raise ValueError("Gaussian NLL requires both sigma and target when use_gll=True.")
            gll_val = gaussian_nll(
                mu=mu,
                sigma=sigma,
                target=target,
                mask=mask,
                w=gll_weights,
                cfg=self.cfg.gll,
            )
            components["gll"] = self.cfg.w_gll * gll_val

        # Bounds / nonnegativity on μ
        if float(self.cfg.nonneg.weight) != 0.0:
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
        for v in components.values():
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
    # example: FGS1 weight on bin 0
    bin_w = torch.ones(N); bin_w[0] = 58.0  # (N,)
    bin_w = bin_w.unsqueeze(0).expand(B, -1)  # (B, N)

    cfg = CompositeLossConfig(
        use_gll=True,
        w_gll=1.0,
        gll=GLLConfig(reduction="mean", sigma_transform="softplus", sigma_floor=1e-5, eps=1e-12),
        nonneg=NonNegConfig(lower=0.0, upper=1.0, mode="soft_hinge", p=2.0, weight=0.5, reduction="mean"),
        smooth=SmoothnessConfig(enable=True, order=2, mode="charbonnier", delta=1e-3, weight=0.1, reduction="mean"),
        band=BandCoherenceConfig(enable=True, bands=[(0, 3), (4, 7), (8, 11), (12, 15)], band_weights=[1.0], mode="var", weight=0.2),
    )
    comp = CompositeLoss(cfg)
    losses = comp(mu=mu, sigma=sigma, target=y, mask=mask, gll_weights=bin_w)
    print({k: float(v) for k, v in losses.items()})
