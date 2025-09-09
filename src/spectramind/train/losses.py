from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import math
import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    # primary terms
    w_gll: float = 1.0       # Gaussian NLL weight (primary leaderboard metric)
    w_tv: float = 0.0        # first-derivative smoothness (total variation)
    w_curv: float = 0.0      # second-derivative smoothness
    w_nonneg: float = 0.0    # non-negativity penalty on μ
    w_calib: float = 0.0     # coverage/uncertainty calibration penalty

    # options
    tv_eps: float = 1e-6     # TV epsilon for differentiable |.| approx
    fgs1_weight: float = 58.0  # up-weight for bin 0 (FGS1 white-light), ~58× per ADR 0002
    clamp_sigma_min: float = 1e-8
    # If True and no custom bin weights provided, apply fgs1_weight to bin 0
    use_fgs1_weight: bool = True


LOG_2PI: float = math.log(2.0 * math.pi)


def _apply_mask_and_weights(x: torch.Tensor,
                            mask: Optional[torch.Tensor],
                            bin_weights: Optional[torch.Tensor]) -> torch.Tensor:
    """
    Broadcast-safe multiply by (optional) mask and bin_weights.
    mask shape can be [B, BINS] or [B, 1] or [BINS] or scalar
    bin_weights shape can be [BINS] or [B, BINS] or [1, BINS]
    """
    if mask is not None:
        x = x * mask
    if bin_weights is not None:
        x = x * bin_weights
    return x


def gaussian_log_likelihood(
    mu: torch.Tensor,            # [B, BINS]
    sigma: torch.Tensor,         # [B, BINS]
    target: torch.Tensor,        # [B, BINS]
    *,
    clamp_sigma_min: float = 1e-8,
    mask: Optional[torch.Tensor] = None,      # broadcastable to [B, BINS]
    bin_weights: Optional[torch.Tensor] = None,  # broadcastable to [B, BINS]
) -> torch.Tensor:
    """
    Stable heteroscedastic Gaussian NLL per sample (sum over bins),
    optionally masked/weighted per-bin.
    Returns: [B] vector.
    """
    sigma = torch.clamp(sigma, min=clamp_sigma_min)
    z = (target - mu) / sigma  # [B, BINS]

    # elementwise NLL per bin
    nll_bin = 0.5 * (z * z + 2.0 * torch.log(sigma) + LOG_2PI)

    # apply optional mask/weights, then sum over bins
    nll_bin = _apply_mask_and_weights(nll_bin, mask, bin_weights)
    return nll_bin.sum(dim=-1)  # [B]


def tv_penalty(mu: torch.Tensor, eps: float = 1e-6,
               mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Total-variation-like penalty along spectral axis (last dim).
    Returns per-sample sum over bins (length-1 diff), shape [B].
    """
    d = torch.diff(mu, dim=-1)
    # If a mask is provided, reduce it to the diff lattice (simple AND of adjacent mask bins)
    if mask is not None:
        m = mask[..., :-1] * mask[..., 1:]
    else:
        m = None
    tv = torch.sqrt(d * d + eps)
    if m is not None:
        tv = tv * m
    return tv.sum(dim=-1)


def curvature_penalty(mu: torch.Tensor,
                      mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Second-difference (discrete curvature) penalty along spectral axis.
    Returns per-sample sum over bins (length-2 diff), shape [B].
    """
    d2 = torch.diff(mu, n=2, dim=-1)
    if mask is not None:
        # shrink mask to second-diff lattice: m[i] = mask[i] & mask[i+1] & mask[i+2]
        m = mask[..., :-2] * mask[..., 1:-1] * mask[..., 2:]
        d2 = d2 * m
    return (d2 * d2).sum(dim=-1)


def nonneg_penalty(mu: torch.Tensor,
                   mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Penalize negative μ entries (enforce physical non-negativity of transit depth).
    """
    x = F.relu(-mu)
    if mask is not None:
        x = x * mask
    return x.sum(dim=-1)


def coverage_calibration_penalty(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    *,
    clamp_sigma_min: float = 1e-8,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Simple, smooth coverage calibration: encourage z^2 mean ≈ 1 across bins per sample.
    penalty = | mean(z^2) - 1 |, where z = (y - μ)/σ. Returns [B].
    """
    sigma = torch.clamp(sigma, min=clamp_sigma_min)
    z2 = ((target - mu) / sigma).square()  # [B, BINS]
    if mask is not None:
        # avoid division by zero: normalize by masked count
        masked_sum = (z2 * mask).sum(dim=-1)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        m_z2 = masked_sum / denom
    else:
        m_z2 = z2.mean(dim=-1)
    return (m_z2 - 1.0).abs()


def _default_bin_weights(
    cfg: LossConfig, bins: int, device: torch.device
) -> torch.Tensor:
    """
    If cfg.use_fgs1_weight, construct [1, BINS] weights with bin 0 = fgs1_weight.
    Otherwise return ones.
    """
    w = torch.ones((1, bins), device=device)
    if cfg.use_fgs1_weight and cfg.fgs1_weight != 1.0 and bins > 0:
        w[..., 0] = cfg.fgs1_weight
    return w


def build_composite_loss(cfg: LossConfig):
    """
    Returns a callable:
        loss_fn(pred, target, *, bin_weights=None, mask=None, reduction='mean') -> (loss, metrics_dict)

    Args:
        pred: Tuple[mu, sigma] with shapes [B, BINS].
        target: Tensor [B, BINS].
        bin_weights: optional weights (broadcastable to [B, BINS]). If None, applies
                     FGS1 up-weighting via cfg.use_fgs1_weight/cfg.fgs1_weight.
        mask: optional mask (broadcastable to [B, BINS]), 1=use, 0=ignore.
        reduction: 'mean' | 'sum' over batch for scalar loss; metrics in dict are reported as floats.

    Notes:
        • Gaussian NLL (GLL) matches leaderboard metric; FGS1 bin is ~58× by default.
        • Smoothness (TV, curvature) discourages unphysical jagged spectra.
        • Nonnegativity enforces μ ≥ 0.
        • Calibration penalty encourages honest σ (z^2 mean ≈ 1).
    """
    def loss_fn(
        pred: Tuple[torch.Tensor, torch.Tensor],
        target: torch.Tensor,
        *,
        bin_weights: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
        reduction: Literal["mean", "sum"] = "mean",
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        mu, sigma = pred  # [B, BINS]
        B, BINS = mu.shape

        # prepare weights (FGS1 ~58× if none provided)
        if bin_weights is None:
            bin_weights = _default_bin_weights(cfg, BINS, mu.device)  # [1, BINS]

        # --- Gaussian NLL (primary)
        gll_vec = gaussian_log_likelihood(
            mu, sigma, target,
            clamp_sigma_min=cfg.clamp_sigma_min,
            mask=mask,
            bin_weights=bin_weights,
        )  # [B]

        # --- Additional terms (per-sample)
        terms_vec = {}
        total_vec = cfg.w_gll * gll_vec
        terms_vec["gll"] = gll_vec

        if cfg.w_tv != 0.0:
            tv_vec = tv_penalty(mu, eps=cfg.tv_eps, mask=mask)
            total_vec = total_vec + cfg.w_tv * tv_vec
            terms_vec["tv"] = tv_vec

        if cfg.w_curv != 0.0:
            curv_vec = curvature_penalty(mu, mask=mask)
            total_vec = total_vec + cfg.w_curv * curv_vec
            terms_vec["curv"] = curv_vec

        if cfg.w_nonneg != 0.0:
            nn_vec = nonneg_penalty(mu, mask=mask)
            total_vec = total_vec + cfg.w_nonneg * nn_vec
            terms_vec["nonneg"] = nn_vec

        if cfg.w_calib != 0.0:
            calib_vec = coverage_calibration_penalty(
                mu, sigma, target,
                clamp_sigma_min=cfg.clamp_sigma_min,
                mask=mask,
            )
            total_vec = total_vec + cfg.w_calib * calib_vec
            terms_vec["calib"] = calib_vec

        # reduction over batch
        if reduction == "mean":
            total = total_vec.mean()
            metrics = {k: float(v.mean().detach().cpu()) for k, v in terms_vec.items()}
        elif reduction == "sum":
            total = total_vec.sum()
            metrics = {k: float(v.sum().detach().cpu()) for k, v in terms_vec.items()}
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        return total, metrics

    return loss_fn