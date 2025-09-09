from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Literal

import math
import torch
import torch.nn.functional as F


@dataclass
class LossConfig:
    # Primary terms
    w_gll: float = 1.0       # Gaussian NLL weight (primary leaderboard metric)
    w_tv: float = 0.0        # first-derivative smoothness (total variation)
    w_curv: float = 0.0      # second-derivative smoothness
    w_nonneg: float = 0.0    # non-negativity penalty on μ
    w_calib: float = 0.0     # coverage/uncertainty calibration penalty
    w_sigma_prior: float = 0.0  # L2 penalty on log_sigma toward a prior mean (sigma scale sanity)

    # Options
    tv_eps: float = 1e-6
    fgs1_weight: float = 58.0            # ADR-0002 (~58× for the FGS1 white-light channel)
    fgs1_bin_index: int = 0              # which bin is the FGS1 “white-light” entry
    use_fgs1_weight: bool = True

    clamp_sigma_min: float = 1e-8        # hard floor for σ for stability
    clamp_sigma_max: Optional[float] = None  # optional σ ceiling (e.g., 1.0)

    clamp_mu_min: Optional[float] = None  # optional μ floor (non-neg is a penalty; this is a hard clamp)
    clamp_mu_max: Optional[float] = None  # optional μ ceiling

    # Coverage (z^2≈1) penalty: use absolute deviation from 1.0 (smooth)
    calib_target: float = 1.0

    # Sigma prior
    sigma_prior_log_mean: float = 0.0     # prior mean for log_sigma (0 ⇒ σ≈1)
    sigma_prior_reduction: Literal["mean", "sum"] = "mean"

    # Per-sample normalization for masked data:
    # if True, divide per-sample binwise sums by number of valid bins (so masked samples contribute comparably)
    normalize_by_valid: bool = True


# Constants (device/dtype-safe at call time)
def _log_2pi(device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(math.log(2.0 * math.pi), device=device, dtype=dtype)


def _apply_mask_and_weights(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    bin_weights: Optional[torch.Tensor],
) -> torch.Tensor:
    """
    Broadcast-safe multiply by (optional) mask and bin_weights.
    mask shape: [B, BINS] or [B, 1] or [BINS] or scalar
    weights shape: [BINS] or [B, BINS] or [1, BINS]
    """
    if mask is not None:
        x = x * mask
    if bin_weights is not None:
        x = x * bin_weights
    return x


def _normalize_per_sample(
    x: torch.Tensor,
    mask: Optional[torch.Tensor],
    normalize_by_valid: bool,
) -> torch.Tensor:
    """
    If normalize_by_valid is True, divide per-sample sums by number of valid bins.
    Otherwise, leave as sums.
    """
    if not normalize_by_valid:
        # sum over bins (last dim) already done by caller in most cases
        return x

    if mask is not None:
        denom = mask.sum(dim=-1).clamp_min(1.0)
    else:
        denom = torch.full((x.shape[0],), x.shape[-1], device=x.device, dtype=x.dtype)
    return x / denom


def gaussian_log_likelihood(
    mu: torch.Tensor,            # [B, BINS]
    sigma: torch.Tensor,         # [B, BINS]
    target: torch.Tensor,        # [B, BINS]
    *,
    clamp_sigma_min: float = 1e-8,
    clamp_sigma_max: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,      # broadcastable to [B, BINS]
    bin_weights: Optional[torch.Tensor] = None,  # broadcastable to [B, BINS]
    normalize_by_valid: bool = True,
) -> torch.Tensor:
    """
    Stable heteroscedastic Gaussian NLL per sample (sum over bins → normalized per-sample if req).
    Returns: [B] vector.
    """
    # Optional μ clamp (hard, for sanity; nonneg penalty is a soft alternative)
    mu = mu
    if clamp_sigma_max is not None:
        sigma = torch.clamp(sigma, min=clamp_sigma_min, max=clamp_sigma_max)
    else:
        sigma = torch.clamp(sigma, min=clamp_sigma_min)

    z = (target - mu) / sigma  # [B, BINS]
    nll_bin = 0.5 * (z * z + 2.0 * torch.log(sigma) + _log_2pi(device=sigma.device, dtype=sigma.dtype))
    nll_bin = _apply_mask_and_weights(nll_bin, mask, bin_weights)  # [B, BINS]
    nll_sum = nll_bin.sum(dim=-1)                                  # [B]
    if normalize_by_valid:
        if mask is not None:
            denom = mask.sum(dim=-1).clamp_min(1.0)
        else:
            denom = torch.full_like(nll_sum, fill_value=float(nll_bin.shape[-1]))
        nll_sum = nll_sum / denom
    return nll_sum


def tv_penalty(
    mu: torch.Tensor,
    eps: float = 1e-6,
    mask: Optional[torch.Tensor] = None,
    normalize_by_valid: bool = True,
) -> torch.Tensor:
    """
    Total-variation-like penalty along spectral axis (last dim).
    Returns [B] per-sample aggregate (normalized if requested).
    """
    d = torch.diff(mu, dim=-1)
    if mask is not None:
        m = mask[..., :-1] * mask[..., 1:]
    else:
        m = None
    tv = torch.sqrt(d * d + eps)
    tv = _apply_mask_and_weights(tv, m, None)
    tv = tv.sum(dim=-1)
    if normalize_by_valid:
        if m is not None:
            denom = m.sum(dim=-1).clamp_min(1.0)
        else:
            denom = torch.full_like(tv, fill_value=float(mu.shape[-1] - 1))
        tv = tv / denom
    return tv


def curvature_penalty(
    mu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    normalize_by_valid: bool = True,
) -> torch.Tensor:
    """
    Second-difference (discrete curvature) penalty along spectral axis.
    Returns [B] per-sample aggregate (normalized if requested).
    """
    d2 = torch.diff(mu, n=2, dim=-1)
    if mask is not None:
        m = mask[..., :-2] * mask[..., 1:-1] * mask[..., 2:]
        d2 = d2 * m
    curv = (d2 * d2).sum(dim=-1)
    if normalize_by_valid:
        if mask is not None:
            denom = (mask[..., :-2] * mask[..., 1:-1] * mask[..., 2:]).sum(dim=-1).clamp_min(1.0)
        else:
            denom = torch.full_like(curv, fill_value=float(mu.shape[-1] - 2))
        curv = curv / denom
    return curv


def nonneg_penalty(
    mu: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    normalize_by_valid: bool = True,
) -> torch.Tensor:
    """
    Penalize negative μ entries (enforce physical non-negativity of transit depth).
    Returns [B] per-sample aggregate (normalized if requested).
    """
    x = F.relu(-mu)  # penalize only negatives
    x = _apply_mask_and_weights(x, mask, None).sum(dim=-1)
    if normalize_by_valid:
        if mask is not None:
            denom = mask.sum(dim=-1).clamp_min(1.0)
        else:
            denom = torch.full_like(x, fill_value=float(mu.shape[-1]))
        x = x / denom
    return x


def coverage_calibration_penalty(
    mu: torch.Tensor,
    sigma: torch.Tensor,
    target: torch.Tensor,
    *,
    clamp_sigma_min: float = 1e-8,
    clamp_sigma_max: Optional[float] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    Smooth coverage calibration: encourage z^2 mean ≈ 1 across bins per sample.
    penalty = | mean(z^2) - 1 |  (absolute deviation).
    Returns [B].
    """
    if clamp_sigma_max is not None:
        sigma = torch.clamp(sigma, min=clamp_sigma_min, max=clamp_sigma_max)
    else:
        sigma = torch.clamp(sigma, min=clamp_sigma_min)

    z2 = ((target - mu) / sigma).square()  # [B, BINS]
    if mask is not None:
        masked_sum = (z2 * mask).sum(dim=-1)
        denom = mask.sum(dim=-1).clamp_min(1.0)
        m_z2 = masked_sum / denom
    else:
        m_z2 = z2.mean(dim=-1)
    return (m_z2 - 1.0).abs()


def sigma_prior_penalty(
    sigma: torch.Tensor,
    log_prior_mean: float = 0.0,
    reduction: Literal["mean", "sum"] = "mean",
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    L2 penalty on log_sigma toward `log_prior_mean`.
    Encourages σ to remain near a reasonable scale; useful when σ can drift.
    Returns [B] vector after aggregating over bins.
    """
    log_sigma = torch.log(sigma.clamp_min(torch.finfo(sigma.dtype).eps))
    diff2 = (log_sigma - log_prior_mean) ** 2
    if mask is not None:
        diff2 = diff2 * mask
    per_sample = diff2.sum(dim=-1)
    if reduction == "mean":
        return per_sample / (mask.sum(dim=-1).clamp_min(1.0) if mask is not None else sigma.new_full((sigma.shape[0],), sigma.shape[-1]))
    return per_sample


def _default_bin_weights(
    cfg: LossConfig, bins: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """
    If cfg.use_fgs1_weight, construct [1, BINS] weights with FGS1 bin index = fgs1_weight.
    Otherwise return ones.
    """
    w = torch.ones((1, bins), device=device, dtype=dtype)
    if cfg.use_fgs1_weight and cfg.fgs1_weight != 1.0 and bins > 0:
        idx = max(0, min(cfg.fgs1_bin_index, bins - 1))
        w[..., idx] = cfg.fgs1_weight
    return w


def build_composite_loss(cfg: LossConfig):
    """
    Returns a callable:
        loss_fn(pred, target, *, bin_weights=None, mask=None, reduction='mean') -> (loss, metrics)

    Args:
        pred: Tuple[mu, sigma] with shapes [B, BINS].
        target: Tensor [B, BINS].
        bin_weights: optional weights (broadcastable to [B, BINS]).
        mask: optional mask (broadcastable to [B, BINS]), 1=use, 0=ignore.
        reduction: 'mean' | 'sum' over the batch for scalar loss; metrics reported as floats.

    Notes:
        • Gaussian NLL (GLL) matches leaderboard metric; FGS1 bin ~58× by default.
        • Smoothness (TV, curvature) discourages unphysical jagged spectra.
        • Nonnegativity enforces μ ≥ 0 softly; you may also set clamp_mu_min for hard floor.
        • Calibration penalty encourages honest σ (z^2 mean ≈ 1).
        • Sigma prior discourages exploding/vanishing σ.
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
        assert mu.shape == sigma.shape == target.shape, "mu/sigma/target must have same [B, BINS] shape"
        B, BINS = mu.shape
        device, dtype = mu.device, mu.dtype

        # Optional hard clamp on μ (soft nonneg is a separate term)
        if cfg.clamp_mu_min is not None or cfg.clamp_mu_max is not None:
            lo = cfg.clamp_mu_min if cfg.clamp_mu_min is not None else -float("inf")
            hi = cfg.clamp_mu_max if cfg.clamp_mu_max is not None else float("inf")
            mu = mu.clamp(min=lo, max=hi)

        # Prepare weights (FGS1 up-weight if none provided)
        if bin_weights is None:
            bin_weights = _default_bin_weights(cfg, BINS, device, dtype)  # [1, BINS]

        # --- Gaussian NLL (primary)
        gll_vec = gaussian_log_likelihood(
            mu, sigma, target,
            clamp_sigma_min=cfg.clamp_sigma_min,
            clamp_sigma_max=cfg.clamp_sigma_max,
            mask=mask,
            bin_weights=bin_weights,
            normalize_by_valid=cfg.normalize_by_valid,
        )  # [B]

        total_vec = cfg.w_gll * gll_vec
        terms_vec: Dict[str, torch.Tensor] = {"gll": gll_vec}

        # --- TV
        if cfg.w_tv != 0.0:
            tv_vec = tv_penalty(mu, eps=cfg.tv_eps, mask=mask, normalize_by_valid=cfg.normalize_by_valid)
            total_vec = total_vec + cfg.w_tv * tv_vec
            terms_vec["tv"] = tv_vec

        # --- Curvature
        if cfg.w_curv != 0.0:
            curv_vec = curvature_penalty(mu, mask=mask, normalize_by_valid=cfg.normalize_by_valid)
            total_vec = total_vec + cfg.w_curv * curv_vec
            terms_vec["curv"] = curv_vec

        # --- Nonnegativity
        if cfg.w_nonneg != 0.0:
            nn_vec = nonneg_penalty(mu, mask=mask, normalize_by_valid=cfg.normalize_by_valid)
            total_vec = total_vec + cfg.w_nonneg * nn_vec
            terms_vec["nonneg"] = nn_vec

        # --- Coverage/Calibration
        if cfg.w_calib != 0.0:
            calib_vec = coverage_calibration_penalty(
                mu, sigma, target,
                clamp_sigma_min=cfg.clamp_sigma_min,
                clamp_sigma_max=cfg.clamp_sigma_max,
                mask=mask,
            )
            # Center target is cfg.calib_target (default 1.0)
            calib_vec = (calib_vec - abs(cfg.calib_target - 1.0)) + abs(cfg.calib_target - 1.0)
            total_vec = total_vec + cfg.w_calib * calib_vec
            terms_vec["calib"] = calib_vec

        # --- Sigma Prior
        if cfg.w_sigma_prior != 0.0:
            sigp_vec = sigma_prior_penalty(
                sigma, log_prior_mean=cfg.sigma_prior_log_mean,
                reduction=cfg.sigma_prior_reduction, mask=mask
            )
            total_vec = total_vec + cfg.w_sigma_prior * sigp_vec
            terms_vec["sigma_prior"] = sigp_vec

        # Batch reduction
        if reduction == "mean":
            total = total_vec.mean()
            metrics = {k: float(v.mean().detach().cpu()) for k, v in terms_vec.items()}
        elif reduction == "sum":
            total = total_vec.sum()
            metrics = {k: float(v.sum().detach().cpu()) for k, v in terms_vec.items()}
        else:
            raise ValueError(f"Unsupported reduction: {reduction}")

        # Extra telemetry
        metrics["loss"] = float(total.detach().cpu())
        # Bins used per sample (average)
        if mask is not None:
            metrics["bins_used_mean"] = float(mask.sum(dim=-1).float().mean().detach().cpu())
        else:
            metrics["bins_used_mean"] = float(torch.tensor(BINS, device=device, dtype=dtype).item())

        # Effective FGS1 weight (useful to confirm the up-weight is applied)
        if cfg.use_fgs1_weight and BINS > 0:
            fgs1_idx = max(0, min(cfg.fgs1_bin_index, BINS - 1))
            w_eff = bin_weights[..., fgs1_idx].mean()
            metrics["fgs1_weight_eff"] = float(w_eff.detach().cpu())

        return total, metrics

    return loss_fn