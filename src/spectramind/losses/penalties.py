# src/spectramind/losses/penalties.py
from __future__ import annotations
from typing import Literal, Optional, Sequence
import torch
from torch import Tensor
from .utils import ensure_var, apply_mask_and_weights, reduce_loss


def smoothness_penalty(
    mu: Tensor,
    *,
    mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """
    L2 curvature (discrete 2nd derivative) penalty over spectral bins.

    curv[i] = mu[i-1] - 2*mu[i] + mu[i+1]; penalty = ||curv||^2
    """
    pad_left = mu[..., :1]
    pad_right = mu[..., -1:]
    mu_pad = torch.cat([pad_left, mu, pad_right], dim=-1)  # [..., C+2]
    curv = mu_pad[..., :-2] - 2.0 * mu_pad[..., 1:-1] + mu_pad[..., 2:]  # [..., C]
    pen = curv**2

    pen = apply_mask_and_weights(pen, mask, weights)
    normalizer = None
    if reduction == "mean" and (mask is not None or weights is not None):
        norm = torch.ones_like(pen)
        norm = apply_mask_and_weights(norm, mask, weights)
        normalizer = norm.sum()
    return reduce_loss(pen, reduction=reduction, normalizer=normalizer)


def nonnegativity_penalty(
    mu: Tensor,
    *,
    mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """
    Penalize negative transit depths: ReLU(-μ)^2
    """
    neg = torch.relu(-mu)
    pen = neg**2

    pen = apply_mask_and_weights(pen, mask, weights)
    normalizer = None
    if reduction == "mean" and (mask is not None or weights is not None):
        norm = torch.ones_like(pen)
        norm = apply_mask_and_weights(norm, mask, weights)
        normalizer = norm.sum()
    return reduce_loss(pen, reduction=reduction, normalizer=normalizer)


def band_coherence_penalty(
    mu: Tensor,
    *,
    band_slices: Optional[Sequence[slice]] = None,
    mode: Literal["tv", "l2"] = "tv",
    reduction: Literal["none", "mean", "sum"] = "mean",
) -> Tensor:
    """
    Encourage coherence within bands.
      - If band_slices is None: apply across the whole spectrum.
      - 'tv': smooth TV ≈ sum sqrt((Δμ)^2 + ε)
      - 'l2': sum (Δμ)^2
    """
    eps = 1e-12

    def _penalize(x: Tensor) -> Tensor:
        d = x[..., 1:] - x[..., :-1]
        return torch.sqrt(d * d + eps).sum(dim=-1) if mode == "tv" else (d * d).sum(dim=-1)

    if band_slices is None:
        pen = _penalize(mu)
    else:
        parts = []
        for s in band_slices:
            if (s.stop - s.start) >= 2:
                parts.append(_penalize(mu[..., s]))
        if not parts:
            return mu.new_tensor(0.0)
        pen = torch.stack(parts, dim=-1).sum(dim=-1)

    return reduce_loss(pen, reduction=reduction)


def calibration_penalty(
    mu: Tensor,
    *,
    y: Tensor,
    sigma: Optional[Tensor] = None,
    var: Optional[Tensor] = None,
    log_var: Optional[Tensor] = None,
    mask: Optional[Tensor] = None,
    weights: Optional[Tensor] = None,
    reduction: Literal["none", "mean", "sum"] = "mean",
    min_sigma: float = 1e-6,
) -> Tensor:
    """
    Penalize variance miscalibration vs residual power:
      L = ( (y-μ)^2 - σ^2 )^2 + ( log((y-μ)^2+ε) - log σ^2 )^2
    """
    v, lv = ensure_var(sigma=sigma, var=var, log_var=log_var, min_sigma=min_sigma)
    resid2 = (y - mu) ** 2

    lin = (resid2 - v) ** 2
    log = (torch.log(resid2 + 1e-12) - lv) ** 2
    pen = lin + log

    pen = apply_mask_and_weights(pen, mask, weights)
    normalizer = None
    if reduction == "mean" and (mask is not None or weights is not None):
        norm = torch.ones_like(pen)
        norm = apply_mask_and_weights(norm, mask, weights)
        normalizer = norm.sum()
    return reduce_loss(pen, reduction=reduction, normalizer=normalizer)
