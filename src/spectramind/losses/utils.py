# src/spectramind/losses/utils.py
from __future__ import annotations
from typing import Literal, Optional, Tuple

import torch
from torch import Tensor


def make_fgs1_weights(
    n_bins: int,
    fgs1_weight: float = 58.0,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
) -> Tensor:
    """
    Build default per-bin weights with the FGS1 (bin 0) up-weighted.

    Args:
        n_bins: number of spectral bins (e.g., 283).
        fgs1_weight: multiplicative weight for bin 0 (~58× by challenge metric).
        device: optional torch device.
        dtype: optional dtype.

    Returns:
        Tensor [n_bins] with w[0] = fgs1_weight, others = 1.0
    """
    w = torch.ones(n_bins, device=device, dtype=dtype)
    if n_bins > 0 and fgs1_weight != 1.0:
        w[0] = float(fgs1_weight)
    return w


def ensure_var(
    *,
    sigma: Optional[Tensor] = None,
    var: Optional[Tensor] = None,
    log_var: Optional[Tensor] = None,
    min_sigma: float = 1e-6,
) -> Tuple[Tensor, Tensor]:
    """
    Resolve one (and only one) of {sigma, var, log_var} into (var, log_var).

    Args:
        sigma: standard deviation tensor (σ).
        var: variance tensor (σ²).
        log_var: log-variance tensor (log σ²).
        min_sigma: strictly positive floor used to clamp σ / √var.

    Returns:
        (var, log_var) tensors broadcastable to the same shape as the inputs.

    Raises:
        ValueError if none or more than one of {sigma, var, log_var} is provided.
    """
    provided = [sigma is not None, var is not None, log_var is not None]
    if sum(provided) != 1:
        raise ValueError("Provide exactly one of {sigma, var, log_var}.")

    if var is not None:
        v = var
        lv = torch.log(v.clamp_min(min_sigma**2))
        return v, lv

    if sigma is not None:
        s = sigma.clamp_min(min_sigma)
        v = s * s
        lv = torch.log(v)
        return v, lv

    # log_var provided
    lv = log_var
    v = torch.exp(lv).clamp_min(min_sigma**2)
    return v, lv


def apply_mask_and_weights(
    x: Tensor,
    mask: Optional[Tensor],
    weights: Optional[Tensor],
) -> Tensor:
    """
    Broadcast and apply mask/weights to a per-bin tensor [..., C].

    Notes:
        - `mask` and `weights` are multiplied into x. They are broadcast to x's shape.
        - If both are provided, they are multiplied together (effective weight).

    Args:
        x: input tensor [..., C]
        mask: optional mask [..., C] (0=ignore, 1=keep) or soft weights
        weights: optional per-bin weights [C] or [..., C]

    Returns:
        Tensor with mask/weights applied.
    """
    out = x
    if mask is not None:
        m = mask
        # broadcast mask to out's shape
        while m.dim() < out.dim():
            m = m.unsqueeze(0)
        out = out * m.to(dtype=out.dtype, device=out.device)

    if weights is not None:
        w = weights
        while w.dim() < out.dim():
            w = w.unsqueeze(0)
        out = out * w.to(dtype=out.dtype, device=out.device)

    return out


def reduce_loss(
    x: Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
    normalizer: Optional[Tensor] = None,
) -> Tensor:
    """
    Reduction helper for per-element losses.

    Args:
        x: per-element loss tensor (any shape).
        reduction: 'none' | 'mean' | 'sum'
        normalizer: optional positive scalar/tenor to use when averaging (e.g., sum of weights).

    Returns:
        Reduced tensor per the requested reduction.
    """
    if reduction == "none":
        return x
    if reduction == "sum":
        return x.sum()
    # mean
    if normalizer is not None:
        return x.sum() / normalizer.clamp_min(1.0)
    return x.mean()
