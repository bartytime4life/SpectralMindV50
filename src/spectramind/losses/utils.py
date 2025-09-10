# src/spectramind/losses/utils.py
from __future__ import annotations
from typing import Literal, Optional, Tuple
import torch
from torch import Tensor


def make_fgs1_weights(n_bins: int, fgs1_weight: float = 58.0, device=None) -> Tensor:
    """
    Default per-bin weights with FGS1 (bin 0) up-weighted, per challenge spec (~58×).
    """
    w = torch.ones(n_bins, device=device)
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
    Resolve one of {sigma, var, log_var} into (var, log_var).
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


def apply_mask_and_weights(x: Tensor, mask: Optional[Tensor], weights: Optional[Tensor]) -> Tensor:
    """
    Broadcast and apply mask/weights to a per-bin tensor [..., C].
    """
    if mask is not None:
        x = x * mask.to(dtype=x.dtype)
    if weights is not None:
        w = weights
        while w.dim() < x.dim():
            w = w.unsqueeze(0)
        x = x * w.to(dtype=x.dtype, device=x.device)
    return x


def reduce_loss(
    x: Tensor,
    reduction: Literal["none", "mean", "sum"] = "mean",
    normalizer: Optional[Tensor] = None,
) -> Tensor:
    """
    Reduction helper. If mean with a provided normalizer, uses sum/normalizer.
    """
    if reduction == "none":
        return x
    if reduction == "sum":
        return x.sum()
    # mean
    if normalizer is not None:
        return x.sum() / normalizer.clamp_min(1.0)
    return x.mean()
