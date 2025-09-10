# src/spectramind/losses/gaussian.py
from __future__ import annotations
from typing import Literal, Optional
import torch
from torch import Tensor
from .utils import ensure_var, apply_mask_and_weights, reduce_loss


def gaussian_nll(
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
    Heteroscedastic Gaussian negative log-likelihood.

    NLL_i = 0.5 * [ log(2π σ_i^2) + (y_i - μ_i)^2 / σ_i^2 ]

    Shapes:
      mu, y: [..., C]
      sigma OR var OR log_var: [..., C]
      mask: [..., C] (0/1)
      weights: [C] or [..., C] (FGS1 up-weight, etc.)
    """
    v, lv = ensure_var(sigma=sigma, var=var, log_var=log_var, min_sigma=min_sigma)

    resid2 = (y - mu) ** 2
    nll = 0.5 * (torch.log(2.0 * torch.pi) + lv + resid2 / v.clamp_min(min_sigma**2))
    nll = apply_mask_and_weights(nll, mask, weights)

    normalizer = None
    if reduction == "mean" and (mask is not None or weights is not None):
        norm = torch.ones_like(nll)
        norm = apply_mask_and_weights(norm, mask, weights)
        normalizer = norm.sum()

    return reduce_loss(nll, reduction=reduction, normalizer=normalizer)
