# src/spectramind/losses/nonneg.py
from __future__ import annotations

"""
Non-negativity & bounds penalties for spectral predictions.

Typical use for SpectraMind V50:
  - Encourage μ >= 0 across all 283 bins
  - Optionally also keep μ <= 1 (or any domain upper bound)
  - Smooth surrogates to avoid brittle gradients

Supports:
  - modes: "soft_hinge" (softplus), "hinge", "relu", "log_barrier"
  - optional upper bound penalty (same modes)
  - masks to exclude bins or weight subsets
  - p-norm style penalties via 'p' (for hinge/relu/soft_hinge)

All functions are torch-only and can be composed inside any training loop or LightningModule.
"""

from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch import Tensor, nn

__all__ = [
    "NonNegConfig",
    "nonneg_penalty",
    "bounds_penalty",
    "NonNegLoss",
    "BoundsLoss",
]


# =================================================================================================
# Config
# =================================================================================================

@dataclass
class NonNegConfig:
    """Configuration for non-negativity penalties."""
    # Lower bound (usually 0.0)
    lower: float = 0.0
    # Optional upper bound (set to None to disable)
    upper: float | None = None
    # Penalty mode: "soft_hinge" | "hinge" | "relu" | "log_barrier"
    mode: str = "soft_hinge"
    # p-exponent for hinge/relu/soft_hinge penalties (e.g. 1 for L1-like, 2 for L2-like)
    p: float = 2.0
    # Smoothing parameter for soft variants
    beta: float = 1.0
    # Epsilon for log-barrier & numerical safety
    eps: float = 1e-6
    # Global weight multiplier
    weight: float = 1.0
    # Reduction: "mean" | "sum" | "none"
    reduction: str = "mean"


# =================================================================================================
# Core penalty helpers
# =================================================================================================

def _reduce(x: Tensor, reduction: str) -> Tensor:
    if reduction == "none":
        return x
    if reduction == "sum":
        return x.sum()
    return x.mean()


def _soft_hinge(x: Tensor, margin: float, p: float = 2.0, beta: float = 1.0) -> Tensor:
    """
    Soft hinge: softplus(margin - x) ** p
    - Smoothly penalizes values below 'margin' (default margin=0)
    """
    z = beta * (margin - x)
    return torch.nn.functional.softplus(z) ** p


def _hinge(x: Tensor, margin: float, p: float = 2.0) -> Tensor:
    """
    Classic hinge: relu(margin - x) ** p
    """
    return torch.relu(margin - x) ** p


def _relu_neg(x: Tensor, margin: float, p: float = 2.0) -> Tensor:
    """
    Penalize only negative part relative to 'margin':
    relu(margin - x) ** p, identical to hinge but named for clarity.
    """
    return torch.relu(margin - x) ** p


def _log_barrier(x: Tensor, lower: float, eps: float = 1e-6) -> Tensor:
    """
    Log-barrier for x >= lower:
      penalty = -log((x - lower) + eps) for x near/below lower.
    This diverges as x -> lower-; combine with soft penalties for stability.
    """
    return -torch.log((x - lower).clamp_min(0.0) + eps)


def nonneg_penalty(
    x: Tensor,
    *,
    lower: float = 0.0,
    mode: str = "soft_hinge",
    p: float = 2.0,
    beta: float = 1.0,
    eps: float = 1e-6,
    mask: Optional[Tensor] = None,
    weight: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Penalize x falling below 'lower'.

    Parameters
    ----------
    x : Tensor
        Input values, e.g. μ predictions of shape (B, Nλ).
    lower : float
        Lower bound to enforce (default: 0).
    mode : str
        "soft_hinge" (default), "hinge", "relu", "log_barrier".
    p : float
        Exponent for hinge-like penalties.
    beta : float
        Smoothness multiplier for softplus in "soft_hinge".
    eps : float
        Numerical epsilon for 'log_barrier'.
    mask : Optional[Tensor]
        Broadcastable mask (1=active; 0=off) or per-entry weights.
    weight : float
        Global weight multiplier applied after reduction.
    reduction : str
        "mean" | "sum" | "none".

    Returns
    -------
    Tensor
        Scalar loss (if reduction != "none") or per-element penalties.
    """
    if mode not in {"soft_hinge", "hinge", "relu", "log_barrier"}:
        raise ValueError(f"Unknown mode '{mode}'")

    if mode == "soft_hinge":
        pen = _soft_hinge(x, margin=lower, p=p, beta=beta)
    elif mode == "hinge":
        pen = _hinge(x, margin=lower, p=p)
    elif mode == "relu":
        pen = _relu_neg(x, margin=lower, p=p)
    else:  # "log_barrier"
        pen = _log_barrier(x, lower=lower, eps=eps)

    if mask is not None:
        pen = pen * mask

    return weight * _reduce(pen, reduction)


def bounds_penalty(
    x: Tensor,
    *,
    lower: float = 0.0,
    upper: float | None = None,
    mode: str = "soft_hinge",
    p: float = 2.0,
    beta: float = 1.0,
    eps: float = 1e-6,
    mask: Optional[Tensor] = None,
    weight: float = 1.0,
    reduction: str = "mean",
) -> Tensor:
    """
    Penalize x outside [lower, upper]. If upper=None, only enforce lower bound.

    Lower side uses 'mode'. Upper side mirrors the same idea by applying
    the penalty to 'x - upper' symmetrically.

    Example:
      bounds_penalty(mu, lower=0.0, upper=1.0, mode="soft_hinge")
      → encourages 0 ≤ μ ≤ 1 smoothly.
    """
    total: Tensor = torch.zeros_like(x, dtype=x.dtype)

    # lower bound
    total = total + nonneg_penalty(
        x,
        lower=lower,
        mode=mode,
        p=p,
        beta=beta,
        eps=eps,
        mask=mask,
        weight=1.0,
        reduction="none",
    )

    # upper bound (x <= upper)
    if upper is not None:
        if mode == "log_barrier":
            upper_pen = -torch.log((upper - x).clamp_min(0.0) + eps)
        elif mode == "soft_hinge":
            upper_pen = torch.nn.functional.softplus((x - upper) * beta) ** p
        elif mode in {"hinge", "relu"}:
            upper_pen = torch.relu(x - upper) ** p
        else:
            raise ValueError(f"Unknown mode '{mode}'")

        if mask is not None:
            upper_pen = upper_pen * mask
        total = total + upper_pen

    return weight * _reduce(total, reduction)


# =================================================================================================
# Modules
# =================================================================================================

class NonNegLoss(nn.Module):
    """
    nn.Module wrapper for non-negativity penalty (x >= lower).
    """

    def __init__(self, cfg: Optional[NonNegConfig] = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = NonNegConfig(**kwargs)
        self.cfg = cfg

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return nonneg_penalty(
            x,
            lower=self.cfg.lower,
            mode=self.cfg.mode,
            p=self.cfg.p,
            beta=self.cfg.beta,
            eps=self.cfg.eps,
            mask=mask,
            weight=self.cfg.weight,
            reduction=self.cfg.reduction,
        )


class BoundsLoss(nn.Module):
    """
    nn.Module wrapper for bounds penalty (lower <= x <= upper).

    Set upper=None to only enforce non-negativity.
    """

    def __init__(self, cfg: Optional[NonNegConfig] = None, **kwargs: Any) -> None:
        super().__init__()
        if cfg is None:
            cfg = NonNegConfig(**kwargs)
        self.cfg = cfg

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        return bounds_penalty(
            x,
            lower=self.cfg.lower,
            upper=self.cfg.upper,
            mode=self.cfg.mode,
            p=self.cfg.p,
            beta=self.cfg.beta,
            eps=self.cfg.eps,
            mask=mask,
            weight=self.cfg.weight,
            reduction=self.cfg.reduction,
        )


# =================================================================================================
# Quick smoke tests
# =================================================================================================

if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.tensor([[0.2, -0.1, 0.5, 1.2]], dtype=torch.float32)  # (1, 4)
    m = torch.tensor([[1.0, 1.0, 0.0, 1.0]], dtype=torch.float32)   # mask out third entry

    # Non-negativity (μ >= 0)
    loss1 = nonneg_penalty(x, lower=0.0, mode="soft_hinge", p=2.0, beta=2.0, mask=m, reduction="mean")
    print("nonneg soft_hinge:", float(loss1))

    # Bounds (0 ≤ μ ≤ 1)
    loss2 = bounds_penalty(x, lower=0.0, upper=1.0, mode="soft_hinge", p=2.0, beta=2.0, mask=m, reduction="mean")
    print("bounds soft_hinge:", float(loss2))

    mod = BoundsLoss(NonNegConfig(lower=0.0, upper=1.0, mode="hinge", p=2.0, weight=0.5))
    loss3 = mod(x, mask=m)
    print("module bounds hinge (w=0.5):", float(loss3))
