# src/spectramind/models/decoder_head.py
from __future__ import annotations

"""
SpectraMind V50 — Heteroscedastic Decoder Head
===============================================

A lightweight PyTorch head that maps fused features → per-wavelength
mean (μ) and uncertainty (σ) over `n_bins` (default: 283), with
physics-aware constraints and optional losses:

- Gaussian NLL (heteroscedastic)
- Smoothness penalty (μ) to encourage spectral continuity
- Non-negativity penalty (σ)
- Band coherence penalty (μ local variation)
- FGS1 up-weighting for the first bin (default ~58× relevance)

This module is intentionally decoupled from the encoder stack so it can
be swapped without touching the rest of the model.

Typical usage
-------------
>>> head = DecoderHead(in_dim=512, n_bins=283)
>>> mu, sigma = head(x)  # x: [B, in_dim]
>>> criterion = HeteroscedasticSpectralLoss(n_bins=283, fgs1_weight=58.0)
>>> loss = criterion(mu=mu, sigma=sigma, target=target_mu)   # target_mu: [B, 283]
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------- Head definition ---------------------------------


class DecoderHead(nn.Module):
    """
    Heteroscedastic decoder head that predicts per-bin μ and σ.

    Features:
    - Two independent linear projections for μ and raw σ (pre-activation)
    - σ constrained via softplus + eps floor (numerical stability)
    - Optional residual MLP (single hidden) for extra capacity
    - Optional per-bin affine calibration (γ, β) on μ

    Args
    ----
    in_dim:
        Input feature dimension (from fusion layer).
    n_bins:
        Number of spectral bins (default: 283).
    hidden_dim:
        Optional hidden layer size for a 1x MLP before heads. If None, uses
        single linear layers directly on `in_dim`.
    activation:
        Activation to use in the optional hidden MLP (default: GELU).
    init_sigma:
        Initial σ scale (used to bias raw σ head so softplus(init) ≈ init_sigma).
    min_sigma:
        Minimum σ (soft floor via clamping after softplus).
    use_calibration:
        If True, applies per-bin affine calibration to μ: μ ← γ ⊙ μ + β.
    """

    def __init__(
        self,
        in_dim: int,
        n_bins: int = 283,
        hidden_dim: Optional[int] = None,
        activation: Optional[nn.Module] = None,
        init_sigma: float = 0.03,
        min_sigma: float = 1e-4,
        use_calibration: bool = True,
    ) -> None:
        super().__init__()
        self.n_bins = int(n_bins)
        self.min_sigma = float(min_sigma)
        self.use_calibration = bool(use_calibration)

        act = activation if activation is not None else nn.GELU()

        if hidden_dim is not None and hidden_dim > 0:
            self.backbone = nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                act,
                nn.LayerNorm(hidden_dim),
            )
            feat_dim = hidden_dim
        else:
            self.backbone = nn.Identity()
            feat_dim = in_dim

        # Heads
        self.mu_head = nn.Linear(feat_dim, self.n_bins)
        self.sigma_head = nn.Linear(feat_dim, self.n_bins)

        # Initialize σ bias so softplus(bias) ≈ init_sigma
        # Solve for bias ~ softplus^{-1}(init_sigma)
        init_bias = torch.as_tensor(init_sigma).log().exp()  # crude warmup
        # Better inverse-softplus approx:
        init_bias = torch.log(torch.expm1(torch.as_tensor(init_sigma)))
        with torch.no_grad():
            self.sigma_head.bias.copy_(init_bias.expand(self.n_bins))

        # Optional per-bin affine calibration for μ
        if self.use_calibration:
            self.mu_gamma = nn.Parameter(torch.ones(self.n_bins))
            self.mu_beta = nn.Parameter(torch.zeros(self.n_bins))
        else:
            self.register_parameter("mu_gamma", None)
            self.register_parameter("mu_beta", None)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor [B, in_dim]
            Fused latent features.

        Returns
        -------
        mu : Tensor [B, n_bins]
            Predicted mean spectrum per bin.
        sigma : Tensor [B, n_bins]
            Predicted uncertainty per bin (strictly positive).
        """
        h = self.backbone(x)
        mu = self.mu_head(h)  # [B, n_bins]
        raw_sigma = self.sigma_head(h)  # [B, n_bins]

        # Enforce σ > 0 with a softplus + clamp floor
        sigma = F.softplus(raw_sigma)
        sigma = torch.clamp(sigma, min=self.min_sigma)

        # Optional per-bin affine calibration on μ
        if self.use_calibration:
            mu = mu * self.mu_gamma + self.mu_beta

        return mu, sigma


# ------------------------------- Loss config ----------------------------------


@dataclass
class SpectralLossConfig:
    """
    Configuration for heteroscedastic spectral loss and penalties.

    Attributes
    ----------
    n_bins:
        Number of spectral bins.
    fgs1_weight:
        Up-weight factor for the first bin (FGS1 broadband). Set to 1.0 to disable.
    eps:
        Numerical stability epsilon for NLL.
    smoothness_lambda:
        Weight for μ smoothness penalty (first derivative).
    curvature_lambda:
        Optional weight for μ curvature penalty (second derivative). Set 0 to disable.
    nonneg_sigma_lambda:
        L1 penalty weight encouraging σ ≥ 0 (post softplus this is typically near 0).
        Kept here as a gentle regularizer towards small σ without collapsing.
    band_coherence_lambda:
        Penalty weight for local band incoherence (higher-order L2 drift).
    """

    n_bins: int = 283
    fgs1_weight: float = 58.0
    eps: float = 1e-8
    smoothness_lambda: float = 1e-3
    curvature_lambda: float = 0.0
    nonneg_sigma_lambda: float = 1e-6
    band_coherence_lambda: float = 0.0


# ------------------------------ Loss function ---------------------------------


class HeteroscedasticSpectralLoss(nn.Module):
    """
    Gaussian NLL with physics/regularization terms for spectra.

    Loss = NLL(mu, sigma | target)
           + λ_smooth * smoothness(μ)
           + λ_curv   * curvature(μ)            [optional]
           + λ_nonneg * ||σ||₁                  [post-softplus σ≥0, acts as shrinkage]
           + λ_coh    * band_coherence(μ)

    Up-weights FGS1 (bin 0) via `fgs1_weight`.

    Notes
    -----
    - Assumes `target` is the ground-truth μ only (per challenge rules).
    - σ is model-predicted aleatoric uncertainty.
    """

    def __init__(self, n_bins: int = 283, fgs1_weight: float = 58.0, **kwargs: Any) -> None:
        super().__init__()
        self.cfg = SpectralLossConfig(n_bins=n_bins, fgs1_weight=fgs1_weight, **kwargs)
        # Precompute per-bin weights (FGS1 emphasis)
        w = torch.ones(self.cfg.n_bins)
        if self.cfg.fgs1_weight > 1.0:
            w[0] = self.cfg.fgs1_weight
        self.register_buffer("_w", w, persistent=False)

    @property
    def device(self) -> torch.device:
        return self._w.device  # type: ignore[attr-defined]

    def forward(
        self,
        *,
        mu: torch.Tensor,
        sigma: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute loss and diagnostics.

        Parameters
        ----------
        mu : Tensor [B, n_bins]
            Predicted mean.
        sigma : Tensor [B, n_bins]
            Predicted std; must be strictly positive.
        target : Tensor [B, n_bins]
            Ground-truth mean spectrum.
        mask : Tensor [B, n_bins], optional
            Binary mask for valid bins (1) vs. missing (0). Broadcastable to [B, n_bins].

        Returns
        -------
        dict with:
            - loss: scalar loss
            - nll: scalar NLL
            - reg_smooth: smoothness penalty
            - reg_curv: curvature penalty (if enabled)
            - reg_nonneg_sigma: sigma shrinkage
            - reg_coherence: band coherence penalty
        """
        assert mu.shape == sigma.shape == target.shape, "mu/sigma/target must share shape [B, n_bins]"
        B, n_bins = mu.shape
        assert n_bins == self.cfg.n_bins, f"expected n_bins={self.cfg.n_bins}, got {n_bins}"

        # Weights
        w = self._w.to(mu.dtype).to(mu.device)  # [n_bins]
        w = w.unsqueeze(0).expand(B, -1)        # [B, n_bins]
        if mask is not None:
            w = w * mask.to(mu.dtype)

        # Gaussian NLL per-bin (avoid log(0) with eps)
        sigma_safe = torch.clamp(sigma, min=self.cfg.eps)
        diff2 = (mu - target) ** 2
        nll_per_bin = 0.5 * (torch.log(2 * torch.pi * sigma_safe**2) + diff2 / (sigma_safe**2))
        nll = ((nll_per_bin * w).sum(dim=1) / (w.sum(dim=1).clamp_min(self.cfg.eps))).mean()

        # Regularizers
        reg_smooth = self._smoothness(mu, w) if self.cfg.smoothness_lambda > 0 else mu.new_tensor(0.0)
        reg_curv = self._curvature(mu, w) if self.cfg.curvature_lambda > 0 else mu.new_tensor(0.0)
        reg_nonneg_sigma = self.cfg.nonneg_sigma_lambda * sigma.abs().mean() if self.cfg.nonneg_sigma_lambda > 0 else mu.new_tensor(0.0)
        reg_coherence = self._band_coherence(mu, w) if self.cfg.band_coherence_lambda > 0 else mu.new_tensor(0.0)

        loss = (
            nll
            + self.cfg.smoothness_lambda * reg_smooth
            + self.cfg.curvature_lambda * reg_curv
            + reg_nonneg_sigma
            + self.cfg.band_coherence_lambda * reg_coherence
        )

        return {
            "loss": loss,
            "nll": nll.detach(),
            "reg_smooth": reg_smooth.detach(),
            "reg_curv": reg_curv.detach(),
            "reg_nonneg_sigma": reg_nonneg_sigma.detach(),
            "reg_coherence": reg_coherence.detach(),
        }

    # -------------------------- Regularizers ----------------------------------

    @staticmethod
    def _smoothness(mu: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        First-derivative smoothness penalty on μ (L2).

        Encourages μ[:, i] ~ μ[:, i-1]. Uses masked per-bin weighting.
        """
        # μ' ≈ μ[i] - μ[i-1]
        d1 = mu[:, 1:] - mu[:, :-1]               # [B, n_bins-1]
        w1 = (w[:, 1:] * w[:, :-1])               # joint validity
        val = ((d1**2) * w1).sum(dim=1) / (w1.sum(dim=1).clamp_min(1.0))
        return val.mean()

    @staticmethod
    def _curvature(mu: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Second-derivative (curvature) penalty on μ (L2).

        Encourages μ[:, i] ~ (μ[:, i-1] + μ[:, i+1]) / 2.
        """
        if mu.shape[1] < 3:
            return mu.new_tensor(0.0)
        d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]  # [B, n_bins-2]
        w2 = (w[:, 2:] * w[:, 1:-1] * w[:, :-2])
        val = ((d2**2) * w2).sum(dim=1) / (w2.sum(dim=1).clamp_min(1.0))
        return val.mean()

    def _band_coherence(self, mu: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        """
        Penalize incoherent local drift beyond a short window.

        Implements an L2 penalty on the difference between immediate slope
        and average local slope (window=5).
        """
        B, N = mu.shape
        if N < 6:
            return mu.new_tensor(0.0)
        # Local slope
        d1 = mu[:, 1:] - mu[:, :-1]  # [B, N-1]
        # Moving average slope with small window
        k = 5
        pad = (k - 1) // 2
        # Average via conv1d on channel dimension
        filt = torch.ones(1, 1, k, device=mu.device, dtype=mu.dtype) / k
        d1_pad = F.pad(d1.unsqueeze(1), (pad, pad), mode="replicate")  # [B,1,N-1+2pad]
        d1_ma = F.conv1d(d1_pad, filt).squeeze(1)                      # [B, N-1]
        # Difference between slope and local mean slope
        incoh = (d1 - d1_ma) ** 2                                      # [B, N-1]
        w_local = (w[:, 1:] * w[:, :-1])
        val = (incoh * w_local).sum(dim=1) / (w_local.sum(dim=1).clamp_min(1.0))
        return val.mean()


# ---------------------------- Convenience factory -----------------------------


def build_decoder_and_loss(
    *,
    in_dim: int,
    n_bins: int = 283,
    hidden_dim: Optional[int] = None,
    activation: Optional[nn.Module] = None,
    init_sigma: float = 0.03,
    min_sigma: float = 1e-4,
    use_calibration: bool = True,
    fgs1_weight: float = 58.0,
    smoothness_lambda: float = 1e-3,
    curvature_lambda: float = 0.0,
    nonneg_sigma_lambda: float = 1e-6,
    band_coherence_lambda: float = 0.0,
) -> Tuple[DecoderHead, HeteroscedasticSpectralLoss]:
    """
    Convenience builder to keep model wiring tidy.

    Returns
    -------
    (head, criterion)
    """
    head = DecoderHead(
        in_dim=in_dim,
        n_bins=n_bins,
        hidden_dim=hidden_dim,
        activation=activation,
        init_sigma=init_sigma,
        min_sigma=min_sigma,
        use_calibration=use_calibration,
    )
    criterion = HeteroscedasticSpectralLoss(
        n_bins=n_bins,
        fgs1_weight=fgs1_weight,
        smoothness_lambda=smoothness_lambda,
        curvature_lambda=curvature_lambda,
        nonneg_sigma_lambda=nonneg_sigma_lambda,
        band_coherence_lambda=band_coherence_lambda,
    )
    return head, criterion
