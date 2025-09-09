# src/spectramind/models/heads/hetero.py
# =============================================================================
# SpectraMind V50 — Heteroscedastic Gaussian Head
# -----------------------------------------------------------------------------
# Produces per-bin mean μ and uncertainty σ for the 283-bin spectrum (or any
# configurable number of bins). Designed for GLL-aligned training and physics-
# informed constraints:
#   • μ: unconstrained real output (post-constraints are handled by loss/penalties)
#   • σ: strictly-positive via softplus (or exp of log-σ), numerically clamped
#
# Utilities include a vectorized Gaussian NLL with an optional FGS1 up-weight
# (default ~58× for bin 0) and masking / reduction support.
# =============================================================================

from __future__ import annotations

from typing import Iterable, Literal, Optional, Sequence, Tuple, Union

import math
import torch
from torch import nn, Tensor
import torch.nn.functional as F


ActivationName = Literal["relu", "gelu", "silu", "tanh", "none"]
SigmaMode = Literal["std", "var", "log_var"]


def _make_activation(name: ActivationName) -> nn.Module:
    if name == "relu":
        return nn.ReLU(inplace=True)
    if name == "gelu":
        return nn.GELU()
    if name == "silu":
        return nn.SiLU(inplace=True)
    if name == "tanh":
        return nn.Tanh()
    return nn.Identity()


def _init_mlp(
    in_dim: int,
    hidden: Sequence[int],
    dropout: float,
    activation: ActivationName,
    layernorm: bool,
) -> nn.Sequential:
    layers: list[nn.Module] = []
    d_prev = in_dim
    for i, d in enumerate(hidden):
        layers.append(nn.Linear(d_prev, d))
        if layernorm:
            layers.append(nn.LayerNorm(d))
        layers.append(_make_activation(activation))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        d_prev = d
    return nn.Sequential(*layers)


def _kaiming_init_linear(m: nn.Module) -> None:
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
        if m.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(m.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(m.bias, -bound, bound)


class HeteroscedasticHead(nn.Module):
    """
    Heteroscedastic Gaussian prediction head.

    Parameters
    ----------
    in_dim : int
        Input feature dimension.
    out_bins : int, default 283
        Number of spectral bins (output channels).
    hidden : int | Iterable[int], default (512,)
        Hidden layer sizes for an optional MLP trunk. If empty or 0, uses a
        single linear projection from `in_dim` to outputs.
    activation : {"relu","gelu","silu","tanh","none"}, default "gelu"
        Nonlinearity for the MLP trunk.
    dropout : float, default 0.0
        Dropout rate applied after hidden layers.
    layernorm : bool, default False
        Apply LayerNorm after each hidden linear layer.
    sigma_activation : {"softplus","exp"}, default "softplus"
        Parameterization for σ. If "softplus", σ = softplus(raw_sigma) + eps.
        If "exp", the head predicts log-σ (unconstrained) and σ = exp(log-σ).
    sigma_min : float, default 1e-6
        Lower bound for σ for numerical stability.
    sigma_max : Optional[float], default None
        Optional upper bound for σ to avoid pathological uncertainties.
    init_log_sigma : float, default -1.0
        Bias init for the σ branch; if sigma_activation="exp", this is the
        initial log-σ; if "softplus", the bias is set so that softplus(bias)≈exp(init_log_sigma).
    fgs1_weight : float, default 58.0
        Default up-weighting factor for bin 0 in the NLL loss.
    return_sigma : {"std","var","log_var"}, default "std"
        Controls format returned in forward().

    Notes
    -----
    • Forward returns a dict with keys: "mu" and one of {"sigma","var","log_var"}.
    • Use `gaussian_nll` below for metric-aligned loss; it supports per-bin weights
      (including FGS1 up-weighting) and masking/broadcasting.
    """

    def __init__(
        self,
        in_dim: int,
        out_bins: int = 283,
        hidden: Union[int, Iterable[int]] = (512,),
        *,
        activation: ActivationName = "gelu",
        dropout: float = 0.0,
        layernorm: bool = False,
        sigma_activation: Literal["softplus", "exp"] = "softplus",
        sigma_min: float = 1e-6,
        sigma_max: Optional[float] = None,
        init_log_sigma: float = -1.0,
        fgs1_weight: float = 58.0,
        return_sigma: SigmaMode = "std",
    ) -> None:
        super().__init__()
        self.out_bins = int(out_bins)
        self.sigma_activation = sigma_activation
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max) if sigma_max is not None else None
        self.return_sigma_mode: SigmaMode = return_sigma
        self.fgs1_weight = float(fgs1_weight)

        if isinstance(hidden, int):
            hidden = () if hidden <= 0 else (hidden,)
        hidden = tuple(int(h) for h in hidden)

        # Trunk (optional MLP)
        self.trunk = _init_mlp(
            in_dim=in_dim,
            hidden=hidden,
            dropout=dropout,
            activation=activation,
            layernorm=layernorm,
        )
        trunk_out = hidden[-1] if len(hidden) > 0 else in_dim

        # Two linear heads: μ and σ-branch (raw)
        self.mu = nn.Linear(trunk_out, out_bins)
        self.raw_sigma = nn.Linear(trunk_out, out_bins)

        # Initialize
        self.apply(_kaiming_init_linear)
        nn.init.zeros_(self.mu.bias)

        # Initialize σ-branch bias for a sensible starting uncertainty.
        if sigma_activation == "exp":
            # raw_sigma outputs log-σ directly
            nn.init.constant_(self.raw_sigma.bias, init_log_sigma)
        else:
            # We want softplus(bias) ≈ exp(init_log_sigma)
            target = math.exp(init_log_sigma)
            # Inverse softplus: for y>0, x ≈ log(exp(y)-1). Protect for small y.
            inv_sp = math.log(math.expm1(target) + 1e-12)
            nn.init.constant_(self.raw_sigma.bias, inv_sp)

    def _sigma_from_raw(self, raw: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """
        Convert raw σ branch output to (sigma, var, log_var), with clamping.

        Returns
        -------
        sigma : Tensor
            Standard deviation, clamped to [sigma_min, sigma_max?].
        var : Tensor
            Variance, clamped accordingly.
        log_var : Tensor
            Log-variance (stable).
        """
        if self.sigma_activation == "exp":
            # raw = log-σ
            log_sigma = raw
            sigma = torch.exp(log_sigma)
        else:
            # raw -> σ via softplus
            sigma = F.softplus(raw)
            # Add a tiny epsilon to preserve strictly positive lower bound
            sigma = sigma + 1e-12
            log_sigma = torch.log(sigma)

        # Clamp σ for stability
        sigma = torch.clamp(sigma, min=self.sigma_min)
        if self.sigma_max is not None:
            sigma = torch.clamp(sigma, max=self.sigma_max)

        var = sigma * sigma
        log_var = torch.log(var)
        return sigma, var, log_var

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        """
        Forward pass.

        Parameters
        ----------
        x : Tensor, shape (B, *, in_dim)
            Input features; any leading batch-like dims are accepted.

        Returns
        -------
        out : dict
            Contains:
              • "mu": (B, *, out_bins)
              • one of {"sigma","var","log_var"} depending on `return_sigma`.
        """
        h = self.trunk(x)
        mu = self.mu(h)
        raw_s = self.raw_sigma(h)

        sigma, var, log_var = self._sigma_from_raw(raw_s)

        out: dict[str, Tensor] = {"mu": mu}
        if self.return_sigma_mode == "std":
            out["sigma"] = sigma
        elif self.return_sigma_mode == "var":
            out["var"] = var
        else:
            out["log_var"] = log_var
        return out

    # -------------------------------------------------------------------------
    # Training utilities
    # -------------------------------------------------------------------------
    @torch.no_grad()
    def default_weights(self, device: Optional[torch.device] = None) -> Tensor:
        """
        Build default per-bin weights with FGS1 up-weight at index 0.

        Returns
        -------
        w : Tensor, shape (out_bins,)
            w[0] = fgs1_weight; w[i>0] = 1.0
        """
        w = torch.ones(self.out_bins, device=device)
        if self.out_bins > 0 and self.fgs1_weight != 1.0:
            w[0] = self.fgs1_weight
        return w

    def gaussian_nll(
        self,
        mu: Tensor,
        sigma: Optional[Tensor] = None,
        *,
        var: Optional[Tensor] = None,
        y: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        reduction: Literal["none", "mean", "sum"] = "mean",
        eps: float = 1e-12,
    ) -> Tensor:
        """
        Vectorized Gaussian negative log-likelihood (per-bin), with optional
        mask and per-bin weights (supports FGS1 up-weighting).

        Parameters
        ----------
        mu : Tensor, shape (..., out_bins)
            Predicted mean.
        sigma : Tensor, shape (..., out_bins), optional
            Predicted std; ignored if `var` is provided.
        var : Tensor, shape (..., out_bins), optional
            Predicted variance; overrides `sigma` if provided.
        y : Tensor, shape (..., out_bins)
            Target spectrum.
        mask : Tensor, shape (..., out_bins), optional
            Boolean or {0,1} mask; masked=0 entries are ignored.
        weights : Tensor, shape (out_bins,) or (..., out_bins), optional
            Per-bin weights. If None, uses `default_weights()`.
        reduction : {"none","mean","sum"}, default "mean"
            Reduction over all dims.

        Returns
        -------
        loss : Tensor
            Scalar if reduced, else per-element NLL with weights/mask applied.

        Notes
        -----
        NLL = 0.5 * (log(2π * σ²) + (y-μ)² / σ²)
        """
        if y is None:
            raise ValueError("`y` (targets) must be provided to compute NLL.")

        if var is None:
            if sigma is None:
                raise ValueError("Provide either `sigma` or `var`.")
            var = sigma * sigma

        var = torch.clamp(var, min=(self.sigma_min ** 2))
        log_var = torch.log(var + eps)

        # Core per-element NLL
        nll = 0.5 * (math.log(2 * math.pi) + log_var + (y - mu) ** 2 / (var + eps))

        # Mask
        if mask is not None:
            mask = mask.to(nll.dtype)
            nll = nll * mask

        # Weights
        if weights is None:
            weights = self.default_weights(device=nll.device)
        # Broadcast weights if needed
        while weights.dim() < nll.dim():
            weights = weights.unsqueeze(0)
        nll = nll * weights

        if reduction == "none":
            return nll
        if reduction == "sum":
            return nll.sum()
        # mean over all valid elements; if masked, normalize by sum of mask*weights
        if mask is not None:
            denom = (mask * weights).sum().clamp_min(1.0)
            return nll.sum() / denom
        else:
            return nll.mean()

    def loss(
        self,
        outputs: dict[str, Tensor],
        y: Tensor,
        *,
        mask: Optional[Tensor] = None,
        weights: Optional[Tensor] = None,
        reduction: Literal["mean", "sum", "none"] = "mean",
    ) -> Tensor:
        """
        Convenience wrapper that reads μ/σ (or μ/var) from `outputs` and calls
        `gaussian_nll`.

        Parameters
        ----------
        outputs : dict
            Must contain "mu" and one of {"sigma","var","log_var"}.
        y : Tensor
            Target spectrum.
        mask : Tensor, optional
            Optional mask.
        weights : Tensor, optional
            Optional per-bin weights.
        reduction : {"mean","sum","none"}, default "mean"

        Returns
        -------
        loss : Tensor
        """
        mu = outputs["mu"]
        if "var" in outputs:
            return self.gaussian_nll(mu, var=outputs["var"], y=y, mask=mask, weights=weights, reduction=reduction)
        if "log_var" in outputs:
            var = torch.exp(outputs["log_var"])
            return self.gaussian_nll(mu, var=var, y=y, mask=mask, weights=weights, reduction=reduction)
        if "sigma" in outputs:
            return self.gaussian_nll(mu, sigma=outputs["sigma"], y=y, mask=mask, weights=weights, reduction=reduction)
        raise KeyError("`outputs` must contain one of {'sigma','var','log_var'}.")

    @torch.no_grad()
    def predict(
        self,
        x: Tensor,
        *,
        return_sigma: SigmaMode | None = None,
    ) -> dict[str, Tensor]:
        """
        Inference helper (no grad). Mirrors forward() but allows overriding the
        sigma return mode.

        Parameters
        ----------
        x : Tensor
            Input features.
        return_sigma : {"std","var","log_var"}, optional
            Override of the configured return mode.

        Returns
        -------
        dict with "mu" and uncertainty tensor.
        """
        ret_mode = return_sigma or self.return_sigma_mode
        prev = self.return_sigma_mode
        self.return_sigma_mode = ret_mode
        try:
            return self.forward(x)
        finally:
            self.return_sigma_mode = prev
