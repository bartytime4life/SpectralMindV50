from __future__ import annotations

"""
FusionXAttn — Cross-attention fusion model for SpectraMind V50.

This module fuses:
  • FGS1 temporal features  : (B, T, D_time)
  • AIRS spectral features  : (B, Nλ, D_spec)    # typically Nλ = 283

The AIRS tokens (queries) attend to FGS1 tokens (keys/values) via MultiheadAttention
and pass through a small Transformer-style FFN. The final per-bin tokens are projected
to (μ, σ) with σ enforced positive via softplus + floor. Optionally, if AIRS tokens are
not provided, learned spectral query tokens are used.

The module is torch-only (no Lightning), so it can be plugged into any training loop
or a LightningModule wrapper (e.g., SpectraSystem).
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch import Tensor, nn


__all__ = [
    "FusionConfig",
    "FusionXAttn",
    "gll_loss",
]


# --------------------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------------------


@dataclass
class FusionConfig:
    bins: int = 283               # number of wavelength bins
    d_time: int = 256             # input feature dim for time tokens (FGS1 encoder output)
    d_spec: int = 256             # input feature dim for spectral tokens (AIRS encoder output)
    d_model: int = 256            # internal model dimension for tokens
    nhead: int = 8                # multi-head attention heads
    num_layers: int = 2           # cross-attention blocks
    ff_mult: int = 4              # FFN width multiplier (d_ff = ff_mult * d_model)
    dropout: float = 0.1          # dropout
    use_learned_spectral_tokens: bool = False  # use learnable queries if AIRS tokens unavailable
    sigma_floor: float = 1e-5     # lower bound for σ after softplus
    # Optional positional encodings
    posenc_time: bool = True
    posenc_spec: bool = True


# --------------------------------------------------------------------------------------------------
# Positional Encoding (sinusoidal)
# --------------------------------------------------------------------------------------------------


class SinusoidalPositionalEncoding(nn.Module):
    """
    Classic transformer sinusoidal positional encoding.

    Inputs are shaped (B, L, D); the same PE is broadcast over batch.
    """

    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L, 1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # even indices
        pe[:, 1::2] = torch.cos(position * div_term)  # odd indices
        # register as buffer (non-trainable)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, L, D)
        """
        L = x.size(1)
        return x + self.pe[:L].unsqueeze(0)  # (1, L, D) + (B, L, D)


# --------------------------------------------------------------------------------------------------
# Cross-attention block: LN → MHA → Residual → LN → FFN → Residual
# --------------------------------------------------------------------------------------------------


class CrossAttnBlock(nn.Module):
    def __init__(self, d_model: int, nhead: int, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.q_ln = nn.LayerNorm(d_model)
        self.kv_ln = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=False)
        self.attn_drop = nn.Dropout(dropout)

        d_ff = ff_mult * d_model
        self.ffn_ln = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, query: Tensor, key: Tensor, value: Tensor) -> Tensor:
        """
        query: (B, Nq, D), key: (B, Nk, D), value: (B, Nk, D)
        returns: (B, Nq, D)
        """
        # Pre-norm
        q = self.q_ln(query)
        kv = self.kv_ln(key), self.kv_ln(value)  # share LN weights for key/value

        # MultiheadAttention expects (L, B, D); convert batch_first=False
        q_t = q.transpose(0, 1)       # (Nq, B, D)
        k_t = kv[0].transpose(0, 1)   # (Nk, B, D)
        v_t = kv[1].transpose(0, 1)   # (Nk, B, D)

        attn_out, _ = self.attn(q_t, k_t, v_t, need_weights=False)
        attn_out = attn_out.transpose(0, 1)  # (B, Nq, D)

        # Residual + FFN
        x = query + self.attn_drop(attn_out)
        x_ff = self.ffn(self.ffn_ln(x))
        return x + x_ff


# --------------------------------------------------------------------------------------------------
# Fusion Model
# --------------------------------------------------------------------------------------------------


class FusionXAttn(nn.Module):
    """
    FusionXAttn

    Parameters
    ----------
    cfg : FusionConfig

    Forward contracts
    -----------------
    • fgs1_tokens: (B, T, D_time)       # produced by FGS1 encoder
    • airs_tokens: (B, Nλ, D_spec) | None
        - if None and cfg.use_learned_spectral_tokens is True, learned queries are used
        - otherwise AIRS tokens are projected and used as queries

    Returns
    -------
    Dict with:
      "mu":    (B, Nλ)
      "sigma": (B, Nλ)  # positive
      "tokens":(B, Nλ, D)
    """

    def __init__(self, cfg: FusionConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.bins = cfg.bins

        # Projections to shared model dim
        self.proj_time = nn.Linear(cfg.d_time, cfg.d_model, bias=True)
        self.proj_spec = nn.Linear(cfg.d_spec, cfg.d_model, bias=True)

        # Positional encodings
        self.pe_time = SinusoidalPositionalEncoding(cfg.d_model) if cfg.posenc_time else nn.Identity()
        self.pe_spec = SinusoidalPositionalEncoding(cfg.d_model) if cfg.posenc_spec else nn.Identity()

        # Optional learned queries (if AIRS tokens not provided)
        if cfg.use_learned_spectral_tokens:
            self.spectral_queries = nn.Parameter(torch.randn(cfg.bins, cfg.d_model) * 0.02)
        else:
            self.spectral_queries = None

        # Cross-attention blocks
        self.blocks = nn.ModuleList(
            [CrossAttnBlock(cfg.d_model, cfg.nhead, cfg.ff_mult, cfg.dropout) for _ in range(cfg.num_layers)]
        )

        # Output head per spectral token
        self.out_mu = nn.Linear(cfg.d_model, 1)
        self.out_sigma = nn.Linear(cfg.d_model, 1)
        self.softplus = nn.Softplus()

        self.drop_in = nn.Dropout(cfg.dropout)
        self._init_weights()

    def _init_weights(self) -> None:
        # Xavier for projections & heads
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.spectral_queries is not None:
            nn.init.normal_(self.spectral_queries, mean=0.0, std=0.02)

    def forward(
        self,
        fgs1_tokens: Tensor,
        airs_tokens: Optional[Tensor] = None,
    ) -> Dict[str, Tensor]:
        """
        fgs1_tokens: (B, T, D_time)
        airs_tokens: (B, Nλ, D_spec) or None

        Returns: dict(mu=(B, Nλ), sigma=(B, Nλ), tokens=(B, Nλ, D))
        """
        B = fgs1_tokens.size(0)
        # Project FGS1 tokens
        k = self.proj_time(fgs1_tokens)                                      # (B, T, D)
        k = self.pe_time(k)
        k = self.drop_in(k)

        # Prepare queries: from AIRS tokens or learned queries
        if airs_tokens is not None:
            assert airs_tokens.size(1) == self.bins, f_
