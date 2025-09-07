from __future__ import annotations

"""
FGS1-Mamba — Temporal encoder for the FGS1 time-series.

Design
------
* Input:  FGS1 photometry or temporal features of shape (B, T) or (B, T, C_in)
* Output: Token sequence of shape (B, T_out, D), where T_out=T/downsample (avg-pooled)

Backbones
---------
* If `mamba_ssm` is installed, we optionally employ a small Mamba block.
* Otherwise, we fall back to a causal depthwise-conv + GLU block with Transformer-style FFN.

Notes
-----
* The module is torch-only (no Lightning), so it can be plugged into any training loop
  or wrapped inside a LightningModule (e.g., SpectraSystem).
* Positional encodings are sinusoidal and broadcast across batch.
"""

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = [
    "FGS1MambaConfig",
    "FGS1MambaEncoder",
    "build_fgs1_mamba",
]


# ======================================================================================
# Optional Mamba import
# ======================================================================================

_HAS_MAMBA = False
try:  # pragma: no cover
    # there are multiple public APIs; keep this conservative
    from mamba_ssm import Mamba  # type: ignore

    _HAS_MAMBA = True
except Exception:
    _HAS_MAMBA = False


# ======================================================================================
# Config
# ======================================================================================


@dataclass
class FGS1MambaConfig:
    d_in: int = 1                  # input feature channels (1 for raw flux)
    d_model: int = 256             # internal token dimension
    n_layers: int = 4              # number of temporal blocks
    dropout: float = 0.1
    downsample: int = 4            # temporal pooling factor (>=1). 1 → no downsampling
    # Fallback CausalConvGLU block params
    kernel_size: int = 7           # depthwise kernel size (odd recommended)
    ff_mult: int = 4               # FFN expansion multiplier
    # Mamba-specific (if library present). These are hints; actual used only if available
    mamba_d_state: int = 16
    mamba_expand: int = 2
    # Positional encoding
    posenc: bool = True


# ======================================================================================
# Positional Encoding
# ======================================================================================


class SinusoidalPositionalEncoding(nn.Module):
    """
    Sinusoidal positional encoding; broadcast over batch.
    """

    def __init__(self, d_model: int, max_len: int = 100000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)  # (L,1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, D)
        """
        T = x.size(1)
        return x + self.pe[:T].unsqueeze(0)


# ======================================================================================
# Fallback Causal Conv + GLU block (pre-norm)
# ======================================================================================


class CausalConvGLUBlock(nn.Module):
    """
    Pre-norm: LN → (causal depthwise conv → GLU) + residual → LN → FFN + residual
    """

    def __init__(self, d_model: int, kernel_size: int = 7, ff_mult: int = 4, dropout: float = 0.1):
        super().__init__()
        assert kernel_size >= 1 and kernel_size % 2 == 1, "kernel_size should be odd for clean causality padding"
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # Depthwise causal conv (channel-wise)
        self.dw = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, groups=d_model, padding=kernel_size - 1)

        # Pointwise projection producing 2*D for GLU
        self.pw = nn.Conv1d(d_model, 2 * d_model, kernel_size=1)

        # FFN
        d_ff = ff_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        x: (B, T, D)
        """
        # Branch 1: Causal conv + GLU
        y = self.ln1(x)
        y = y.transpose(1, 2)                        # (B,D,T)
        y = self.dw(y)                               # conv with left padding k-1
        y = y[:, :, : x.size(1)]                     # trim right to keep causality
        y = self.pw(y)                               # (B,2D,T)
        y = y.transpose(1, 2)                        # (B,T,2D)
        a, b = y.chunk(2, dim=-1)
        y = a * torch.sigmoid(b)                     # GLU
        y = self.dropout(y)
        x = x + y                                    # residual

        # Branch 2: FFN
        z = self.ln2(x)
        z = self.ffn(z)
        return x + z


# ======================================================================================
# Mamba block wrapper (optional)
# ======================================================================================


class _MambaBlock(nn.Module):  # pragma: no cover - used only when mamba_ssm is present
    def __init__(self, d_model: int, dropout: float, d_state: int, expand: int):
        super().__init__()
        # Light wrapper; the actual API can differ across versions.
        self.ln = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=4,
            expand=expand,
            dropout=dropout,
        )

    def forward(self, x: Tensor) -> Tensor:
        # Mamba often expects (B, T, D); keep pre-norm & residual
        y = self.mamba(self.ln(x))
        return x + y


# ======================================================================================
# Encoder
# ======================================================================================


class FGS1MambaEncoder(nn.Module):
    """
    FGS1-Mamba Encoder

    Inputs
    ------
    • x: (B, T) or (B, T, C_in)

    Returns
    -------
    dict(tokens=(B, T_out, D), pooled=(optional)), where T_out = ceil(T / downsample)
    """

    def __init__(self, cfg: FGS1MambaConfig):
        super().__init__()
        self.cfg = cfg
        self.d_model = cfg.d_model
        self.downsample = max(int(cfg.downsample), 1)

        # Input projection: (B,T,C_in) -> (B,T,D)
        self.in_proj = nn.Linear(cfg.d_in, cfg.d_model)

        # Positional encoding
        self.posenc = SinusoidalPositionalEncoding(cfg.d_model) if cfg.posenc else nn.Identity()
        self.in_drop = nn.Dropout(cfg.dropout)

        # Temporal blocks
        blocks = []
        for _ in range(cfg.n_layers):
            if _HAS_MAMBA:
                blocks.append(_MambaBlock(cfg.d_model, cfg.dropout, cfg.mamba_d_state, cfg.mamba_expand))
            else:
                blocks.append(CausalConvGLUBlock(cfg.d_model, cfg.kernel_size, cfg.ff_mult, cfg.dropout))
        self.blocks = nn.ModuleList(blocks)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        x: (B, T) or (B, T, C_in)
        """
        if x.dim() == 2:  # (B, T) -> (B, T, 1)
            x = x.unsqueeze(-1)
        assert x.dim() == 3 and x.size(-1) == self.cfg.d_in, \
            f"expected input (B,T,{self.cfg.d_in}) but got {tuple(x.shape)}"

        # Project to d_model
        y = self.in_proj(x)              # (B,T,D)
        y = self.posenc(y)               # (B,T,D)
        y = self.in_drop(y)

        # Temporal stack
        for blk in self.blocks:
            y = blk(y)                   # (B,T,D)

        # Downsample (avg pool) if needed
        if self.downsample > 1:
            y = F.avg_pool1d(y.transpose(1, 2), kernel_size=self.downsample, stride=self.downsample)
            y = y.transpose(1, 2)        # (B,T_out,D)

        return {"tokens": y}

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Convenience: returns just tokens."""
        self.eval()
        return self.forward(x)["tokens"]


# ======================================================================================
# Factory + Smoke test
# ======================================================================================


def build_fgs1_mamba(cfg_dict: Dict[str, Any]) -> FGS1MambaEncoder:
    """
    Factory helper from a plain dict (e.g., OmegaConf.to_container()).
    """
    cfg = FGS1MambaConfig(**cfg_dict)
    return FGS1MambaEncoder(cfg)


if __name__ == "__main__":  # quick smoke
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = FGS1MambaConfig(d_in=1, d_model=128, n_layers=3, downsample=4, kernel_size=7)
    enc = FGS1MambaEncoder(cfg).to(device)

    B, T = 2, 1024
    x = torch.randn(B, T, cfg.d_in, device=device)

    out = enc(x)
    tok = out["tokens"]
    print("tokens:", tok.shape)  # (B, T/4, 128)
