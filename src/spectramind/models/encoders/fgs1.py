# src/spectramind/models/encoders/fgs1.py
# =============================================================================
# SpectraMind V50 — FGS1 Encoder
# -----------------------------------------------------------------------------
# Processes the white-light photometric time-series (FGS1) into a latent code.
#   • Input shape: [B, T] or [B, T, 1]  (batch, time[, channel=1])
#   • Backbone (configurable): BiLSTM (default) or TCN (temporal Conv1D stack)
#   • Optional Transformer refinement for long-range context
#   • Output: [B, D] or [B, T, D] (if pool=False)
#
# Motivation:
#   - Dual-encoder architecture (FGS1 + AIRS), then fuse [oai_citation:0‡SpectraMind V50 – Production-Grade Repository Blueprint.pdf](file-service://file-SrT96eN7UDBfjfxRBtdtjG) [oai_citation:1‡ADR 0004 — Dual Encoder Fusion (FGS1 + AIRS).pdf](file-service://file-4CBsvxoriyyazqtkG3ekUJ)
#   - FGS1 branch specializes in global transit depth/timing anchoring calibration [oai_citation:2‡ADR 0004 — Dual Encoder Fusion (FGS1 + AIRS).pdf](file-service://file-4CBsvxoriyyazqtkG3ekUJ)
# =============================================================================

from __future__ import annotations
from typing import Literal, Optional

import torch
import torch.nn as nn


class _TemporalConvBlock(nn.Module):
    """1D Temporal Conv block: Conv1d -> BN -> GELU with residual."""
    def __init__(self, channels: int, kernel_size: int = 5, dilation: int = 1) -> None:
        super().__init__()
        pad = (kernel_size - 1) // 2 * dilation
        self.net = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=pad, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class FGS1Encoder(nn.Module):
    """
    Encoder for the FGS1 white-light lightcurve.

    Args:
        backbone: 'lstm' (BiLSTM) or 'tcn' (temporal conv stack).
        d_model: latent embedding size for fusion/decoders.
        lstm_hidden: hidden size for the (bi)lstm.
        lstm_layers: number of stacked LSTM layers.
        lstm_dropout: dropout between LSTM layers (PyTorch semantics).
        tcn_width: internal channel width for temporal conv.
        tcn_layers: number of temporal conv residual blocks.
        tcn_kernel: kernel size for temporal conv blocks.
        transformer_layers: number of Transformer encoder layers (0 disables).
        nhead: attention heads (if transformer enabled).
        pool: mean-pool across time to [B, D] if True, else return [B, T, D].
    """

    def __init__(
        self,
        *,
        backbone: Literal["lstm", "tcn"] = "lstm",
        d_model: int = 256,
        lstm_hidden: int = 128,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        tcn_width: int = 128,
        tcn_layers: int = 4,
        tcn_kernel: int = 5,
        transformer_layers: int = 1,
        nhead: int = 8,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.pool = pool
        self.backbone = backbone

        if backbone == "lstm":
            # BiLSTM over time; input feature dim = 1 (flux)
            self.lstm = nn.LSTM(
                input_size=1,
                hidden_size=lstm_hidden,
                num_layers=lstm_layers,
                dropout=lstm_dropout if lstm_layers > 1 else 0.0,
                batch_first=True,
                bidirectional=True,
            )
            self.proj = nn.Linear(2 * lstm_hidden, d_model)
            self.tcn = None
        elif backbone == "tcn":
            # TCN on [B, C=1, T] with widening to tcn_width, then residual blocks
            self.stem = nn.Conv1d(1, tcn_width, kernel_size=3, padding=1)
            blocks = []
            for i in range(tcn_layers):
                blocks.append(_TemporalConvBlock(tcn_width, kernel_size=tcn_kernel, dilation=2**i))
            self.tcn = nn.Sequential(*blocks)
            self.proj = nn.Linear(tcn_width, d_model)
            self.lstm = None
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # Optional Transformer refinement (batch_first=True)
        if transformer_layers > 0:
            enc_layer = nn.TransformerEncoderLayer(
                d_model=d_model, nhead=nhead, batch_first=True, norm_first=True
            )
            self.transformer = nn.TransformerEncoder(enc_layer, num_layers=transformer_layers)
        else:
            self.transformer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, T] or [B, T, 1] lightcurve (normalized flux vs time)
        Returns:
            [B, D] if pooled, else [B, T, D]
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)          # [B, T, 1]
        elif x.dim() == 3 and x.size(-1) == 1:
            pass
        else:
            raise ValueError(f"FGS1Encoder expects [B,T] or [B,T,1], got {tuple(x.shape)}")

        if self.backbone == "lstm":
            # BiLSTM over time
            feats, _ = self.lstm(x)       # [B, T, 2*hidden]
            feats = self.proj(feats)      # [B, T, D]
        else:
            # TCN expects [B, C, T]
            z = x.transpose(1, 2)         # [B, 1, T]
            z = self.stem(z)              # [B, W, T]
            z = self.tcn(z)               # [B, W, T]
            feats = z.transpose(1, 2)     # [B, T, W]
            feats = self.proj(feats)      # [B, T, D]

        if self.transformer is not None:
            feats = self.transformer(feats)  # [B, T, D]

        if self.pool:
            return feats.mean(dim=1)        # [B, D]
        return feats                         # [B, T, D]
