# src/spectramind/models/encoders/airs.py
# =============================================================================
# SpectraMind V50 — AIRS Encoder
# -----------------------------------------------------------------------------
# Processes 283-channel spectroscopic (AIRS) inputs into a latent representation.
#   • Input shape: [B, T, C]  (batch, time, channels) where C≈283
#   • Backbone: 1D CNN stack + optional Transformer encoder
#   • Output: latent embedding [B, D] or [B, T, D] (configurable)
#
# References:
#   - SpectraMind V50 Production Repo Blueprint [oai_citation:3‡SpectraMind V50 – Production-Grade Repository Blueprint.pdf](file-service://file-SrT96eN7UDBfjfxRBtdtjG)
#   - ADR 0004 (Dual Encoder Fusion, FGS1 + AIRS) [oai_citation:4‡ADR 0004 — Dual Encoder Fusion (FGS1 + AIRS).pdf](file-service://file-4CBsvxoriyyazqtkG3ekUJ)
# =============================================================================

from __future__ import annotations
from typing import Optional

import torch
import torch.nn as nn


class AIRSEncoder(nn.Module):
    """
    Convolutional + Transformer encoder for AIRS spectral time-series.

    Args:
        in_channels: number of spectral channels (default=283).
        cnn_dim: hidden size of CNN feature maps.
        num_layers: number of CNN conv layers.
        kernel_size: size of CNN kernels.
        transformer_layers: number of Transformer encoder layers.
        nhead: number of attention heads (if transformer enabled).
        d_model: latent embedding size.
        pool: whether to mean-pool over time dimension.
    """

    def __init__(
        self,
        in_channels: int = 283,
        cnn_dim: int = 128,
        num_layers: int = 3,
        kernel_size: int = 5,
        transformer_layers: int = 2,
        nhead: int = 8,
        d_model: int = 256,
        pool: bool = True,
    ) -> None:
        super().__init__()
        self.pool = pool

        # --- CNN frontend -----------------------------------------------------
        layers: list[nn.Module] = []
        in_ch = in_channels
        for _ in range(num_layers):
            layers += [
                nn.Conv1d(in_ch, cnn_dim, kernel_size, padding=kernel_size // 2),
                nn.BatchNorm1d(cnn_dim),
                nn.GELU(),
            ]
            in_ch = cnn_dim
        self.cnn = nn.Sequential(*layers)

        # --- Projection to transformer embedding ------------------------------
        self.proj = nn.Linear(cnn_dim, d_model)

        # --- Optional Transformer encoder ------------------------------------
        if transformer_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                batch_first=True,
                norm_first=True,
            )
            self.transformer = nn.TransformerEncoder(
                encoder_layer, num_layers=transformer_layers
            )
        else:
            self.transformer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor [B, T, C] = (batch, time, channels)
        Returns:
            Tensor [B, D] if pooled, else [B, T, D]
        """
        # CNN expects [B, C, T]
        x = x.transpose(1, 2)          # [B, C, T]
        feats = self.cnn(x)            # [B, H, T]
        feats = feats.transpose(1, 2)  # [B, T, H]

        # Project to d_model
        feats = self.proj(feats)       # [B, T, D]

        # Transformer context
        if self.transformer is not None:
            feats = self.transformer(feats)  # [B, T, D]

        if self.pool:
            return feats.mean(dim=1)  # [B, D]
        return feats                   # [B, T, D]
