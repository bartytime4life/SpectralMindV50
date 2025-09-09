# src/spectramind/models/fusion/xattn.py
# =============================================================================
# SpectraMind V50 — Cross-Attention Fusion (ADR-0004)
# -----------------------------------------------------------------------------
# Cross-attention fusion between FGS1 (photometry) and AIRS (spectroscopy).
# This module projects both modalities into a common d_model space and applies
# one-way or bi-directional multi-head cross-attention with residual MLPs and
# normalization. It returns sequence-wise fused streams and a pooled joint code
# suitable for downstream decoders/heads.
#
# Shapes (defaults; batch_first=True):
#   FGS1  : (B, T_fgs1, d_fgs1)
#   AIRS  : (B, S_airs, d_airs)
#   Masks : key padding masks are boolean with True = PAD/ignore
#           fgs1_mask: (B, T_fgs1) / airs_mask: (B, S_airs)
#
# Returns:
#   {
#     "fgs1_fused": (B, T_fgs1, d_model),
#     "airs_fused": (B, S_airs, d_model),
#     "joint":      (B, d_joint)  # pooled fusion (concat or projected)
#   }
# =============================================================================

from __future__ import annotations

from typing import Literal, Optional, Tuple

import math
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def _make_norm(norm: Literal["layer", "rms"], d_model: int) -> nn.Module:
    if norm == "layer":
        return nn.LayerNorm(d_model)
    if norm == "rms":
        # Simple RMSNorm (no bias)
        class RMSNorm(nn.Module):
            def __init__(self, dim: int, eps: float = 1e-6) -> None:
                super().__init__()
                self.weight = nn.Parameter(torch.ones(dim))
                self.eps = eps

            def forward(self, x: Tensor) -> Tensor:
                # x: (..., dim)
                rms = x.pow(2).mean(dim=-1, keepdim=True).add(self.eps).sqrt()
                return self.weight * (x / rms)

        return RMSNorm(d_model)
    raise ValueError(f"Unknown norm='{norm}'")


def _positional_encoding_sinusoids(seq_len: int, d_model: int, device=None) -> Tensor:
    """Classic sinusoidal PE (Vaswani et al.). Returns (1, L, D)."""
    position = torch.arange(seq_len, device=device).unsqueeze(1)  # (L, 1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device, dtype=torch.float32)
        * (-math.log(10000.0) / d_model)
    )  # (D/2,)
    pe = torch.zeros(seq_len, d_model, device=device, dtype=torch.float32)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)  # (1, L, D)


class _FFN(nn.Module):
    def __init__(self, d_model: int, d_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_hidden, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class _CrossAttnBlock(nn.Module):
    """
    A single fusion block with (optionally) bi-directional cross-attention:
      - fgs1 <- attends to AIRS  (queries=FGS1, keys/values=AIRS)
      - airs <- attends to FGS1  (queries=AIRS, keys/values=FGS1)
    Each stream applies PreNorm -> MHA -> Residual -> FFN -> Residual.
    """

    def __init__(
        self,
        d_model: int,
        nhead: int,
        ff_ratio: float,
        dropout: float,
        norm_type: Literal["layer", "rms"] = "layer",
        attn_bias: bool = False,
        attend: Literal["bi", "fgs1_to_airs", "airs_to_fgs1"] = "bi",
    ) -> None:
        super().__init__()
        self.attend = attend

        self.norm_f_q = _make_norm(norm_type, d_model)
        self.norm_f_kv = _make_norm(norm_type, d_model)
        self.norm_a_q = _make_norm(norm_type, d_model)
        self.norm_a_kv = _make_norm(norm_type, d_model)

        self.mha_f = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=attn_bias
        )
        self.mha_a = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True, bias=attn_bias
        )

        d_ff = max(1, int(d_model * ff_ratio))
        self.ff_f = _FFN(d_model, d_ff, dropout)
        self.ff_a = _FFN(d_model, d_ff, dropout)

        self.drop = nn.Dropout(dropout)

    def _attend(
        self,
        q: Tensor,
        kv: Tensor,
        q_norm: nn.Module,
        kv_norm: nn.Module,
        mha: nn.MultiheadAttention,
        q_mask: Optional[Tensor],
        kv_mask: Optional[Tensor],
    ) -> Tensor:
        q_in = q
        q = q_norm(q)
        kv = kv_norm(kv)
        # key_padding_mask expects True for positions that are to be ignored
        out, _ = mha(
            query=q,
            key=kv,
            value=kv,
            key_padding_mask=kv_mask,  # (B, L_kv) with True = PAD
            attn_mask=None,
            need_weights=False,
        )
        q = q_in + self.drop(out)
        q = q + self.drop(self.ff_f(q)) if mha is self.mha_f else q + self.drop(self.ff_a(q))
        return q

    def forward(
        self,
        fgs1: Tensor,  # (B, T, D)
        airs: Tensor,  # (B, S, D)
        fgs1_mask: Optional[Tensor],  # (B, T) True = PAD
        airs_mask: Optional[Tensor],  # (B, S) True = PAD
    ) -> Tuple[Tensor, Tensor]:
        f, a = fgs1, airs

        if self.attend in ("bi", "fgs1_to_airs"):
            # FGS1 queries AIRS
            f = self._attend(
                q=f, kv=a,
                q_norm=self.norm_f_q, kv_norm=self.norm_f_kv,
                mha=self.mha_f,
                q_mask=fgs1_mask, kv_mask=airs_mask,
            )
        if self.attend in ("bi", "airs_to_fgs1"):
            # AIRS queries FGS1
            a = self._attend(
                q=a, kv=f,
                q_norm=self.norm_a_q, kv_norm=self.norm_a_kv,
                mha=self.mha_a,
                q_mask=airs_mask, kv_mask=fgs1_mask,
            )
        return f, a


class CrossAttentionFusion(nn.Module):
    """
    Cross-attention fusion module between FGS1 and AIRS streams.

    Parameters
    ----------
    d_fgs1 : int
        Input channel dimension for FGS1 tokens.
    d_airs : int
        Input channel dimension for AIRS tokens.
    d_model : int, default 256
        Common model dimension after projection.
    nhead : int, default 8
        Number of attention heads.
    num_layers : int, default 2
        Number of stacked cross-attention fusion blocks.
    ff_ratio : float, default 4.0
        Feed-forward width multiplier.
    dropout : float, default 0.0
        Dropout rate for attention and FFN.
    norm : {"layer","rms"}, default "layer"
        Normalization type.
    attend : {"bi","fgs1_to_airs","airs_to_fgs1"}, default "bi"
        Attention directionality.
    positional_encoding : bool, default True
        Add sinusoidal positional encodings to each stream before fusion.
    pool : {"mean","cls","mean+proj"}, default "mean+proj"
        Strategy for producing a fixed-size joint code:
          - "mean": concat(mean_f, mean_a)
          - "cls":  prepend learned CLS tokens (one per stream) and concat
          - "mean+proj": concat means, then project to d_model

    Returns
    -------
    dict with:
      - "fgs1_fused": (B, T, d_model)
      - "airs_fused": (B, S, d_model)
      - "joint":      (B, d_joint)
    """

    def __init__(
        self,
        d_fgs1: int,
        d_airs: int,
        *,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 2,
        ff_ratio: float = 4.0,
        dropout: float = 0.0,
        norm: Literal["layer", "rms"] = "layer",
        attend: Literal["bi", "fgs1_to_airs", "airs_to_fgs1"] = "bi",
        positional_encoding: bool = True,
        pool: Literal["mean", "cls", "mean+proj"] = "mean+proj",
        attn_bias: bool = False,
    ) -> None:
        super().__init__()
        self.d_model = int(d_model)
        self.pool = pool
        self.use_pe = positional_encoding

        # Projections to a shared space
        self.proj_f = nn.Linear(d_fgs1, d_model, bias=True)
        self.proj_a = nn.Linear(d_airs, d_model, bias=True)

        # Optional CLS tokens for both streams (if using "cls" pooling)
        if self.pool == "cls":
            self.cls_f = nn.Parameter(torch.zeros(1, 1, d_model))
            self.cls_a = nn.Parameter(torch.zeros(1, 1, d_model))
            nn.init.trunc_normal_(self.cls_f, std=0.02)
            nn.init.trunc_normal_(self.cls_a, std=0.02)
        else:
            self.cls_f = None
            self.cls_a = None

        # Fusion blocks
        self.blocks = nn.ModuleList(
            [
                _CrossAttnBlock(
                    d_model=d_model,
                    nhead=nhead,
                    ff_ratio=ff_ratio,
                    dropout=dropout,
                    norm_type=norm,
                    attn_bias=attn_bias,
                    attend=attend,
                )
                for _ in range(num_layers)
            ]
        )

        # Final norms
        self.norm_f_out = _make_norm(norm, d_model)
        self.norm_a_out = _make_norm(norm, d_model)

        # Project joint if requested
        if self.pool == "mean+proj":
            self.joint_proj = nn.Linear(2 * d_model, d_model)
        else:
            self.joint_proj = None

        # (Lazy) cache for sinusoidal PE lengths
        self._pe_f_len = 0
        self._pe_a_len = 0
        self.register_buffer("_pe_f_cache", torch.zeros(1, 1, d_model), persistent=False)
        self.register_buffer("_pe_a_cache", torch.zeros(1, 1, d_model), persistent=False)

    def _maybe_pe(self, x: Tensor, cache_len: int, cache: Tensor) -> Tuple[Tensor, int, Tensor]:
        # x: (B, L, D)
        if not self.use_pe:
            return x, cache_len, cache
        L, D = x.shape[1], x.shape[2]
        if L > cache_len or cache.shape[1] != L or cache.shape[2] != D:
            pe = _positional_encoding_sinusoids(L, D, device=x.device)
            cache = pe
            cache_len = L
        return x + cache, cache_len, cache

    def _masked_mean(self, x: Tensor, mask: Optional[Tensor]) -> Tensor:
        # x: (B, L, D); mask: (B, L) True = PAD
        if mask is None:
            return x.mean(dim=1)
        valid = (~mask).to(x.dtype)  # 1 for valid, 0 for pad
        denom = valid.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (x * valid.unsqueeze(-1)).sum(dim=1) / denom

    def forward(
        self,
        fgs1: Tensor,                   # (B, T, d_fgs1)
        airs: Tensor,                   # (B, S, d_airs)
        *,
        fgs1_mask: Optional[Tensor] = None,  # (B, T) True = PAD
        airs_mask: Optional[Tensor] = None,  # (B, S) True = PAD
    ) -> dict:
        B, T, _ = fgs1.shape
        B2, S, _ = airs.shape
        assert B == B2, "FGS1 and AIRS batch sizes must match"

        f = self.proj_f(fgs1)  # (B, T, D)
        a = self.proj_a(airs)  # (B, S, D)

        # Optionally prepend CLS tokens
        if self.pool == "cls":
            cls_f = self.cls_f.expand(B, -1, -1)  # (B, 1, D)
            cls_a = self.cls_a.expand(B, -1, -1)  # (B, 1, D)
            f = torch.cat([cls_f, f], dim=1)  # (B, 1+T, D)
            a = torch.cat([cls_a, a], dim=1)  # (B, 1+S, D)
            if fgs1_mask is not None:
                fgs1_mask = torch.cat([torch.zeros(B, 1, dtype=fgs1_mask.dtype, device=fgs1_mask.device), fgs1_mask], dim=1)
            if airs_mask is not None:
                airs_mask = torch.cat([torch.zeros(B, 1, dtype=airs_mask.dtype, device=airs_mask.device), airs_mask], dim=1)

        # Positional encodings
        f, self._pe_f_len, self._pe_f_cache = self._maybe_pe(f, self._pe_f_len, self._pe_f_cache)
        a, self._pe_a_len, self._pe_a_cache = self._maybe_pe(a, self._pe_a_len, self._pe_a_cache)

        # Stacked cross-attention fusion
        for blk in self.blocks:
            f, a = blk(f, a, fgs1_mask, airs_mask)

        # Final norms
        f = self.norm_f_out(f)
        a = self.norm_a_out(a)

        # Pooled joint code
        if self.pool == "mean":
            f_pool = self._masked_mean(f, fgs1_mask)
            a_pool = self._masked_mean(a, airs_mask)
            joint = torch.cat([f_pool, a_pool], dim=-1)  # (B, 2D)
        elif self.pool == "cls":
            # Take CLS positions (index 0)
            f_pool = f[:, 0, :]  # (B, D)
            a_pool = a[:, 0, :]  # (B, D)
            joint = torch.cat([f_pool, a_pool], dim=-1)  # (B, 2D)
            # Remove CLS from sequences for downstream if desired:
            f = f[:, 1:, :]
            a = a[:, 1:, :]
            if fgs1_mask is not None:
                fgs1_mask = fgs1_mask[:, 1:]
            if airs_mask is not None:
                airs_mask = airs_mask[:, 1:]
        else:  # "mean+proj"
            f_pool = self._masked_mean(f, fgs1_mask)
            a_pool = self._masked_mean(a, airs_mask)
            joint = self.joint_proj(torch.cat([f_pool, a_pool], dim=-1))  # (B, D)

        return {
            "fgs1_fused": f,   # (B, T, D) (or T unchanged if no CLS)
            "airs_fused": a,   # (B, S, D) (or S unchanged if no CLS)
            "joint": joint,    # (B, 2D) for "mean"/"cls", or (B, D) for "mean+proj"
        }
