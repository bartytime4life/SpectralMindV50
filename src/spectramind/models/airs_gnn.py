from __future__ import annotations

"""
AIRS-GNN — Spectral encoder for the AIRS channel.

Design
------
* Input:  AIRS features per wavelength bin, shaped (B, Nλ) or (B, Nλ, C_in)
* Output: Token sequence shaped (B, Nλ, D)

Backbones
---------
* If `torch_geometric` is installed, we use a lightweight GNN (SAGE/GCN) over
  a static graph of wavelength bins:
    - chain graph by default (i <-> i+1)
    - optional kNN graph using provided wavelengths (in nm/um/etc.)
* If `torch_geometric` is not present, we fall back to a 1-D Conv residual stack.

Notes
-----
* Module is torch-only and can be wrapped in a LightningModule.
* Edge topology is static and reused across steps; batched graphs are supported
  by offsetting edge indices per sample.
"""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

__all__ = ["AIRSGNNConfig", "AIRSGNNEncoder", "build_airs_gnn"]

# --------------------------------------------------------------------------------------
# Optional PyG
# --------------------------------------------------------------------------------------

_HAS_PYG = False
try:  # pragma: no cover (import varies by env)
    from torch_geometric.nn import SAGEConv, GCNConv  # type: ignore
    _HAS_PYG = True
except Exception:
    _HAS_PYG = False


# --------------------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------------------

@dataclass
class AIRSGNNConfig:
    bins: int = 283               # number of wavelength bins (Nλ)
    d_in: int = 1                 # input feature dim per bin (1 if raw intensity)
    d_model: int = 256            # output token dim per bin
    n_layers: int = 3             # graph/conv layers
    dropout: float = 0.1

    # Graph options (only used if torch_geometric is available)
    use_gnn: bool = True
    conv_type: str = "sage"       # "sage" | "gcn"
    use_knn: bool = False
    knn_k: int = 4
    wavelengths: Optional[Sequence[float]] = None  # if provided and use_knn=True

    # Fallback conv options
    kernel_size: int = 5
    ff_mult: int = 2

    # Positional encoding (sinusoidal) for fallback path
    posenc: bool = True


# --------------------------------------------------------------------------------------
# Utils: graph construction
# --------------------------------------------------------------------------------------

def _make_chain_edges(n: int, device: torch.device) -> Tensor:
    """
    Undirected chain graph edges: (i <-> i+1).
    Returns edge_index of shape (2, E).
    """
    if n < 2:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    src = torch.arange(n - 1, device=device, dtype=torch.long)
    dst = src + 1
    # make bidirectional
    edge_index = torch.stack([torch.cat([src, dst]), torch.cat([dst, src])], dim=0)
    return edge_index


def _make_knn_edges(
    wavelengths: Tensor, k: int, device: torch.device
) -> Tensor:
    """
    kNN edges (symmetric) from wavelength coordinates.
    wavelengths: (N,) float tensor of sorted (or unsorted) wavelength values
    """
    N = wavelengths.numel()
    if N == 0 or k <= 0:
        return torch.empty(2, 0, dtype=torch.long, device=device)
    # compute pairwise |λi - λj|
    diff = wavelengths.view(-1, 1) - wavelengths.view(1, -1)
    dist = diff.abs()
    # for each node, select k nearest excluding itself
    topk = torch.topk(-dist, k=k + 1, dim=-1)  # negative distances -> largest are nearest
    idxs = topk.indices[:, 1:]  # drop self index (first)
    rows = torch.arange(N, device=device).unsqueeze(1).expand_as(idxs)
    edge_index = torch.stack([rows.reshape(-1), idxs.reshape(-1)], dim=0)
    # make symmetric
    edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)
    # deduplicate
    edge_index = torch.unique(edge_index, dim=1)
    return edge_index.to(device=device, dtype=torch.long)


def _expand_edge_index_for_batch(edge_index: Tensor, n_nodes: int, batch_size: int) -> Tensor:
    """
    Expand edge_index (2, E) for a batch by offsetting indices per sample.
    """
    if batch_size == 1:
        return edge_index
    offsets = torch.arange(batch_size, device=edge_index.device, dtype=torch.long) * n_nodes  # (B,)
    # expand and add offsets
    ei = edge_index.unsqueeze(0).repeat(batch_size, 1, 1)  # (B, 2, E)
    ei[ :, 0, :] += offsets.view(-1, 1)
    ei[ :, 1, :] += offsets.view(-1, 1)
    # collapse batch to (2, B*E)
    return ei.permute(1, 0, 2).reshape(2, -1)


# --------------------------------------------------------------------------------------
# Fallback block (1D Conv) with residual + FFN
# --------------------------------------------------------------------------------------

class Conv1dResidualBlock(nn.Module):
    """
    Pre-norm LayerNorm along feature dim (last), 1D conv along bins, residual + FFN.
    Input/Output: (B, N, D)
    """

    def __init__(self, d_model: int, kernel_size: int = 5, ff_mult: int = 2, dropout: float = 0.1):
        super().__init__()
        assert kernel_size % 2 == 1, "Use odd kernel_size for symmetric padding"
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        pad = kernel_size // 2
        self.conv = nn.Conv1d(d_model, d_model, kernel_size=kernel_size, padding=pad, groups=1)
        d_ff = ff_mult * d_model
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        # Conv branch
        y = self.ln1(x)                    # (B,N,D)
        y = y.transpose(1, 2)              # (B,D,N)
        y = self.conv(y)
        y = y.transpose(1, 2)              # (B,N,D)
        y = self.dropout(y)
        x = x + y                          # residual

        # FFN branch
        z = self.ln2(x)
        z = self.ffn(z)
        return x + z


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 10000):
        super().__init__()
        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        pos = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32) * (-torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: Tensor) -> Tensor:
        return x + self.pe[: x.size(1)].unsqueeze(0)


# --------------------------------------------------------------------------------------
# Encoder
# --------------------------------------------------------------------------------------

class AIRSGNNEncoder(nn.Module):
    """
    AIRS-GNN Encoder

    Inputs
    ------
    * x: (B, Nλ) or (B, Nλ, C_in)

    Returns
    -------
    dict(tokens=(B, Nλ, D))
    """

    def __init__(self, cfg: AIRSGNNConfig):
        super().__init__()
        self.cfg = cfg
        self.bins = cfg.bins
        self.d_model = cfg.d_model

        # Input projection
        self.in_proj = nn.Linear(cfg.d_in, cfg.d_model)

        # GNN path
        self.use_gnn = bool(cfg.use_gnn and _HAS_PYG)
        if self.use_gnn:
            # Build base edge index (chain or kNN)
            device = torch.device("cpu")
            if cfg.use_knn and cfg.wavelengths is not None:
                w = torch.tensor(cfg.wavelengths, dtype=torch.float32, device=device).flatten()
                assert w.numel() == cfg.bins, "wavelengths length must match bins"
                base_edge = _make_knn_edges(w, k=int(cfg.knn_k), device=device)
            else:
                base_edge = _make_chain_edges(cfg.bins, device=device)
            self.register_buffer("base_edge_index", base_edge, persistent=False)

            # Conv stack
            self.convs = nn.ModuleList()
            in_dim = cfg.d_model
            for i in range(cfg.n_layers):
                if cfg.conv_type.lower() == "gcn":
                    self.convs.append(GCNConv(in_dim, cfg.d_model))
                else:
                    self.convs.append(SAGEConv(in_dim, cfg.d_model))
                in_dim = cfg.d_model
            self.act = nn.GELU()
            self.drop = nn.Dropout(cfg.dropout)
            self.ln = nn.LayerNorm(cfg.d_model)
        else:
            # Fallback conv path
            self.posenc = SinusoidalPositionalEncoding(cfg.d_model) if cfg.posenc else nn.Identity()
            self.blocks = nn.ModuleList(
                [Conv1dResidualBlock(cfg.d_model, cfg.kernel_size, cfg.ff_mult, cfg.dropout) for _ in range(cfg.n_layers)]
            )
            self.drop_in = nn.Dropout(cfg.dropout)

        self._init_weights()

    def _init_weights(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        """
        x: (B, Nλ) or (B, Nλ, C_in)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)                  # (B,Nλ,1)
        assert x.shape[1] == self.cfg.bins, f"Nλ mismatch: expected {self.cfg.bins}, got {x.shape[1]}"
        assert x.shape[-1] == self.cfg.d_in, f"d_in mismatch: expected {self.cfg.d_in}, got {x.shape[-1]}"

        # Project to model dim
        h = self.in_proj(x)                      # (B,Nλ,D)

        if self.use_gnn:
            # Flatten batch as disjoint graphs
            B, N, D = h.shape
            h_flat = h.reshape(B * N, D)        # (B*N, D)
            # Expand edge index for the full batch
            edge_index = _expand_edge_index_for_batch(self.base_edge_index, N, B)

            # Pass through conv stack
            for conv in self.convs:
                h_flat = conv(h_flat, edge_index)
                h_flat = self.act(h_flat)
                h_flat = self.drop(h_flat)

            h = h_flat.view(B, N, D)            # (B,Nλ,D)
            h = self.ln(h)
            return {"tokens": h}

        # Fallback conv path
        h = self.drop_in(h)
        for blk in self.blocks:
            h = blk(h)
        # Optional sinusoidal PE (applied at the end if configured)
        if isinstance(self.posenc, SinusoidalPositionalEncoding):
            h = self.posenc(h)
        return {"tokens": h}

    @torch.no_grad()
    def encode(self, x: Tensor) -> Tensor:
        """Convenience method returning only tokens."""
        self.eval()
        return self.forward(x)["tokens"]


# --------------------------------------------------------------------------------------
# Factory + Smoke test
# --------------------------------------------------------------------------------------

def build_airs_gnn(cfg_dict: Dict[str, Any]) -> AIRSGNNEncoder:
    """Factory from plain dict (e.g., OmegaConf.to_container())."""
    cfg = AIRSGNNConfig(**cfg_dict)
    return AIRSGNNEncoder(cfg)


if __name__ == "__main__":  # quick smoke test
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    cfg = AIRSGNNConfig(
        bins=283,
        d_in=1,
        d_model=128,
        n_layers=3,
        dropout=0.1,
        use_gnn=True,
        conv_type="sage",
        use_knn=True,
        knn_k=4,
        wavelengths=[float(i) for i in range(283)],
    )
    enc = AIRSGNNEncoder(cfg).to(device)

    B = 2
    x = torch.randn(B, cfg.bins, cfg.d_in, device=device)
    out = enc(x)
    print("tokens:", out["tokens"].shape)
