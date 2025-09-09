# src/spectramind/train/collate.py
# =============================================================================
# SpectraMind V50 — Collate Functions for DataLoader
# -----------------------------------------------------------------------------
# Builds a robust collate_fn that:
#   • Pads variable-length time-series (FGS1/AIRS) along time dim to batch max
#   • Emits boolean masks for padded positions
#   • Stacks fixed-shape targets (e.g. 283-bin spectrum) and optional fields
#   • Accepts numpy arrays or torch tensors and coerces dtype/device safely
#
# Design notes:
#   • Guarded torch import (module import-safe without torch)
#   • Configurable keys and pad value through CollateConfig
#   • No assumptions on AIRS rank beyond time on dim=0
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple, Union

try:  # pragma: no cover
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # type: ignore

try:  # pragma: no cover
    import torch
    from torch import Tensor
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    Tensor = Any  # type: ignore


ArrayLike = Union["np.ndarray", "Tensor"]  # type: ignore[name-defined]
Sample = Mapping[str, Any]
Batch = Dict[str, Any]


# =============================================================================
# Helpers
# =============================================================================
def _require_torch() -> None:
    if torch is None:  # pragma: no cover
        raise RuntimeError("This collate module requires PyTorch at runtime.")


def _is_array(x: Any) -> bool:
    return (np is not None and isinstance(x, np.ndarray)) or (torch is not None and isinstance(x, torch.Tensor))


def _to_tensor(x: Any, dtype: Optional[torch.dtype] = None) -> Tensor:
    _require_torch()
    if isinstance(x, torch.Tensor):
        return x.to(dtype=dtype) if dtype is not None else x
    if np is not None and isinstance(x, np.ndarray):
        t = torch.from_numpy(x)
        return t.to(dtype=dtype) if dtype is not None else t
    # Python scalars or lists
    t = torch.tensor(x)
    return t.to(dtype=dtype) if dtype is not None else t


def _pad_time_dim(
    seqs: Sequence[Tensor],
    pad_value: float,
) -> Tuple[Tensor, Tensor]:
    """
    Pad variable-length sequences along time dim=0 to the batch max length.

    Args:
      seqs: list of tensors shaped [Ti, ...]
      pad_value: value for padded region

    Returns:
      padded: [B, T_max, ...]
      mask:   [B, T_max]  True for valid positions, False for pad
    """
    _require_torch()
    lengths = [int(s.size(0)) for s in seqs]
    b = len(seqs)
    t_max = max(lengths) if lengths else 0
    if b == 0 or t_max == 0:
        # Empty batch fallback
        return torch.empty((0, 0), dtype=torch.bool), torch.empty((0, 0), dtype=torch.bool)  # type: ignore[return-value]

    # Infer trailing shape from first tensor
    tail_shape = tuple(seqs[0].size()[1:])
    dtype = seqs[0].dtype
    device = seqs[0].device

    padded = torch.full((b, t_max, *tail_shape), fill_value=pad_value, dtype=dtype, device=device)
    mask = torch.zeros((b, t_max), dtype=torch.bool, device=device)

    for i, (s, L) in enumerate(zip(seqs, lengths)):
        padded[i, :L, ...] = s
        mask[i, :L] = True

    return padded, mask


def _stack_if_present(
    xs: List[Optional[ArrayLike]],
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tensor]:
    present = [x for x in xs if x is not None]
    if not present:
        return None
    ts = [_to_tensor(x, dtype=dtype) for x in present]
    return torch.stack(ts, dim=0)


def _gather_key(batch: Sequence[Sample], key: str) -> List[Any]:
    return [s.get(key, None) for s in batch]


def _coerce_optional_tensor_list(
    items: List[Any],
    dtype: Optional[torch.dtype] = None,
) -> Optional[Tensor]:
    if all(x is None for x in items):
        return None
    ts: List[Tensor] = []
    for x in items:
        if x is None:
            # fill with a zero of shape []
            ts.append(_to_tensor(0.0, dtype=dtype))
        else:
            ts.append(_to_tensor(x, dtype=dtype))
    return torch.stack(ts, dim=0)


# =============================================================================
# Config
# =============================================================================
@dataclass
class CollateConfig:
    # dataset keys
    key_fgs1: str = "fgs1"          # variable-length time series [T, ...]
    key_airs: str = "airs"          # variable-length time series [T, ...]
    key_target: str = "y"           # fixed-shape spectrum [N_bins]
    key_sigma: Optional[str] = None # optional per-bin sigma [N_bins]
    key_id: Optional[str] = "sample_id"  # optional string/ID

    # padding
    pad_value: float = 0.0

    # dtypes
    dtype_time: Optional[str] = "float32"  # for time-series tensors
    dtype_target: Optional[str] = "float32"

    # additional passthrough keys (stacked if tensor/ndarray or kept as list otherwise)
    extra_keys: Tuple[str, ...] = tuple()


def _to_torch_dtype(name: Optional[str]) -> Optional[torch.dtype]:
    if torch is None or name is None:
        return None
    name = name.lower()
    return {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float64": torch.float64,
        "fp64": torch.float64,
        "int64": torch.int64,
        "int32": torch.int32,
        "int16": torch.int16,
        "int8": torch.int8,
        "bool": torch.bool,
    }.get(name, None)


# =============================================================================
# Collate builder
# =============================================================================
def build_collate_fn(cfg: Optional[CollateConfig] = None) -> Callable[[Sequence[Sample]], Batch]:
    """
    Build a collate_fn for DataLoader.

    Example
    -------
    >>> collate_fn = build_collate_fn(CollateConfig(pad_value=0.0))
    >>> loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)
    """
    _require_torch()
    cfg = cfg or CollateConfig()
    time_dtype = _to_torch_dtype(cfg.dtype_time)
    target_dtype = _to_torch_dtype(cfg.dtype_target)

    def _collate(batch: Sequence[Sample]) -> Batch:
        # Defensive: empty batch
        if not batch:
            return {}

        out: Batch = {}

        # -------- FGS1 (variable-length time) --------
        fgs_list_raw = _gather_key(batch, cfg.key_fgs1)
        fgs_list: List[Tensor] = []
        for x in fgs_list_raw:
            if x is None:
                fgs_list.append(_to_tensor([], dtype=time_dtype))  # empty
            else:
                t = _to_tensor(x, dtype=time_dtype)
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                fgs_list.append(t)
        if any(t.numel() > 0 for t in fgs_list):
            fgs_padded, fgs_mask = _pad_time_dim(fgs_list, pad_value=float(cfg.pad_value))
            out[cfg.key_fgs1] = fgs_padded
            out[f"{cfg.key_fgs1}_mask"] = fgs_mask

        # -------- AIRS (variable-length time) --------
        airs_list_raw = _gather_key(batch, cfg.key_airs)
        airs_list: List[Tensor] = []
        for x in airs_list_raw:
            if x is None:
                airs_list.append(_to_tensor([], dtype=time_dtype))  # empty
            else:
                t = _to_tensor(x, dtype=time_dtype)
                if t.ndim == 0:
                    t = t.unsqueeze(0)
                airs_list.append(t)
        if any(t.numel() > 0 for t in airs_list):
            airs_padded, airs_mask = _pad_time_dim(airs_list, pad_value=float(cfg.pad_value))
            out[cfg.key_airs] = airs_padded
            out[f"{cfg.key_airs}_mask"] = airs_mask

        # -------- Target (fixed shape) --------
        if cfg.key_target:
            target_list = _gather_key(batch, cfg.key_target)
            y = _stack_if_present(target_list, dtype=target_dtype)
            if y is not None:
                out[cfg.key_target] = y

        # -------- Optional sigma (fixed shape) --------
        if cfg.key_sigma:
            sigma_list = _gather_key(batch, cfg.key_sigma)
            sigma = _stack_if_present(sigma_list, dtype=target_dtype)
            if sigma is not None:
                out[cfg.key_sigma] = sigma

        # -------- Optional ID (list of strings/ints) --------
        if cfg.key_id:
            ids = _gather_key(batch, cfg.key_id)
            if any(x is not None for x in ids):
                out[cfg.key_id] = ids

        # -------- Extra passthrough keys --------
        for k in cfg.extra_keys:
            vals = _gather_key(batch, k)
            # If all arrays/tensors and same shape, stack; else keep as list
            if all(_is_array(v) for v in vals if v is not None):
                try:
                    out[k] = torch.stack([_to_tensor(v) for v in vals], dim=0)  # type: ignore[arg-type]
                except Exception:
                    out[k] = vals
            else:
                out[k] = vals

        return out

    return _collate
