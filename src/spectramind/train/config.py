# src/spectramind/train/ckpt.py
# =============================================================================
# SpectraMind V50 — Checkpoint Utilities
# -----------------------------------------------------------------------------
# Lightning/PyTorch-agnostic helpers for saving/loading checkpoints,
# resuming training, discovering best/last checkpoints in a directory,
# and averaging multiple checkpoints into a single weights file.
#
# Design goals:
#   • Guarded imports (module is import-safe without torch/lightning).
#   • Works with PL Trainer when available, falls back to torch-only.
#   • Repro-friendly: explicit map_location, strict toggle, ignore/include prefixes.
#   • Robust "best" selection (mode=min|max) compatible with filename schema
#     like:  epoch{epoch:03d}-{monitor:.5f}.ckpt  (our default).
#   • Utilities to average (mean or EMA) multiple ckpts for inference stability.
#   • Small, composable API for Kaggle/CI and local dev.
# =============================================================================

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

# --- Optional heavy deps (import only when needed) --------------------------------
try:  # pragma: no cover
    import torch
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:  # pragma: no cover
    import pytorch_lightning as pl
except Exception:  # pragma: no cover
    pl = None  # type: ignore


# =============================================================================
# Types / dataclasses
# =============================================================================
@dataclass(frozen=True)
class CkptPick:
    path: Path
    score: Optional[float] = None
    epoch: Optional[int] = None


# =============================================================================
# Internal helpers
# =============================================================================
def _require_torch() -> None:
    if torch is None:
        raise RuntimeError(
            "This operation requires `torch` at runtime. Install PyTorch to proceed."
        )


def _is_ckpt(p: Path) -> bool:
    return p.is_file() and p.suffix == ".ckpt"


def _cpu_map_location() -> Any:
    # Works on CUDA/CPU nodes; keeps CPU for deterministic resume if desired.
    return "cpu"


_MONITOR_FLOAT_RE = re.compile(r"[-_](?P<val>[-+]?\d+(?:\.\d+)?)(?:\.ckpt)$")
_EPOCH_FROM_NAME_RE = re.compile(r"epoch(?P<ep>\d+)")
_EPOCH_FROM_STATE_RE_KEYS = ("epoch", "global_step")


def _try_parse_monitor_from_name(p: Path) -> Optional[float]:
    # expects filenames like epoch012-0.12345.ckpt (default template)
    m = _MONITOR_FLOAT_RE.search(p.name)
    if m:
        try:
            return float(m.group("val"))
        except Exception:
            return None
    return None


def _try_parse_epoch_from_name(p: Path) -> Optional[int]:
    m = _EPOCH_FROM_NAME_RE.search(p.stem)
    if m:
        try:
            return int(m.group("ep"))
        except Exception:
            return None
    return None


def _best_by_mode(items: Sequence[CkptPick], mode: str) -> Optional[CkptPick]:
    mode = (mode or "min").strip().lower()
    candidates = [x for x in items if x.score is not None]
    if not candidates:
        return None
    if mode == "max":
        return max(candidates, key=lambda x: (x.score, x.epoch or -1))
    # default min
    return min(candidates, key=lambda x: (x.score, x.epoch or -1))


def _load_torch_file(path: Path, map_location: Any) -> Dict[str, Any]:
    _require_torch()
    return torch.load(str(path), map_location=map_location)  # type: ignore[no-any-return]


def _state_dict_from_checkpoint(ckpt: Mapping[str, Any]) -> Optional[Mapping[str, Any]]:
    # Lightning checkpoints typically store model state under 'state_dict'
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], Mapping):
        return ckpt["state_dict"]
    # Plain state dict checkpoint (torch.save(model.state_dict()))
    maybe = {k: v for k, v in ckpt.items() if hasattr(v, "size") or hasattr(v, "shape")}
    # Heuristic: if enough tensor-like keys present, treat as state_dict
    if len(maybe) >= 10 or all(isinstance(v, (list, tuple)) is False for v in maybe.values()):
        return ckpt  # type: ignore[return-value]
    return None


def _strip_prefix(state: Mapping[str, Any], prefix: str) -> Dict[str, Any]:
    plen = len(prefix)
    out: Dict[str, Any] = {}
    for k, v in state.items():
        out[k[plen:] if k.startswith(prefix) else k] = v
    return out


def _filter_ignore_prefixes(
    state: Mapping[str, Any],
    ignore_prefixes: Optional[Sequence[str]],
) -> Dict[str, Any]:
    if not ignore_prefixes:
        return dict(state)
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if any(k.startswith(pref) for pref in ignore_prefixes):
            continue
        out[k] = v
    return out


def _filter_include_prefixes(
    state: Mapping[str, Any],
    include_prefixes: Optional[Sequence[str]],
) -> Dict[str, Any]:
    if not include_prefixes:
        return dict(state)
    out: Dict[str, Any] = {}
    for k, v in state.items():
        if any(k.startswith(pref) for pref in include_prefixes):
            out[k] = v
    return out


# =============================================================================
# Public: discovery (best/last)
# =============================================================================
def find_all_checkpoints(ckpt_dir: Union[str, Path]) -> List[Path]:
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    return sorted([p for p in d.iterdir() if _is_ckpt(p)])


def find_checkpoints_by_glob(ckpt_dir: Union[str, Path], pattern: str = "*.ckpt") -> List[Path]:
    """Find checkpoints using a glob pattern (e.g., 'epoch*.ckpt' or '*-val_loss*.ckpt')."""
    d = Path(ckpt_dir)
    if not d.exists():
        return []
    return sorted([p for p in d.glob(pattern) if _is_ckpt(p)])


def find_best_checkpoint(
    ckpt_dir: Union[str, Path],
    monitor: Optional[str] = None,
    mode: str = "min",
) -> Optional[Path]:
    """
    Try to choose a 'best' checkpoint from a directory.

    Strategy:
      1) If filenames encode the monitor value (default template "epoch{epoch}-{monitor}.ckpt"),
         parse and select min/max accordingly.
      2) If parsing fails for all, return None.
    """
    items: List[CkptPick] = []
    for p in find_all_checkpoints(ckpt_dir):
        score = _try_parse_monitor_from_name(p)
        epoch = _try_parse_epoch_from_name(p)
        items.append(CkptPick(path=p, score=score, epoch=epoch))
    pick = _best_by_mode(items, mode=mode)
    return pick.path if pick else None


def find_last_checkpoint(ckpt_dir: Union[str, Path]) -> Optional[Path]:
    """
    Best-effort "last" checkpoint:
      • Prefer higher epoch parsed from filename.
      • Fallback to most recently modified file.
    """
    cks = find_all_checkpoints(ckpt_dir)
    if not cks:
        return None
    scored: List[Tuple[int, Path]] = []
    for p in cks:
        ep = _try_parse_epoch_from_name(p)
        if ep is not None:
            scored.append((ep, p))
    if scored:
        scored.sort(key=lambda t: t[0], reverse=True)
        return scored[0][1]
    # fallback: newest by mtime
    return max(cks, key=lambda p: p.stat().st_mtime)


# =============================================================================
# Public: save / load / resume
# =============================================================================
def save_checkpoint(
    trainer: Optional["pl.Trainer"],  # type: ignore[name-defined]
    path: Union[str, Path],
    *,  # kw-only
    weights_only: bool = False,
) -> None:
    """
    Save a checkpoint via Lightning if trainer provided; otherwise raises.

    Note:
      • For torch-only model saving, call `torch.save(model.state_dict(), path)`.
    """
    path = Path(path)
    if trainer is None or pl is None:
        raise RuntimeError("Lightning Trainer is required to save a full checkpoint.")
    path.parent.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(str(path), weights_only=weights_only)


def save_state_dict(
    state_dict: Mapping[str, Any],
    path: Union[str, Path],
) -> None:
    """Save a raw state_dict (torch-only) to disk."""
    _require_torch()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dict(state_dict), str(path))


def load_weights_into_model(
    model: Any,
    ckpt_path: Union[str, Path],
    *,
    map_location: Any = None,
    strict: bool = False,
    ignore_prefixes: Optional[Sequence[str]] = None,
    include_prefixes: Optional[Sequence[str]] = None,
    strip_module_prefix: bool = True,
    report: bool = True,
) -> Dict[str, Any]:
    """
    Load (Lightning or raw) checkpoint weights into a torch.nn.Module.

    Args:
      model: torch.nn.Module-like with `load_state_dict`.
      ckpt_path: file path to .ckpt
      map_location: map location (default: CPU)
      strict: pass-through to `load_state_dict`
      ignore_prefixes: skip any state_dict entries whose keys start with these prefixes
      include_prefixes: if provided, only keep entries with these prefixes
      strip_module_prefix: remove 'module.' (DDP) prefix if present
      report: print missing/unexpected keys to help debugging (rank-zero assumed)

    Returns:
      The return value from `model.load_state_dict` (Missing/Unexpected keys).
    """
    _require_torch()
    import torch.nn as nn  # local to avoid import at module import time

    if not isinstance(model, nn.Module):
        raise TypeError("`model` must be a `torch.nn.Module`.")

    map_location = map_location or _cpu_map_location()
    ckpt = _load_torch_file(Path(ckpt_path), map_location=map_location)

    state = _state_dict_from_checkpoint(ckpt)
    if state is None:
        # Not a Lightning "state_dict" style; assume it's raw state_dict
        if isinstance(ckpt, Mapping):
            state = ckpt  # type: ignore[assignment]
        else:
            raise RuntimeError(f"Unrecognized checkpoint structure at {ckpt_path}.")

    # Optional DP unwrap
    if strip_module_prefix and any(k.startswith("module.") for k in state.keys()):
        state = _strip_prefix(state, "module.")

    # Optional include/ignore filters
    state = _filter_include_prefixes(state, include_prefixes)
    state = _filter_ignore_prefixes(state, ignore_prefixes)

    result = model.load_state_dict(state, strict=strict)
    if report and hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
        missing = list(getattr(result, "missing_keys"))
        unexpected = list(getattr(result, "unexpected_keys"))
        if missing or unexpected:
            print(f"[ckpt] load report for: {ckpt_path}")
            if missing:
                print(f"  - missing_keys ({len(missing)}): {missing[:8]}{' ...' if len(missing) > 8 else ''}")
            if unexpected:
                print(f"  - unexpected_keys ({len(unexpected)}): {unexpected[:8]}{' ...' if len(unexpected) > 8 else ''}")
    return result  # type: ignore[return-value]


def load_lightning_checkpoint(
    ckpt_path: Union[str, Path],
    *,
    map_location: Any = None,
) -> Dict[str, Any]:
    """
    Load a Lightning checkpoint (dict) safely with map_location.
    Returns the full checkpoint mapping.
    """
    map_location = map_location or _cpu_map_location()
    return _load_torch_file(Path(ckpt_path), map_location=map_location)


def resume_trainer_if_available(
    trainer: "pl.Trainer",  # type: ignore[name-defined]
    ckpt_dir: Union[str, Path],
    *,
    prefer: str = "best",  # 'best' | 'last'
    monitor: Optional[str] = None,
    mode: str = "min",
) -> Optional[Path]:
    """
    If a checkpoint exists, set `trainer.ckpt_path` (PL >= 2) and return the chosen path.

    Args:
      prefer: 'best' or 'last'
      monitor/mode: used for 'best' selection

    Returns:
      The checkpoint path chosen, or None if none found.
    """
    if pl is None:
        return None
    ckpt_path: Optional[Path] = None
    if prefer == "best":
        ckpt_path = find_best_checkpoint(ckpt_dir, monitor=monitor, mode=mode)
    if ckpt_path is None:
        ckpt_path = find_last_checkpoint(ckpt_dir)

    if ckpt_path is None:
        return None

    # Lightning 2.x accepts `ckpt_path` in Trainer.fit(...); here we store on trainer for later.
    try:
        setattr(trainer, "ckpt_path", str(ckpt_path))
    except Exception:
        pass
    return ckpt_path


# =============================================================================
# Public: averaging checkpoints
# =============================================================================
def average_checkpoints(
    ckpt_paths: Sequence[Union[str, Path]],
    *,
    out_path: Optional[Union[str, Path]] = None,
    map_location: Any = None,
    ema_beta: Optional[float] = None,
    cast_dtype: Optional[str] = None,        # e.g., "float32", "float16", "bfloat16"
    device: Optional[str] = None,            # e.g., "cpu", "cuda"
) -> Dict[str, Any]:
    """
    Average multiple checkpoints' `state_dict` into a single weights dict.

    Modes:
      • Mean (default): simple arithmetic mean across all provided ckpts.
      • EMA: if `ema_beta` in (0,1) is given, computes exponential moving average across
             checkpoints sorted by epoch (ascending). Useful for stabilizing inference.

    Args:
      ckpt_paths: list of .ckpt files
      out_path: optional file to write the resulting raw `state_dict` via `torch.save`
      map_location: map location (default CPU)
      ema_beta: optional EMA factor in (0, 1). If None => mean.
      cast_dtype: optionally cast resulting tensors to dtype (name string)
      device: optionally move resulting state to device name

    Returns:
      The averaged `state_dict` (a dict of tensor weights).
    """
    _require_torch()
    map_location = map_location or _cpu_map_location()
    paths: List[Path] = [Path(p) for p in ckpt_paths if p]
    if not paths:
        raise ValueError("No checkpoints provided for averaging.")

    # Load all state dicts
    pieces: List[Tuple[Path, Dict[str, Any], Optional[int]]] = []
    for p in paths:
        ckpt = _load_torch_file(p, map_location=map_location)
        state = _state_dict_from_checkpoint(ckpt)
        if state is None:
            if isinstance(ckpt, Mapping):
                state = ckpt  # type: ignore
            else:
                raise RuntimeError(f"Unrecognized checkpoint structure: {p}")
        # Epoch provenance (if present)
        epoch = _try_parse_epoch_from_name(p)
        for key in _EPOCH_FROM_STATE_RE_KEYS:
            if epoch is None and isinstance(ckpt, Mapping) and key in ckpt:
                try:
                    epoch = int(ckpt[key])  # type: ignore[arg-type]
                except Exception:
                    pass
        pieces.append((p, dict(state), epoch))

    # Sort for EMA if needed
    if ema_beta is not None:
        if not (0.0 < ema_beta < 1.0):
            raise ValueError("ema_beta must be in (0, 1).")
        pieces.sort(key=lambda t: (t[2] if t[2] is not None else -1))

        avg: Dict[str, Any] = {}
        for _, sd, _ in pieces:
            if not avg:
                avg = {k: v.clone() if hasattr(v, "clone") else v for k, v in sd.items()}
                continue
            for k, v in sd.items():
                if k not in avg:
                    avg[k] = v.clone() if hasattr(v, "clone") else v
                else:
                    # avg = beta * avg + (1 - beta) * v
                    if hasattr(avg[k], "mul_") and hasattr(v, "mul"):
                        avg[k].mul_(ema_beta).add_(v, alpha=(1.0 - ema_beta))
                    else:  # fallback (not in-place)
                        avg[k] = ema_beta * avg[k] + (1.0 - ema_beta) * v  # type: ignore[operator]
        out_state = avg
    else:
        # Arithmetic mean
        sum_state: Dict[str, Any] = {}
        counts: Dict[str, int] = {}
        for _, sd, _ in pieces:
            for k, v in sd.items():
                if k not in sum_state:
                    sum_state[k] = v.clone() if hasattr(v, "clone") else v
                    counts[k] = 1
                else:
                    if hasattr(sum_state[k], "add_"):
                        sum_state[k].add_(v)
                    else:
                        sum_state[k] = sum_state[k] + v  # type: ignore[operator]
                    counts[k] += 1
        out_state: Dict[str, Any] = {}
        for k, v in sum_state.items():
            c = counts.get(k, 1)
            if hasattr(v, "div_"):
                out_state[k] = v.div(c)
            else:
                out_state[k] = v / c  # type: ignore[operator]

    # Optional dtype/device cast
    if cast_dtype is not None:
        to_dtype = getattr(torch, cast_dtype, None)
        if to_dtype is None:
            raise ValueError(f"Unknown cast_dtype: {cast_dtype}")
        for k, v in out_state.items():
            if hasattr(v, "to"):
                out_state[k] = v.to(dtype=to_dtype)
    if device is not None:
        for k, v in out_state.items():
            if hasattr(v, "to"):
                out_state[k] = v.to(device=device)

    # Optional write
    if out_path is not None:
        Path(out_path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(out_state, str(out_path))  # type: ignore[arg-type]

    return out_state