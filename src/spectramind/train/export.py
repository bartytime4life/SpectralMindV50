# src/spectramind/train/export.py
# =============================================================================
# SpectraMind V50 — Export Utilities
# -----------------------------------------------------------------------------
# Utilities to restore a trained LightningModule from a checkpoint and export it
# as:
#   • raw state_dict (.pt)
#   • TorchScript (script or trace)
#   • ONNX (with optional dynamic axes)
#
# Highlights / upgrades
# ---------------------
# • Guarded imports & side-effect free (Kaggle/CI safe)
# • Hydra-friendly (but works with plain dicts)
# • Prefix-robust state loading (handles 'model.' / 'module.' etc.)
# • Optional cast / freeze before export (dtype/device)
# • Flexible example-inputs support (tensor / tuple / dict / shape spec)
# • Simple dynamic-axes helper for ONNX
# • Optional multi-ckpt averaging (mean/EMA) before export
# • Clear, composable API surface
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

# --- Torch / Lightning (guarded) ------------------------------------------------
try:  # pragma: no cover
    import torch
    import pytorch_lightning as pl  # noqa: F401  (only for type/availability check)
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    pl = None  # type: ignore
    _TORCH_IMPORT_ERROR = _e
else:
    _TORCH_IMPORT_ERROR = None

# --- Hydra / OmegaConf (optional) ----------------------------------------------
try:  # pragma: no cover
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
except Exception:
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore

    def instantiate(*_a, **_k):  # type: ignore
        raise RuntimeError("Hydra is required to instantiate objects from config.")

# --- Local registry/builders (for non-_target_ models) --------------------------
from .registry import (
    get_model_builder,
    get_loss_builder,
)

# --- Reuse checkpoint averaging if needed --------------------------------------
from .ckpt import average_checkpoints  # re-export for convenience


# =============================================================================
# Helpers
# =============================================================================

def _require_torch() -> None:
    if _TORCH_IMPORT_ERROR is not None:
        raise RuntimeError("torch / pytorch_lightning are required for export.") from _TORCH_IMPORT_ERROR


def _to_pure_container(cfg: Any) -> Dict[str, Any]:
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        return dict(OmegaConf.to_container(cfg, resolve=True))
    return dict(cfg)


def _maybe_instantiate(node: Any, **overrides: Any) -> Any:
    """
    Hydra instantiate if `_target_` is present; else return None.
    """
    try:
        container = _to_pure_container(node)
    except Exception:
        container = {}
    if isinstance(container, dict) and container.get("_target_"):
        return instantiate({**container, **overrides})
    return None


def _build_model_from_cfg(cfg: Mapping[str, Any]) -> Any:
    """
    Build model using the same logic as training:
      1) If `model._target_` present → instantiate
      2) Else via registry: `model.name` (+ optional `loss.name`)
    """
    model_cfg = cfg.get("model")
    if model_cfg is None:
        raise RuntimeError("Missing `model` section in config for export.")
    # Prefer Hydra
    model = _maybe_instantiate(model_cfg)
    if model is not None:
        return model

    # Registry fallback
    model_name = model_cfg.get("name")
    if not model_name:
        raise RuntimeError("Provide either `model._target_` or `model.name` for export.")
    loss_name = cfg.get("loss", {}).get("name")

    model_builder = get_model_builder(model_name)
    criterion = get_loss_builder(loss_name)(cfg=cfg) if loss_name else None
    return model_builder(cfg=cfg, criterion=criterion)


def _strip_prefix(sd: Dict[str, Any], prefixes: Sequence[str]) -> Dict[str, Any]:
    out = sd
    for pref in prefixes:
        if any(k.startswith(pref) for k in out.keys()):
            out = {k[len(pref):] if k.startswith(pref) else k: v for k, v in out.items()}
    return out


def _state_from_any_checkpoint(ckpt: Mapping[str, Any]) -> Dict[str, Any]:
    """
    Get a state_dict from a Lightning or raw checkpoint mapping.
    """
    if "state_dict" in ckpt and isinstance(ckpt["state_dict"], Mapping):
        sd = dict(ckpt["state_dict"])
    else:
        # treat full mapping as state_dict if all (or most) values are tensors
        tensor_like = [k for k, v in ckpt.items() if hasattr(v, "size") or hasattr(v, "shape")]
        if tensor_like and len(tensor_like) >= max(8, int(0.7 * len(ckpt))):
            sd = dict(ckpt)
        else:
            raise RuntimeError("Unrecognized checkpoint structure (no 'state_dict' and not tensor-like mapping).")
    # be robust with prefixes
    sd = _strip_prefix(sd, ["model.", "module."])
    return sd


def _load_state_into_model(model: Any,
                           ckpt_path: Union[str, Path],
                           strict: bool = True) -> Dict[str, Any]:
    """
    Load state dict from a Lightning checkpoint or a raw state_dict checkpoint.
    Returns the raw checkpoint dict that was loaded.
    """
    _require_torch()
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise RuntimeError(f"Unexpected checkpoint type at {ckpt_path}")

    state_dict = _state_from_any_checkpoint(checkpoint)
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if getattr(missing, "__len__", None) and getattr(unexpected, "__len__", None):
        if missing or unexpected:
            raise RuntimeError(
                "State dict mismatch after load.\n"
                f"Missing keys: {list(missing)[:8]}{' ...' if len(missing) > 8 else ''}\n"
                f"Unexpected keys: {list(unexpected)[:8]}{' ...' if len(unexpected) > 8 else ''}"
            )
    return checkpoint


def _ensure_eval(model: Any, device: Optional[str] = None, dtype: Optional[str] = None, freeze: bool = True) -> Any:
    """
    Move to device/dtype, optionally freeze parameters, and set eval().
    """
    model.eval()
    # device
    if device is not None:
        model.to(device=device)
    else:
        # default to CPU for portability
        model.to(device="cpu")
    # dtype
    if dtype is not None:
        to_dtype = getattr(torch, dtype, None)
        if to_dtype is None:
            raise ValueError(f"Unknown dtype string: {dtype}")
        model.to(dtype=to_dtype)
    # freeze
    if freeze:
        for p in model.parameters():
            p.requires_grad_(False)
    return model


def _build_example_from_shape(shape: Sequence[int]) -> torch.Tensor:
    """
    Construct a zero tensor for tracing from a shape spec, e.g. [1, 283] or (1, T, C).
    """
    _require_torch()
    return torch.zeros(*shape)


# =============================================================================
# Public API
# =============================================================================

def restore_model_from_checkpoint(cfg: Union[Dict[str, Any], DictConfig],
                                  ckpt_path: Union[str, Path],
                                  strict: bool = True,
                                  *,
                                  device: Optional[str] = None,
                                  dtype: Optional[str] = None,
                                  freeze: bool = True) -> Any:
    """
    Build a model from `cfg` and load weights from `ckpt_path`.
    Returns the model in eval mode (device/dtype applied).
    """
    _require_torch()
    cfg_pure = _to_pure_container(cfg)
    model = _build_model_from_cfg(cfg_pure)
    _load_state_into_model(model, ckpt_path, strict=strict)
    return _ensure_eval(model, device=device, dtype=dtype, freeze=freeze)


def export_state_dict(model: Any,
                      out_path: Union[str, Path],
                      extra: Optional[Dict[str, Any]] = None) -> Path:
    """
    Save a raw `state_dict` checkpoint (.pt). Optionally bundles `extra` metadata (JSON sidecar).
    """
    _require_torch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_path))

    if extra:
        with (out_path.with_suffix(out_path.suffix + ".json")).open("w", encoding="utf-8") as f:
            json.dump(extra, f, indent=2, sort_keys=True)
    return out_path


def export_torchscript(model: Any,
                       out_path: Union[str, Path],
                       example_inputs: Optional[Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]]] = None,
                       method: str = "script") -> Path:
    """
    Export to TorchScript. `method` in {"script","trace"}.
    If tracing, `example_inputs` is required and can be a Tensor / tuple / dict.
    """
    _require_torch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if method not in ("script", "trace"):
        raise ValueError("TorchScript export method must be 'script' or 'trace'.")

    if method == "script":
        scripted = torch.jit.script(model)
    else:
        if example_inputs is None:
            raise ValueError("Tracing requires `example_inputs`.")
        scripted = torch.jit.trace(model, example_inputs)  # type: ignore[arg-type]

    scripted.save(str(out_path))
    return out_path


def export_onnx(model: Any,
                out_path: Union[str, Path],
                example_inputs: Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]],
                *,
                opset: int = 17,
                dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                input_names: Optional[Sequence[str]] = None,
                output_names: Optional[Sequence[str]] = None) -> Path:
    """
    Export to ONNX with optional dynamic axes. Requires torch.onnx.
    """
    _require_torch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    torch.onnx.export(
        model,
        example_inputs,
        str(out_path),
        opset_version=opset,
        input_names=list(input_names) if input_names else None,
        output_names=list(output_names) if output_names else None,
        dynamic_axes=dynamic_axes or None,
        do_constant_folding=True,
    )
    return out_path


# ------------------------ convenience: dynamic axes ------------------------ #

def make_dynamic_axes(batch_dim: Optional[int] = 0,
                      time_dim: Optional[int] = None,
                      input_names: Optional[Sequence[str]] = None,
                      output_names: Optional[Sequence[str]] = None) -> Dict[str, Dict[int, str]]:
    """
    Convenience builder for ONNX dynamic_axes mapping.
      • If batch_dim is not None: mark that dim as 'batch'
      • If time_dim is not None: mark that dim as 'time'
    """
    axes: Dict[str, Dict[int, str]] = {}
    def _mark(name: str) -> Dict[int, str]:
        marks: Dict[int, str] = {}
        if batch_dim is not None:
            marks[int(batch_dim)] = "batch"
        if time_dim is not None:
            marks[int(time_dim)] = "time"
        return marks

    for name in list(input_names or []):
        axes[name] = _mark(name)
    for name in list(output_names or []):
        axes[name] = _mark(name)
    return axes


# =============================================================================
# Higher-level convenience (one-shot export)
# =============================================================================

def export_all_from_cfg(cfg: Union[Dict[str, Any], DictConfig],
                        ckpt_path: Union[str, Path],
                        out_dir: Union[str, Path],
                        formats: Sequence[str] = ("state_dict",),
                        *,
                        example_inputs: Optional[Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]]] = None,
                        example_input_shape: Optional[Sequence[int]] = None,
                        torchscript_method: str = "script",
                        onnx_opset: int = 17,
                        onnx_dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                        input_names: Optional[Sequence[str]] = None,
                        output_names: Optional[Sequence[str]] = None,
                        strict: bool = True,
                        device: Optional[str] = None,
                        dtype: Optional[str] = None,
                        freeze: bool = True,
                        # multi-ckpt support (optional)
                        avg_ckpts: Optional[Sequence[Union[str, Path]]] = None,
                        avg_out_path: Optional[Union[str, Path]] = None,
                        avg_ema_beta: Optional[float] = None) -> Dict[str, str]:
    """
    One-shot export using a training config and a checkpoint path (or averaged checkpoints).

    Parameters
    ----------
    cfg : DictConfig or dict
        Composed Hydra config used for building the model.
    ckpt_path : str|Path
        Path to a (best or last) checkpoint to restore from.
    out_dir : str|Path
        Output directory to place exported files.
    formats : list[str]
        Any of {"state_dict", "torchscript", "onnx"}.
    example_inputs : Tensor | tuple | dict
        Explicit example inputs (for tracing/onnx); takes precedence over shape.
    example_input_shape : list[int]
        Shape spec fallback to construct a dummy tensor if tracing/onnx needs it.
    torchscript_method : str
        "script" or "trace".
    onnx_opset : int
        Opset version for ONNX export.
    onnx_dynamic_axes : dict
        Optional dynamic axes mapping for ONNX export (see make_dynamic_axes).
    input_names / output_names : list[str]
        Optional names for ONNX inputs/outputs.
    strict : bool
        Strict loading of state dict.
    device : str
        Target device for export (e.g. "cpu", "cuda").
    dtype : str
        Target dtype for export (e.g. "float32", "float16", "bfloat16").
    freeze : bool
        Freeze model params (requires_grad=False) for export.
    avg_ckpts : list[path]
        Optional list of checkpoints to average before export (mean/EMA). If provided,
        these are averaged and loaded into the model (ckpt_path is still recorded in manifest).
    avg_out_path : path
        Optional file path to save the averaged state_dict when avg_ckpts is used.
    avg_ema_beta : float
        EMA beta in (0,1) to use exponential moving average instead of mean.

    Returns
    -------
    Dict[str, str] : mapping of format → output filepath
    """
    _require_torch()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build fresh model
    model = _build_model_from_cfg(_to_pure_container(cfg))

    # Optionally average checkpoints before export
    if avg_ckpts:
        avg_state = average_checkpoints(
            avg_ckpts, out_path=avg_out_path, ema_beta=avg_ema_beta, map_location="cpu"
        )
        model.load_state_dict(avg_state, strict=strict)
    else:
        _load_state_into_model(model, ckpt_path, strict=strict)

    # Finalize eval/device/dtype & freeze
    model = _ensure_eval(model, device=device, dtype=dtype, freeze=freeze)

    results: Dict[str, str] = {}

    # state_dict
    if "state_dict" in formats:
        sd_path = out_dir / "model.state_dict.pt"
        export_state_dict(model, sd_path, extra={"source_ckpt": str(ckpt_path)})
        results["state_dict"] = str(sd_path)

    # TorchScript
    if "torchscript" in formats:
        ts_path = out_dir / f"model.torchscript.{torchscript_method}.pt"
        if torchscript_method == "trace":
            ex = example_inputs
            if ex is None:
                if example_input_shape is None:
                    raise ValueError("Tracing requires `example_inputs` or `example_input_shape`.")
                ex = _build_example_from_shape(example_input_shape)
            export_torchscript(model, ts_path, example_inputs=ex, method="trace")
        else:
            export_torchscript(model, ts_path, example_inputs=None, method="script")
        results["torchscript"] = str(ts_path)

    # ONNX
    if "onnx" in formats:
        onnx_path = out_dir / "model.onnx"
        ex = example_inputs
        if ex is None:
            if example_input_shape is None:
                raise ValueError("ONNX export requires `example_inputs` or `example_input_shape`.")
            ex = _build_example_from_shape(example_input_shape)
        export_onnx(
            model,
            onnx_path,
            example_inputs=ex,
            opset=onnx_opset,
            dynamic_axes=onnx_dynamic_axes,
            input_names=input_names,
            output_names=output_names,
        )
        results["onnx"] = str(onnx_path)

    # Artifacts manifest
    manifest = {
        "source_ckpt": str(ckpt_path),
        "averaged_from": [str(p) for p in (avg_ckpts or [])],
        "formats": list(formats),
        "artifacts": results,
        "dtype": dtype,
        "device": device or "cpu",
    }
    with (out_dir / "export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return results


__all__ = [
    "restore_model_from_checkpoint",
    "export_state_dict",
    "export_torchscript",
    "export_onnx",
    "make_dynamic_axes",
    "average_checkpoints",
    "export_all_from_cfg",
]