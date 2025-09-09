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
# The functions are Hydra-friendly but do not *require* Hydra — you can call
# them with plain dicts as well. This module avoids side effects (no DVC/IO
# beyond explicit paths) and uses guarded imports to behave well on Kaggle/CI.
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence, Tuple, Union

# --- Torch / Lightning (guarded) ------------------------------------------------
try:  # pragma: no cover
    import torch
    import pytorch_lightning as pl
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


def _load_state_into_model(model: Any, ckpt_path: Union[str, Path], strict: bool = True) -> Dict[str, Any]:
    """
    Load state dict from a Lightning checkpoint or a raw state_dict checkpoint.
    Returns the raw checkpoint dict that was loaded.
    """
    _require_torch()
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(str(ckpt_path), map_location="cpu")

    state_dict = None
    if isinstance(checkpoint, dict):
        # Lightning checkpoint typically has 'state_dict'
        if "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            state_dict = checkpoint["state_dict"]
        else:
            # try to treat the entire dict as state_dict
            state_dict = checkpoint
    if state_dict is None:
        raise RuntimeError(f"Unrecognized checkpoint format: {ckpt_path}")

    # If keys have 'model.'/'module.' prefixes adjust automatically
    missing, unexpected = model.load_state_dict(state_dict, strict=strict)
    if missing or unexpected:
        # Try a shallow fix: remove leading 'model.' or 'module.'
        def _strip_prefix(sd: Dict[str, torch.Tensor], pref: str) -> Dict[str, torch.Tensor]:
            return {k[len(pref):] if k.startswith(pref) else k: v for k, v in sd.items()}

        altered = False
        for pref in ("model.", "module."):
            _sd = _strip_prefix(state_dict, pref)
            try:
                _missing, _unexpected = model.load_state_dict(_sd, strict=strict)
                if not _missing and not _unexpected:
                    state_dict = _sd
                    altered = True
                    break
            except Exception:
                pass
        if not altered and (missing or unexpected):
            raise RuntimeError(
                "State dict mismatch after load.\n"
                f"Missing keys: {missing}\nUnexpected keys: {unexpected}"
            )
    return checkpoint


def _ensure_eval_cpu(model: Any) -> Any:
    model.eval()
    # Move recursively to CPU (Lightning / nn.Module agnostic)
    for p in model.parameters():
        if p.device.type != "cpu":
            p.data = p.data.cpu()
    for b in model.buffers():
        if b.device.type != "cpu":
            b.data = b.data.cpu()
    return model


# =============================================================================
# Public API
# =============================================================================

def restore_model_from_checkpoint(cfg: Union[Dict[str, Any], DictConfig],
                                  ckpt_path: Union[str, Path],
                                  strict: bool = True) -> Any:
    """
    Build a model from `cfg` and load weights from `ckpt_path`.
    Returns the model in eval mode on CPU.
    """
    _require_torch()
    cfg_pure = _to_pure_container(cfg)
    model = _build_model_from_cfg(cfg_pure)
    _load_state_into_model(model, ckpt_path, strict=strict)
    return _ensure_eval_cpu(model)


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
                       example_inputs: Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]],
                       method: str = "script") -> Path:
    """
    Export to TorchScript. `method` in {"script","trace"}.
    """
    _require_torch()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    model = _ensure_eval_cpu(model)
    if method == "script":
        scripted = torch.jit.script(model)
    elif method == "trace":
        scripted = torch.jit.trace(model, example_inputs)
    else:
        raise ValueError("TorchScript export method must be 'script' or 'trace'.")

    scripted.save(str(out_path))
    return out_path


def export_onnx(model: Any,
                out_path: Union[str, Path],
                example_inputs: Union[torch.Tensor, Tuple, Dict[str, torch.Tensor]],
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

    model = _ensure_eval_cpu(model)
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


def strip_lightning_checkpoint(ckpt_path: Union[str, Path],
                               out_path: Union[str, Path]) -> Path:
    """
    Create a slim checkpoint from a Lightning checkpoint by keeping only `state_dict`.
    """
    _require_torch()
    ckpt_path = Path(ckpt_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ckpt = torch.load(str(ckpt_path), map_location="cpu")
    if not isinstance(ckpt, dict):
        raise RuntimeError("Unexpected checkpoint format.")
    state = ckpt.get("state_dict", None)
    if state is None:
        # If it looks like a pure state_dict already, just rewrite it
        if all(isinstance(v, torch.Tensor) for v in ckpt.values()):
            state = ckpt
        else:
            raise RuntimeError("No 'state_dict' found in Lightning checkpoint.")
    torch.save(state, str(out_path))
    return out_path


# =============================================================================
# Higher-level convenience
# =============================================================================

def export_all_from_cfg(cfg: Union[Dict[str, Any], DictConfig],
                        ckpt_path: Union[str, Path],
                        out_dir: Union[str, Path],
                        formats: Sequence[str] = ("state_dict",),
                        example_input_shape: Optional[Sequence[int]] = None,
                        torchscript_method: str = "script",
                        onnx_opset: int = 17,
                        onnx_dynamic_axes: Optional[Dict[str, Dict[int, str]]] = None,
                        input_names: Optional[Sequence[str]] = None,
                        output_names: Optional[Sequence[str]] = None,
                        strict: bool = True) -> Dict[str, str]:
    """
    One-shot export using a training config and a checkpoint path.

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
    example_input_shape : list[int]
        If provided and formats include torchscript/onnx with tracing, we construct a
        dummy tensor: torch.zeros(*shape). For dict/tuple inputs, pass via the lower-level
        `export_torchscript` / `export_onnx` APIs directly.
    torchscript_method : str
        "script" or "trace".
    onnx_opset : int
        Opset version for ONNX export.
    onnx_dynamic_axes : dict
        Optional dynamic axes mapping for ONNX export.
    input_names / output_names : list[str]
        Optional names for ONNX inputs/outputs.
    strict : bool
        Strict loading of state dict.

    Returns
    -------
    Dict[str, str] : mapping of format → output filepath
    """
    _require_torch()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = restore_model_from_checkpoint(cfg, ckpt_path, strict=strict)
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
            if example_input_shape is None:
                raise ValueError("Tracing requires `example_input_shape` or example inputs.")
            example = torch.zeros(*example_input_shape)
            export_torchscript(model, ts_path, example_inputs=example, method="trace")
        else:
            export_torchscript(model, ts_path, example_inputs=None, method="script")
        results["torchscript"] = str(ts_path)

    # ONNX
    if "onnx" in formats:
        onnx_path = out_dir / "model.onnx"
        # For tracing-based ONNX export, we need example inputs
        if example_input_shape is None:
            raise ValueError("ONNX export requires `example_input_shape` or explicit example inputs.")
        example = torch.zeros(*example_input_shape)
        export_onnx(
            model,
            onnx_path,
            example_inputs=example,
            opset=onnx_opset,
            dynamic_axes=onnx_dynamic_axes,
            input_names=input_names,
            output_names=output_names,
        )
        results["onnx"] = str(onnx_path)

    # Artifacts manifest
    with (out_dir / "export_manifest.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "source_ckpt": str(ckpt_path),
                "formats": list(formats),
                "artifacts": results,
            },
            f,
            indent=2,
            sort_keys=True,
        )
    return results


__all__ = [
    "restore_model_from_checkpoint",
    "export_state_dict",
    "export_torchscript",
    "export_onnx",
    "strip_lightning_checkpoint",
    "average_checkpoints",
    "export_all_from_cfg",
]
