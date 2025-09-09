# src/spectramind/train/loop.py
# =============================================================================
# SpectraMind V50 — Training Loop Utilities
# -----------------------------------------------------------------------------
# Provides high-level orchestration around PyTorch Lightning's Trainer
# for the NeurIPS 2025 Ariel Data Challenge. Handles:
#   • Hydra-configurable Trainer creation (instantiate or dict → kwargs)
#   • Callback and logger registration (append-safe)
#   • Reproducibility toggles (deterministic/benchmark/fast_dev_run passthrough)
#   • Graceful resume from checkpoints
#   • Rank-zero safe printing/logging
#
# Design: Keep orchestration minimal here; heavy logic belongs in
# datasets.py, loggers.py, ckpt.py, and train.py.
# =============================================================================

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, List, Tuple

try:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping  # noqa: F401
    from pytorch_lightning.loggers import Logger  # noqa: F401
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except Exception as _e:  # pragma: no cover
    pl = None  # type: ignore

    def rank_zero_only(fn):  # type: ignore
        def _wrap(*a, **k):  # noqa: D401
            return fn(*a, **k)
        return _wrap

    Callback = object  # type: ignore
    ModelCheckpoint = object  # type: ignore
    EarlyStopping = object  # type: ignore
    Logger = object  # type: ignore
    _PL_IMPORT_ERROR = _e
else:
    _PL_IMPORT_ERROR = None

try:  # pragma: no cover
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, DictConfig
except Exception:
    instantiate = None  # type: ignore
    OmegaConf = None  # type: ignore
    DictConfig = Any  # type: ignore


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _ensure_pl() -> None:
    if _PL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "This utility requires `pytorch_lightning` at runtime."
        ) from _PL_IMPORT_ERROR


@rank_zero_only
def _rz_print(msg: str) -> None:
    print(msg, flush=True)


def _to_pure(obj: Any) -> Dict[str, Any]:
    if OmegaConf is not None and isinstance(obj, DictConfig):
        return dict(OmegaConf.to_container(obj, resolve=True))
    return dict(obj)


def _append_callbacks(trainer: "pl.Trainer", cbs: Optional[List["Callback"]]) -> None:  # type: ignore[name-defined]
    if not cbs:
        return
    # Lightning >= 1.7 exposes trainer.callbacks list
    for cb in cbs:
        trainer.callbacks.append(cb)


def _append_loggers(trainer: "pl.Trainer", lgs: Optional[List["Logger"]]) -> None:  # type: ignore[name-defined]
    if not lgs:
        return
    # attach additional logger instances without clobbering the primary one
    if not hasattr(trainer, "loggers"):
        # older versions: single logger; if one exists, keep it
        if getattr(trainer, "logger", None) and trainer.logger is not True:
            # nothing we can do; user should pass via Trainer(logger=[...])
            return
        trainer.logger = lgs[0] if len(lgs) == 1 else lgs  # type: ignore[attr-defined]
        return
    # newer PL supports .loggers list
    for lg in lgs:
        trainer.loggers.append(lg)


def _maybe_resume(trainer: "pl.Trainer", ckpt_path: Optional[Path]) -> None:  # type: ignore[name-defined]
    if not ckpt_path:
        return
    ckpt_path = Path(ckpt_path)
    if not ckpt_path.exists():
        _rz_print(f"⚠️  Resume checkpoint not found: {ckpt_path}")
        return
    _rz_print(f"▶️  Resuming from checkpoint: {ckpt_path}")
    # Lightning 2.x: prefer passing via fit(..., ckpt_path=...)
    # We keep also on trainer for convenience.
    setattr(trainer, "ckpt_path", str(ckpt_path))


# -----------------------------------------------------------------------------
# Trainer Construction
# -----------------------------------------------------------------------------
def _build_trainer_from_cfg(trainer_cfg: Any) -> "pl.Trainer":  # type: ignore[name-defined]
    """
    Best-effort builder:
      • If Hydra `instantiate` available and `_target_` present → instantiate
      • Else treat as dict of kwargs → pl.Trainer(**kwargs)
    """
    _ensure_pl()
    # Hydra route
    if instantiate is not None:
        try:
            box = _to_pure(trainer_cfg)
        except Exception:
            box = {}
        if isinstance(box, dict) and box.get("_target_"):
            return instantiate(box)  # type: ignore[no-any-return]
    # Plain kwargs route
    kwargs = _to_pure(trainer_cfg) if trainer_cfg is not None else {}
    # Guard against stray _target_ if present
    kwargs.pop("_target_", None)
    # Common safety toggles can live in cfg; PL will ignore unknown ones
    return pl.Trainer(**kwargs)  # type: ignore[call-arg]


def build_trainer(
    cfg: Dict[str, Any],
    callbacks: Optional[List["Callback"]] = None,  # type: ignore[name-defined]
    loggers: Optional[List["Logger"]] = None,      # type: ignore[name-defined]
    resume_from: Optional[Path] = None,
) -> "pl.Trainer":  # type: ignore[name-defined]
    """
    Build a PyTorch Lightning Trainer from config.

    Args:
        cfg: Dict/Hydra config (expects `trainer` section)
        callbacks: Extra callbacks to register (appended)
        loggers: Extra loggers to register (appended)
        resume_from: Optional checkpoint path to resume

    Returns:
        Configured pl.Trainer instance.
    """
    _ensure_pl()
    trainer_cfg = cfg.get("trainer", {}) or {}
    trainer = _build_trainer_from_cfg(trainer_cfg)
    _append_callbacks(trainer, callbacks)
    _append_loggers(trainer, loggers)
    _maybe_resume(trainer, resume_from)
    return trainer


# -----------------------------------------------------------------------------
# Training Execution
# -----------------------------------------------------------------------------
def run_training(
    trainer: "pl.Trainer",                  # type: ignore[name-defined]
    model: "pl.LightningModule",           # type: ignore[name-defined]
    datamodule: "pl.LightningDataModule",  # type: ignore[name-defined]
    *,
    ckpt_path: Optional[Union[str, Path]] = None,
) -> Tuple[Optional[Any], Optional[Any]]:
    """
    Execute training given a Trainer, model, and datamodule.

    Args:
        trainer: Lightning Trainer.
        model: LightningModule (SpectraMind model).
        datamodule: LightningDataModule providing loaders.
        ckpt_path: optional ckpt path (overrides trainer.ckpt_path if provided)

    Returns:
        (fit_result, val_result) as returned by pl.Trainer.fit/validate (PL may return None).
    """
    _ensure_pl()
    _rz_print("🚀 Starting training loop...")
    fit_result = None
    val_result = None
    try:
        fit_result = trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(ckpt_path) if ckpt_path else getattr(trainer, "ckpt_path", None))  # type: ignore[arg-type]
    except KeyboardInterrupt:
        _rz_print("⏸️  Training interrupted by user.")
    finally:
        _rz_print("🏁 Training finished.")
    # Some workflows want an immediate validation pass with best ckpt; caller can perform it.
    return fit_result, val_result


# -----------------------------------------------------------------------------
# Default factories (optional use)
# -----------------------------------------------------------------------------
def default_callbacks(cfg: Dict[str, Any]) -> List["Callback"]:  # type: ignore[name-defined]
    """
    Instantiate callbacks from a Hydra-like section: cfg['callbacks'] is a mapping of nodes.
    """
    if instantiate is None:
        return []
    cb_cfgs = cfg.get("callbacks", {}) or {}
    out: List["Callback"] = []
    for node in cb_cfgs.values():
        try:
            # Support both Hydra nodes and plain kwargs { _target_: ..., ... }
            obj = instantiate(node)  # type: ignore
            out.append(obj)
        except Exception:
            continue
    return out


def default_loggers(cfg: Dict[str, Any]) -> List["Logger"]:  # type: ignore[name-defined]
    """
    Instantiate loggers from a Hydra-like section: cfg['loggers'] is a mapping of nodes.
    """
    if instantiate is None:
        return []
    lg_cfgs = cfg.get("loggers", {}) or {}
    out: List["Logger"] = []
    for node in lg_cfgs.values():
        try:
            obj = instantiate(node)  # type: ignore
            out.append(obj)
        except Exception:
            continue
    return out