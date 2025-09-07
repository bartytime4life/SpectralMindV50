# src/spectramind/train/callbacks.py
"""
SpectraMind V50 — Training Callbacks
------------------------------------
Factory helpers to build PyTorch Lightning callbacks from config, plus
lightweight custom callbacks for JSONL metrics logging and epoch timing.

Exports
-------
- build_callbacks(cfg: dict, ckpt_dir: Path, logs_dir: Path)
    -> tuple[list[pl.Callback], ModelCheckpoint]

- get_checkpoint_callback(cfg: dict, ckpt_dir: Path) -> ModelCheckpoint
- get_early_stopping_callback(cfg: dict) -> Optional[EarlyStopping]
- get_lr_monitor(cfg: dict) -> Optional[LearningRateMonitor]

Custom callbacks
----------------
- JsonlMetricsLogger: writes per-epoch metrics to a JSONL file (train/val/test)
- EpochTimeCallback: measures epoch durations (sec) and logs them

Notes
-----
- This module assumes PyTorch Lightning is installed when used. We guard the
  imports to avoid import errors on non-training contexts.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Optional heavy deps (required at runtime when building callbacks) ---
try:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
except Exception as _e:  # pragma: no cover
    pl = None
    ModelCheckpoint = object  # type: ignore
    EarlyStopping = object  # type: ignore
    LearningRateMonitor = object  # type: ignore
    _IMPORT_ERROR = _e
else:
    _IMPORT_ERROR = None


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------

def _require_pl() -> None:
    if _IMPORT_ERROR is not None:
        raise RuntimeError(
            "This module requires `pytorch-lightning`. "
            "Please install it to use training callbacks."
        ) from _IMPORT_ERROR


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def _as_bool(val: Any, default: bool) -> bool:
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.strip().lower() in {"1", "true", "yes", "y", "on"}
    if isinstance(val, (int, float)):
        return bool(val)
    return default


# ---------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------

class JsonlMetricsLogger(pl.Callback):  # type: ignore[name-defined]
    """
    Write scalar metrics to a JSONL file at the end of train/val/test epochs.

    Each line is a JSON object: {"phase": "train|val|test", "epoch": int, **metrics}
    """

    def __init__(self, jsonl_path: Path) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    def _write_row(self, row: Dict[str, Any]) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if _is_scalar(v)}
        self._write_row({"phase": "train", "epoch": epoch, **metrics})

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if _is_scalar(v)}
        self._write_row({"phase": "val", "epoch": epoch, **metrics})

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        epoch = trainer.current_epoch
        metrics = {k: float(v) for k, v in trainer.callback_metrics.items() if _is_scalar(v)}
        self._write_row({"phase": "test", "epoch": epoch, **metrics})


def _is_scalar(x: Any) -> bool:
    try:
        # Torch/NumPy scalars, plain numbers
        float(x)
        return True
    except Exception:
        return False


class EpochTimeCallback(pl.Callback):  # type: ignore[name-defined]
    """
    Measure training epoch durations and log them as 'epoch_time_sec'.

    Adds the metric to `trainer.callback_metrics` so it is visible to loggers
    and checkpoint monitors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._t0: Optional[float] = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._t0 is not None:
            dt = time.perf_counter() - self._t0
            # Use log_dict to be logger-agnostic
            pl_module.log_dict({"epoch_time_sec": dt}, prog_bar=False, logger=True)


# ---------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------

def get_checkpoint_callback(cfg: Dict[str, Any], ckpt_dir: Path) -> ModelCheckpoint:
    """Construct a ModelCheckpoint from config."""
    _require_pl()
    ck_cfg = _cfg_get(cfg, "callbacks.checkpoint", {}) or {}
    monitor = ck_cfg.get("monitor", "val_loss")
    mode = ck_cfg.get("mode", "min")
    filename = ck_cfg.get("filename", "epoch{epoch:03d}-val{val_loss:.5f}")

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=int(ck_cfg.get("save_top_k", 1)),
        save_last=_as_bool(ck_cfg.get("save_last", True), True),
        auto_insert_metric_name=True,
    )
    return ckpt


def get_early_stopping_callback(cfg: Dict[str, Any]) -> Optional[EarlyStopping]:
    """Construct an EarlyStopping callback if enabled."""
    _require_pl()
    es_cfg = _cfg_get(cfg, "callbacks.early_stopping", {}) or {}
    enable = _as_bool(es_cfg.get("enable", True), True)
    if not enable:
        return None
    return EarlyStopping(
        monitor=es_cfg.get("monitor", "val_loss"),
        mode=es_cfg.get("mode", "min"),
        patience=int(es_cfg.get("patience", 5)),
        min_delta=float(es_cfg.get("min_delta", 0.0)),
        verbose=False,
    )


def get_lr_monitor(cfg: Dict[str, Any]) -> Optional[LearningRateMonitor]:
    """Construct a LearningRateMonitor callback if enabled."""
    _require_pl()
    if not _as_bool(_cfg_get(cfg, "callbacks.lr_monitor", True), True):
        return None
    return LearningRateMonitor(logging_interval="step")


def build_callbacks(
    cfg: Dict[str, Any],
    ckpt_dir: Path,
    logs_dir: Path,
) -> Tuple[list, ModelCheckpoint]:
    """
    Build the full callback list from config.

    Includes:
      - ModelCheckpoint (returned separately as well)
      - optional EarlyStopping
      - optional LearningRateMonitor
      - JsonlMetricsLogger at logs_dir/metrics.jsonl
      - EpochTimeCallback
    """
    _require_pl()
    callbacks: list = []

    ckpt_cb = get_checkpoint_callback(cfg, ckpt_dir)
    callbacks.append(ckpt_cb)

    es_cb = get_early_stopping_callback(cfg)
    if es_cb:
        callbacks.append(es_cb)

    lr_cb = get_lr_monitor(cfg)
    if lr_cb:
        callbacks.append(lr_cb)

    # Always enable compact, local JSONL logging (small, portable)
    metrics_jsonl = Path(logs_dir) / "metrics.jsonl"
    callbacks.append(JsonlMetricsLogger(metrics_jsonl))

    # Epoch timing (helps profiling in Kaggle/CI)
    callbacks.append(EpochTimeCallback())

    return callbacks, ckpt_cb
