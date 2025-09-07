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
- get_swa_callback(cfg: dict) -> Optional[StochasticWeightAveraging]
- get_model_summary(cfg: dict) -> Optional[ModelSummary]

Custom callbacks
----------------
- JsonlMetricsLogger: writes per-epoch metrics to a JSONL file (train/val/test)
- EpochTimeCallback: measures epoch durations (sec) and logs them

Notes
-----
- Imports are guarded so importing this module won’t fail in non-training contexts.
- Writing to disk is rank-zero only (safe for DDP).
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

# --- Optional heavy deps (required only when building callbacks) ---
try:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
    # Optional extras: SWA, ModelSummary
    try:
        from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelSummary
    except Exception:  # older PL
        StochasticWeightAveraging = None  # type: ignore
        ModelSummary = None  # type: ignore

    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except Exception:
        # Fallback shim
        def rank_zero_only(fn):  # type: ignore
            def _wrap(*args, **kwargs):
                return fn(*args, **kwargs)
            return _wrap
    _PL_IMPORT_ERROR: Optional[BaseException] = None
except Exception as _e:  # pragma: no cover
    pl = None  # type: ignore
    ModelCheckpoint = object  # type: ignore
    EarlyStopping = object  # type: ignore
    LearningRateMonitor = object  # type: ignore
    StochasticWeightAveraging = None  # type: ignore
    ModelSummary = None  # type: ignore
    def rank_zero_only(fn):  # type: ignore
        def _wrap(*args, **kwargs):
            return fn(*args, **kwargs)
        return _wrap
    _PL_IMPORT_ERROR = _e


# ---------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------
def _require_pl() -> None:
    if _PL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "This module requires `pytorch-lightning` at runtime to build callbacks. "
            "Please install it (e.g., `pip install pytorch-lightning`)."
        ) from _PL_IMPORT_ERROR


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


def _now_iso() -> str:
    # Always UTC for portability/repro
    return datetime.now(tz=timezone.utc).isoformat(timespec="seconds")


def _is_number_like(x: Any) -> bool:
    try:
        float(x)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------
# Custom callbacks
# ---------------------------------------------------------------------
class JsonlMetricsLogger(pl.Callback):  # type: ignore[name-defined]
    """
    Write scalar metrics to a JSONL file at the end of train/val/test epochs.

    Each line: {"ts":"...Z","phase":"train|val|test","epoch":int,"global_step":int, **metrics}
    Only rank-zero writes (safe under DDP).
    """

    def __init__(self, jsonl_path: Path) -> None:
        super().__init__()
        self.jsonl_path = Path(jsonl_path)
        self.jsonl_path.parent.mkdir(parents=True, exist_ok=True)

    @rank_zero_only
    def _write_row(self, row: Dict[str, Any]) -> None:
        # Ensure ASCII off so keys/values are readable; one JSON per line
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")

    def _collect_metrics(self, trainer) -> Dict[str, float]:
        # callback_metrics may contain tensors; coerce scalars only
        out: Dict[str, float] = {}
        for k, v in dict(trainer.callback_metrics).items():
            if _is_number_like(v):
                out[k] = float(v)  # type: ignore[arg-type]
        return out

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        row = {
            "ts": _now_iso(),
            "phase": "train",
            "epoch": int(trainer.current_epoch),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            **self._collect_metrics(trainer),
        }
        self._write_row(row)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        row = {
            "ts": _now_iso(),
            "phase": "val",
            "epoch": int(trainer.current_epoch),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            **self._collect_metrics(trainer),
        }
        self._write_row(row)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        row = {
            "ts": _now_iso(),
            "phase": "test",
            "epoch": int(trainer.current_epoch),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            **self._collect_metrics(trainer),
        }
        self._write_row(row)


class EpochTimeCallback(pl.Callback):  # type: ignore[name-defined]
    """
    Measure training epoch durations and log them as 'epoch_time_sec'.
    Adds to logger via `pl_module.log_dict(...)`; visible to checkpoint monitors.
    """

    def __init__(self) -> None:
        super().__init__()
        self._t0: Optional[float] = None

    def on_train_epoch_start(self, trainer, pl_module) -> None:
        self._t0 = time.perf_counter()

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        if self._t0 is not None:
            dt = time.perf_counter() - self._t0
            pl_module.log_dict({"epoch_time_sec": float(dt)}, prog_bar=False, logger=True)


# ---------------------------------------------------------------------
# Factory functions (each individually buildable from cfg)
# ---------------------------------------------------------------------
def get_checkpoint_callback(cfg: Dict[str, Any], ckpt_dir: Path) -> ModelCheckpoint:
    """Construct a ModelCheckpoint from config."""
    _require_pl()
    ck_cfg = _cfg_get(cfg, "callbacks.checkpoint", {}) or {}
    monitor = ck_cfg.get("monitor", "val_loss")
    mode = ck_cfg.get("mode", "min")
    filename = ck_cfg.get("filename", "epoch{epoch:03d}-{monitor:.5f}")

    ckpt = ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=int(ck_cfg.get("save_top_k", 1)),
        save_last=_as_bool(ck_cfg.get("save_last", True), True),
        auto_insert_metric_name=True,
        every_n_epochs=int(ck_cfg.get("every_n_epochs", 1)),
        enable_version_counter=_as_bool(ck_cfg.get("enable_version_counter", True), True),
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
        check_on_train_epoch_end=_as_bool(es_cfg.get("check_on_train_epoch_end", False), False),
        verbose=_as_bool(es_cfg.get("verbose", False), False),
    )


def get_lr_monitor(cfg: Dict[str, Any]) -> Optional[LearningRateMonitor]:
    """Construct a LearningRateMonitor callback if enabled."""
    _require_pl()
    lr_cfg = _cfg_get(cfg, "callbacks.lr_monitor", {}) or {}
    enable = _as_bool(lr_cfg.get("enable", True), True)
    if not enable:
        return None
    interval = lr_cfg.get("logging_interval", "step")  # "step" or "epoch"
    log_momentum = _as_bool(lr_cfg.get("log_momentum", False), False)
    return LearningRateMonitor(logging_interval=interval, log_momentum=log_momentum)


def get_swa_callback(cfg: Dict[str, Any]) -> Optional["StochasticWeightAveraging"]:
    """
    Optional Stochastic Weight Averaging. Safe no-op if unsupported PL version.
    """
    _require_pl()
    if StochasticWeightAveraging is None:
        return None
    swa_cfg = _cfg_get(cfg, "callbacks.swa", {}) or {}
    enable = _as_bool(swa_cfg.get("enable", False), False)
    if not enable:
        return None
    # Defaults follow PL recommendations; user may override in cfg
    swa_lrs = float(swa_cfg.get("swa_lrs", 1e-3))
    anneal_epochs = int(swa_cfg.get("anneal_epochs", 10))
    anneal_strategy = swa_cfg.get("anneal_strategy", "cos")
    return StochasticWeightAveraging(swa_lrs=swa_lrs, anneal_epochs=anneal_epochs, anneal_strategy=anneal_strategy)


def get_model_summary(cfg: Dict[str, Any]) -> Optional["ModelSummary"]:
    """
    Optional model summary printer (depth configurable). No-op on older PL.
    """
    _require_pl()
    if ModelSummary is None:
        return None
    ms_cfg = _cfg_get(cfg, "callbacks.model_summary", {}) or {}
    enable = _as_bool(ms_cfg.get("enable", True), True)
    if not enable:
        return None
    max_depth = int(ms_cfg.get("max_depth", 2))
    return ModelSummary(max_depth=max_depth)


# ---------------------------------------------------------------------
# Main builder
# ---------------------------------------------------------------------
def build_callbacks(
    cfg: Dict[str, Any],
    ckpt_dir: Path,
    logs_dir: Path,
) -> Tuple[List["pl.Callback"], "ModelCheckpoint"]:
    """
    Build the full callback list from config.

    Includes:
      - ModelCheckpoint (returned separately as well)
      - optional EarlyStopping
      - optional LearningRateMonitor
      - optional StochasticWeightAveraging
      - optional ModelSummary
      - JsonlMetricsLogger at logs_dir/metrics.jsonl (always)
      - EpochTimeCallback (always)
    """
    _require_pl()
    callbacks: List["pl.Callback"] = []

    # Core: checkpoint
    ckpt_cb = get_checkpoint_callback(cfg, ckpt_dir)
    callbacks.append(ckpt_cb)

    # Optional blocks driven by cfg
    es_cb = get_early_stopping_callback(cfg)
    if es_cb:
        callbacks.append(es_cb)

    lr_cb = get_lr_monitor(cfg)
    if lr_cb:
        callbacks.append(lr_cb)

    swa_cb = get_swa_callback(cfg)
    if swa_cb:
        callbacks.append(swa_cb)

    ms_cb = get_model_summary(cfg)
    if ms_cb:
        callbacks.append(ms_cb)

    # Always-on lightweight local logging
    metrics_jsonl = Path(logs_dir) / "metrics.jsonl"
    callbacks.append(JsonlMetricsLogger(metrics_jsonl))

    # Epoch timing (helps profiling in Kaggle/CI)
    callbacks.append(EpochTimeCallback())

    return callbacks, ckpt_cb
