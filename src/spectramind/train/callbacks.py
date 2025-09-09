# src/spectramind/train/callbacks.py
"""
SpectraMind V50 — Training Callbacks
------------------------------------
Factory helpers to build PyTorch Lightning callbacks from config, plus
lightweight custom callbacks for JSONL metrics logging, epoch timing,
and (optional) Kaggle artifacts export.

Exports
-------
- build_callbacks(cfg: dict, ckpt_dir: Path, logs_dir: Path)
    -> tuple[list[pl.Callback], ModelCheckpoint]

- get_checkpoint_callback(cfg: dict, ckpt_dir: Path) -> ModelCheckpoint
- get_early_stopping_callback(cfg: dict) -> Optional[EarlyStopping]
- get_lr_monitor(cfg: dict) -> Optional[LearningRateMonitor]
- get_swa_callback(cfg: dict) -> Optional[StochasticWeightAveraging]
- get_model_summary(cfg: dict) -> Optional[ModelSummary]
- get_kaggle_artifacts_callback(cfg: dict, ckpt_dir: Path, logs_dir: Path)
    -> Optional[KaggleArtifactsCallback]

Custom callbacks
----------------
- JsonlMetricsLogger: writes per-epoch metrics to a JSONL file (train/val/test)
- EpochTimeCallback: measures epoch durations (sec) and logs them
- KaggleArtifactsCallback: exports best checkpoint + run summary to a folder
  (useful for downloading from Kaggle after the run)

Notes
-----
- Imports are guarded so importing this module won’t fail in non-training contexts.
- Writing to disk is rank-zero only (safe for DDP).
"""

from __future__ import annotations

import json
import shutil
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
    # Optional extras: SWA, ModelSummary (version-dependent)
    try:
        from pytorch_lightning.callbacks import StochasticWeightAveraging, ModelSummary
    except Exception:  # older PL
        StochasticWeightAveraging = None  # type: ignore
        ModelSummary = None  # type: ignore

    try:
        from pytorch_lightning.utilities.rank_zero import rank_zero_only
    except Exception:  # pragma: no cover
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
        if x is None:
            return False
        # torch / numpy safety without importing them at module import time
        if hasattr(x, "item"):  # torch.Tensor / numpy scalar
            x = x.item()
        float(x)
        return True
    except Exception:
        return False


def _to_float(x: Any) -> float:
    if hasattr(x, "item"):
        # torch / numpy scalar
        return float(x.item())
    return float(x)


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

    def _collect_metrics(self, trainer: "pl.Trainer") -> Dict[str, float]:  # type: ignore[name-defined]
        # callback_metrics may contain tensors; coerce scalars only
        out: Dict[str, float] = {}
        for k, v in dict(getattr(trainer, "callback_metrics", {})).items():
            if _is_number_like(v):
                try:
                    out[k] = _to_float(v)
                except Exception:
                    # be defensive: skip weird/unserializable metrics
                    continue
        return out

    def on_train_epoch_end(self, trainer, pl_module) -> None:
        row = {
            "ts": _now_iso(),
            "phase": "train",
            "epoch": int(getattr(trainer, "current_epoch", -1) or -1),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            **self._collect_metrics(trainer),
        }
        self._write_row(row)

    def on_validation_epoch_end(self, trainer, pl_module) -> None:
        row = {
            "ts": _now_iso(),
            "phase": "val",
            "epoch": int(getattr(trainer, "current_epoch", -1) or -1),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            **self._collect_metrics(trainer),
        }
        self._write_row(row)

    def on_test_epoch_end(self, trainer, pl_module) -> None:
        row = {
            "ts": _now_iso(),
            "phase": "test",
            "epoch": int(getattr(trainer, "current_epoch", -1) or -1),
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


class KaggleArtifactsCallback(pl.Callback):  # type: ignore[name-defined]
    """
    Export best checkpoint and a small run summary JSON into an artifacts directory,
    so they can be downloaded easily from Kaggle after the run.

    Config path:
      callbacks.kaggle_artifacts:
        enable: true
        dir: "./artifacts"
        best_name: "best.ckpt"
        summary_name: "run_summary.json"
    """
    def __init__(self, artifacts_dir: Path, best_ckpt_name: str = "best.ckpt",
                 summary_name: str = "run_summary.json") -> None:
        super().__init__()
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.best_ckpt_name = best_ckpt_name
        self.summary_name = summary_name

    @rank_zero_only
    def _write_summary(self, trainer: "pl.Trainer") -> None:  # type: ignore[name-defined]
        summary = {
            "ts": _now_iso(),
            "current_epoch": int(getattr(trainer, "current_epoch", -1) or -1),
            "global_step": int(getattr(trainer, "global_step", 0) or 0),
            "best_model_path": None,
            "best_monitor": None,
            "best_score": None,
        }
        # Try to resolve the active checkpoint callback
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if ckpt_cb and hasattr(ckpt_cb, "best_model_path"):
            summary["best_model_path"] = getattr(ckpt_cb, "best_model_path", None)
            summary["best_monitor"] = getattr(ckpt_cb, "monitor", None)
            score = getattr(ckpt_cb, "best_model_score", None)
            if score is not None:
                try:
                    summary["best_score"] = float(score.cpu().item())
                except Exception:
                    try:
                        summary["best_score"] = float(score)
                    except Exception:
                        summary["best_score"] = None

        with (self.artifacts_dir / self.summary_name).open("w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)

    @rank_zero_only
    def _copy_best_checkpoint(self, trainer: "pl.Trainer") -> None:  # type: ignore[name-defined]
        ckpt_cb = getattr(trainer, "checkpoint_callback", None)
        if not ckpt_cb or not getattr(ckpt_cb, "best_model_path", None):
            return
        src = Path(ckpt_cb.best_model_path)
        if not src.exists():
            return
        dst = self.artifacts_dir / self.best_ckpt_name
        try:
            shutil.copy2(src, dst)
        except Exception:
            # Best-effort fallback
            try:
                shutil.copy(src, dst)
            except Exception:
                pass

    def on_fit_end(self, trainer, pl_module) -> None:
        self._copy_best_checkpoint(trainer)
        self._write_summary(trainer)


# ---------------------------------------------------------------------
# Factory functions (each individually buildable from cfg)
# ---------------------------------------------------------------------
def get_checkpoint_callback(cfg: Dict[str, Any], ckpt_dir: Path) -> "ModelCheckpoint":
    """Construct a ModelCheckpoint from config."""
    _require_pl()
    ck_cfg = _cfg_get(cfg, "callbacks.checkpoint", {}) or {}
    enable = _as_bool(ck_cfg.get("enable", True), True)
    if not enable:
        # Still return a checkpoint object with disabled saving to keep type stable
        return ModelCheckpoint(dirpath=str(ckpt_dir), save_top_k=0, monitor=None)

    monitor = ck_cfg.get("monitor", "val_loss")
    mode = ck_cfg.get("mode", "min")
    filename = ck_cfg.get("filename", "epoch{epoch:03d}-{monitor:.5f}")

    # Some PL versions expect None to disable version counter
    enable_version_counter = _as_bool(ck_cfg.get("enable_version_counter", True), True)

    # Optional advanced toggles with sensible defaults
    save_top_k = int(ck_cfg.get("save_top_k", 1))
    save_last = _as_bool(ck_cfg.get("save_last", True), True)
    every_n_epochs = int(ck_cfg.get("every_n_epochs", 1))
    every_n_train_steps = ck_cfg.get("every_n_train_steps", None)
    if every_n_train_steps is not None:
        try:
            every_n_train_steps = int(every_n_train_steps)
        except Exception:
            every_n_train_steps = None
    save_weights_only = _as_bool(ck_cfg.get("save_weights_only", False), False)

    return ModelCheckpoint(
        dirpath=str(ckpt_dir),
        filename=filename,
        monitor=monitor,
        mode=mode,
        save_top_k=save_top_k,
        save_last=save_last,
        auto_insert_metric_name=True,
        every_n_epochs=every_n_epochs if every_n_train_steps is None else 0,
        every_n_train_steps=every_n_train_steps,
        enable_version_counter=enable_version_counter,
        save_weights_only=save_weights_only,
    )


def get_early_stopping_callback(cfg: Dict[str, Any]) -> Optional["EarlyStopping"]:
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


def get_lr_monitor(cfg: Dict[str, Any]) -> Optional["LearningRateMonitor"]:
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
    swa_lrs = float(swa_cfg.get("swa_lrs", 1e-3))
    anneal_epochs = int(swa_cfg.get("anneal_epochs", 10))
    anneal_strategy = swa_cfg.get("anneal_strategy", "cos")
    return StochasticWeightAveraging(
        swa_lrs=swa_lrs,
        anneal_epochs=anneal_epochs,
        anneal_strategy=anneal_strategy,
    )


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


def get_kaggle_artifacts_callback(
    cfg: Dict[str, Any],
    ckpt_dir: Path,
    logs_dir: Path,
) -> Optional["KaggleArtifactsCallback"]:
    """
    Optional Kaggle artifacts exporter. Writes best checkpoint (+ a JSON summary)
    into a folder (default: ./artifacts) for convenient download from Kaggle.

    callbacks.kaggle_artifacts:
      enable: true
      dir: "./artifacts"
      best_name: "best.ckpt"
      summary_name: "run_summary.json"
    """
    _require_pl()
    ka_cfg = _cfg_get(cfg, "callbacks.kaggle_artifacts", {}) or {}
    enable = _as_bool(ka_cfg.get("enable", False), False)
    if not enable:
        return None
    artifacts_dir = Path(ka_cfg.get("dir", "./artifacts"))
    best_name = str(ka_cfg.get("best_name", "best.ckpt"))
    summary_name = str(ka_cfg.get("summary_name", "run_summary.json"))
    return KaggleArtifactsCallback(artifacts_dir=artifacts_dir, best_ckpt_name=best_name, summary_name=summary_name)


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
      - optional KaggleArtifactsCallback (if enabled)
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

    # Optional: export artifacts for Kaggle
    ka_cb = get_kaggle_artifacts_callback(cfg, ckpt_dir, logs_dir)
    if ka_cb:
        callbacks.append(ka_cb)

    return callbacks, ckpt_cb