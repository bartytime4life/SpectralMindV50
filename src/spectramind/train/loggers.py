# src/spectramind/train/loggers.py
# =============================================================================
# SpectraMind V50 — Logger Builders
# -----------------------------------------------------------------------------
# Builds a list of PyTorch Lightning loggers driven by Hydra/Dict config. Guarded
# imports keep Kaggle/CI happy (no heavy deps unless explicitly enabled).
#
# Supported out of the box:
#   • CSVLogger
#   • TensorBoardLogger
#   • WandbLogger (optional; requires `wandb`)
#   • MLFlowLogger (optional; requires `mlflow`)
#
# Example (configs/train.yaml):
#
# logging:
#   csv:
#     enable: true
#     name: v50
#   tensorboard:
#     enable: true
#     default_hp_metric: false
#   wandb:
#     enable: false
#     project: spectramind-v50
#     entity: your_team
#     group: ${paths.exp_name}
#     tags: [v50, ariel, neurips2025]
#     mode: ${env:WANDB_MODE, "online"}  # "offline"|"dryrun"|"online"
#     log_model: false
#   mlflow:
#     enable: false
#     tracking_uri: ${env:MLFLOW_TRACKING_URI, null}
#     experiment_name: spectramind-v50
#     run_name: ${paths.run_name}
#     tags:
#       repo: spectramind-v50
#
# Notes:
#   • All loggers pick sensible defaults for save_dir/name/version based on `paths`.
#   • `paths` is the object returned by resolve_train_paths(cfg) (run_dir/logs_dir/exp_name/timestamp).
#   • This module does not import optional loggers unless enabled in config.
# =============================================================================

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

# ---- Guarded Lightning import ------------------------------------------------
try:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.loggers import CSVLogger, TensorBoardLogger
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except Exception as _e:  # pragma: no cover
    pl = None  # type: ignore

    def rank_zero_only(fn):  # type: ignore
        def _wrap(*a, **k):  # noqa: D401
            return fn(*a, **k)
        return _wrap

    CSVLogger = object  # type: ignore
    TensorBoardLogger = object  # type: ignore
    _PL_IMPORT_ERROR = _e
else:
    _PL_IMPORT_ERROR = None

# Optional backends are imported lazily only if enabled
_WANDB_AVAILABLE = False
_MLFLOW_AVAILABLE = False


# -----------------------------------------------------------------------------
# Internals
# -----------------------------------------------------------------------------
@rank_zero_only
def _log(msg: str) -> None:
    print(f"[SpectraMind][loggers] {msg}")


def _ensure_pl() -> None:
    if _PL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "`pytorch_lightning` is required to build loggers."
        ) from _PL_IMPORT_ERROR


def _as_path(p: Optional[str | Path]) -> Path:
    return Path(p) if isinstance(p, (str, Path)) else Path(".")


def _default_name_version(paths: Any, cfg_name: Optional[str], cfg_version: Optional[str]) -> tuple[str, str]:
    """
    Compute default (name, version) pair for loggers. If config provides name/version
    they take precedence. Otherwise use experiment/run identifiers from paths.
    """
    # name
    if cfg_name is not None:
        name = cfg_name
    else:
        name = getattr(paths, "exp_name", None) or getattr(paths, "exp", None) or "spectramind"
    # version
    if cfg_version is not None:
        version = cfg_version
    else:
        # timestamped run_dir folder can serve as a stable version tag
        version = getattr(paths, "timestamp", None) or getattr(paths, "run_name", None) or "run"
    return str(name), str(version)


def _resolve_save_dir(paths: Any, explicit: Optional[str | Path]) -> Path:
    """
    Determine a save_dir for loggers:
      1) explicit path if provided
      2) paths.logs_dir if present
      3) paths.run_dir/logs as fallback
    """
    if explicit:
        return _as_path(explicit)
    if hasattr(paths, "logs_dir") and getattr(paths, "logs_dir"):
        return _as_path(getattr(paths, "logs_dir"))
    if hasattr(paths, "run_dir") and getattr(paths, "run_dir"):
        return _as_path(getattr(paths, "run_dir")) / "logs"
    return Path("logs")


def _get_logging_cfg(cfg: Dict[str, Any] | Any) -> Dict[str, Any]:
    """Return cfg.logging as a plain dict without depending on OmegaConf at import time."""
    try:
        if hasattr(cfg, "logging"):
            return dict(getattr(cfg, "logging"))
        return dict(cfg.get("logging", {}))
    except Exception:
        return {}


# -----------------------------------------------------------------------------
# CSV
# -----------------------------------------------------------------------------
def _maybe_csv_logger(cfg: Dict[str, Any], paths: Any) -> Optional["pl.loggers.Logger"]:
    opts = cfg.get("csv", {})
    if not bool(opts.get("enable", False)):
        return None
    save_dir = _resolve_save_dir(paths, opts.get("save_dir"))
    name, version = _default_name_version(paths, opts.get("name"), opts.get("version"))
    save_dir.mkdir(parents=True, exist_ok=True)
    _log(f"CSVLogger → dir={save_dir}, name={name}, version={version}")
    return CSVLogger(save_dir=str(save_dir), name=name, version=version)


# -----------------------------------------------------------------------------
# TensorBoard
# -----------------------------------------------------------------------------
def _maybe_tensorboard_logger(cfg: Dict[str, Any], paths: Any) -> Optional["pl.loggers.Logger"]:
    opts = cfg.get("tensorboard", {})
    if not bool(opts.get("enable", False)):
        return None
    save_dir = _resolve_save_dir(paths, opts.get("save_dir"))
    name, version = _default_name_version(paths, opts.get("name"), opts.get("version"))
    default_hp_metric = bool(opts.get("default_hp_metric", False))
    log_graph = bool(opts.get("log_graph", False))
    save_dir.mkdir(parents=True, exist_ok=True)
    _log(f"TensorBoardLogger → dir={save_dir}, name={name}, version={version}")
    return TensorBoardLogger(
        save_dir=str(save_dir),
        name=name,
        version=version,
        default_hp_metric=default_hp_metric,
        log_graph=log_graph,
    )


# -----------------------------------------------------------------------------
# Weights & Biases
# -----------------------------------------------------------------------------
def _maybe_import_wandb() -> bool:
    global _WANDB_AVAILABLE
    if _WANDB_AVAILABLE:
        return True
    try:  # pragma: no cover
        import wandb  # noqa: F401
        from pytorch_lightning.loggers import WandbLogger  # noqa: F401
    except Exception:
        return False
    else:
        _WANDB_AVAILABLE = True
        return True


def _apply_wandb_mode(mode: Optional[str]) -> None:
    """
    Respect config/env WANDB_MODE. Allowed: "online" | "offline" | "dryrun"
    """
    if not mode:
        mode = os.environ.get("WANDB_MODE", "").strip().lower() or None
    if mode:
        os.environ["WANDB_MODE"] = mode


def _maybe_wandb_logger(cfg: Dict[str, Any], paths: Any) -> Optional["pl.loggers.Logger"]:
    opts = cfg.get("wandb", {})
    if not bool(opts.get("enable", False)):
        return None
    if not _maybe_import_wandb():
        _log("WandB logger requested but `wandb` not installed. Skipping.")
        return None

    from pytorch_lightning.loggers import WandbLogger  # type: ignore

    # Apply WANDB mode (offline/online/dryrun)
    _apply_wandb_mode(opts.get("mode"))

    project = opts.get("project", "spectramind-v50")
    entity = opts.get("entity", None)
    group = opts.get("group", getattr(paths, "exp_name", None))
    job_type = opts.get("job_type", "train")
    tags = list(opts.get("tags", []))
    resume = opts.get("resume", "allow")  # 'allow'|'never'|'must'|run_id

    # Provide save_dir/name/version for consistency with other loggers
    save_dir = _resolve_save_dir(paths, opts.get("save_dir"))
    name, version = _default_name_version(paths, opts.get("name"), opts.get("version"))
    save_dir.mkdir(parents=True, exist_ok=True)

    _log(f"WandbLogger → project={project}, entity={entity}, group={group}, name={name}, version={version}, resume={resume}")
    return WandbLogger(
        project=project,
        entity=entity,
        group=group,
        job_type=job_type,
        tags=tags,
        save_dir=str(save_dir),
        name=name,
        version=version,
        log_model=bool(opts.get("log_model", False)),
        resume=resume,
        settings=opts.get("settings", None),
    )


# -----------------------------------------------------------------------------
# MLflow
# -----------------------------------------------------------------------------
def _maybe_import_mlflow() -> bool:
    global _MLFLOW_AVAILABLE
    if _MLFLOW_AVAILABLE:
        return True
    try:  # pragma: no cover
        import mlflow  # noqa: F401
        from pytorch_lightning.loggers import MLFlowLogger  # noqa: F401
    except Exception:
        return False
    else:
        _MLFLOW_AVAILABLE = True
        return True


def _maybe_mlflow_logger(cfg: Dict[str, Any], paths: Any) -> Optional["pl.loggers.Logger"]:
    opts = cfg.get("mlflow", {})
    if not bool(opts.get("enable", False)):
        return None
    if not _maybe_import_mlflow():
        _log("MLflow logger requested but `mlflow` not installed. Skipping.")
        return None

    from pytorch_lightning.loggers import MLFlowLogger  # type: ignore

    tracking_uri = opts.get("tracking_uri", None)
    experiment_name = opts.get("experiment_name", getattr(paths, "exp_name", "spectramind"))
    run_name = opts.get("run_name", getattr(paths, "run_name", None))
    # Save dir is not directly used by MLflow, but we keep consistency
    _resolve_save_dir(paths, opts.get("save_dir")).mkdir(parents=True, exist_ok=True)

    _log(f"MLFlowLogger → experiment={experiment_name}, uri={tracking_uri}, run_name={run_name}")
    return MLFlowLogger(
        experiment_name=experiment_name,
        tracking_uri=tracking_uri,
        run_name=run_name or f"{experiment_name}-{getattr(paths, 'timestamp', 'run')}",
        tags=opts.get("tags", None),
    )


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def build_loggers(cfg: Dict[str, Any] | Any, paths: Any) -> List["pl.loggers.Logger"]:
    """
    Build a list of Lightning loggers from `cfg.logging` and return them.
    `paths` is the TrainPaths-like object (run_dir/logs_dir/exp_name/timestamp).
    """
    _ensure_pl()
    logging_cfg = _get_logging_cfg(cfg)
    loggers: List["pl.loggers.Logger"] = []

    # CSV
    csv_logger = _maybe_csv_logger(logging_cfg, paths)
    if csv_logger is not None:
        loggers.append(csv_logger)

    # TensorBoard
    tb_logger = _maybe_tensorboard_logger(logging_cfg, paths)
    if tb_logger is not None:
        loggers.append(tb_logger)

    # Weights & Biases
    wb_logger = _maybe_wandb_logger(logging_cfg, paths)
    if wb_logger is not None:
        loggers.append(wb_logger)

    # MLflow
    mf_logger = _maybe_mlflow_logger(logging_cfg, paths)
    if mf_logger is not None:
        loggers.append(mf_logger)

    if not loggers:
        _log("No loggers enabled; proceeding without external logging.")

    return loggers


def add_loggers_to_trainer_kwargs(trainer_kwargs: Dict[str, Any], cfg: Dict[str, Any] | Any, paths: Any) -> None:
    """
    Mutates `trainer_kwargs` to include built loggers if not already set.
    If `trainer_kwargs['logger']` is True, Lightning uses its default CSVLogger.
    """
    if "logger" in trainer_kwargs and trainer_kwargs["logger"] is not True:
        # respect explicit override (True means 'use default')
        return
    loggers = build_loggers(cfg, paths)
    if loggers:
        trainer_kwargs["logger"] = loggers
    else:
        trainer_kwargs["logger"] = True  # keep Lightning internal default


__all__ = [
    "build_loggers",
    "add_loggers_to_trainer_kwargs",
]