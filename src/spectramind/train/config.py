# src/spectramind/train/config.py
# =============================================================================
# SpectraMind V50 — Training Config Utilities
# -----------------------------------------------------------------------------
# Utilities to:
#   • resolve training output directories (run_dir / ckpt_dir / logs_dir)
#   • seed all RNGs deterministically from config
#   • translate a Hydra-style config dict into pl.Trainer(**kwargs)
#
# Design notes:
#   • Guarded imports: module is import-safe without torch/lightning/numpy
#   • No logger construction here (loggers are configured elsewhere);
#     this module returns Trainer kwargs and filesystem paths only.
#   • Matches repo’s CLI-first, config-driven, reproducible workflow.
# =============================================================================

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple, Union

# --- Optional heavy deps (import only at runtime) --------------------------------
try:  # pragma: no cover
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover
    _np = None  # type: ignore

try:  # pragma: no cover
    import torch as _torch  # type: ignore
except Exception:  # pragma: no cover
    _torch = None  # type: ignore

try:  # pragma: no cover
    import pytorch_lightning as _pl  # type: ignore
    from pytorch_lightning.utilities.seed import seed_everything as _pl_seed  # type: ignore
except Exception:  # pragma: no cover
    _pl = None  # type: ignore
    _pl_seed = None  # type: ignore


# =============================================================================
# Dataclasses
# =============================================================================
@dataclass(frozen=True)
class TrainPaths:
    root: Path          # experiment root (e.g., artifacts/)
    exp_name: str       # experiment group
    run_name: str       # run identifier (human-friendly)
    timestamp: str      # YYYYmmdd-HHMMSS
    run_dir: Path       # root/exp_name/run_name__YYYYmmdd-HHMMSS/
    ckpt_dir: Path      # run_dir/checkpoints/
    logs_dir: Path      # run_dir/logs/


# =============================================================================
# Internal helpers
# =============================================================================
def _cfg_get(cfg: Mapping[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, Mapping):
            return default
        cur = cur.get(key, default)
    return cur


def _now_stamp() -> str:
    # UTC-like stable timestamp (local tz, but reproducible format)
    return time.strftime("%Y%m%d-%H%M%S", time.localtime())


def _ensure_dir(p: Union[str, Path]) -> Path:
    path = Path(p)
    path.mkdir(parents=True, exist_ok=True)
    return path


def _normalize_experiment_name(s: Optional[str]) -> str:
    if not s:
        return "experiment"
    s = str(s).strip()
    return s or "experiment"


def _normalize_run_name(s: Optional[str]) -> str:
    if not s:
        return "run"
    s = str(s).strip()
    return s or "run"


def _normalize_precision(x: Any) -> Union[int, str]:
    """
    Map user precision config to something Trainer understands.
    Accepts: 32|16|64, '32', '16', 'bf16', 'bf16-mixed', '16-mixed', '64-true', etc.
    Defaults to 32-bit if unset.
    """
    if x is None:
        return 32
    if isinstance(x, (int, float)):
        v = int(x)
        return v if v in (16, 32, 64) else 32
    s = str(x).strip().lower()
    # Common aliases
    if s in {"32", "fp32", "float32"}:
        return 32
    if s in {"64", "fp64", "float64"}:
        return 64
    if s in {"16", "fp16", "float16", "16-mixed"}:
        # PL 2.x: '16-mixed' or 16 (AMP) both acceptable
        return "16-mixed"
    if s in {"bf16", "bf16-mixed", "bfloat16"}:
        return "bf16-mixed"
    return 32


def _infer_devices(devices: Any) -> Any:
    """
    Return devices value compatible with Trainer. Handles:
      - None: auto (1 if cpu)
      - int/str: pass through if sensible
      - 'auto' => 'auto'
      - list => pass-through
    """
    if devices is None:
        # If torch cuda available and env allows, leaving None lets PL choose; otherwise fallback to 1
        try:
            if _torch is not None and _torch.cuda.is_available():
                return "auto"
        except Exception:
            pass
        return 1
    return devices


def _parse_bool(x: Any, default: bool) -> bool:
    if x is None:
        return default
    if isinstance(x, bool):
        return x
    s = str(x).strip().lower()
    return s in {"1", "true", "yes", "y", "on"}


# =============================================================================
# Public: resolve output paths
# =============================================================================
def resolve_train_paths(cfg: Mapping[str, Any]) -> TrainPaths:
    """
    Resolve run output directories from config. Sane defaults:
      root:   cfg.paths.artifacts or cfg.train.output_dir or "./artifacts"
      exp:    cfg.experiment.name or cfg.train.experiment or "experiment"
      run:    cfg.run.name or cfg.train.run_name or derived from model name
      stamp:  cfg.run.timestamp or generated (YYYYmmdd-HHMMSS)
    """
    root = _cfg_get(cfg, "paths.artifacts") or _cfg_get(cfg, "train.output_dir") or "artifacts"
    exp_name = (
        _cfg_get(cfg, "experiment.name")
        or _cfg_get(cfg, "train.experiment")
        or _cfg_get(cfg, "model.name")
        or "experiment"
    )
    run_name = _cfg_get(cfg, "run.name") or _cfg_get(cfg, "train.run_name") or "run"
    stamp = _cfg_get(cfg, "run.timestamp") or _now_stamp()

    exp_name = _normalize_experiment_name(exp_name)
    run_name = _normalize_run_name(run_name)

    root_p = _ensure_dir(root)
    run_dir = _ensure_dir(root_p / exp_name / f"{run_name}__{stamp}")
    ckpt_dir = _ensure_dir(run_dir / "checkpoints")
    logs_dir = _ensure_dir(run_dir / "logs")

    return TrainPaths(
        root=root_p,
        exp_name=exp_name,
        run_name=run_name,
        timestamp=stamp,
        run_dir=run_dir,
        ckpt_dir=ckpt_dir,
        logs_dir=logs_dir,
    )


# =============================================================================
# Public: seeding
# =============================================================================
def seed_everything_from_cfg(cfg: Mapping[str, Any]) -> int:
    """
    Seed Python, NumPy, (optional) Torch and PL deterministically from cfg.train.seed.
    Returns the seed actually used.
    """
    seed = int(_cfg_get(cfg, "train.seed", 42) or 42)

    # Python
    try:
        import random as _random  # local
        _random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
    except Exception:
        pass

    # NumPy
    try:
        if _np is not None:
            _np.random.seed(seed)  # type: ignore[attr-defined]
    except Exception:
        pass

    # Torch
    try:
        if _torch is not None:
            _torch.manual_seed(seed)
            if _torch.cuda.is_available():
                _torch.cuda.manual_seed_all(seed)
            # optional torch backends deterministic flags (can be slow)
            if _parse_bool(_cfg_get(cfg, "train.deterministic", False), False):
                _torch.backends.cudnn.deterministic = True  # type: ignore[attr-defined]
                _torch.backends.cudnn.benchmark = False    # type: ignore[attr-defined]
    except Exception:
        pass

    # Lightning
    try:
        if _pl_seed is not None:
            # PL controls python + numpy + torch in one go (we keep prior steps for robustness)
            _pl_seed(seed, workers=_parse_bool(_cfg_get(cfg, "train.seed_workers", True), True))
    except Exception:
        pass

    return seed


# =============================================================================
# Public: translate config -> Trainer kwargs
# =============================================================================
def build_trainer_kwargs(
    cfg: Mapping[str, Any],
    *,
    default_ckpt_dir: Optional[Union[str, Path]] = None,
) -> Dict[str, Any]:
    """
    Translate Hydra-style config dict into `pytorch_lightning.Trainer` kwargs.

    Examples of read keys:
      train:
        max_epochs: 50
        accelerator: "auto"
        strategy: null
        devices: "auto"
        precision: "bf16"
        gradient_clip_val: 1.0
        accumulate_grad_batches: 1
        deterministic: false
        detect_anomaly: false
        log_every_n_steps: 50
        check_val_every_n_epoch: 1
        val_check_interval: null
        enable_checkpointing: true
        num_sanity_val_steps: 2
        overfit_batches: 0.0
        limit_train_batches: 1.0
        limit_val_batches: 1.0
        limit_test_batches: 1.0
        profiler: null|"simple"|"advanced"
    """
    # Core scheduling
    max_epochs = int(_cfg_get(cfg, "train.max_epochs", 50))
    max_steps = _cfg_get(cfg, "train.max_steps", None)
    if max_steps is not None:
        max_steps = int(max_steps)

    # Resources / performance
    accelerator = _cfg_get(cfg, "train.accelerator", "auto")
    strategy = _cfg_get(cfg, "train.strategy", None)
    devices = _infer_devices(_cfg_get(cfg, "train.devices", None))
    precision = _normalize_precision(_cfg_get(cfg, "train.precision", 32))

    gradient_clip_val = float(_cfg_get(cfg, "train.gradient_clip_val", 0.0) or 0.0)
    accumulate_grad_batches = int(_cfg_get(cfg, "train.accumulate_grad_batches", 1))

    # Determinism / debugging
    deterministic = _parse_bool(_cfg_get(cfg, "train.deterministic", False), False)
    detect_anomaly = _parse_bool(_cfg_get(cfg, "train.detect_anomaly", False), False)

    # Logging cadence
    log_every_n_steps = int(_cfg_get(cfg, "train.log_every_n_steps", 50))
    check_val_every_n_epoch = int(_cfg_get(cfg, "train.check_val_every_n_epoch", 1))
    val_check_interval = _cfg_get(cfg, "train.val_check_interval", None)  # float | int | None

    # Data limits / sanity checks
    limit_train_batches = _cfg_get(cfg, "train.limit_train_batches", 1.0)
    limit_val_batches = _cfg_get(cfg, "train.limit_val_batches", 1.0)
    limit_test_batches = _cfg_get(cfg, "train.limit_test_batches", 1.0)
    overfit_batches = _cfg_get(cfg, "train.overfit_batches", 0.0)
    num_sanity_val_steps = int(_cfg_get(cfg, "train.num_sanity_val_steps", 2))
    enable_checkpointing = _parse_bool(_cfg_get(cfg, "train.enable_checkpointing", True), True)

    profiler = _cfg_get(cfg, "train.profiler", None)

    # Checkpointing directory (Lightning uses this if checkpoint callbacks are constructed)
    default_root_dir = str(default_ckpt_dir) if default_ckpt_dir is not None else None

    kwargs: Dict[str, Any] = {
        "max_epochs": max_epochs,
        "max_steps": max_steps,
        "accelerator": accelerator,
        "strategy": strategy,
        "devices": devices,
        "precision": precision,
        "gradient_clip_val": gradient_clip_val,
        "accumulate_grad_batches": accumulate_grad_batches,
        "deterministic": deterministic,
        "detect_anomaly": detect_anomaly,
        "log_every_n_steps": log_every_n_steps,
        "check_val_every_n_epoch": check_val_every_n_epoch,
        "val_check_interval": val_check_interval,
        "limit_train_batches": limit_train_batches,
        "limit_val_batches": limit_val_batches,
        "limit_test_batches": limit_test_batches,
        "overfit_batches": overfit_batches,
        "num_sanity_val_steps": num_sanity_val_steps,
        "enable_checkpointing": enable_checkpointing,
        "profiler": profiler,
        "default_root_dir": default_root_dir,
    }

    # Drop Nones so Trainer doesn’t complain
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return kwargs
