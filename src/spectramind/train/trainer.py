# src/spectramind/train/trainer.py
"""
SpectraMind V50 — Training Service
----------------------------------
Config-driven training entry for the V50 pipeline (PyTorch Lightning).

What this does
--------------
- Seeds everything for reproducibility
- Builds model & datamodule from factory helpers (Hydra/_target_ or registry fallback)
- Configures Lightning Trainer + callbacks (checkpoint, early stop, LR monitor, etc.)
- Injects loggers (CSV/TensorBoard/W&B/MLflow) from cfg.logging
- Writes run manifests (config snapshot, hashes, metrics)
- Plays nice with Kaggle/CI constraints (progress bars, deterministic, no internet)
- Tolerates PL 1.7 → 2.x API deltas (accelerator/devices/ckpt_path/etc.)

Expected Config (subset)
-----------------------
training:
  max_epochs: 20
  accelerator: auto
  devices: auto
  precision: 16
  accumulate_grad_batches: 1
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  gradient_clip_val: 0.0
  deterministic: true
  num_sanity_val_steps: 2
  log_every_n_steps: 50
  profiler: null
  resume_from: null
  strategy: null
  detect_anomaly: false
  inference_mode: true
  enable_model_summary: true
  enable_progress_bar: auto      # auto/true/false

callbacks:
  checkpoint:
    monitor: val_loss
    mode: min
    save_top_k: 1
    save_last: true
    filename: "epoch{epoch:03d}-val{val_loss:.5f}"

logging:
  csv:
    enable: true
  tensorboard:
    enable: false
  wandb:
    enable: false
  mlflow:
    enable: false

paths:
  workdir: "outputs/run"                # base dir for artifacts
  checkpoints: "${paths.workdir}/ckpt"  # checkpoint dir
  logs: "${paths.workdir}/logs"
  manifest: "${paths.workdir}/run_manifest.json"

seed: 42

model: {}        # fed to model builder
data:  {}        # fed to datamodule builder

Notes
-----
- Prefer Hydra `_target_` configs for model/datamodule; registry fallback is provided.
- If external factories exist, they are used:
    spectramind.models.__models__.build_model(cfg: dict) -> nn.Module | LightningModule
    spectramind.data.datamodule.build_datamodule(cfg: dict) -> LightningDataModule
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

# --- Lightning (guarded) ------------------------------------------------------
try:  # pragma: no cover
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "Training requires PyTorch Lightning. Install `pytorch-lightning`."
    ) from _e

# --- Optional Hydra -----------------------------------------------------------
try:  # pragma: no cover
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
except Exception:
    DictConfig = Any  # type: ignore
    OmegaConf = None  # type: ignore

    def instantiate(*_a, **_k):  # type: ignore
        raise RuntimeError("Hydra is required to instantiate Hydra nodes (_target_).")

# --- Local: central factories & utilities ------------------------------------
from .callbacks import build_callbacks
from .loggers import add_loggers_to_trainer_kwargs
from .ckpt import resume_trainer_if_available
from .config import (
    resolve_train_paths,
    seed_everything_from_cfg,
    build_trainer_kwargs,
)
# Fallback model/datamodule builders (same logic as train.py)
from .train import (
    _build_model_from_cfg_or_registry as _fallback_build_model,
    _build_datamodule_from_cfg as _fallback_build_dm,
)

# --- Optional external factories (if present in user repo) --------------------
_EXT_MODEL_BUILDER = None
_EXT_DM_BUILDER = None
try:  # pragma: no cover
    from spectramind.models.__models__ import build_model as _EXT_MODEL_BUILDER  # type: ignore
except Exception:
    pass
try:  # pragma: no cover
    from spectramind.data.datamodule import build_datamodule as _EXT_DM_BUILDER  # type: ignore
except Exception:
    pass


# ──────────────────────────────────────────────────────────────────────────────
# Small helpers (no external deps)
# ──────────────────────────────────────────────────────────────────────────────

def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Small helper to read nested keys: 'a.b.c' -> cfg['a']['b']['c']."""
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def _is_ci_or_kaggle() -> bool:
    if os.environ.get("CI") or os.environ.get("GITHUB_ACTIONS"):
        return True
    if "KAGGLE_KERNEL_RUN_TYPE" in os.environ or Path("/kaggle").exists():
        return True
    return False


def _as_bool_auto(val: Any, default: bool) -> bool:
    """Accept 'auto' as 'not CI', strings as booleans, fallback to default."""
    if val is None:
        return default
    if isinstance(val, str):
        v = val.strip().lower()
        if v == "auto":
            return not _is_ci_or_kaggle()
        return v in {"1", "true", "t", "yes", "y", "on"}
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return default


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _write_json(obj: Dict[str, Any], path: Path) -> Path:
    _ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
    return path


@rank_zero_only
def _write_run_manifest(
    cfg_dict: Dict[str, Any],
    workdir: Path,
    ckpt_path: Optional[Path],
    metrics: Dict[str, Any],
    status: str,
    t_start: float,
    t_end: float,
) -> Path:
    manifest_path = _cfg_get(cfg_dict, "paths.manifest", workdir / "run_manifest.json")
    manifest_path = Path(manifest_path)

    # Include a resolved, pure-python config snapshot
    if OmegaConf is not None and isinstance(cfg_dict, DictConfig):
        cfg_snapshot = OmegaConf.to_container(cfg_dict, resolve=True)  # type: ignore
    else:
        cfg_snapshot = cfg_dict

    snapshot = {
        "config": cfg_snapshot,
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "metrics": metrics,
        "status": status,  # "ok" | "failed"
        "run_id": int(t_start),
        "time_sec": max(0.0, t_end - t_start),
    }
    return _write_json(snapshot, manifest_path)


def _save_hydra_snapshot(cfg: DictConfig, run_dir: Path) -> None:
    """Persist a config snapshot (yaml+json) similar to train.train."""
    _ensure_dir(run_dir)
    if OmegaConf is not None and isinstance(cfg, DictConfig):
        OmegaConf.save(config=cfg, f=str(run_dir / "config_snapshot.yaml"))
        try:
            cfg_dict = OmegaConf.to_container(cfg, resolve=True)  # type: ignore
        except Exception:
            cfg_dict = dict(cfg)
    else:
        cfg_dict = dict(cfg)
    _write_json(cfg_dict, run_dir / "config_snapshot.json")


def _build_trainer(cfg: Dict[str, Any], callbacks, loggers, default_root_dir: Path) -> pl.Trainer:
    """Build trainer kwargs from cfg.training, probe for keys PL supports, and create a Trainer."""
    tr_cfg = cfg.get("training", {}) or {}

    enable_progress_bar = _as_bool_auto(tr_cfg.get("enable_progress_bar", "auto"), True)

    # Baseline kwargs (common across PL 1.7 → 2.x)
    trainer_kwargs: Dict[str, Any] = dict(
        max_epochs=int(tr_cfg.get("max_epochs", 20)),
        log_every_n_steps=int(tr_cfg.get("log_every_n_steps", 50)),
        accumulate_grad_batches=int(tr_cfg.get("accumulate_grad_batches", 1)),
        gradient_clip_val=float(tr_cfg.get("gradient_clip_val", 0.0)),
        deterministic=bool(tr_cfg.get("deterministic", True)),
        num_sanity_val_steps=int(tr_cfg.get("num_sanity_val_steps", 2)),
        limit_train_batches=tr_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=tr_cfg.get("limit_val_batches", 1.0),
        callbacks=callbacks,
        logger=loggers if loggers else True,
        enable_checkpointing=True,
        default_root_dir=str(default_root_dir),
        detect_anomaly=bool(tr_cfg.get("detect_anomaly", False)),
        inference_mode=bool(tr_cfg.get("inference_mode", True)),
        enable_model_summary=bool(tr_cfg.get("enable_model_summary", True)),
    )

    # Optional keys (probe to avoid TypeError on older PL)
    def _try_add(key: str, value: Any) -> None:
        try:
            pl.Trainer(**{key: value})
            trainer_kwargs[key] = value
        except TypeError:
            pass

    _try_add("enable_progress_bar", enable_progress_bar)
    _try_add("enable_fault_tolerance", True)

    # Precision / strategy
    if "precision" in tr_cfg:
        trainer_kwargs["precision"] = tr_cfg.get("precision")
    if tr_cfg.get("strategy"):
        trainer_kwargs["strategy"] = tr_cfg["strategy"]

    # Accelerator/devices (PL >=1.7 preferred) with fallback to legacy gpus
    if "accelerator" in tr_cfg:
        trainer_kwargs["accelerator"] = tr_cfg["accelerator"]
    if "devices" in tr_cfg:
        trainer_kwargs["devices"] = tr_cfg["devices"]
    elif "gpus" in tr_cfg:
        g = tr_cfg["gpus"]
        trainer_kwargs["accelerator"] = "gpu" if g else "cpu"
        trainer_kwargs["devices"] = g or 0

    # Profiler passthrough if requested
    if tr_cfg.get("profiler"):
        trainer_kwargs["profiler"] = tr_cfg["profiler"]

    return pl.Trainer(**trainer_kwargs)


# ──────────────────────────────────────────────────────────────────────────────
# Service
# ──────────────────────────────────────────────────────────────────────────────

class TrainerService:
    """
    Facade to train a model using (Hydra) config dicts.

    Example
    -------
    >>> svc = TrainerService(cfg)
    >>> ckpt_path, metrics = svc.run()
    """

    def __init__(self, cfg: Dict[str, Any] | DictConfig):
        self.cfg = cfg
        # Resolve run/ckpt/log dirs via our central helper
        self.paths = resolve_train_paths(cfg if isinstance(cfg, DictConfig) else DictConfig(cfg))  # type: ignore

    def _seed(self) -> int:
        return seed_everything_from_cfg(self.cfg)

    def _build_datamodule(self):
        # Prefer external factory if the repo provides one
        if _EXT_DM_BUILDER is not None:
            return _EXT_DM_BUILDER(self.cfg.get("data", {}) or {})
        # Fallback to our universal builder
        return _fallback_build_dm(self.cfg, self.paths)

    def _build_model(self):
        # Prefer external factory if the repo provides one
        if _EXT_MODEL_BUILDER is not None:
            return _EXT_MODEL_BUILDER(self.cfg.get("model", {}) or {})
        # Fallback to our registry/Hydra logic
        return _fallback_build_model(self.cfg)

    def run(self) -> Tuple[Optional[Path], Dict[str, Any]]:
        t0 = time.perf_counter()
        # 1) seed
        seed = self._seed()
        _ensure_dir(self.paths.run_dir)
        _save_hydra_snapshot(self.cfg, self.paths.run_dir)
        print(f"[SpectraMind][trainer] Seed set to {seed}")

        # 2) datamodule + model
        print("[SpectraMind][trainer] Building datamodule...")
        datamodule = self._build_datamodule()

        print("[SpectraMind][trainer] Building model...")
        model = self._build_model()

        # 3) callbacks + loggers + trainer kwargs
        callbacks, ckpt_cb = build_callbacks(self.cfg, ckpt_dir=self.paths.ckpt_dir, logs_dir=self.paths.logs_dir)
        trainer_kwargs = build_trainer_kwargs(self.cfg, default_ckpt_dir=self.paths.ckpt_dir)
        add_loggers_to_trainer_kwargs(trainer_kwargs, self.cfg, self.paths)

        # 4) trainer
        trainer = pl.Trainer(callbacks=callbacks, **trainer_kwargs)

        # 5) resume (best/last selection supported)
        prefer = _cfg_get(self.cfg, "train.resume.prefer", "best")
        monitor = _cfg_get(self.cfg, "callbacks.checkpoint.monitor", "val_loss")
        mode = _cfg_get(self.cfg, "callbacks.checkpoint.mode", "min")
        resume_path = resume_trainer_if_available(trainer, self.paths.ckpt_dir, prefer=prefer, monitor=monitor, mode=mode)
        if resume_path:
            print(f"[SpectraMind][trainer] Resuming from checkpoint: {resume_path}")

        # 6) fit + optional validate
        status = "ok"
        val_metrics: Dict[str, Any] = {}
        try:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=getattr(trainer, "ckpt_path", None))
            # optional validate
            do_validate = bool(_cfg_get(self.cfg, "train.validate_after_fit", True))
            if do_validate:
                try:
                    res = trainer.validate(model=model, datamodule=datamodule, verbose=False)
                    if isinstance(res, list) and res:
                        # JSON-serializable
                        val_metrics = {k: float(v) for k, v in res[0].items() if isinstance(v, (int, float))}
                except Exception as ve:
                    print(f"[SpectraMind][trainer] Validation failed: {ve!r}")
        except Exception as e:
            status = "failed"
            print(f"[SpectraMind][trainer] Training loop failed: {e!r}")

        # 7) best/last ckpt
        ckpt_path = None
        try:
            ckpt_path = getattr(ckpt_cb, "best_model_path", None) or getattr(ckpt_cb, "last_model_path", None)
        except Exception:
            pass
        ckpt_file = Path(ckpt_path) if ckpt_path else None

        # 8) manifest
        t1 = time.perf_counter()
        _write_run_manifest(
            cfg_dict=(self.cfg if not isinstance(self.cfg, DictConfig) else self.cfg),
            workdir=self.paths.run_dir,
            ckpt_path=ckpt_file,
            metrics=val_metrics,
            status=status,
            t_start=t0,
            t_end=t1,
        )
        if ckpt_file:
            print(f"[SpectraMind][trainer] Best checkpoint: {ckpt_file}")

        return ckpt_file, val_metrics


# ---- Convenience function for simple usage -----------------------------------

def train_from_config(cfg: Dict[str, Any] | DictConfig) -> Tuple[Optional[Path], Dict[str, Any]]:
    """
    One-shot training convenience for CLI/Notebook usage.

    Returns
    -------
    ckpt_path : Optional[Path]
        Path to best/last checkpoint.
    metrics : Dict[str, Any]
        Final validation metrics (if any).
    """
    return TrainerService(cfg).run()