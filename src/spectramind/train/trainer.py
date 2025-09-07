# src/spectramind/train/trainer.py
"""
SpectraMind V50 — Training Service
----------------------------------
Config-driven training entry for the V50 pipeline (PyTorch Lightning).
- Seeds everything for reproducibility
- Builds model & datamodule from factory helpers
- Configures Lightning Trainer + callbacks
- Writes run manifests (config snapshot, hashes, metrics)
- Plays nice with Kaggle/CI constraints

Expected Config (subset)
-----------------------
training:
  max_epochs: 20
  gpus: 1              # or devices/accelerator per PL >=1.7
  precision: 16
  accumulate_grad_batches: 1
  limit_train_batches: 1.0
  limit_val_batches: 1.0
  gradient_clip_val: 0.0
  deterministic: true
  num_sanity_val_steps: 2
  log_every_n_steps: 50
  profiler: null
  resume_from: null      # optional checkpoint path
  strategy: null         # e.g. "auto", "ddp", "fsdp"
  detect_anomaly: false
  inference_mode: true
  enable_model_summary: true
  enable_progress_bar: auto  # auto/true/false

callbacks:
  checkpoint:
    monitor: val_loss
    mode: min
    save_top_k: 1
    save_last: true
    filename: "epoch{epoch:03d}-val{val_loss:.5f}"
  early_stopping:
    enable: true
    monitor: val_loss
    mode: min
    patience: 5
    min_delta: 0.0
  lr_monitor: true

paths:
  workdir: "outputs/run"                # base dir for artifacts
  checkpoints: "${paths.workdir}/ckpt"  # checkpoint dir
  logs: "${paths.workdir}/logs"
  manifest: "${paths.workdir}/run_manifest.json"

seed: 42

model: {}        # passed to build_model(...)
data: {}         # passed to build_datamodule(...)

Notes
-----
- Model & Data factories must exist:
    spectramind.models.__models__.build_model(cfg: dict) -> nn.Module | LightningModule
    spectramind.data.datamodule.build_datamodule(cfg: dict) -> LightningDataModule
- If your model is a plain nn.Module, wrap it in a LightningModule in your factory.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from spectramind.utils.logging import get_logger
from spectramind.utils.seed import set_global_seed
from spectramind.utils.timer import timeit
from spectramind.utils.hashing import config_snapshot_hash
from spectramind.utils.io import p, ensure_dir, write_json

# --- Optional heavy deps (guarded) ---
try:  # pragma: no cover
    import pytorch_lightning as pl
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "Training requires PyTorch Lightning. Install `pytorch-lightning`."
    ) from _e

# Factories (must be implemented in your repo)
from spectramind.models.__models__ import build_model  # type: ignore
from spectramind.data.datamodule import build_datamodule  # type: ignore
from spectramind.train.callbacks import build_callbacks  # central callback factory


# ---------- small helpers ----------

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
        return v in {"1", "true", "yes", "y", "on"}
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    return default


# ---------- trainer construction ----------

def _build_trainer(cfg: Dict[str, Any], callbacks, logger, default_root_dir: Path) -> pl.Trainer:
    tr_cfg = cfg.get("training", {}) or {}

    enable_progress_bar = _as_bool_auto(tr_cfg.get("enable_progress_bar", "auto"), True)

    trainer_kwargs = dict(
        max_epochs=tr_cfg.get("max_epochs", 20),
        log_every_n_steps=tr_cfg.get("log_every_n_steps", 50),
        accumulate_grad_batches=tr_cfg.get("accumulate_grad_batches", 1),
        gradient_clip_val=tr_cfg.get("gradient_clip_val", 0.0),
        deterministic=tr_cfg.get("deterministic", True),
        num_sanity_val_steps=tr_cfg.get("num_sanity_val_steps", 2),
        limit_train_batches=tr_cfg.get("limit_train_batches", 1.0),
        limit_val_batches=tr_cfg.get("limit_val_batches", 1.0),
        callbacks=callbacks,
        logger=logger,  # use Python logging handlers already configured
        enable_checkpointing=True,
        enable_progress_bar=enable_progress_bar,
        default_root_dir=str(default_root_dir),
        detect_anomaly=bool(tr_cfg.get("detect_anomaly", False)),
        inference_mode=bool(tr_cfg.get("inference_mode", True)),
        enable_model_summary=bool(tr_cfg.get("enable_model_summary", True)),
        enable_fault_tolerance=True,  # safe default across envs
    )

    # Precision
    if "precision" in tr_cfg:
        trainer_kwargs["precision"] = tr_cfg.get("precision")

    # Strategy
    if tr_cfg.get("strategy"):
        trainer_kwargs["strategy"] = tr_cfg["strategy"]

    # Accelerator/devices (PL >=1.7 preferred)
    if "accelerator" in tr_cfg:
        trainer_kwargs["accelerator"] = tr_cfg["accelerator"]
    if "devices" in tr_cfg:
        trainer_kwargs["devices"] = tr_cfg["devices"]
    else:
        # Backward-compat fallback to 'gpus'
        if "gpus" in tr_cfg:
            g = tr_cfg["gpus"]
            trainer_kwargs["accelerator"] = "gpu" if g else "cpu"
            trainer_kwargs["devices"] = g or 0

    # Profiler if requested
    if tr_cfg.get("profiler"):
        trainer_kwargs["profiler"] = tr_cfg["profiler"]

    # ckpt_path handled at .fit(...) time (Lightning supports ckpt_path=None)
    return pl.Trainer(**trainer_kwargs)


def _write_run_manifest(
    cfg: Dict[str, Any],
    workdir: Path,
    ckpt_path: Optional[Path],
    metrics: Dict[str, Any],
    status: str,
    t_start: float,
    t_end: float,
) -> Path:
    ensure_dir(workdir)
    manifest_path = p(_cfg_get(cfg, "paths.manifest", workdir / "run_manifest.json"))
    snapshot = {
        "config": cfg,
        "config_hash": config_snapshot_hash(cfg),
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "metrics": metrics,
        "status": status,  # "ok" | "failed"
        "run_id": int(t_start),
        "time_sec": max(0.0, t_end - t_start),
    }
    return write_json(snapshot, manifest_path)


# ---------- service ----------

class TrainerService:
    """
    Facade to train a model using config dicts.

    Example
    -------
    svc = TrainerService(cfg)
    ckpt_path, metrics = svc.run()
    """

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.logger = get_logger(__name__)

        # Paths
        self.workdir = p(_cfg_get(cfg, "paths.workdir", "outputs/run"))
        self.ckpt_dir = p(_cfg_get(cfg, "paths.checkpoints", self.workdir / "ckpt"))
        self.logs_dir = p(_cfg_get(cfg, "paths.logs", self.workdir / "logs"))

    def _seed(self):
        seed = int(self.cfg.get("seed", 42))
        self.logger.info(f"[Seed] Setting global random seed = {seed}")
        set_global_seed(seed)

    @timeit("TrainerService.run")
    def run(self) -> Tuple[Optional[Path], Dict[str, Any]]:
        t0 = time.perf_counter()
        self._seed()

        # Factories
        self.logger.info("Building datamodule...")
        datamodule = build_datamodule(self.cfg.get("data", {}) or {})

        self.logger.info("Building model...")
        model = build_model(self.cfg.get("model", {}) or {})

        # Callbacks + Trainer (centralized & JSONL metrics)
        callbacks, ckpt_cb = build_callbacks(self.cfg, self.ckpt_dir, self.logs_dir)

        # Build trainer
        pl_logger = True  # forward to Python logging handlers
        trainer = _build_trainer(self.cfg, callbacks, pl_logger, self.workdir)

        # Fit (with resume support via ckpt_path)
        self.logger.info("Starting training loop...")
        status = "ok"
        val_metrics: Dict[str, Any] = {}
        ckpt_resume = _cfg_get(self.cfg, "training.resume_from", None)
        try:
            trainer.fit(model=model, datamodule=datamodule, ckpt_path=str(p(ckpt_resume)) if ckpt_resume else None)

            # Validate at end (optional but handy for manifest)
            self.logger.info("Final validation...")
            try:
                res = trainer.validate(model=model, datamodule=datamodule, verbose=False)
                if isinstance(res, list) and res:
                    # Ensure serialization to JSON (floats/ints only)
                    val_metrics = {k: float(v) for k, v in res[0].items() if isinstance(v, (int, float))}
            except Exception as ve:
                self.logger.warning("Validation step failed: %s", ve)
        except Exception as e:
            status = "failed"
            self.logger.exception("Training loop failed: %s", e)

        # Checkpoint path from callback
        ckpt_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path  # type: ignore[attr-defined]
        ckpt_file = Path(ckpt_path) if ckpt_path else None

        # Write manifest regardless of success/failure
        t1 = time.perf_counter()
        manifest = _write_run_manifest(self.cfg, self.workdir, ckpt_file, val_metrics, status, t0, t1)
        self.logger.info("Run manifest written to %s", manifest)

        return ckpt_file, val_metrics


# ---- Convenience function for simple usage -----------------------------------

def train_from_config(cfg: Dict[str, Any]) -> Tuple[Optional[Path], Dict[str, Any]]:
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
