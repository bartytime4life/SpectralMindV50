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
- If your model is a plain nn.Module, we recommend wrapping it in a LightningModule
  in the build_model factory so Trainer logic stays uniform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from spectramind.utils.logging import get_logger
from spectramind.utils.seed import set_global_seed
from spectramind.utils.timer import timeit
from spectramind.utils.hashing import config_snapshot_hash
from spectramind.utils.io import p, ensure_dir, write_json

# --- Optional heavy deps (guarded) ---
try:  # pragma: no cover
    import torch
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import (
        ModelCheckpoint,
        EarlyStopping,
        LearningRateMonitor,
    )
except Exception as _e:  # pragma: no cover
    raise RuntimeError(
        "Training requires PyTorch and PyTorch Lightning. "
        "Please install `torch` and `pytorch-lightning`."
    ) from _e

# Factories (must be implemented in your repo)
from spectramind.models.__models__ import build_model  # type: ignore
from spectramind.data.datamodule import build_datamodule  # type: ignore


def _cfg_get(cfg: Dict[str, Any], path: str, default: Any = None) -> Any:
    """Small helper to read nested keys: 'a.b.c' -> cfg['a']['b']['c']."""
    cur: Any = cfg
    for key in path.split("."):
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key, default)
    return cur


def _build_callbacks(cfg: Dict[str, Any], ckpt_dir: Path):
    cb_cfg = cfg.get("callbacks", {}) or {}

    # Checkpoint
    ck_cfg = cb_cfg.get("checkpoint", {}) or {}
    ckpt = ModelCheckpoint(
        dirpath=str(ensure_dir(ckpt_dir)),
        filename=ck_cfg.get("filename", "epoch{epoch:03d}-{monitor}{"+ck_cfg.get("monitor", "val_loss")+"}"),
        monitor=ck_cfg.get("monitor", "val_loss"),
        mode=ck_cfg.get("mode", "min"),
        save_top_k=ck_cfg.get("save_top_k", 1),
        save_last=ck_cfg.get("save_last", True),
        auto_insert_metric_name=True,
    )

    # Early Stopping
    es = None
    es_cfg = cb_cfg.get("early_stopping", {}) or {}
    if es_cfg.get("enable", True):
        es = EarlyStopping(
            monitor=es_cfg.get("monitor", "val_loss"),
            mode=es_cfg.get("mode", "min"),
            patience=es_cfg.get("patience", 5),
            min_delta=es_cfg.get("min_delta", 0.0),
            verbose=False,
        )

    # LR monitor
    lr = None
    if cb_cfg.get("lr_monitor", True):
        lr = LearningRateMonitor(logging_interval="step")

    callbacks = [ckpt]
    if es:
        callbacks.append(es)
    if lr:
        callbacks.append(lr)
    return callbacks, ckpt


def _build_trainer(cfg: Dict[str, Any], callbacks, logger) -> pl.Trainer:
    tr_cfg = cfg.get("training", {}) or {}

    # Lightning Trainer arguments (compatible across 1.7+)
    # Prefer 'devices'/'accelerator' if users provide them; fallback to legacy flags where common.
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
        logger=logger,  # Let PL forward to whatever logger handlers you prefer
        enable_checkpointing=True,
        enable_progress_bar=True,
    )

    # Precision
    if "precision" in tr_cfg:
        trainer_kwargs["precision"] = tr_cfg.get("precision")

    # Accelerator/devices (flexible config)
    if "accelerator" in tr_cfg:
        trainer_kwargs["accelerator"] = tr_cfg["accelerator"]
    if "devices" in tr_cfg:
        trainer_kwargs["devices"] = tr_cfg["devices"]
    else:
        # Backward-compat fallback to 'gpus' if provided
        if "gpus" in tr_cfg:
            g = tr_cfg["gpus"]
            trainer_kwargs["accelerator"] = "gpu" if g else "cpu"
            trainer_kwargs["devices"] = g or 0

    # Resume
    resume = tr_cfg.get("resume_from", None)
    if resume:
        trainer_kwargs["ckpt_path"] = str(p(resume))

    # Profiler if requested
    if "profiler" in tr_cfg and tr_cfg["profiler"]:
        trainer_kwargs["profiler"] = tr_cfg["profiler"]

    return pl.Trainer(**trainer_kwargs)


def _write_run_manifest(
    cfg: Dict[str, Any],
    workdir: Path,
    ckpt_path: Optional[Path],
    metrics: Dict[str, Any],
) -> Path:
    ensure_dir(workdir)
    manifest_path = p(_cfg_get(cfg, "paths.manifest", workdir / "run_manifest.json"))
    snapshot = {
        "config": cfg,
        "config_hash": config_snapshot_hash(cfg),
        "checkpoint": str(ckpt_path) if ckpt_path else None,
        "metrics": metrics,
    }
    return write_json(snapshot, manifest_path)


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
        set_global_seed(seed)

    @timeit("TrainerService.run")
    def run(self) -> Tuple[Optional[Path], Dict[str, Any]]:
        self._seed()

        # Factories
        self.logger.info("Building datamodule...")
        datamodule = build_datamodule(self.cfg.get("data", {}) or {})

        self.logger.info("Building model...")
        model = build_model(self.cfg.get("model", {}) or {})

        # Callbacks + Trainer
        callbacks, ckpt_cb = _build_callbacks(self.cfg, self.ckpt_dir)

        # Integrate with Python logging via our Rich/JSON handlers
        pl_logger = True  # use python logging handlers already configured
        trainer = _build_trainer(self.cfg, callbacks, pl_logger)

        # Fit
        self.logger.info("Starting training loop...")
        trainer.fit(model=model, datamodule=datamodule)

        # Validate at end (optional but handy for manifest)
        self.logger.info("Final validation...")
        val_metrics = {}
        try:
            res = trainer.validate(model=model, datamodule=datamodule, verbose=False)
            if isinstance(res, list) and res:
                val_metrics = res[0]
        except Exception:
            pass

        # Checkpoint path from callback
        ckpt_path = ckpt_cb.best_model_path or ckpt_cb.last_model_path
        ckpt_file = Path(ckpt_path) if ckpt_path else None

        manifest = _write_run_manifest(self.cfg, self.workdir, ckpt_file, val_metrics)
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