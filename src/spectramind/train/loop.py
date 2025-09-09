
# src/spectramind/train/loop.py
# =============================================================================
# SpectraMind V50 — Training Loop Utilities
# -----------------------------------------------------------------------------
# Provides high-level orchestration around PyTorch Lightning's Trainer
# for the NeurIPS 2025 Ariel Data Challenge. Handles:
#   • Hydra-configurable Trainer creation
#   • Callback and logger registration
#   • Reproducibility (seeding, deterministic flags)
#   • Graceful resume from checkpoints
#   • Rank-zero safe printing/logging
#
# Design: Keep orchestration minimal here; heavy logic belongs in
# datasets.py, loggers.py, ckpt.py, and train.py.
# =============================================================================

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import pytorch_lightning as pl
    from pytorch_lightning.callbacks import Callback, ModelCheckpoint, EarlyStopping
    from pytorch_lightning.loggers import Logger
except Exception:  # pragma: no cover
    pl, Callback, ModelCheckpoint, EarlyStopping, Logger = None, object, object, object, object

try:
    from hydra.utils import instantiate
except Exception:  # pragma: no cover
    instantiate = None  # type: ignore

log = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _rank_zero_info(msg: str) -> None:
    """Print info only from rank zero (safe for multi-GPU)."""
    if pl and pl.utilities.rank_zero.rank_zero_only.rank == 0:  # type: ignore
        print(msg)
    else:
        log.info(msg)


# -----------------------------------------------------------------------------
# Trainer Construction
# -----------------------------------------------------------------------------

def build_trainer(
    cfg: Dict[str, Any],
    callbacks: Optional[list[Callback]] = None,
    loggers: Optional[list[Logger]] = None,
    resume_from: Optional[Path] = None,
) -> pl.Trainer:
    """
    Build a PyTorch Lightning Trainer from Hydra config.

    Args:
        cfg: Dict-like config (Hydra/OmegaConf).
        callbacks: Extra callbacks to register.
        loggers: Extra loggers to register.
        resume_from: Optional checkpoint path.

    Returns:
        Configured pl.Trainer instance.
    """
    if pl is None or instantiate is None:
        raise RuntimeError("PyTorch Lightning + Hydra required but not available.")

    trainer_cfg = cfg.get("trainer", {})
    trainer: pl.Trainer = instantiate(trainer_cfg)

    # Register loggers
    if loggers:
        for lg in loggers:
            trainer.logger.experiment  # trigger attach
            trainer.loggers.append(lg)

    # Register callbacks
    if callbacks:
        for cb in callbacks:
            trainer.callbacks.append(cb)

    # Handle checkpoint resume
    if resume_from:
        resume_from = Path(resume_from)
        if not resume_from.exists():
            _rank_zero_info(f"⚠️ Resume checkpoint not found: {resume_from}")
        else:
            _rank_zero_info(f"▶️  Resuming from checkpoint: {resume_from}")
            trainer.ckpt_path = str(resume_from)

    return trainer


# -----------------------------------------------------------------------------
# Training Execution
# -----------------------------------------------------------------------------

def run_training(
    trainer: pl.Trainer,
    model: pl.LightningModule,
    datamodule: pl.LightningDataModule,
) -> None:
    """
    Execute training given a Trainer, model, and datamodule.

    Args:
        trainer: Lightning Trainer.
        model: LightningModule (SpectraMind model).
        datamodule: LightningDataModule providing loaders.
    """
    _rank_zero_info("🚀 Starting training loop...")
    try:
        trainer.fit(model=model, datamodule=datamodule)
    except KeyboardInterrupt:
        _rank_zero_info("⏸️ Training interrupted by user.")
    finally:
        _rank_zero_info("🏁 Training finished.")


# -----------------------------------------------------------------------------
# Default Callbacks
# -----------------------------------------------------------------------------

def default_callbacks(cfg: Dict[str, Any]) -> list[Callback]:
    """Instantiate default callbacks (checkpoint, early stopping, etc.) from Hydra config."""
    if instantiate is None:
        return []
    cb_cfgs = cfg.get("callbacks", {})
    return [instantiate(cb) for cb in cb_cfgs.values()]


def default_loggers(cfg: Dict[str, Any]) -> list[Logger]:
    """Instantiate default loggers (WandB, CSV, JSONL, etc.) from Hydra config."""
    if instantiate is None:
        return []
    lg_cfgs = cfg.get("loggers", {})
    return [instantiate(lg) for lg in lg_cfgs.values()]