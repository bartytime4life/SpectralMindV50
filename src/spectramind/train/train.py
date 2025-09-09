# src/spectramind/train/train.py
# =============================================================================
# SpectraMind V50 — Training Entrypoint
# -----------------------------------------------------------------------------
# Orchestrates model & datamodule construction, callbacks, Trainer creation,
# checkpoint resume, and training using Hydra-configured settings.
#
# Design goals:
#   • Clean separation of config → instantiate/build → fit
#   • Rank-zero logging, friendly error messages
#   • Works with either registry builders or Hydra `_target_` instantiation
#   • Deterministic seeding & reproducible run directories
#   • Kaggle/CI friendly (no hardcoded local paths, guarded heavy imports)
# =============================================================================

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional

# --- Hydra / OmegaConf
try:
    from omegaconf import DictConfig, OmegaConf
    from hydra.utils import instantiate
except Exception as _e:  # pragma: no cover
    DictConfig = Any  # type: ignore
    def instantiate(*args, **kwargs):  # type: ignore
        raise RuntimeError("Hydra is required to instantiate configs") from _e
    OmegaConf = None  # type: ignore

# --- PyTorch / Lightning (guarded imports)
try:  # pragma: no cover
    import torch
    from torch.utils.data import DataLoader
    import pytorch_lightning as pl
    from pytorch_lightning.utilities.rank_zero import rank_zero_only
except Exception as _e:  # pragma: no cover
    torch = None  # type: ignore
    pl = None  # type: ignore
    DataLoader = None  # type: ignore
    def rank_zero_only(fn):  # type: ignore
        def _wrap(*a, **k):
            return fn(*a, **k)
        return _wrap
    _PL_IMPORT_ERROR = _e
else:
    _PL_IMPORT_ERROR = None

# --- Local training utilities
from .registry import (
    get_model_builder,
    get_loss_builder,
    get_optimizer_builder,
    get_scheduler_builder,
)
from .callbacks import build_callbacks
from .ckpt import resume_trainer_if_available
from .config import (
    resolve_train_paths,
    seed_everything_from_cfg,
    build_trainer_kwargs,
    TrainPaths,  # for typing
)
from .collate import build_collate_fn, CollateConfig


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #

@rank_zero_only
def _log_rank_zero(msg: str) -> None:
    print(f"[SpectraMind][train] {msg}")


def _ensure_pl() -> None:
    if _PL_IMPORT_ERROR is not None:
        raise RuntimeError(
            "This module requires `pytorch-lightning` and `torch` at runtime."
        ) from _PL_IMPORT_ERROR


def _save_config_snapshot(cfg: DictConfig, run_dir: Path) -> None:
    """
    Persist a human- and machine-readable snapshot of the composed Hydra config
    for reproducibility. (JSON + YAML)
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    # YAML
    if OmegaConf is not None:
        yaml_path = run_dir / "config_snapshot.yaml"
        with yaml_path.open("w", encoding="utf-8") as f:
            OmegaConf.save(config=cfg, f=f.name)
    # JSON (resolves OmegaConf to pure dict)
    try:
        cfg_dict = OmegaConf.to_container(cfg, resolve=True) if OmegaConf else dict(cfg)
    except Exception:
        cfg_dict = dict(cfg)
    json_path = run_dir / "config_snapshot.json"
    with json_path.open("w", encoding="utf-8") as f:
        json.dump(cfg_dict, f, indent=2, sort_keys=True)


def _maybe_instantiate(obj_cfg: Any, **overrides: Any) -> Any:
    """
    Try Hydra instantiate if `_target_` is present; else return None.
    Useful for optional components (model/datamodule) which may come
    from registry instead.
    """
    if obj_cfg is None:
        return None
    if isinstance(obj_cfg, dict) and "_target_" in obj_cfg:
        merged = {**obj_cfg, **overrides}
        return instantiate(merged)
    # OmegaConf DictConfig supports attribute-style; handle that too
    if hasattr(obj_cfg, "_get_full_key") or hasattr(obj_cfg, "get"):
        try:
            if "_target_" in obj_cfg:
                merged = {**OmegaConf.to_container(obj_cfg, resolve=True), **overrides}
                return instantiate(merged)
        except Exception:
            pass
    return None


def _build_datamodule_from_cfg(cfg: DictConfig, paths: TrainPaths) -> Any:
    """
    Construct a LightningDataModule (preferred) or ad-hoc dataloaders from cfg.

    Supports:
      - Hydra instantiate: cfg.data.datamodule._target_
      - Passing a collate_fn via CollateConfig
      - Fallback: instantiate datasets + wrap in DataLoader
    """
    collate_cfg = None
    if "data" in cfg and "collate" in cfg.data:
        # Convert to CollateConfig dataclass if fields match
        try:
            collate_cfg = CollateConfig(**OmegaConf.to_container(cfg.data.collate, resolve=True))  # type: ignore
        except Exception:
            collate_cfg = CollateConfig()
    collate_fn = build_collate_fn(collate_cfg)

    # 1) Preferred: LightningDataModule via Hydra
    dm = _maybe_instantiate(getattr(cfg.data, "datamodule", None), collate_fn=collate_fn)
    if dm is not None:
        return dm

    # 2) Fallback: datasets + DataLoader via Hydra
    train_ds = _maybe_instantiate(getattr(cfg.data, "train_dataset", None))
    val_ds = _maybe_instantiate(getattr(cfg.data, "val_dataset", None))
    if train_ds is not None and val_ds is not None:
        batch_size = int(getattr(cfg.data, "batch_size", 32))
        num_workers = int(getattr(cfg.data, "num_workers", 2))
        shuffle = bool(getattr(cfg.data, "shuffle", True))

        train_loader = DataLoader(
            train_ds, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )
        val_loader = DataLoader(
            val_ds, batch_size=batch_size, shuffle=False,
            num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
        )

        class _AdHocDM(pl.LightningDataModule):  # type: ignore
            def __init__(self):
                super().__init__()
                self._train = train_loader
                self._val = val_loader

            def train_dataloader(self):
                return self._train

            def val_dataloader(self):
                return self._val

        return _AdHocDM()

    raise RuntimeError(
        "Could not build datamodule. Provide either `cfg.data.datamodule._target_` "
        "or both `cfg.data.train_dataset._target_` and `cfg.data.val_dataset._target_`."
    )


def _build_model_from_cfg_or_registry(cfg: DictConfig) -> Any:
    """
    Build the LightningModule either via registry builders (model+loss glued inside
    a custom module) or via Hydra instantiate on a `_target_`.

    Expected config options:
      - cfg.model: name or Hydra node
      - cfg.loss: name or Hydra node (if needed)
    """
    # 1) Hydra instantiate (preferred if provided)
    model = _maybe_instantiate(getattr(cfg, "model", None))
    if model is not None:
        return model

    # 2) Registry path (classic builder pattern)
    model_name = getattr(cfg.model, "name", None) if hasattr(cfg, "model") else None
    loss_name = getattr(cfg.loss, "name", None) if hasattr(cfg, "loss") else None
    if not model_name:
        raise RuntimeError("Model config is missing. Provide `model._target_` or `model.name`.")

    model_builder = get_model_builder(model_name)
    loss_builder = get_loss_builder(loss_name) if loss_name else None

    # Build loss first (if separate)
    criterion = loss_builder(cfg=cfg) if loss_builder else None
    # Build model with optional criterion
    model = model_builder(cfg=cfg, criterion=criterion)
    return model


def _attach_optimizers_if_needed(model: Any, cfg: DictConfig) -> None:
    """
    If the LightningModule doesn't implement `configure_optimizers`, build optimizer/
    scheduler via registry (or Hydra) and attach to the model (expects setters or attributes).
    """
    if hasattr(model, "configure_optimizers"):
        # Assume model fully handles optimizers; nothing to do.
        return

    # Try registry builders first
    opt_name = getattr(cfg.optimizer, "name", None) if hasattr(cfg, "optimizer") else None
    sch_name = getattr(cfg.scheduler, "name", None) if hasattr(cfg, "scheduler") else None

    optimizer = None
    scheduler = None

    if opt_name:
        optimizer_builder = get_optimizer_builder(opt_name)
        optimizer = optimizer_builder(cfg=cfg, params=model.parameters())

    # Optional scheduler
    if sch_name:
        scheduler_builder = get_scheduler_builder(sch_name)
        scheduler = scheduler_builder(cfg=cfg, optimizer=optimizer)

    # Attach to model; Lightning will pick it up if `configure_optimizers` inspects attributes,
    # or the training loop wrapper can pass explicitly (custom wrappers).
    setattr(model, "_external_optimizer", optimizer)
    setattr(model, "_external_scheduler", scheduler)


# ----------------------------------------------------------------------------- #
# Public API
# ----------------------------------------------------------------------------- #

def train_from_config(cfg: DictConfig) -> Dict[str, Any]:
    """
    Main training entrypoint used by the spectramind CLI and CI.

    Returns a summary dict with run directories and (optionally) metrics.
    """
    _ensure_pl()

    # Paths & seeding
    paths = resolve_train_paths(cfg)
    seed = seed_everything_from_cfg(cfg)
    _log_rank_zero(f"Resolved run dir: {paths.run_dir}")
    _log_rank_zero(f"Seeded everything with: {seed}")

    # Persist config snapshot
    _save_config_snapshot(cfg, paths.run_dir)

    # Build datamodule
    _log_rank_zero("Building datamodule...")
    datamodule = _build_datamodule_from_cfg(cfg, paths)

    # Build model (LightningModule)
    _log_rank_zero("Building model...")
    model = _build_model_from_cfg_or_registry(cfg)
    _attach_optimizers_if_needed(model, cfg)

    # Trainer configuration
    trainer_kwargs = build_trainer_kwargs(cfg, default_ckpt_dir=paths.ckpt_dir)
    callbacks, ckpt_cb = build_callbacks(cfg, ckpt_dir=paths.ckpt_dir, logs_dir=paths.logs_dir)
    trainer = pl.Trainer(callbacks=callbacks, **trainer_kwargs)

    # Resume logic (best/last)
    prefer = getattr(getattr(cfg, "train", {}), "resume", {}).get("prefer", "best")
    monitor = getattr(getattr(cfg, "callbacks", {}), "checkpoint", {}).get("monitor", "val_loss")
    mode = getattr(getattr(cfg, "callbacks", {}), "checkpoint", {}).get("mode", "min")
    resume_path = resume_trainer_if_available(
        trainer, paths.ckpt_dir, prefer=prefer, monitor=monitor, mode=mode
    )
    if resume_path:
        _log_rank_zero(f"Resuming from checkpoint: {resume_path}")

    # Fit
    _log_rank_zero("Starting training...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=getattr(trainer, "ckpt_path", None))
    _log_rank_zero("Training finished.")

    # Optional: Validate/Test after fit if configured
    do_validate = bool(getattr(getattr(cfg, "train", {}), "validate_after_fit", True))
    if do_validate:
        _log_rank_zero("Running validation...")
        trainer.validate(model=model, datamodule=datamodule, ckpt_path="best")

    summary = {
        "run_dir": str(paths.run_dir),
        "ckpt_dir": str(paths.ckpt_dir),
        "logs_dir": str(paths.logs_dir),
        "seed": seed,
        "resumed_from": str(resume_path) if resume_path else None,
    }
    return summary


# ----------------------------------------------------------------------------- #
# Optional Hydra entrypoint (CLI usually calls train_from_config)
# ----------------------------------------------------------------------------- #

if __name__ == "__main__":  # pragma: no cover
    # We avoid hardcoding config_path because the spectramind CLI typically calls this
    # function after Hydra composition. This block is provided for direct manual runs.
    try:
        import hydra
        from omegaconf import OmegaConf

        @hydra.main(config_path="../../../configs", config_name="train", version_base="1.3")
        def _main(cfg: DictConfig) -> None:
            _ = train_from_config(cfg)

        _main()
    except Exception as e:
        raise SystemExit(f"[SpectraMind][train] Failed in __main__: {e}")