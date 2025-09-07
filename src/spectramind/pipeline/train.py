# src/spectramind/pipeline/train.py
from __future__ import annotations

import os
import sys
import time
import json
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pytorch_lightning as pl
import torch
from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint, RichModelSummary
from pytorch_lightning.loggers import CSVLogger

# Optional loggers/callbacks (loaded lazily)
try:
    from pytorch_lightning.loggers import WandbLogger  # type: ignore
except Exception:  # pragma: no cover
    WandbLogger = None  # type: ignore

__all__ = ["run"]


# --------------------------------------------------------------------------------------
# Payload & Entrypoint
# --------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class TrainPayload:
    config_name: str
    overrides: List[str]
    seed: Optional[int]
    devices: Optional[str]
    precision: Optional[str]
    epochs: Optional[int]
    resume_from: Optional[str]
    log_dir: Optional[str]
    dry_run: bool
    strict: bool
    quiet: bool
    env: Dict[str, Any]


def run(
    *,
    config_name: str = "train",
    overrides: Iterable[str] | None = None,
    seed: int | None = None,
    devices: str | None = None,
    precision: str | None = None,
    epochs: int | None = None,
    resume_from: str | None = None,
    log_dir: str | None = None,
    dry_run: bool = False,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Training runner (Hydra + Lightning). Called by the CLI thin wrapper.

    Responsibilities
    ---------------
    • Compose Hydra config (config_name + overrides)
    • Set seeds & deterministic behavior when requested
    • Instantiate DataModule, LightningModule, loggers, callbacks
    • Enforce Kaggle/CI guardrails (no internet, reduced workers/verbosity, stable precision)
    • Launch PL.Trainer.fit() and persist artifacts/metrics

    Notes
    -----
    • Business logic lives here; CLI remains thin and import-cheap.
    • Config/schema should remain the single source of truth.
    """
    payload = TrainPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        seed=seed,
        devices=devices,
        precision=precision,
        epochs=epochs,
        resume_from=resume_from,
        log_dir=log_dir,
        dry_run=dry_run,
        strict=strict,
        quiet=quiet,
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    # Merge/override runtime toggles from CLI
    if payload.seed is not None:
        cfg.training.seed = payload.seed
    if payload.devices is not None:
        cfg.training.devices = payload.devices
    if payload.precision is not None:
        cfg.training.precision = payload.precision
    if payload.epochs is not None:
        cfg.training.epochs = payload.epochs
    if payload.log_dir is not None:
        cfg.logger.dir = payload.log_dir
    if payload.resume_from is not None:
        cfg.training.resume_from = payload.resume_from

    # Kaggle/CI guardrails
    _apply_env_guardrails(cfg, payload.env, payload.quiet, payload.dry_run)

    # Seeds & determinism
    _seed_everything(cfg.training.seed)

    # Instantiate components (keep imports local to provide clear errors)
    try:
        from spectramind.data.datamodule import SpectraDataModule  # type: ignore[attr-defined]
        from spectramind.models.system import SpectraSystem  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to import pipeline modules: {type(e).__name__}: {e}") from e

    # Data
    datamodule = SpectraDataModule(cfg.data, cfg.calib, num_workers=cfg.training.num_workers)

    # Model (LightningModule)
    model = SpectraSystem(
        model_cfg=cfg.model,
        loss_cfg=cfg.loss,
        optimizer_cfg=cfg.training.optimizer,
        scheduler_cfg=cfg.training.scheduler,
        metrics_cfg=cfg.training.metrics,
    )

    # Loggers
    loggers = _build_loggers(cfg, repo_root, payload.quiet)

    # Callbacks
    callbacks = _build_callbacks(cfg)

    # Trainer
    trainer = _build_trainer(cfg, loggers, callbacks, payload.dry_run)

    # Fit
    _print_banner(cfg, payload, repo_root, payload.quiet)
    trainer.fit(model, datamodule=datamodule, ckpt_path=cfg.training.resume_from or None)

    # Persist config snapshot & run manifest
    _persist_config_snapshot(cfg, repo_root, loggers)


# --------------------------------------------------------------------------------------
# Hydra & Config Utilities
# --------------------------------------------------------------------------------------


def _compose_hydra_config(repo_root: Path, payload: TrainPayload) -> DictConfig:
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(f"Missing Hydra config directory: {config_dir}")

    # Hydra 1.3 style composition
    with initialize(config_path=str(config_dir), version_base=None):
        try:
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)
        except Exception:
            if payload.strict:
                raise
            # Non-strict: try a minimal compose without overrides
            cfg = compose(config_name=payload.config_name, overrides=[])

    # Enforce minimally expected sections with safe defaults
    _ensure_defaults(cfg)
    _validate_minimal_schema(cfg, strict=payload.strict)

    # Emit k/v for debugging if not quiet
    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved training config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")

    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    # Training defaults
    cfg.setdefault("training", {})
    cfg.training.setdefault("seed", 42)
    cfg.training.setdefault("epochs", 20)
    cfg.training.setdefault("devices", "auto")
    cfg.training.setdefault("precision", "32")
    cfg.training.setdefault("num_workers", max(1, os.cpu_count() // 4 if os.cpu_count() else 1))
    cfg.training.setdefault("resume_from", None)
    cfg.training.setdefault("optimizer", {})
    cfg.training.setdefault("scheduler", {})
    cfg.training.setdefault("metrics", {})
    cfg.training.setdefault(
        "early_stopping",
        {"enabled": True, "monitor": "val/loss", "patience": 10, "mode": "min"},
    )
    cfg.training.setdefault(
        "checkpoint",
        {"monitor": "val/loss", "mode": "min", "save_top_k": 1},
    )
    # Optional extras
    cfg.training.setdefault("grad_clip_val", None)
    cfg.training.setdefault("detect_anomaly", False)

    # Logger defaults (DVC-friendly)
    cfg.setdefault("logger", {})
    cfg.logger.setdefault("dir", "outputs")
    cfg.logger.setdefault("name", "train")
    cfg.logger.setdefault("wandb", {"enabled": False, "project": "spectramind-v50", "entity": None, "tags": []})

    # Model/Data defaults
    cfg.setdefault("data", {})
    cfg.setdefault("calib", {})
    cfg.setdefault("model", {})
    cfg.setdefault("loss", {})


def _validate_minimal_schema(cfg: DictConfig, *, strict: bool) -> None:
    missing: List[str] = []
    for key in ("training", "logger", "data", "model", "loss"):
        if key not in cfg:
            missing.append(key)
    for key in ("epochs", "devices", "precision", "num_workers"):
        if "training" in cfg and key not in cfg.training:
            missing.append(f"training.{key}")

    if missing:
        msg = f"[train] Missing required config keys: {', '.join(missing)}"
        if strict:
            raise KeyError(msg)
        sys.stderr.write(msg + " (continuing)\n")


# --------------------------------------------------------------------------------------
# Trainer / Loggers / Callbacks
# --------------------------------------------------------------------------------------


def _build_trainer(
    cfg: DictConfig,
    loggers: List[pl.loggers.Logger],
    callbacks: List[pl.Callback],
    dry_run: bool,
) -> pl.Trainer:
    accelerator, devices = _parse_devices(cfg.training.devices)
    prec = _normalize_precision(cfg.training.precision)

    # Fast smoke test
    fast_dev_run = bool(dry_run)

    trainer = pl.Trainer(
        accelerator=accelerator,
        devices=devices,
        max_epochs=1 if fast_dev_run else int(cfg.training.epochs),
        precision=prec,
        logger=loggers,
        callbacks=callbacks,
        deterministic=True,  # prefer deterministic kernels when available
        log_every_n_steps=10,
        enable_progress_bar=not fast_dev_run,
        fast_dev_run=fast_dev_run,
        gradient_clip_val=getattr(cfg.training, "grad_clip_val", None),
        detect_anomaly=getattr(cfg.training, "detect_anomaly", False),
    )
    return trainer


def _build_loggers(cfg: DictConfig, repo_root: Path, quiet: bool) -> List[pl.loggers.Logger]:
    out_dir = Path(cfg.logger.dir)
    out_dir = out_dir if out_dir.is_absolute() else (repo_root / out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    csv = CSVLogger(save_dir=str(out_dir), name=str(cfg.logger.name or "train"), flush_logs_every_n_steps=50)
    loggers: List[pl.loggers.Logger] = [csv]

    if bool(cfg.logger.wandb.enabled) and WandbLogger is not None:
        try:
            wandb_logger = WandbLogger(
                project=cfg.logger.wandb.project,
                entity=cfg.logger.wandb.entity,
                tags=list(cfg.logger.wandb.tags or []),
                save_dir=str(out_dir),
                name=cfg.logger.name or f"run-{int(time.time())}",
                log_model=False,  # storing large ckpts on wandb can be expensive
                mode="offline" if os.environ.get("WANDB_MODE") == "offline" else "online",
            )
            loggers.append(wandb_logger)
        except Exception as e:  # pragma: no cover
            if not quiet:
                sys.stderr.write(f"[logger] WandB unavailable: {e}\n")

    return loggers


def _build_callbacks(cfg: DictConfig) -> List[pl.Callback]:
    callbacks: List[pl.Callback] = [
        RichModelSummary(max_depth=2),
    ]

    if cfg.training.early_stopping.enabled:
        callbacks.append(
            EarlyStopping(
                monitor=str(cfg.training.early_stopping.monitor),
                patience=int(cfg.training.early_stopping.patience),
                mode=str(cfg.training.early_stopping.mode),
                verbose=False,
            )
        )

    ck = cfg.training.checkpoint
    filename_tmpl = "{epoch:03d}-{" + str(ck.monitor) + ":.5f}"
    callbacks.append(
        ModelCheckpoint(
            monitor=str(ck.monitor),
            mode=str(ck.mode),
            save_top_k=int(ck.save_top_k),
            save_last=True,
            filename=filename_tmpl,
            auto_insert_metric_name=False,
        )
    )

    return callbacks


# --------------------------------------------------------------------------------------
# Guardrails, Seeding, Utils
# --------------------------------------------------------------------------------------


def _apply_env_guardrails(cfg: DictConfig, env: Dict[str, Any], quiet: bool, dry_run: bool) -> None:
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))

    # Quiet default in non-interactive environments
    if is_kaggle or is_ci:
        cfg.training.setdefault("progress_bar", False)

    # Workers: keep conservative on Kaggle/CI
    if is_kaggle or is_ci:
        cfg.training.num_workers = min(int(cfg.training.num_workers), 2)

    # Precision defaults on Kaggle T4/CPU
    if is_kaggle and str(cfg.training.precision).lower() in {"16", "16-mixed", "fp16"}:
        # Avoid surprise OOM / missing AMP
        cfg.training.precision = "32"

    # Be extra conservative for dry runs
    if dry_run:
        cfg.training.num_workers = 0


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    pl.seed_everything(int(seed), workers=True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _normalize_precision(p: str | int | None) -> str | int:
    if p is None:
        return 32
    s = str(p).lower()
    if s in {"16", "16-true"}:
        return 16
    if s in {"32", "32-true"}:
        return 32
    if s in {"bf16", "bfloat16"}:
        return "bf16-mixed"
    if s in {"fp16", "16-mixed"}:
        return "16-mixed"
    return s


def _parse_devices(spec: Any) -> Tuple[str, Any]:
    """
    Returns (accelerator, devices) suitable for PL.Trainer.
    """
    if spec is None or str(spec).lower() == "auto":
        return "auto", "auto"

    if isinstance(spec, int):
        return ("gpu" if torch.cuda.is_available() else "cpu"), spec

    s = str(spec).strip().lower()
    if s in {"cpu", "gpu", "auto"}:
        return ("gpu" if s == "gpu" else s), 1 if s in {"cpu", "gpu"} else "auto"

    # Comma-separated GPU indices, e.g. "0,1"
    if "," in s:
        devs = [int(x) for x in s.split(",") if x.strip() != ""]
        return ("gpu" if torch.cuda.is_available() else "cpu"), devs

    # Single index string
    if s.isdigit():
        return ("gpu" if torch.cuda.is_available() else "cpu"), int(s)

    return "auto", "auto"


def _find_repo_root() -> Path:
    """
    Heuristic: start from this file, walk up until we hit a project marker.

    Accepts:
      • pyproject.toml (preferred)
      • setup.cfg
      • .git
    """
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            return p.parent if (p / "src").exists() and (p.name == "src") else p
    # Fallback: three levels up from this file (keeps Kaggle/Colab from exploding)
    return here.parents[3]


def _print_banner(cfg: DictConfig, payload: TrainPayload, repo_root: Path, quiet: bool) -> None:
    if quiet:
        return
    sys.stderr.write(
        "\n"
        "──────────────────────────────────────────────────────────────\n"
        "  SpectraMind V50 — Training\n"
        "──────────────────────────────────────────────────────────────\n"
        f"Repo:     {repo_root}\n"
        f"Config:   {payload.config_name}\n"
        f"Overrides:{' ' if payload.overrides else ''}{', '.join(payload.overrides) if payload.overrides else '(none)'}\n"
        f"Seed:     {cfg.training.seed}\n"
        f"Devices:  {cfg.training.devices}\n"
        f"Precision:{cfg.training.precision}\n"
        f"Epochs:   {cfg.training.epochs}\n"
        f"Log dir:  {cfg.logger.dir}\n"
        "──────────────────────────────────────────────────────────────\n"
    )


def _persist_config_snapshot(cfg: DictConfig, repo_root: Path, loggers: List[pl.loggers.Logger]) -> None:
    # Determine run directory from first CSV logger
    run_dir: Optional[Path] = None
    for lg in loggers:
        if isinstance(lg, CSVLogger):
            run_dir = Path(lg.log_dir) / lg.name / f"version_{lg.version}"
            break
    if run_dir is None:
        return

    run_dir.mkdir(parents=True, exist_ok=True)
    # Save the fully resolved config (YAML)
    (run_dir / "config_resolved.yaml").write_text(OmegaConf.to_yaml(cfg, resolve=True))

    # Save a minimal manifest (JSON) with git SHA + timestamp
    manifest = {
        "schema": "spectramind/train_manifest@v1",
        "run_dir": str(run_dir),
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_sha": _git_sha(repo_root),
        "overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
        "training": {
            "epochs": int(cfg.training.get("epochs", 0)),
            "devices": str(cfg.training.get("devices", "auto")),
            "precision": str(cfg.training.get("precision", "32")),
            "num_workers": int(cfg.training.get("num_workers", 0)),
        },
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))


def _git_sha(repo_root: Path) -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(repo_root))
            .decode("utf-8")
            .strip()
        )
    except Exception:
        return "unknown"
