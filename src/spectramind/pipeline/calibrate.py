# src/spectramind/pipeline/calibrate.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


@dataclass
class CalibPayload:
    config_name: str
    overrides: list[str]
    out_dir: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


def run(
    *,
    config_name: str = "calibrate",
    overrides: Iterable[str] | None = None,
    out_dir: str | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Calibration runner: raw inputs → calibrated cubes/tensors.

    Responsibilities:
      • Compose Hydra config (data + calib + outputs)
      • Dispatch to internal calibration pipeline
      • Write artifacts in a DVC-friendly structure
    """
    payload = CalibPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        out_dir=out_dir,
        strict=strict,
        quiet=quiet,
        env=env or {},
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    if payload.out_dir is not None:
        cfg.outputs.dir = payload.out_dir

    _apply_guardrails(cfg, payload.env)

    # Defer heavy imports and provide clear errors
    try:
        # Expect you to implement this orchestrator:
        # `spectramind.calib.pipeline.run(cfg=cfg)`
        from spectramind.calib.pipeline import run as run_calibration  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            f"Calibration entrypoint not found. Implement `spectramind.calib.pipeline.run(cfg)` "
            f"or adjust this import. Cause: {type(e).__name__}: {e}"
        ) from e

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved calibration config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")

    run_calibration(cfg=cfg)


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: CalibPayload) -> DictConfig:
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(f"Missing Hydra config directory: {config_dir}")

    with initialize(config_path=str(config_dir), version_base=None):
        try:
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)
        except Exception:
            if payload.strict:
                raise
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)

    _ensure_defaults(cfg)
    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    cfg.setdefault("data", {})
    cfg.setdefault("calib", {})
    cfg.setdefault("outputs", {})
    cfg.outputs.setdefault("dir", "data/processed")
    cfg.outputs.setdefault("format", "zarr")  # or parquet/npz


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))

    # Default to conservative workers on non-interactive envs
    cfg.setdefault("runtime", {})
    cfg.runtime.setdefault("num_workers", 2 if (is_kaggle or is_ci) else 4)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            return p.parent if (p / "src").exists() and (p.name == "src") else p
    return here.parents[3]
