# src/spectramind/pipeline/predict.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


@dataclass
class PredictPayload:
    config_name: str
    overrides: list[str]
    checkpoint: Optional[str]
    out_path: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


def run(
    *,
    config_name: str = "predict",
    overrides: Iterable[str] | None = None,
    checkpoint: str | None = None,
    out_path: str | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Inference runner: checkpoint + test data → predictions CSV/Parquet.
    """
    payload = PredictPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        checkpoint=checkpoint,
        out_path=out_path,
        strict=strict,
        quiet=quiet,
        env=env or {},
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    if payload.checkpoint is not None:
        cfg.predict.checkpoint = payload.checkpoint
    if payload.out_path is not None:
        cfg.predict.out_path = payload.out_path

    _apply_guardrails(cfg, payload.env)

    try:
        # Expect orchestrator:
        # `spectramind.inference.predict.run(cfg=cfg)`
        from spectramind.inference.predict import run as run_inference  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Inference entrypoint not found. Implement `spectramind.inference.predict.run(cfg)` "
            f"or adjust this import. Cause: {type(e).__name__}: {e}"
        ) from e

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved prediction config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")

    run_inference(cfg=cfg)


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: PredictPayload) -> DictConfig:
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
    cfg.setdefault("predict", {})
    cfg.predict.setdefault("batch_size", 64)
    cfg.predict.setdefault("checkpoint", None)
    cfg.predict.setdefault("out_path", "outputs/predictions.csv")


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))
    cfg.predict.setdefault("num_workers", 2 if (is_kaggle or is_ci) else 4)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            return p.parent if (p / "src").exists() and (p.name == "src") else p
    return here.parents[3]
