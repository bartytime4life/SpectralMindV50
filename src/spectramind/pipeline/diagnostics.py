# src/spectramind/pipeline/diagnostics.py
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


@dataclass
class DiagPayload:
    config_name: str
    overrides: list[str]
    report_dir: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


def run(
    *,
    config_name: str = "diagnose",
    overrides: Iterable[str] | None = None,
    report_dir: str | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Diagnostics runner: metrics/plots/HTML report (FFT/UMAP/SHAP, etc.)
    """
    payload = DiagPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        report_dir=report_dir,
        strict=strict,
        quiet=quiet,
        env=env or {},
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    if payload.report_dir is not None:
        cfg.diagnostics.report_dir = payload.report_dir

    _apply_guardrails(cfg, payload.env)

    try:
        # Expect orchestrator:
        # `spectramind.diagnostics.report.generate(cfg=cfg)`
        from spectramind.diagnostics.report import generate as generate_report  # type: ignore
    except Exception as e:
        raise RuntimeError(
            f"Diagnostics entrypoint not found. Implement `spectramind.diagnostics.report.generate(cfg)` "
            f"or adjust this import. Cause: {type(e).__name__}: {e}"
        ) from e

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved diagnostics config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")

    generate_report(cfg=cfg)


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: DiagPayload) -> DictConfig:
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
    cfg.setdefault("diagnostics", {})
    cfg.diagnostics.setdefault("report_dir", "outputs/diagnostics")
    cfg.diagnostics.setdefault("modules", ["metrics", "fft", "umap", "shap"])


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))
    cfg.diagnostics.setdefault("num_workers", 1 if (is_kaggle or is_ci) else 2)


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            return p.parent if (p / "src").exists() and (p.name == "src") else p
    return here.parents[3]
