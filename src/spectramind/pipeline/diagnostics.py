# src/spectramind/pipeline/diagnostics.py
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

__all__ = ["run"]


# ----------------------------- Data model -------------------------------------


@dataclass(frozen=True, slots=True)
class DiagPayload:
    """Strongly-typed diagnostics invocation payload."""
    config_name: str
    overrides: list[str]
    report_dir: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


# --------------------------------- API ----------------------------------------


def run(
    *,
    config_name: str = "diagnose",
    overrides: Iterable[str] | None = None,
    report_dir: str | os.PathLike[str] | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Diagnostics runner: metrics/plots/HTML report (FFT/UMAP/SHAP, etc.)

    Responsibilities
    ---------------
    • Compose Hydra config (diagnostics + data + model refs)
    • Dispatch to internal reporting pipeline
    • Write artifacts in a DVC-friendly structure

    Parameters
    ----------
    config_name
        Name of the top-level Hydra config under `configs/` (default: "diagnose").
    overrides
        Hydra override strings, e.g. `["env=kaggle", "diagnostics.modules=[metrics,fft]"]`.
    report_dir
        Optional report output directory. When provided, overrides `cfg.diagnostics.report_dir`.
    strict
        If True, propagate Hydra composition/validation errors; otherwise warn and continue with defaults.
    quiet
        Suppress resolved config echoing to stderr.
    env
        Optional execution context, e.g. {"is_kaggle": True, "is_ci": False}.

    Raises
    ------
    FileNotFoundError
        When `configs/` cannot be found at repo root.
    RuntimeError
        When the internal diagnostics orchestrator is missing.
    """
    payload = DiagPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        report_dir=str(report_dir) if report_dir is not None else None,
        strict=strict,
        quiet=quiet,
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    # honor explicit report_dir if provided
    if payload.report_dir is not None:
        cfg.diagnostics.report_dir = payload.report_dir

    _apply_guardrails(cfg, payload.env)

    # prepare report root + emit config snapshot/metadata for traceability
    report_root = _prepare_report_root(repo_root, cfg)
    _emit_config_snapshot(report_root, cfg)

    # Defer heavy imports and provide clear errors
    try:
        # Expect orchestrator:
        # `spectramind.diagnostics.report.generate(cfg=cfg)`
        from spectramind.diagnostics.report import generate as generate_report  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Diagnostics entrypoint not found. Implement "
            "`spectramind.diagnostics.report.generate(cfg)` or adjust this import. "
            f"Cause: {type(e).__name__}: {e}"
        ) from e

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved diagnostics config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")
        sys.stderr.write(f"[diagnostics] report_dir = {report_root}\n")

    generate_report(cfg=cfg)


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: DiagPayload) -> DictConfig:
    """Compose Hydra config from repo `configs/` with robust defaults + strict mode."""
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per repo blueprint)"
        )

    with initialize(config_path=str(config_dir), version_base=None):
        try:
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)
        except Exception:
            if payload.strict:
                raise
            # Best-effort fallback: attempt to compose base config with no overrides
            cfg = compose(config_name=payload.config_name, overrides=[])

    _ensure_defaults(cfg)
    _validate_minimal_schema(cfg, strict=payload.strict)
    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    """Inject minimal defaults so downstream code doesn’t explode."""
    cfg.setdefault("diagnostics", {})
    cfg.diagnostics.setdefault("report_dir", "outputs/diagnostics")
    cfg.diagnostics.setdefault("modules", ["metrics", "fft", "umap", "shap"])
    cfg.diagnostics.setdefault("num_workers", 2)
    cfg.setdefault("runtime", {})
    cfg.runtime.setdefault("seed", 42)


def _validate_minimal_schema(cfg: DictConfig, *, strict: bool) -> None:
    """Minimal schema checks; raise in strict mode, warn otherwise."""
    missing: list[str] = []
    if "diagnostics" not in cfg:
        missing.append("diagnostics")
    else:
        if "report_dir" not in cfg.diagnostics:
            missing.append("diagnostics.report_dir")
        if "modules" not in cfg.diagnostics:
            missing.append("diagnostics.modules")

    if missing:
        msg = f"[diagnostics] Missing required config keys: {', '.join(missing)}"
        if strict:
            raise KeyError(msg)
        sys.stderr.write(msg + " (continuing)\n")


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    """Apply environment-aware guardrails (workers, determinism, etc.)."""
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))

    # Conservative workers on non-interactive envs
    if "num_workers" not in cfg.diagnostics:
        cfg.diagnostics.num_workers = 1 if (is_kaggle or is_ci) else 2

    # Respect common CI/Kaggle env hints
    cfg.runtime.setdefault("deterministic", bool(is_ci))
    if is_kaggle:
        cfg.runtime.setdefault("low_mem_mode", True)


def _prepare_report_root(repo_root: Path, cfg: DictConfig) -> Path:
    """Create report root under repo root (DVC-friendly); return absolute path."""
    raw = str(cfg.diagnostics.report_dir)
    report_root = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    report_root.mkdir(parents=True, exist_ok=True)

    # common subdirs (safe to create; DVC decides tracking)
    for sub in ("assets", "plots", "html", "logs", "snapshots"):
        (report_root / sub).mkdir(exist_ok=True)
    return report_root


def _emit_config_snapshot(report_root: Path, cfg: DictConfig) -> None:
    """Write resolved config & minimal run metadata for traceability."""
    try:
        # YAML snapshot (human-friendly)
        snap_yaml = report_root / "snapshots" / "config_snapshot.yaml"
        OmegaConf.save(cfg, snap_yaml, resolve=True)

        # JSON run metadata (machine-friendly)
        meta = {
            "schema": "spectramind/diagnostics_config_snapshot@v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_root": str(_nearest_git_root(report_root) or ""),
            "report_dir": str(report_root),
            "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
            "runtime": dict(getattr(cfg, "runtime", {})),
            "modules": list(getattr(cfg, "diagnostics", {}).get("modules", [])),
        }
        meta_json = report_root / "snapshots" / "config_snapshot.json"
        meta_json.write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[diagnostics] warning: failed to write config snapshot: {e}\n")


def _nearest_git_root(path: Path) -> Path | None:
    """Return nearest parent that contains `.git`, else None."""
    for p in [path, *path.parents]:
        if (p / ".git").exists():
            return p
    return None


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
            # If we’re sitting in .../src, prefer its parent as root (layout: root/src/...)
            if p.name == "src":
                return p.parent
            return p
    # Fallback: best-effort three levels up (keeps Kaggle/colab from exploding)
    return here.parents[3]


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))
