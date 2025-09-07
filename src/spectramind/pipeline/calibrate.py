# src/spectramind/pipeline/calibrate.py
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
class CalibPayload:
    """Strongly-typed calibration invocation payload."""

    config_name: str
    overrides: list[str]
    out_dir: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


# --------------------------------- API ----------------------------------------


def run(
    *,
    config_name: str = "calibrate",
    overrides: Iterable[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Calibration runner: raw inputs → calibrated cubes/tensors.

    Responsibilities
    ---------------
    • Compose Hydra config (data + calib + outputs)
    • Dispatch to internal calibration pipeline
    • Write artifacts in a DVC-friendly structure

    Parameters
    ----------
    config_name
        Name of the top-level Hydra config under `configs/` (default: "calibrate").
    overrides
        Hydra override strings, e.g. `["env=kaggle", "calib=fast"]`.
    out_dir
        Optional output directory. When provided, overrides `cfg.outputs.dir`.
    strict
        If True, propagate Hydra composition errors; otherwise try to continue with defaults.
    quiet
        Suppress resolved config echoing to stderr.
    env
        Optional execution context, e.g. {"is_kaggle": True, "is_ci": False}.

    Raises
    ------
    FileNotFoundError
        When `configs/` cannot be found at repo root.
    RuntimeError
        When the internal calibration orchestrator is missing.
    """
    payload = CalibPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        out_dir=str(out_dir) if out_dir is not None else None,
        strict=strict,
        quiet=quiet,
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    # honor explicit output dir if provided
    if payload.out_dir is not None:
        cfg.outputs.dir = payload.out_dir

    _apply_guardrails(cfg, payload.env)

    # Prepare outputs dir (DVC-friendly) + snapshot config for traceability
    outputs_dir = _prepare_outputs_dir(repo_root, cfg)
    _emit_config_snapshot(outputs_dir, cfg)

    # Defer heavy imports and provide clear errors
    try:
        # Expect you to implement this orchestrator:
        # `spectramind.calib.pipeline.run(cfg=cfg)`
        from spectramind.calib.pipeline import run as run_calibration  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Calibration entrypoint not found. Implement "
            "`spectramind.calib.pipeline.run(cfg)` or adjust this import. "
            f"Cause: {type(e).__name__}: {e}"
        ) from e

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved calibration config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")
        sys.stderr.write(f"[calibrate] outputs.dir = {outputs_dir}\n")

    run_calibration(cfg=cfg)


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: CalibPayload) -> DictConfig:
    """Compose Hydra config from repo `configs/` with robust defaults + strict mode."""
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per repo blueprint)"
        )

    # hydra.initialize accepts absolute paths since Hydra 1.1+
    with initialize(config_path=str(config_dir), version_base=None):
        try:
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)
        except Exception:
            if payload.strict:
                raise
            # Best-effort: attempt to compose base config with no overrides
            cfg = compose(config_name=payload.config_name, overrides=[])

    _ensure_defaults(cfg)
    _validate_minimal_schema(cfg, strict=payload.strict)
    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    """Inject minimal defaults so downstream code doesn’t explode."""
    cfg.setdefault("data", {})
    cfg.setdefault("calib", {})
    cfg.setdefault("outputs", {})
    cfg.outputs.setdefault("dir", "data/processed")
    cfg.outputs.setdefault("format", "zarr")  # or parquet/npz
    cfg.setdefault("runtime", {})
    cfg.runtime.setdefault("num_workers", 4)
    cfg.runtime.setdefault("seed", 42)


def _validate_minimal_schema(cfg: DictConfig, *, strict: bool) -> None:
    """Minimal schema checks; raise in strict mode, warn otherwise."""
    missing: list[str] = []
    if "outputs" not in cfg or "dir" not in cfg.outputs:
        missing.append("outputs.dir")
    if "data" not in cfg:
        missing.append("data")
    if "calib" not in cfg:
        missing.append("calib")

    if missing:
        msg = f"[calibrate] Missing required config keys: {', '.join(missing)}"
        if strict:
            raise KeyError(msg)
        sys.stderr.write(msg + " (continuing)\n")


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    """Apply environment-aware guardrails (workers, etc.)."""
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))

    # Default to conservative workers on non-interactive envs
    cfg.setdefault("runtime", {})
    if "num_workers" not in cfg.runtime:
        cfg.runtime.num_workers = 2 if (is_kaggle or is_ci) else 4

    # Respect common CI/Kaggle env hints (no internet, small memory)
    if is_kaggle:
        cfg.runtime.setdefault("allow_internet", False)
        cfg.runtime.setdefault("low_mem_mode", True)
    if is_ci:
        cfg.runtime.setdefault("deterministic", True)


def _prepare_outputs_dir(repo_root: Path, cfg: DictConfig) -> Path:
    """Create outputs dir under repo root (DVC-friendly); return absolute path."""
    raw = str(cfg.outputs.dir)
    outputs_dir = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)

    # common subdirs for clarity / DVC stages (safe to create; DVC decides tracking)
    for sub in ("calib", "logs", "snapshots"):
        (outputs_dir / sub).mkdir(exist_ok=True)
    return outputs_dir


def _emit_config_snapshot(outputs_dir: Path, cfg: DictConfig) -> None:
    """Write resolved config & minimal run metadata for traceability."""
    try:
        # YAML snapshot (human-friendly)
        snap_yaml = outputs_dir / "snapshots" / "config_snapshot.yaml"
        OmegaConf.save(cfg, snap_yaml, resolve=True)

        # JSON run metadata (machine-friendly; e.g., for events table)
        meta = {
            "schema": "spectramind/config_snapshot@v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_root": str(_nearest_git_root(outputs_dir) or ""),
            "outputs_dir": str(outputs_dir),
            "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
            "runtime": dict(getattr(cfg, "runtime", {})),
        }
        meta_json = outputs_dir / "snapshots" / "config_snapshot.json"
        meta_json.write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[calibrate] warning: failed to write config snapshot: {e}\n")


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
