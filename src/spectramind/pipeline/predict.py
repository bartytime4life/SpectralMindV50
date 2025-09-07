# src/spectramind/pipeline/predict.py
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
class PredictPayload:
    """Strongly-typed prediction invocation payload."""
    config_name: str
    overrides: list[str]
    checkpoint: Optional[str]
    out_path: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


# --------------------------------- API ----------------------------------------


def run(
    *,
    config_name: str = "predict",
    overrides: Iterable[str] | None = None,
    checkpoint: str | None = None,
    out_path: str | os.PathLike[str] | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Inference runner: checkpoint + test data → predictions CSV/Parquet.

    Responsibilities
    ---------------
    • Compose Hydra config (predict + data + model refs)
    • Dispatch to internal inference pipeline
    • Write artifacts in a DVC-friendly structure

    Parameters
    ----------
    config_name
        Name of the top-level Hydra config under `configs/` (default: "predict").
    overrides
        Hydra override strings, e.g. `["env=kaggle", "predict.batch_size=32"]`.
    checkpoint
        Optional explicit checkpoint path; overrides `cfg.predict.checkpoint`.
    out_path
        Optional output file path (CSV/Parquet). Overrides `cfg.predict.out_path`.
    strict
        If True, propagate composition/validation errors; else warn and continue with defaults.
    quiet
        Suppress resolved config echoing to stderr.
    env
        Optional execution context, e.g. {"is_kaggle": True, "is_ci": False}.

    Raises
    ------
    FileNotFoundError
        When `configs/` cannot be found at repo root.
    RuntimeError
        When the internal inference orchestrator is missing.
    """
    payload = PredictPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        checkpoint=checkpoint,
        out_path=str(out_path) if out_path is not None else None,
        strict=strict,
        quiet=quiet,
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    # honor explicit CLI overrides
    if payload.checkpoint is not None:
        cfg.predict.checkpoint = payload.checkpoint
    if payload.out_path is not None:
        cfg.predict.out_path = payload.out_path

    _apply_guardrails(cfg, payload.env)
    _validate_minimal_schema(cfg, strict=payload.strict)

    # Prepare output destination (DVC-friendly) + snapshot config for traceability
    out_file = _prepare_output_path(repo_root, cfg)
    _emit_config_snapshot(out_file, cfg)

    try:
        # Expect orchestrator:
        # `spectramind.inference.predict.run(cfg=cfg)`
        from spectramind.inference.predict import run as run_inference  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "Inference entrypoint not found. Implement "
            "`spectramind.inference.predict.run(cfg)` or adjust this import. "
            f"Cause: {type(e).__name__}: {e}"
        ) from e

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved prediction config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")
        sys.stderr.write(f"[predict] out_path = {out_file}\n")

    run_inference(cfg=cfg)


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: PredictPayload) -> DictConfig:
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
            # Best-effort: attempt to compose base config with no overrides
            cfg = compose(config_name=payload.config_name, overrides=[])

    _ensure_defaults(cfg)
    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    """Inject minimal defaults so downstream code doesn’t explode."""
    cfg.setdefault("predict", {})
    cfg.predict.setdefault("batch_size", 64)
    cfg.predict.setdefault("num_workers", 4)
    cfg.predict.setdefault("checkpoint", None)
    # DVC-/Kaggle-friendly default under repo
    cfg.predict.setdefault("out_path", "outputs/predictions/predictions.csv")
    cfg.setdefault("runtime", {})
    cfg.runtime.setdefault("seed", 42)


def _validate_minimal_schema(cfg: DictConfig, *, strict: bool) -> None:
    """Minimal schema checks; raise in strict mode, warn otherwise."""
    missing: list[str] = []
    if "predict" not in cfg:
        missing.append("predict")
    else:
        if "out_path" not in cfg.predict:
            missing.append("predict.out_path")
        if "batch_size" not in cfg.predict:
            missing.append("predict.batch_size")
        if not cfg.predict.get("checkpoint"):
            missing.append("predict.checkpoint")

    if missing:
        msg = f"[predict] Missing required config keys: {', '.join(missing)}"
        if strict:
            raise KeyError(msg)
        sys.stderr.write(msg + " (continuing)\n")


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    """Apply environment-aware guardrails (workers, batch-size, determinism)."""
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))

    # Conservative workers on non-interactive envs
    if "num_workers" not in cfg.predict:
        cfg.predict.num_workers = 2 if (is_kaggle or is_ci) else 4

    # Soften batch-size in low-mem environments
    if is_kaggle or is_ci:
        bs = int(cfg.predict.get("batch_size", 64))
        cfg.predict.batch_size = min(bs, 32)

    cfg.runtime.setdefault("deterministic", bool(is_ci))
    if is_kaggle:
        cfg.runtime.setdefault("low_mem_mode", True)


def _prepare_output_path(repo_root: Path, cfg: DictConfig) -> Path:
    """
    Ensure parent dirs exist for output file under repo root (DVC-friendly).
    Returns absolute path to target file.
    """
    raw = str(cfg.predict.out_path)
    out_file = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # also prepare sibling structure for cleanliness
    for sub in ("logs", "snapshots"):
        (out_file.parent / sub).mkdir(exist_ok=True)

    # normalize extension defaults (.csv or .parquet)
    if out_file.suffix.lower() not in {".csv", ".parquet", ".pq"}:
        # default to CSV if not explicitly set
        out_file = out_file.with_suffix(".csv")
        cfg.predict.out_path = str(out_file)
    return out_file


def _emit_config_snapshot(out_file: Path, cfg: DictConfig) -> None:
    """Write resolved config & minimal run metadata for traceability."""
    try:
        snap_dir = out_file.parent / "snapshots"
        snap_dir.mkdir(exist_ok=True)

        # YAML snapshot (human-friendly)
        snap_yaml = snap_dir / "config_snapshot.yaml"
        OmegaConf.save(cfg, snap_yaml, resolve=True)

        # JSON run metadata (machine-friendly)
        meta = {
            "schema": "spectramind/predict_config_snapshot@v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_root": str(_nearest_git_root(out_file.parent) or ""),
            "out_path": str(out_file),
            "checkpoint": cfg.predict.get("checkpoint"),
            "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
            "runtime": dict(getattr(cfg, "runtime", {})),
            "batch_size": int(cfg.predict.get("batch_size", 64)),
            "num_workers": int(cfg.predict.get("num_workers", 4)),
        }
        meta_json = snap_dir / "config_snapshot.json"
        meta_json.write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[predict] warning: failed to write config snapshot: {e}\n")


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
            if p.name == "src":
                return p.parent
            return p
    # Fallback: best-effort three levels up (keeps Kaggle/colab from exploding)
    return here.parents[3]


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))
