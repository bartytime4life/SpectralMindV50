# src/spectramind/pipeline/predict.py
from __future__ import annotations

"""
SpectraMind V50 — Prediction Runner (Hydra + Inference Orchestrator)
====================================================================

Key capabilities
----------------
• Hydra config composition with robust defaults
• Kaggle/CI guardrails (workers, batch-size, determinism, low-mem)
• Checkpoint & output path resolution (DVC-/CI-friendly)
• Config snapshot + artifact manifest + SHA256
• Optional JSONL event logging (start/end) and run manifest

Notes
-----
• CLI stays thin; this module holds the orchestration/business logic.
• The actual inference is delegated to `spectramind.inference.predict.run(cfg)`.
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# SpectraMind optional logging backplane (manifest + JSONL events)
try:
    from spectramind.logging.manifest import generate_manifest, save_manifest  # type: ignore
    from spectramind.logging.event_logger import EventLogger  # type: ignore

    _HAS_SM_LOGGING = True
except Exception:  # pragma: no cover
    _HAS_SM_LOGGING = False

__all__ = ["run"]


# ======================================================================================
# Data model
# ======================================================================================


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


# ======================================================================================
# Public API
# ======================================================================================


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
    Inference runner: checkpoint + test data → predictions (CSV/Parquet).

    Responsibilities
    ----------------
    • Compose Hydra config (predict + data + model refs)
    • Resolve/validate checkpoint and output destination
    • Apply Kaggle/CI guardrails (workers, batch size, deterministic flags)
    • Emit config snapshot & call internal orchestrator
    • Verify output was written; persist inference manifest & JSONL events
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

    # Honor explicit CLI overrides
    if payload.checkpoint is not None:
        cfg.predict.checkpoint = payload.checkpoint
    if payload.out_path is not None:
        cfg.predict.out_path = payload.out_path

    # Detect CI/Kaggle if not explicitly set
    _auto_env_flags(payload.env)

    # Apply guardrails & check minimal schema
    _apply_guardrails(cfg, payload.env)
    _validate_minimal_schema(cfg, strict=payload.strict)

    # Resolve paths & compose traceability
    ckpt_file = _resolve_checkpoint(repo_root, cfg, strict=payload.strict)
    out_file = _prepare_output_path(repo_root, cfg)

    # Config snapshot (human+machine)
    _emit_config_snapshot(out_file, cfg)

    # Optional JSONL event logger (linked to output dir/version)
    evt_logger: Optional[EventLogger] = None
    run_id = out_file.stem
    if _HAS_SM_LOGGING:
        evt_logger = EventLogger.for_run(stage="predict", run_id=run_id)
        evt_logger.info(
            "predict/start",
            data={
                "config_name": payload.config_name,
                "overrides": payload.overrides,
                "checkpoint": str(ckpt_file),
                "out_path": str(out_file),
                "env": payload.env,
                "batch_size": int(cfg.predict.batch_size),
                "num_workers": int(cfg.predict.num_workers),
                "deterministic": bool(cfg.runtime.deterministic),
                "low_mem_mode": bool(cfg.runtime.low_mem_mode),
            },
        )

    # Dispatch to user orchestrator
    try:
        # Expected orchestrator: spectramind.inference.predict.run(cfg=cfg)
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
        sys.stderr.write(f"[predict] checkpoint = {ckpt_file}\n")
        sys.stderr.write(f"[predict] out_path   = {out_file}\n")

    # Orchestrator may optionally return the final artifact path (string/Path)
    result = run_inference(cfg=cfg)
    if result:
        try:
            candidate = Path(str(result)).resolve()
            if candidate.is_file():
                out_file = candidate  # prefer orchestrator's path if present
        except Exception:
            pass

    # Verify output exists
    if not out_file.exists():
        if evt_logger is not None:
            evt_logger.error("predict/error", message="output_missing", data={"expected_path": str(out_file)})
            evt_logger.close()
        raise RuntimeError(f"[predict] orchestrator did not produce output: {out_file}")

    # Persist an inference manifest (DVC/CI-friendly)
    _write_inference_manifest(out_file, cfg)

    # Optional: save a richer run manifest
    if _HAS_SM_LOGGING:
        manifest = generate_manifest(
            stage="predict",
            config_snapshot=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            extra={
                "predictions_path": str(out_file),
                "predictions_sha256": _sha256_of_file(out_file),
                "checkpoint": str(cfg.predict.get("checkpoint") or ""),
                "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
                "env": payload.env,
            },
        )
        save_manifest(manifest, out_file.parent / "manifest.json")

    if evt_logger is not None:
        evt_logger.info(
            "predict/end",
            data={
                "predictions_path": str(out_file),
                "predictions_sha256": _sha256_of_file(out_file),
            },
        )
        evt_logger.close()


# ======================================================================================
# Hydra / Defaults / Guardrails
# ======================================================================================


def _compose_hydra_config(repo_root: Path, payload: PredictPayload) -> DictConfig:
    """Compose Hydra config from repo `configs/` with robust defaults + strict mode."""
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per repo blueprint)"
        )

    # Prefer absolute-dir initializer (stable CWD)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
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

    # runtime toggles (determinism & low-memory hints)
    cfg.setdefault("runtime", {})
    cfg.runtime.setdefault("seed", 42)
    cfg.runtime.setdefault("deterministic", False)
    cfg.runtime.setdefault("low_mem_mode", False)


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


def _auto_env_flags(env: Dict[str, Any]) -> None:
    """Detect Kaggle/CI if not given."""
    if "is_kaggle" not in env:
        env["is_kaggle"] = bool(os.environ.get("KAGGLE_URL_BASE") or os.environ.get("KAGGLE_KERNEL_INTEGRATIONS"))
    if "is_ci" not in env:
        env["is_ci"] = bool(os.environ.get("GITHUB_ACTIONS") or os.environ.get("CI"))


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    """Apply environment-aware guardrails (workers, batch-size, determinism)."""
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))

    # Conservative workers on non-interactive envs
    if "num_workers" not in cfg.predict:
        cfg.predict.num_workers = 2 if (is_kaggle or is_ci) else 4
    else:
        if is_kaggle or is_ci:
            cfg.predict.num_workers = min(int(cfg.predict.num_workers), 2)

    # Soften batch-size in low-mem environments
    if is_kaggle or is_ci:
        bs = int(cfg.predict.get("batch_size", 64))
        cfg.predict.batch_size = min(bs, 32)

    # Determinism by default for CI; low mem hint on Kaggle
    if is_ci:
        cfg.runtime.deterministic = True
    if is_kaggle:
        cfg.runtime.low_mem_mode = True


# ======================================================================================
# Path resolution / Snapshots / Manifests
# ======================================================================================


def _resolve_checkpoint(repo_root: Path, cfg: DictConfig, *, strict: bool) -> Path:
    raw = str(cfg.predict.checkpoint or "")
    if not raw:
        raise KeyError("[predict] checkpoint path is empty")
    ckpt = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    if not ckpt.exists():
        msg = f"[predict] checkpoint not found: {ckpt}"
        if strict:
            raise FileNotFoundError(msg)
        sys.stderr.write(msg + " (continuing — orchestrator may handle)\n")
    return ckpt


def _prepare_output_path(repo_root: Path, cfg: DictConfig) -> Path:
    """
    Ensure parent dirs exist for output file under repo root (DVC-friendly).
    Returns absolute path to target file and normalizes extension (.csv/.parquet).
    """
    raw = str(cfg.predict.out_path)
    out_file = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    out_file.parent.mkdir(parents=True, exist_ok=True)

    # also prepare sibling structure for cleanliness
    for sub in ("logs", "snapshots"):
        (out_file.parent / sub).mkdir(exist_ok=True)

    # normalize extension defaults (.csv or .parquet)
    if out_file.suffix.lower() not in {".csv", ".parquet", ".pq"}:
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
        (snap_dir / "config_snapshot.json").write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[predict] warning: failed to write config snapshot: {e}\n")


def _write_inference_manifest(out_file: Path, cfg: DictConfig) -> None:
    """Write a small manifest alongside predictions describing the artifact and inputs."""
    try:
        sha256 = _sha256_of_file(out_file)
    except Exception:
        sha256 = ""
    manifest = {
        "schema": "spectramind/predict_manifest@v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions_path": str(out_file),
        "predictions_sha256": sha256,
        "checkpoint": str(cfg.predict.get("checkpoint") or ""),
        "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
    }
    (out_file.parent / "manifest.json").write_text(json.dumps(manifest, indent=2))


def _sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


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
    # Fallback: best-effort two levels up (keeps notebooks happy)
    return here.parents[2]


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))
