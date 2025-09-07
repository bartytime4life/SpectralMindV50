# src/spectramind/pipeline/calibrate.py
from __future__ import annotations

import json
import os
import random
import sys
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize_config_dir
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
    dry_run: bool
    env: Dict[str, Any]


# ------------------------------- Public API -----------------------------------


def run(
    *,
    config_name: str = "calibrate",
    overrides: Iterable[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    strict: bool = True,
    quiet: bool = False,
    dry_run: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Calibration runner: raw inputs → calibrated cubes/tensors.

    Responsibilities
    ---------------
    • Compose Hydra config (data + calib + outputs)
    • Apply environment-aware guardrails (Kaggle, CI)
    • Initialize deterministic seeds
    • Emit lineage/config snapshots + events (JSONL)
    • Dispatch to internal calibration orchestrator
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
    dry_run
        If True, do not execute the orchestrator; still composes config and emits snapshots.
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
        dry_run=dry_run,
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    config_dir = _locate_config_dir(repo_root)
    cfg = _compose_hydra_config(config_dir, payload)

    # honor explicit output dir if provided
    if payload.out_dir is not None:
        cfg.outputs.dir = payload.out_dir

    # environment guardrails & seeds
    _apply_guardrails(cfg, payload.env)
    _set_deterministic_seeds(cfg.runtime.get("seed", 42))

    # Prepare outputs dir (DVC-friendly) + snapshot config for traceability
    outputs_dir = _prepare_outputs_dir(repo_root, cfg)
    _emit_config_snapshot(outputs_dir, cfg)

    # Events logger
    events_path = outputs_dir / "logs" / "events.jsonl"
    _emit_event(events_path, "calibration_start", {"payload": asdict(payload)})

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved calibration config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")
        sys.stderr.write(f"[calibrate] outputs.dir = {outputs_dir}\n")

    if payload.dry_run:
        _emit_event(events_path, "calibration_skip", {"reason": "dry_run"})
        return

    # Defer heavy imports and provide clear errors
    try:
        # Expect: `spectramind.calib.pipeline.run(cfg=cfg)`
        from spectramind.calib.pipeline import run as run_calibration  # type: ignore
    except Exception as e:  # pragma: no cover
        _emit_event(events_path, "calibration_error", {"error": f"{type(e).__name__}: {e}"})
        raise RuntimeError(
            "Calibration entrypoint not found. Implement "
            "`spectramind.calib.pipeline.run(cfg)` or adjust this import. "
            f"Cause: {type(e).__name__}: {e}"
        ) from e

    # Optional minimal JSON Schema validation (best-effort)
    _validate_minimal_schema_jsonschema(cfg, strict=payload.strict)

    try:
        run_calibration(cfg=cfg)
        _emit_event(events_path, "calibration_end", {"status": "ok"})
    except Exception as e:
        _emit_event(events_path, "calibration_error", {"error": f"{type(e).__name__}: {e}"})
        raise


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(config_dir: Path, payload: CalibPayload) -> DictConfig:
    """Compose Hydra config from absolute `config_dir` with robust defaults + strict mode."""
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per repo blueprint)"
        )

    # Initialize with explicit config dir (works in Kaggle datasets and repo checkouts)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
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
    cfg.runtime.setdefault("deterministic", False)
    cfg.runtime.setdefault("allow_internet", False)
    cfg.runtime.setdefault("low_mem_mode", False)


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


def _validate_minimal_schema_jsonschema(cfg: DictConfig, *, strict: bool = False) -> None:
    """
    Optional JSON Schema validation if `jsonschema` is available.
    Validates minimal 'outputs.dir', 'data', 'calib' contract.
    """
    try:
        import jsonschema  # type: ignore
    except Exception:
        return  # Not installed; silently skip

    schema = {
        "type": "object",
        "properties": {
            "outputs": {
                "type": "object",
                "properties": {"dir": {"type": "string"}},
                "required": ["dir"],
            },
            "data": {"type": "object"},
            "calib": {"type": "object"},
        },
        "required": ["outputs", "data", "calib"],
    }
    try:
        jsonschema.validate(OmegaConf.to_container(cfg, resolve=True), schema)
    except Exception as e:
        if strict:
            raise
        sys.stderr.write(f"[calibrate] schema warning: {e}\n")


def _apply_guardrails(cfg: DictConfig, env: Dict[str, Any]) -> None:
    """Apply environment-aware guardrails (workers, determinism, etc.)."""
    # Best-effort inference if not provided
    inferred_kaggle = os.environ.get("KAGGLE_KERNEL_RUN_TYPE") is not None
    is_kaggle = bool(env.get("is_kaggle", inferred_kaggle))
    is_ci = bool(env.get("is_ci", os.environ.get("CI")))

    cfg.setdefault("runtime", {})
    # Workers
    if "num_workers" not in cfg.runtime:
        cfg.runtime.num_workers = 2 if (is_kaggle or is_ci) else 4
    # Environment policies
    if is_kaggle:
        cfg.runtime.setdefault("allow_internet", False)
        cfg.runtime.setdefault("low_mem_mode", True)
    if is_ci:
        cfg.runtime.setdefault("deterministic", True)


def _set_deterministic_seeds(seed: int) -> None:
    """Set seeds across stdlib/random, numpy and torch if available."""
    try:
        random.seed(seed)
    except Exception:
        pass
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch  # type: ignore
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # extra determinism (may slow down)
        torch.backends.cudnn.deterministic = True  # type: ignore
        torch.backends.cudnn.benchmark = False  # type: ignore
    except Exception:
        pass


def _prepare_outputs_dir(repo_root: Path, cfg: DictConfig) -> Path:
    """Create outputs dir under repo root (DVC-friendly); return absolute path."""
    raw = str(cfg.outputs.dir)
    outputs_dir = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    outputs_dir.mkdir(parents=True, exist_ok=True)
    # Common subdirs for clarity / DVC stages
    for sub in ("calib", "logs", "snapshots"):
        (outputs_dir / sub).mkdir(exist_ok=True)
    return outputs_dir


def _emit_config_snapshot(outputs_dir: Path, cfg: DictConfig) -> None:
    """Write resolved config & minimal run metadata for traceability."""
    try:
        # YAML snapshot (human-friendly)
        snap_yaml = outputs_dir / "snapshots" / "config_snapshot.yaml"
        OmegaConf.save(cfg, snap_yaml, resolve=True)

        # JSON run metadata (machine-friendly)
        meta = {
            "schema": "spectramind/config_snapshot@v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_root": str(_nearest_git_root(outputs_dir) or ""),
            "outputs_dir": str(outputs_dir),
            "hydra_overrides": list(_extract_hydra_overrides(cfg)),
            "runtime": dict(getattr(cfg, "runtime", {})),
        }
        meta_json = outputs_dir / "snapshots" / "config_snapshot.json"
        meta_json.write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[calibrate] warning: failed to write config snapshot: {e}\n")


def _extract_hydra_overrides(cfg: DictConfig) -> Iterable[str]:
    """Extract best-effort hydra overrides from composed cfg (if present)."""
    try:
        # hydra might be embedded at cfg.hydra; this is best-effort only
        hydra_node = getattr(cfg, "hydra", None)
        if hydra_node and hasattr(hydra_node, "overrides"):
            return list(hydra_node.overrides.task)
    except Exception:
        pass
    return []


def _emit_event(path: Path, kind: str, payload: Dict[str, Any]) -> None:
    """Append a structured event JSON line to the given file."""
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "kind": kind,
            "payload": payload,
        }
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[calibrate] warn: failed to write event: {e}\n")


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
      • (fallback) assume parent of 'src' is project root
    """
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            return p if p.name != "src" else p.parent
    # Fallback: avoid exploding in Kaggle/colab
    return here.parents[3]


def _locate_config_dir(repo_root: Path) -> Path:
    """
    Locate the absolute Hydra config directory.

    Allows shipping code as a Kaggle dataset (config lives at <root>/configs).
    """
    cand = repo_root / "configs"
    if cand.exists():
        return cand
    # As a last resort, look up one level (some datasets nest the project)
    up = repo_root.parent / "configs"
    if up.exists():
        return up
    raise FileNotFoundError(f"Missing Hydra config directory under {repo_root} or {repo_root.parent}")


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))
