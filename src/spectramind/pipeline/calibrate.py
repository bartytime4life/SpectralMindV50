from __future__ import annotations

"""
SpectraMind V50 — Calibration Runner (Hydra + Orchestrator)
===========================================================

Key capabilities
----------------
• Hydra config composition with robust defaults
• Kaggle/CI guardrails (workers, determinism, low-mem)
• DVC-/CI-friendly output planning (calibration root)
• Config snapshot + artifact manifest + SHA256 digests + run_id
• Optional JSONL event logging (start/end) and run manifest
• Deterministic seeding for numpy/torch

Notes
-----
• Thin runner; the actual calibration logic lives in `spectramind.calibration.run(cfg)`.
• This module returns a dict payload (artifacts + metadata) for pipeline tooling.
"""

import hashlib
import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

# Optional SpectraMind logging backplane (manifest + JSONL events)
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
class CalibPayload:
    """Strongly-typed calibration invocation payload."""
    config_name: str
    overrides: list[str]
    out_dir: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


# ======================================================================================
# Public API
# ======================================================================================

def run(
    *,
    config_name: str = "calibrate",
    overrides: Iterable[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Orchestrate calibration: raw data + calib config → calibrated artifacts (tensors, cubes, json).

    Returns
    -------
    dict with shape:
      {
        "ok": bool,
        "run_id": str,
        "out_dir": str,
        "artifacts": {
          "primary": [paths...],
          "tensors": [paths...],
          "cubes": [paths...],
          "summaries": [paths...],
        }
      }
    """
    payload = CalibPayload(
        config_name=config_name,
        overrides=_normalize_overrides(overrides),
        out_dir=str(out_dir) if out_dir is not None else None,
        strict=strict,
        quiet=quiet,
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    # honor explicit out_dir if provided
    if payload.out_dir is not None:
        cfg.calib.out_dir = payload.out_dir

    # detect env if not provided & apply guardrails
    _auto_env_flags(payload.env)
    _apply_guardrails(cfg, payload.env)

    # deterministic seeding
    _seed_everything(int(cfg.runtime.seed))

    # normalize modules (accept comma-separated string or list[str])
    cfg.calib.modules = _normalize_modules(cfg.calib.modules)

    # prepare calibration root & snapshot config
    calib_root = _prepare_calib_root(repo_root, cfg)
    _emit_config_snapshot(calib_root, cfg)

    # Optional JSONL event logger (tie to out dir name)
    evt_logger: Optional[EventLogger] = None
    run_id = Path(cfg.calib.out_dir).name
    if _HAS_SM_LOGGING:
        evt_logger = EventLogger.for_run(stage="calibrate", run_id=run_id)
        evt_logger.info(
            "calibrate/start",
            data={
                "config_name": payload.config_name,
                "overrides": payload.overrides,
                "out_dir": str(calib_root),
                "modules": list(cfg.calib.modules),
                "env": payload.env,
                "num_workers": int(cfg.calib.num_workers),
                "deterministic": bool(cfg.runtime.deterministic),
                "low_mem_mode": bool(cfg.runtime.low_mem_mode),
                "seed": int(cfg.runtime.seed),
            },
        )

    # Defer heavy imports and provide clear errors
    try:
        # Expect orchestrator:
        #   spectramind.calibration.run(cfg=cfg) -> Optional[path | dict]
        from spectramind.calibration import run as run_calibration  # type: ignore[attr-defined]
    except Exception as e:  # pragma: no cover
        if evt_logger is not None:
            evt_logger.error("calibrate/error", message="import_failure", data={"cause": f"{type(e).__name__}: {e}"})
            evt_logger.close()
        raise RuntimeError(
            "Calibration entrypoint not found. Implement "
            "`spectramind.calibration.run(cfg)` or adjust this import. "
            f"Cause: {type(e).__name__}: {e}"
        ) from e

    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved calibration config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")
        sys.stderr.write(f"[calibrate] out_dir = {calib_root}\n")

    # ---- Run orchestrator ----
    result = run_calibration(cfg=cfg)

    # ---- Verify outputs (allow orchestrator to define primary output) ----
    outputs = _discover_outputs(calib_root, result)

    if not outputs.get("primary"):
        msg = f"[calibrate] no calibrated artifacts found in {calib_root}"
        if payload.strict:
            if evt_logger is not None:
                evt_logger.error("calibrate/error", message="no_outputs", data={"out_dir": str(calib_root)})
                evt_logger.close()
            raise RuntimeError(msg)
        if not payload.quiet:
            sys.stderr.write(msg + " (continuing)\n")

    # ---- Persist a manifest for CI/DVC provenance ----
    _write_calibration_manifest(calib_root, cfg, outputs)

    # ---- Optional richer run manifest ----
    if _HAS_SM_LOGGING:
        manifest = generate_manifest(
            stage="calibrate",
            config_snapshot=OmegaConf.to_container(cfg, resolve=True),  # type: ignore
            extra={
                "out_dir": str(calib_root),
                "modules": list(cfg.calib.modules),
                "outputs": outputs,
                "env": payload.env,
                "seed": int(cfg.runtime.seed),
            },
        )
        save_manifest(manifest, calib_root / "manifest.json")

    if evt_logger is not None:
        evt_logger.info("calibrate/end", data={"outputs": outputs})
        evt_logger.close()

    return {
        "ok": True,
        "run_id": run_id,
        "out_dir": str(calib_root),
        "artifacts": outputs,
    }


# ======================================================================================
# Hydra / Defaults / Guardrails
# ======================================================================================

def _normalize_overrides(ovr: Iterable[str] | None) -> list[str]:
    if not ovr:
        return []
    out: list[str] = []
    for s in ovr:
        s = str(s).strip()
        if s:
            out.append(s)
    return out


def _compose_hydra_config(repo_root: Path, payload: CalibPayload) -> DictConfig:
    """Compose Hydra config from repo `configs/` with robust defaults + strict mode."""
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per repo blueprint)"
        )

    # Use absolute-dir initializer (stable across CWDs and notebooks)
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
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
    cfg.setdefault("calib", {})
    cfg.calib.setdefault("out_dir", "outputs/calib")
    # Nominal pipeline per design: ADC→CDS→Dark→Flat→Photometry→Trace→Phase
    cfg.calib.setdefault("modules", ["adc", "cds", "dark", "flat", "photometry", "trace", "phase"])
    cfg.calib.setdefault("num_workers", 2)

    # Input roots (can be overridden in configs)
    cfg.calib.setdefault("input", {})
    cfg.calib.input.setdefault("raw_dir", "data/raw")
    cfg.calib.input.setdefault("metadata", None)

    # Outputs detail (optional)
    cfg.calib.setdefault("outputs", {})
    cfg.calib.outputs.setdefault("tensors_dir", "tensors")
    cfg.calib.outputs.setdefault("cubes_dir", "cubes")

    cfg.setdefault("runtime", {})
    cfg.runtime.setdefault("seed", 42)
    cfg.runtime.setdefault("deterministic", False)
    cfg.runtime.setdefault("low_mem_mode", False)


def _validate_minimal_schema(cfg: DictConfig, *, strict: bool) -> None:
    """Minimal schema checks; raise in strict mode, warn otherwise."""
    missing: list[str] = []
    if "calib" not in cfg:
        missing.append("calib")
    else:
        for k in ("out_dir", "modules", "num_workers"):
            if k not in cfg.calib:
                missing.append(f"calib.{k}")
        if "input" not in cfg.calib or "raw_dir" not in cfg.calib.input:
            missing.append("calib.input.raw_dir")

    if missing:
        msg = f"[calibrate] Missing required config keys: {', '.join(missing)}"
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
    """Apply environment-aware guardrails (workers, determinism, etc.)."""
    is_kaggle = bool(env.get("is_kaggle"))
    is_ci = bool(env.get("is_ci"))

    # Conservative workers on non-interactive envs
    if "num_workers" not in cfg.calib:
        cfg.calib.num_workers = 1 if (is_kaggle or is_ci) else 2
    else:
        if is_kaggle or is_ci:
            cfg.calib.num_workers = min(int(cfg.calib.num_workers), 2)

    # Determinism in CI; low-mem hint on Kaggle
    if is_ci:
        cfg.runtime.deterministic = True
    if is_kaggle:
        cfg.runtime.low_mem_mode = True


def _seed_everything(seed: int) -> None:
    """Best-effort deterministic seeding (numpy, torch if present)."""
    try:
        import numpy as _np
        _np.random.seed(seed)
    except Exception:
        pass
    try:
        import torch as _torch  # pragma: no cover
        _torch.manual_seed(seed)
        _torch.cuda.manual_seed_all(seed)
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


# ======================================================================================
# Planning, Snapshots, Discovery, Manifest
# ======================================================================================

def _prepare_calib_root(repo_root: Path, cfg: DictConfig) -> Path:
    """Create calibration root under repo root (DVC-friendly); return absolute path."""
    raw = str(cfg.calib.out_dir)
    calib_root = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    calib_root.mkdir(parents=True, exist_ok=True)

    # common subdirs (safe to create; DVC decides tracking)
    for sub in ("tensors", "cubes", "assets", "logs", "snapshots"):
        (calib_root / sub).mkdir(exist_ok=True)
    return calib_root


def _emit_config_snapshot(calib_root: Path, cfg: DictConfig) -> None:
    """Write resolved config & minimal run metadata for traceability."""
    try:
        snap_dir = calib_root / "snapshots"
        snap_dir.mkdir(exist_ok=True)

        # YAML snapshot (human-friendly)
        snap_yaml = snap_dir / "config_snapshot.yaml"
        OmegaConf.save(cfg, snap_yaml, resolve=True)

        # JSON run metadata (machine-friendly)
        meta = {
            "schema": "spectramind/calib_config_snapshot@v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_root": str(_nearest_git_root(calib_root) or ""),
            "out_dir": str(calib_root),
            "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
            "runtime": dict(getattr(cfg, "runtime", {})),
            "modules": list(getattr(cfg, "calib", {}).get("modules", [])),
            "input": dict(getattr(cfg, "calib", {}).get("input", {})),
        }
        (snap_dir / "config_snapshot.json").write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[calibrate] warning: failed to write config snapshot: {e}\n")


def _normalize_modules(mods: Any) -> List[str]:
    """Accept list[str] or comma-separated string; return normalized list[str]."""
    if mods is None:
        return []
    if isinstance(mods, str):
        parts = [p.strip() for p in mods.split(",") if p.strip()]
        return parts
    if isinstance(mods, Sequence):
        return [str(m).strip() for m in mods if str(m).strip()]
    return [str(mods).strip()]


def _discover_outputs(calib_root: Path, orchestrator_ret: Any) -> Dict[str, List[str]]:
    """
    Return discovered outputs:
      • primary: list[str] (main calibrated artifacts)
      • tensors: list[str]
      • cubes: list[str]
      • summaries: list[str]
    Orchestrator may return:
      - a path to a primary artifact
      - a dict of paths (e.g., {'primary': '...', 'summary': '...'})
    """
    outputs: Dict[str, List[str]] = {"primary": [], "tensors": [], "cubes": [], "summaries": []}

    # Honor orchestrator hints
    try:
        if orchestrator_ret:
            if isinstance(orchestrator_ret, (str, os.PathLike)):
                p = Path(str(orchestrator_ret)).resolve()
                if p.exists():
                    outputs["primary"].append(str(p))
            elif isinstance(orchestrator_ret, dict):
                for _, v in orchestrator_ret.items():
                    try:
                        p = Path(str(v)).resolve()
                        if p.exists():
                            if p.is_file() and p.suffix.lower() == ".json":
                                outputs["summaries"].append(str(p))
                            elif p.is_dir():
                                # classify by parent name
                                sp = str(p).lower()
                                if "tensor" in sp:
                                    outputs["tensors"].append(str(p))
                                elif "cube" in sp:
                                    outputs["cubes"].append(str(p))
                                else:
                                    outputs["primary"].append(str(p))
                            else:
                                outputs["primary"].append(str(p))
                    except Exception:
                        continue
    except Exception:
        pass

    # Scan common locations
    for d in [calib_root, calib_root / "tensors", calib_root / "cubes"]:
        if d.exists():
            for p in d.rglob("*"):
                if p.is_file() and p.suffix.lower() in {".npz", ".npy", ".pt", ".pth", ".parquet"}:
                    parent = str(p.parent).lower()
                    if "tensors" in parent:
                        outputs["tensors"].append(str(p.resolve()))
                    elif "cubes" in parent:
                        outputs["cubes"].append(str(p.resolve()))
                    else:
                        outputs["primary"].append(str(p.resolve()))
                if p.is_file() and p.suffix.lower() == ".json":
                    outputs["summaries"].append(str(p.resolve()))

    # dedupe
    for k in outputs:
        outputs[k] = sorted(set(outputs[k]))
    return outputs


def _write_calibration_manifest(calib_root: Path, cfg: DictConfig, outputs: Dict[str, List[str]]) -> None:
    """Write a manifest (CI/DVC-friendly) with discovered outputs and sha256 digests."""
    def _digests(items: List[str]) -> List[Dict[str, str]]:
        out: List[Dict[str, str]] = []
        for s in items:
            try:
                sha = _sha256_of_file(Path(s))
            except Exception:
                sha = ""
            out.append({"path": s, "sha256": sha})
        return out

    manifest = {
        "schema": "spectramind/calib_manifest@v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "out_dir": str(calib_root),
        "modules": list(getattr(cfg, "calib", {}).get("modules", [])),
        "outputs": {
            "primary": _digests(outputs.get("primary", [])),
            "tensors": _digests(outputs.get("tensors", [])),
            "cubes": _digests(outputs.get("cubes", [])),
            "summaries": _digests(outputs.get("summaries", [])),
        },
        "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
    }
    (calib_root / "manifest.json").write_text(json.dumps(manifest, indent=2))


# ======================================================================================
# Common helpers
# ======================================================================================

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
    # Fallback: best-effort two levels up (keeps notebooks/colab from exploding)
    return here.parents[2]


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))


def _sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    try:
        with path.open("rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
    except Exception:
        return ""
    return h.hexdigest()
