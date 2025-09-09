from __future__ import annotations

"""
SpectraMind V50 — Pipeline Entrypoints
======================================

This package exposes clean, lazy-loaded accessors for each pipeline stage's
`run(**kwargs)` function, plus a tiny router.

Stages
------
- calibrate  → sensor calibration (FGS1 + AIRS)
- train      → model training (dual encoders + decoder)
- predict    → inference & spectral μ/σ outputs
- diagnostics→ reporting & evaluation
- submit     → package validated submission bundle (csv/zip/manifest)

Usage
-----
>>> from spectramind.pipeline import train, predict, submit
>>> train(config_name="train", overrides=["+env=local"])
>>> predict(checkpoint="artifacts/ckpts/last.ckpt")
>>> submit(predictions="outputs/predictions/predictions.csv")

Advanced
--------
>>> from spectramind.pipeline import run_stage, available_stages
>>> run_stage("predict", checkpoint="...", strict=True)
>>> available_stages()
['calibrate', 'diagnostics', 'predict', 'submit', 'train']

Design notes
------------
- We *lazy import* stage modules to keep CLI startup snappy and Kaggle/CI-safe.
- We expose a `__getattr__` per PEP 562 to resolve attributes like `train`
  to the underlying `run` function at first touch.
- No heavy deps at import-time; stages can import torch, lightning, etc.
- Stages can be overridden via code or env: SPECTRAMIND_PIPELINE_STAGES='{"train":"pkg.mod"}'
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
import json
import os
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

__all__ = [
    # stage callables (lazy; see __getattr__)
    "train",
    "calibrate",
    "predict",
    "diagnostics",
    "submit",
    # helpers
    "run_stage",
    "available_stages",
    "get_version",
    "get_stage_help",
    "override_stage_modules",
    "reset_stage_cache",
]

# -----------------------------------------------------------------------------#
# Internal registry and lazy resolver
# -----------------------------------------------------------------------------#


class StageResolutionError(AttributeError):
    """Raised when a pipeline stage cannot be resolved or lacks a callable 'run'."""


# base stage name -> module path (may be overridden via env or API)
_STAGE_MODULES: Dict[str, str] = {
    "calibrate": "spectramind.pipeline.calibrate",
    "train": "spectramind.pipeline.train",
    "predict": "spectramind.pipeline.predict",
    "diagnostics": "spectramind.pipeline.diagnostics",
    "submit": "spectramind.pipeline.submit",
}

# Load overrides from env (JSON mapping), if present
_env_override = os.getenv("SPECTRAMIND_PIPELINE_STAGES")
if _env_override:
    try:
        mapping = json.loads(_env_override)
        if isinstance(mapping, dict):
            # only accept known stages and string module paths
            for k, v in mapping.items():
                if k in _STAGE_MODULES and isinstance(v, str) and v:
                    _STAGE_MODULES[k] = v
    except Exception:
        # ignore invalid JSON; keep defaults
        pass

# Cache resolved callables after first import to avoid repeated import overhead.
_RESOLVED: MutableMapping[str, Callable[..., Any]] = {}


def _resolve(stage: str) -> Callable[..., Any]:
    """
    Resolve a stage name to its `run` callable, importing lazily.
    Raises StageResolutionError with a clear message on failure.
    """
    if stage in _RESOLVED:
        return _RESOLVED[stage]

    if stage not in _STAGE_MODULES:
        raise StageResolutionError(
            f"Unknown pipeline stage '{stage}'. Known: {sorted(_STAGE_MODULES)}"
        )

    mod_path = _STAGE_MODULES[stage]
    try:
        mod = import_module(mod_path)
    except Exception as e:  # pragma: no cover
        raise StageResolutionError(
            f"Failed to import module for stage '{stage}': '{mod_path}' ({type(e).__name__}: {e})"
        ) from e

    fn = getattr(mod, "run", None)
    if not callable(fn):
        raise StageResolutionError(
            f"Module '{mod_path}' does not export a callable 'run' for stage '{stage}'"
        )

    _RESOLVED[stage] = fn  # cache
    return fn


# -----------------------------------------------------------------------------#
# Public helpers
# -----------------------------------------------------------------------------#

def available_stages() -> List[str]:
    """Return a sorted list of available stage names."""
    return sorted(_STAGE_MODULES.keys())


def run_stage(stage: str, /, *args: Any, **kwargs: Any) -> Any:
    """
    Dispatch to a stage's `run` with positional/keyword passthrough.

    Example:
        run_stage("predict", checkpoint="...", strict=True)
    """
    fn = _resolve(stage)
    return fn(*args, **kwargs)


def get_stage_help(stage: str) -> Optional[str]:
    """
    Return a short help string for a stage if its module defines `HELP` or a module docstring.

    Example:
        >>> get_stage_help("train")
        'Train model; writes checkpoints and logs'
    """
    if stage not in _STAGE_MODULES:
        return None
    mod_path = _STAGE_MODULES[stage]
    try:
        mod = import_module(mod_path)
    except Exception:  # pragma: no cover
        return None
    help_text = getattr(mod, "HELP", None)  # convention: module-level HELP string
    if isinstance(help_text, str):
        return help_text.strip()
    if isinstance(mod.__doc__, str):
        return mod.__doc__.strip().splitlines()[0] if mod.__doc__ else None
    return None


def get_version(default: str = "0.0.0") -> str:
    """
    Return the installed package version if available (pip/PEP 621 metadata),
    else fall back to reading a VERSION file (repo root) or `default`.
    """
    # Try package metadata
    for pkg in ("spectramind-v50", "spectramind"):
        try:
            return _pkg_version(pkg)
        except PackageNotFoundError:
            pass

    # Fallback: VERSION file in repo (…/src/spectramind/pipeline/__init__.py → repo)
    try:
        repo_root = Path(__file__).resolve().parents[3]
        ver_file = repo_root / "VERSION"
        if ver_file.exists():
            return ver_file.read_text(encoding="utf-8").strip()
    except Exception:
        pass

    return default


def override_stage_modules(mapping: Mapping[str, str]) -> None:
    """
    Programmatically override stage → module resolution. Useful in tests or custom deployments.
    Only keys in `available_stages()` are applied; others ignored.
    """
    for k, v in mapping.items():
        if k in _STAGE_MODULES and isinstance(v, str) and v:
            _STAGE_MODULES[k] = v
    # Clear cache so next resolution imports fresh targets
    _RESOLVED.clear()


def reset_stage_cache() -> None:
    """Clear the resolver cache (e.g., after overriding stage modules)."""
    _RESOLVED.clear()


# -----------------------------------------------------------------------------#
# PEP 562: Module-level attribute access for lazy stage functions
# -----------------------------------------------------------------------------#

def __getattr__(name: str) -> Any:
    """
    Allow `from spectramind.pipeline import train` to resolve lazily.

    Returns the stage `run` callable when `name` matches a stage.
    """
    if name in _STAGE_MODULES:
        return _resolve(name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> List[str]:
    """So that dir(spectramind.pipeline) lists stage names and exports."""
    return sorted(set(globals().keys()) | set(_STAGE_MODULES.keys()))
