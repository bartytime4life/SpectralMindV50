# src/spectramind/pipeline/__init__.py
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
['calibrate', 'train', 'predict', 'diagnostics', 'submit']

Design notes
------------
- We *lazy import* stage modules to keep CLI startup snappy and Kaggle/CI-safe.
- We expose a `__getattr__` per PEP 562 to resolve attributes like `train`
  to the underlying `run` function at first touch.
- No heavy deps at import-time; stages can import torch, lightning, etc.
"""

from importlib import import_module
from importlib.metadata import PackageNotFoundError, version as _pkg_version
from typing import Any, Callable, Dict, Iterable, List, Mapping, MutableMapping, Optional

__all__ = [
    "train",
    "calibrate",
    "predict",
    "diagnostics",
    "submit",
    "run_stage",
    "available_stages",
    "get_version",
]

# -----------------------------------------------------------------------------#
# Internal registry and lazy resolver
# -----------------------------------------------------------------------------#

# stage name -> module path
_STAGE_MODULES: Mapping[str, str] = {
    "calibrate": "spectramind.pipeline.calibrate",
    "train": "spectramind.pipeline.train",
    "predict": "spectramind.pipeline.predict",
    "diagnostics": "spectramind.pipeline.diagnostics",
    "submit": "spectramind.pipeline.submit",
}

# Cache resolved callables after first import to avoid repeated import overhead.
_RESOLVED: MutableMapping[str, Callable[..., Any]] = {}


def _resolve(stage: str) -> Callable[..., Any]:
    """Resolve a stage name to its `run` callable, importing lazily."""
    if stage in _RESOLVED:
        return _RESOLVED[stage]
    if stage not in _STAGE_MODULES:
        raise AttributeError(f"Unknown pipeline stage '{stage}'. "
                             f"Known: {sorted(_STAGE_MODULES)}")
    mod_path = _STAGE_MODULES[stage]
    mod = import_module(mod_path)
    if not hasattr(mod, "run") or not callable(getattr(mod, "run")):
        raise AttributeError(f"Module '{mod_path}' does not export a callable 'run'")
    _RESOLVED[stage] = getattr(mod, "run")  # type: ignore[assignment]
    return _RESOLVED[stage]


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


def get_version(default: str = "0.0.0") -> str:
    """
    Return the installed package version if available (pip/PEP 621 metadata),
    else fall back to `default`.
    """
    try:
        return _pkg_version("spectramind-v50")
    except PackageNotFoundError:
        return default


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
