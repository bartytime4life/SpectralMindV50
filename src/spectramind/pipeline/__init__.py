# src/spectramind/pipeline/__init__.py
from __future__ import annotations

"""
SpectraMind V50 — Pipeline Entrypoints
======================================

This module re-exports the `run` entrypoints of each pipeline stage so that
users and internal code can access them directly:

    >>> from spectramind.pipeline import train, calibrate, predict, diagnostics, submit
    >>> train(config_name="train")

Each stage is defined in its own module under `src/spectramind/pipeline/`
and implements a `run(**kwargs)` function.

Stages:
--------
- train         → model training (dual encoders + decoder)
- calibrate     → sensor calibration (FGS1 + AIRS)
- predict       → inference & spectral μ/σ outputs
- diagnostics   → reporting & evaluation
- submit        → Kaggle packaging & validation

This keeps the CLI (`spectramind ...`) and Python API (`python -m spectramind.pipeline`) consistent.
"""

# Explicit re-exports (import-time safety: modules must define `run`)
from .train import run as train            # noqa: F401
from .calibrate import run as calibrate    # noqa: F401
from .predict import run as predict        # noqa: F401
from .diagnostics import run as diagnostics  # noqa: F401
from .submit import run as submit          # noqa: F401

__all__: list[str] = [
    "train",
    "calibrate",
    "predict",
    "diagnostics",
    "submit",
]
