# src/spectramind/cli/__init__.py
"""
SpectraMind V50 — CLI Package
=============================

This package contains subcommand implementations for the unified CLI.

Design:
- CLI-first: Typer-based entrypoint (`spectramind`) with subcommands.
- Thin wrappers: Each subcommand delegates to internal APIs in
  `src/spectramind/{pipeline,models,inference,...}`.
- Hydra integration: Config composition is resolved in caller layer,
  passed into business logic here (no hard-coded params).
- Kaggle/CI safe: No internet calls, deterministic defaults, loud errors.

Modules:
    calibrate   : Run raw → calibrated cubes pipeline.
    train       : Train dual-channel models (FGS1 + AIRS).
    predict     : Run inference from checkpoints to CSV.
    diagnostics : Post-hoc analysis, FFT/UMAP/SHAP, HTML reports.
    submit      : Package & validate Kaggle submissions.
"""

from __future__ import annotations

import typer

# Create the Typer app for CLI subcommands.
# This is imported in `src/spectramind/cli.py` to register commands.
cli_app = typer.Typer(
    name="spectramind",
    help="SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge CLI",
    no_args_is_help=True,
    add_completion=False,
)

# Explicitly expose submodules for clean imports
__all__ = [
    "cli_app",
    "calibrate",
    "train",
    "predict",
    "diagnostics",
    "submit",
]

# Lazy import subcommand modules so they can register themselves
from . import calibrate, train, predict, diagnostics, submit  # noqa: E402,F401
