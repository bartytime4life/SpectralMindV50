# src/spectramind/submit/__init__.py
"""
Submission toolkit for SpectraMind V50.

This package provides a clean boundary between raw model outputs and the
Kaggle-ready submission artifacts, including:
  • formatting (arrays → DataFrame with required columns/order),
  • validation (JSON schema + physics/num checks),
  • packaging (CSV + manifest.json + optional ZIP).

Typical usage:

    from spectramind.submit import (
        format_predictions, validate_dataframe, package_submission, N_BINS_DEFAULT
    )

    df = format_predictions(sample_ids, mu, sigma, n_bins=N_BINS_DEFAULT)
    validate_dataframe(df).raise_if_failed()
    package_submission(df, "artifacts/submit")

All functions are safe to call from CI and Kaggle (offline) environments.
"""

from __future__ import annotations

# Re-exports from submodules
from .format import format_predictions, build_expected_columns
from .validate import (
    N_BINS_DEFAULT,
    ValidationErrorReport,
    validate_dataframe,
    validate_csv,
    validate_row_dict,
)
from .package import package_submission

# Optional CLI (do not hard-require Typer in library contexts)
try:  # pragma: no cover - thin convenience re-export
    # If your repo wires a standalone CLI in this package, expose it here.
    # This avoids import errors in environments that lack Typer.
    from .cli import app as cli_app  # type: ignore
except Exception:  # pragma: no cover
    cli_app = None  # Sentinel; callers can check and wire only if available.

__all__ = [
    # format
    "format_predictions",
    "build_expected_columns",
    # validate
    "N_BINS_DEFAULT",
    "ValidationErrorReport",
    "validate_dataframe",
    "validate_csv",
    "validate_row_dict",
    # package
    "package_submission",
    # optional CLI handle
    "cli_app",
]