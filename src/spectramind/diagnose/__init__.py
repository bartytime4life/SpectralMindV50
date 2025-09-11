from __future__ import annotations

from .io import (
    read_predictions_any,
    read_truth_any,
    to_wide_predictions,
)
from .metrics import (
    compute_sanity_checks,
    compute_smoothness_score,
    compute_coverage,
    compute_gll_simple,
)
from .run import run_diagnostics

__all__ = [
    # I/O
    "read_predictions_any",
    "read_truth_any",
    "to_wide_predictions",
    # Metrics
    "compute_sanity_checks",
    "compute_smoothness_score",
    "compute_coverage",
    "compute_gll_simple",
    # Orchestrator
    "run_diagnostics",
]
