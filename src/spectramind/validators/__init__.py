from __future__ import annotations

# Re-export the common result type and helpers
from .base import ValidationResult, ValidationError

# Submission CSV (wide/narrow) validator (single source of truth)
from .submission import validate_csv, validate_submission

# Physics/spectra sanity validators
from .spectra_physics import (
    check_sigma_positive,
    check_mu_nonnegative,
    check_smoothness_tv,
    check_spike_robust_zscore,
)

# Config & dataset validators
from .config_schema import validate_config  # optional; no hard dep if pydantic/jsonschema missing
from .dataset import validate_dataset_basic, validate_split_disjointness

# Kaggle runtime guardrails
from .kaggle_runtime import validate_kaggle_runtime

__all__ = [
    "ValidationResult", "ValidationError",
    "validate_csv", "validate_submission",
    "check_sigma_positive", "check_mu_nonnegative",
    "check_smoothness_tv", "check_spike_robust_zscore",
    "validate_config", "validate_dataset_basic", "validate_split_disjointness",
    "validate_kaggle_runtime",
]
