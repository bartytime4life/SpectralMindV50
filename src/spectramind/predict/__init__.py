# src/spectramind/predict/__init__.py
# =============================================================================
# SpectraMind V50 — Prediction API
# -----------------------------------------------------------------------------
# Provides the public interface for generating predictions from trained models.
# Exports configuration dataclasses, high-level predict functions, and utilities
# to produce Kaggle-ready submissions.
#
# Design:
#   • Clean separation: config → predict_to_dataframe → predict_to_submission
#   • Reproducible outputs (schema-aligned, validated)
#   • Works in local, Kaggle, or CI environments
#   • Consistent with train/submit packages (Hydra-driven, CLI-safe)
# =============================================================================

from __future__ import annotations

from .core import (
    PredictConfig,
    predict_to_dataframe,
    predict_to_submission,
)
from .utils import load_checkpoint_safe, seed_everything

__all__: list[str] = [
    # Core config & entry points
    "PredictConfig",
    "predict_to_dataframe",
    "predict_to_submission",
    # Utilities
    "load_checkpoint_safe",
    "seed_everything",
]
