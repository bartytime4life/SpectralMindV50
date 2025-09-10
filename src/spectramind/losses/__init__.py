# src/spectramind/losses/__init__.py
# =============================================================================
# SpectraMind V50 — Losses Public API
# -----------------------------------------------------------------------------
# Re-export a clean, stable surface so downstream code can import:
#   from spectramind.losses import gaussian_nll, CompositeLoss, ...
# =============================================================================

from .gaussian import gaussian_nll
from .penalties import (
    smoothness_penalty,
    nonnegativity_penalty,
    band_coherence_penalty,
    calibration_penalty,
)
from .composite import CompositeLoss, CompositeLossConfig
from .utils import make_fgs1_weights

__all__ = [
    # base
    "gaussian_nll",
    # penalties
    "smoothness_penalty",
    "nonnegativity_penalty",
    "band_coherence_penalty",
    "calibration_penalty",
    # composite
    "CompositeLoss",
    "CompositeLossConfig",
    # helpers
    "make_fgs1_weights",
]
