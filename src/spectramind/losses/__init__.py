# src/spectramind/losses/__init__.py
# Re-export public API for convenient imports.

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
