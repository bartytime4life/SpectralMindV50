# src/spectramind/calib/__init__.py
# =============================================================================
# SpectraMind V50 — Calibration Module
# -----------------------------------------------------------------------------
# This package implements all instrument calibration stages for the
# NeurIPS 2025 Ariel Data Challenge:
#
#   • adc        → Analog-to-digital conversion (gain/offset correction)
#   • dark       → Dark current modeling & subtraction
#   • flat       → Flat-field calibration (pixel sensitivity correction)
#   • cds        → Correlated double sampling (readout noise suppression)
#   • photometry → FGS1 photometry extraction & detrending
#   • trace      → Wavelength trace / dispersion solution calibration
#   • phase      → Phase-locked systematics modeling & correction
#
# Design notes:
#   - Each submodule is backend-agnostic (NumPy or PyTorch arrays).
#   - All functions are Hydra-configurable and reproducible (DVC tracked).
#   - Exports are explicit via __all__ for clean namespace usage.
#
# Example:
#   from spectramind.calib import photometry
#   flux = photometry.extract_fgs1_lightcurve(stack, config)
# =============================================================================

from __future__ import annotations

__all__ = [
    "adc",
    "dark",
    "flat",
    "cds",
    "photometry",
    "trace",
    "phase",
]