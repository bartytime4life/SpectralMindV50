# src/spectramind/validators/physics.py
# =============================================================================
# SpectraMind V50 — Physics-Informed Sanity Checks
# -----------------------------------------------------------------------------
# Validates predicted spectra (μ, σ) against basic astrophysical constraints.
#
# Checks implemented:
#   • Non-negativity: μ ≥ 0 (no negative transit depths)
#   • Boundedness: μ ≤ 1 (depths are fractional, not >100%)
#   • Positive uncertainties: σ > 0
#   • Finite values: no NaN/Inf in μ or σ
#   • FGS1 anchor: bin 0 mean > 0 (white-light transit must exist)
#
# Returns a dict of check_name -> bool
# Can be extended with smoothness, calibration, or band penalties (see ADR 0002).
# =============================================================================

from __future__ import annotations
import numpy as np
from typing import Dict


def run_physics_checks(mu: np.ndarray, sigma: np.ndarray) -> Dict[str, bool]:
    """
    Run physics-informed checks on predicted spectra.

    Parameters
    ----------
    mu : np.ndarray, shape [B, N]
        Predicted mean transit depths across spectral bins.
    sigma : np.ndarray, shape [B, N]
        Predicted standard deviations across spectral bins.

    Returns
    -------
    Dict[str, bool]
        Mapping of check names to pass/fail booleans.
    """
    if mu.ndim != 2 or sigma.ndim != 2:
        raise ValueError(f"Expected 2D arrays [B, N], got mu={mu.shape}, sigma={sigma.shape}")

    checks: Dict[str, bool] = {}

    # --- Core physical validity ---
    checks["non_negative_depths"] = bool(np.all(mu >= 0.0))
    checks["bounded_depths"] = bool(np.all(mu <= 1.0))
    checks["positive_uncertainties"] = bool(np.all(sigma > 0.0))
    checks["no_nans"] = bool(np.isfinite(mu).all() and np.isfinite(sigma).all())

    # --- FGS1 anchor validity (bin 0) ---
    try:
        checks["fgs1_anchor_valid"] = bool(np.mean(mu[:, 0]) > 0.0)
    except Exception:
        checks["fgs1_anchor_valid"] = False

    return checks


__all__ = ["run_physics_checks"]
