from __future__ import annotations
import numpy as np
from typing import Dict

def run_physics_checks(mu: np.ndarray, sigma: np.ndarray) -> Dict[str, bool]:
    """
    Run physics-informed checks on predicted spectra.

    Args:
        mu: Predicted means [B, 283].
        sigma: Predicted stddevs [B, 283].

    Returns:
        Dict of check_name -> pass/fail.
    """
    checks = {}
    checks["non_negative_depths"] = bool(np.all(mu >= 0))
    checks["bounded_depths"] = bool(np.all(mu <= 1))
    checks["positive_uncertainties"] = bool(np.all(sigma > 0))
    checks["no_nans"] = bool(np.isfinite(mu).all() and np.isfinite(sigma).all())
    checks["fgs1_anchor_valid"] = bool(mu[:, 0].mean() > 0)  # FGS1 bin sanity
    return checks
