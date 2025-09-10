# src/spectramind/validators/physics.py
# =============================================================================
# SpectraMind V50 — Physics-Informed Sanity Checks (Upgraded)
# -----------------------------------------------------------------------------
# Validates predicted spectra (μ, σ) against astrophysical & numerical constraints.
#
# Always-on checks (booleans):
#   • non_negative_depths:       μ ≥ 0
#   • bounded_depths:            μ ≤ 1
#   • positive_uncertainties:    σ > 0
#   • no_nans:                   μ, σ finite
#   • fgs1_anchor_valid:         mean(μ[:, fgs1_index]) > 0
#
# Optional checks (thresholds are None by default ⇒ check “passes” but stats recorded):
#   • sigma_min_ok:     min(σ)  ≥ sigma_min
#   • sigma_max_ok:     max(σ)  ≤ sigma_max
#   • tv_rel_ok:        mean relative total variation ≤ tv_rel_thresh
#   • curvature_rel_ok: mean relative 2nd-diff energy ≤ curvature_rel_thresh
#   • intervals_in_bounds_ok:   fraction of [μ±kσ] within [0,1] ≥ intervals_min_frac
#
# Returns:
#   {
#     "checks": {name: bool, ...},
#     "stats": {... per-batch summary metrics ...},
#     "all_passed": bool
#   }
#
# Notes:
#   • All computations use float64 deterministically.
#   • Optional thresholds let ADR-0002 loss/validator stay in sync:
#       - sigma_min ~ 1e-6 to avoid overconfident zeros
#       - tv_rel_thresh/curvature_rel_thresh relate to smoothness priors
#   • “Relative” metrics normalize by mean(μ) per spectrum (ε-guarded).
# =============================================================================

from __future__ import annotations
from typing import Dict, Any, Optional, Tuple
import numpy as np


def _as_2d(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"Expected 2D array for {name} [B, N], got shape {x.shape}")
    return x


def _finite(x: np.ndarray) -> bool:
    return np.isfinite(x).all()


def _eps_like(x: np.ndarray, base: float = 1e-12) -> float:
    # Small, stable epsilon relative to typical magnitudes
    m = float(np.nanmean(np.abs(x))) if x.size else 1.0
    return max(base, 1e-9 * m)


def _total_variation_rel(mu: np.ndarray) -> np.ndarray:
    """
    Relative total variation per spectrum:
      TV_rel_i = sum_j |Δμ_ij| / ((N-1) * max(mean(μ_i), ε))
    """
    d = np.diff(mu, axis=1)                        # [B, N-1]
    tv = np.sum(np.abs(d), axis=1)                 # [B]
    mean_level = np.mean(mu, axis=1)               # [B]
    eps = np.maximum(np.full_like(mean_level, 1e-12), 1e-9 * np.maximum(mean_level, 1.0))
    return tv / ((mu.shape[1] - 1) * np.maximum(mean_level, eps))


def _curvature_rel(mu: np.ndarray) -> np.ndarray:
    """
    Relative second-difference L2 per spectrum:
      C_rel_i = ||Δ² μ_i||_2 / (sqrt(N-2) * max(mean(μ_i), ε))
    """
    if mu.shape[1] < 3:
        return np.zeros(mu.shape[0], dtype=np.float64)
    d2 = np.diff(mu, n=2, axis=1)                  # [B, N-2]
    num = np.linalg.norm(d2, axis=1)               # [B]
    mean_level = np.mean(mu, axis=1)               # [B]
    eps = np.maximum(np.full_like(mean_level, 1e-12), 1e-9 * np.maximum(mean_level, 1.0))
    return num / (np.sqrt(max(mu.shape[1] - 2, 1)) * np.maximum(mean_level, eps))


def _intervals_in_bounds_frac(mu: np.ndarray, sigma: np.ndarray, k: float = 1.0) -> float:
    """
    Fraction of (μ±kσ) entries that fall within [0, 1].
    """
    lo = mu - k * sigma
    hi = mu + k * sigma
    ok = (lo >= 0.0) & (hi <= 1.0) & np.isfinite(lo) & np.isfinite(hi)
    return float(np.mean(ok))


def run_physics_checks(
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    fgs1_index: int = 0,
    # Optional thresholds (None ⇒ do not enforce, but compute stats)
    sigma_min: Optional[float] = None,         # e.g., 1e-6 to discourage overconfidence
    sigma_max: Optional[float] = None,         # e.g., 0.5 to avoid absurdly large σ
    tv_rel_thresh: Optional[float] = None,     # e.g., 0.25 (unitless, relative)
    curvature_rel_thresh: Optional[float] = None,  # e.g., 0.15 (unitless, relative)
    intervals_k: float = 1.0,
    intervals_min_frac: Optional[float] = None # e.g., 0.95 (≥95% of μ±kσ within [0,1])
) -> Dict[str, Any]:
    """
    Run physics-informed checks on predicted spectra.

    Parameters
    ----------
    mu : np.ndarray, shape [B, N]
        Predicted mean transit depths across spectral bins (fractional).
    sigma : np.ndarray, shape [B, N]
        Predicted standard deviations across spectral bins (fractional).
    fgs1_index : int, default=0
        Index of the white-light (FGS1) anchor bin.
    sigma_min, sigma_max : Optional[float]
        If provided, enforce σ bounds (strict > sigma_min, ≤ sigma_max).
    tv_rel_thresh, curvature_rel_thresh : Optional[float]
        If provided, enforce smoothness (relative TV/curvature) thresholds.
    intervals_k : float, default=1.0
        k for μ±kσ interval check.
    intervals_min_frac : Optional[float]
        If provided, enforce min fraction of intervals within [0,1].

    Returns
    -------
    Dict[str, Any]
        {
          "checks": { name: bool, ... },
          "stats":  { ... summary metrics ... },
          "all_passed": bool
        }
    """
    mu = _as_2d(mu, "mu")
    sigma = _as_2d(sigma, "sigma")
    if mu.shape != sigma.shape:
        raise ValueError(f"Shape mismatch: mu {mu.shape} vs sigma {sigma.shape}")

    B, N = mu.shape

    # Core checks (always enforced)
    non_negative_depths   = bool(np.all(mu >= 0.0))
    bounded_depths        = bool(np.all(mu <= 1.0))
    positive_uncertainty  = bool(np.all(sigma > 0.0))
    finite_ok             = bool(_finite(mu) and _finite(sigma))

    # FGS1 anchor
    try:
        fgs1_anchor_valid = bool(np.mean(mu[:, int(fgs1_index)]) > 0.0)
    except Exception:
        fgs1_anchor_valid = False

    # Optional sigma floor/cap
    sigma_min_ok = True
    sigma_max_ok = True
    if sigma_min is not None:
        sigma_min_ok = bool(np.all(sigma > float(sigma_min)))
    if sigma_max is not None:
        sigma_max_ok = bool(np.all(sigma <= float(sigma_max)))

    # Smoothness metrics (relative, per-spectrum → mean across batch)
    tv_rel_per = _total_variation_rel(mu)                 # [B]
    curv_rel_per = _curvature_rel(mu)                     # [B]
    tv_rel_mean = float(np.mean(tv_rel_per)) if B else 0.0
    curv_rel_mean = float(np.mean(curv_rel_per)) if B else 0.0

    tv_rel_ok = True if tv_rel_thresh is None else bool(tv_rel_mean <= float(tv_rel_thresh))
    curvature_rel_ok = True if curvature_rel_thresh is None else bool(curv_rel_mean <= float(curvature_rel_thresh))

    # Interval-in-bounds fraction
    intervals_frac = _intervals_in_bounds_frac(mu, sigma, k=float(intervals_k))
    intervals_in_bounds_ok = True if intervals_min_frac is None else bool(intervals_frac >= float(intervals_min_frac))

    checks: Dict[str, bool] = {
        "non_negative_depths": non_negative_depths,
        "bounded_depths": bounded_depths,
        "positive_uncertainties": positive_uncertainty,
        "no_nans": finite_ok,
        "fgs1_anchor_valid": fgs1_anchor_valid,
        "sigma_min_ok": sigma_min_ok,
        "sigma_max_ok": sigma_max_ok,
        "tv_rel_ok": tv_rel_ok,
        "curvature_rel_ok": curvature_rel_ok,
        "intervals_in_bounds_ok": intervals_in_bounds_ok,
    }

    stats: Dict[str, Any] = {
        "batch_size": int(B),
        "n_bins": int(N),
        "mu_min": float(np.min(mu)) if mu.size else float("nan"),
        "mu_max": float(np.max(mu)) if mu.size else float("nan"),
        "mu_mean": float(np.mean(mu)) if mu.size else float("nan"),
        "sigma_min": float(np.min(sigma)) if sigma.size else float("nan"),
        "sigma_max": float(np.max(sigma)) if sigma.size else float("nan"),
        "sigma_mean": float(np.mean(sigma)) if sigma.size else float("nan"),
        "tv_rel_mean": tv_rel_mean,
        "tv_rel_median": float(np.median(tv_rel_per)) if B else 0.0,
        "curvature_rel_mean": curv_rel_mean,
        "curvature_rel_median": float(np.median(curv_rel_per)) if B else 0.0,
        "intervals_k": float(intervals_k),
        "intervals_in_bounds_frac": float(intervals_frac),
        "fgs1_index": int(fgs1_index),
        "fgs1_mean": float(np.mean(mu[:, int(fgs1_index)])) if B and 0 <= int(fgs1_index) < N else float("nan"),
    }

    all_passed = all(checks.values())
    return {"checks": checks, "stats": stats, "all_passed": bool(all_passed)}


__all__ = ["run_physics_checks"]