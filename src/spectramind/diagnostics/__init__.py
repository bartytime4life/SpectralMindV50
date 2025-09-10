# src/spectramind/diagnostics/__init__.py
# =============================================================================
# SpectraMind V50 — Diagnostics Package (Upgraded)
# -----------------------------------------------------------------------------
# Post-hoc analysis utilities for predictions and calibration artifacts.
# Primary entrypoint: run_diagnostics(preds_csv, truth_csv?, out_dir).
#
# Key capabilities
#   • Column/order validation (mu_*, sigma_*)
#   • Physics checks (ADR-0002): non-negativity, bounds, σ guardrails, smoothness
#   • Likelihood metrics: GLL (mean) and FGS1-weighted GLL (default ~58× on bin 0)
#   • Residual & z-score diagnostics (bias/variance health)
#   • Coverage calibration @1σ/@2σ + deviation from Gaussian ideals
#   • Reproducible reports: canonical JSON (+ .sha256) and styled HTML if available
#
# CI/Kaggle safe: NumPy + pandas only; Rich/HTML reporter optional.
# =============================================================================

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, List

import numpy as np

# Optional rich console for CLI integration (fallback to print if missing)
try:  # pragma: no cover
    from rich.console import Console
    _console: Optional[Console] = Console()
except Exception:  # pragma: no cover
    _console = None

# Prefer the richer report generator if present
try:  # pragma: no cover
    from .report import generate_diagnostics_report, generate_json_and_html  # type: ignore
except Exception:  # pragma: no cover
    generate_diagnostics_report = None  # type: ignore
    generate_json_and_html = None  # type: ignore

# Physics checks (ADR-0002 aware) — optional; fallback if not present
try:  # pragma: no cover
    from ..validators.physics import run_physics_checks  # type: ignore
except Exception:  # pragma: no cover
    run_physics_checks = None  # type: ignore


# -----------------------------------------------------------------------------
# Small console helper
# -----------------------------------------------------------------------------
def _print(msg: str) -> None:
    if _console:
        _console.print(msg)
    else:
        print(msg)


# -----------------------------------------------------------------------------
# CSV loading / validation
# -----------------------------------------------------------------------------
def _load_mu_sigma_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray | None]:
    """
    Load mu_*, sigma_* columns (optionally sample_id/id) from a CSV file.

    Returns:
        mus [N, B], sigmas [N, B], bin_names, sample_ids (or None)
    """
    import pandas as pd

    if not csv_path.exists():
        raise FileNotFoundError(f"File not found: {csv_path}")

    df = pd.read_csv(csv_path)
    mu_cols = [c for c in df.columns if c.startswith("mu_")]
    sigma_cols = [c for c in df.columns if c.startswith("sigma_")]

    if not mu_cols or not sigma_cols:
        raise ValueError(
            f"{csv_path} must contain mu_* and sigma_* columns. "
            f"Found mu: {len(mu_cols)}, sigma: {len(sigma_cols)}"
        )
    if len(mu_cols) != len(sigma_cols):
        raise ValueError(f"Column count mismatch: {len(mu_cols)} mu vs {len(sigma_cols)} sigma in {csv_path}")

    # Sort by trailing 3-digit (or int) index; malformed names pushed to end
    def _key(c: str) -> int:
        try:
            return int(c.split("_")[-1])
        except Exception:
            return 10**9

    mu_cols = sorted(mu_cols, key=_key)
    sigma_cols = sorted(sigma_cols, key=_key)

    mus = df[mu_cols].to_numpy(dtype=np.float64, copy=False)
    sigmas = df[sigma_cols].to_numpy(dtype=np.float64, copy=False)

    sample_id = None
    for sid_key in ("sample_id", "id"):
        if sid_key in df.columns:
            sample_id = df[sid_key].to_numpy()
            break

    return mus, sigmas, mu_cols, sample_id


def _finite_and_counts(arr: np.ndarray) -> Tuple[np.ndarray, Dict[str, int]]:
    mask = np.isfinite(arr)
    stats = {
        "nan_count": int(np.isnan(arr).sum()),
        "inf_count": int(np.isinf(arr).sum()),
        "finite_count": int(mask.sum()),
        "total_count": int(arr.size),
    }
    return mask, stats


# -----------------------------------------------------------------------------
# Metrics
# -----------------------------------------------------------------------------
def _smoothness_l2_second_diff(mu: np.ndarray) -> float:
    """Mean L2 of 2nd finite differences over batch. Lower ⇒ smoother."""
    if mu.shape[1] < 3:
        return 0.0
    d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
    return float(np.mean(d2**2))


def _gll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """Unweighted Gaussian log-likelihood (mean per element)."""
    eps = 1e-12
    s2 = np.maximum(sigma, eps) ** 2
    term = ((y - mu) ** 2) / s2 + np.log(2.0 * np.pi * s2)
    return float(-0.5 * np.mean(term))


def _gll_fgs1_weighted(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, fgs1_weight: float = 58.0) -> float:
    """FGS1-weighted GLL (~58× weight on bin 0)."""
    eps = 1e-12
    s2 = np.maximum(sigma, eps) ** 2
    term = ((y - mu) ** 2) / s2 + np.log(2.0 * np.pi * s2)  # [N, B]
    w = np.ones_like(term)
    if term.shape[1] > 0:
        w[:, 0] = float(fgs1_weight)
    return float(-0.5 * np.sum(w * term) / np.sum(w))


def _coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, k: float) -> float:
    """Empirical coverage fraction for |y − μ| ≤ k σ."""
    eps = 1e-12
    s = np.maximum(sigma, eps)
    ok = np.abs(y - mu) <= k * s
    return float(np.mean(ok))


def _zscore(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Elementwise z = (y − μ)/σ, σ clamped for stability."""
    eps = 1e-12
    s = np.maximum(sigma, eps)
    return (y - mu) / s


# -----------------------------------------------------------------------------
# Minimal HTML fallback (used only if rich reporter isn't available)
# -----------------------------------------------------------------------------
def _write_html_report_minimal(summary: Dict[str, Any], out_html: Path) -> Path:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>SpectraMind V50 Diagnostics</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px}"
        "code,pre{background:#f6f8fa;padding:6px;border-radius:6px;overflow:auto}</style></head><body>",
        "<h1>SpectraMind V50 — Diagnostics Report</h1>",
        "<h2>Summary</h2><pre>",
        json.dumps(summary, indent=2),
        "</pre>",
        "</body></html>",
    ]
    out_html.write_text("\n".join(html), encoding="utf-8")
    return out_html


# -----------------------------------------------------------------------------
# Diagnostics entrypoint
# -----------------------------------------------------------------------------
def run_diagnostics(
    preds_path: Path,
    truth_path: Optional[Path] = None,
    out_dir: Path = Path("artifacts/diagnostics"),
    report_name: str = "report.html",
    *,
    fgs1_weight: Optional[float] = None,      # default from env or 58.0
    physics_thresholds: Optional[Dict[str, float]] = None,  # see below
) -> Dict[str, Any]:
    """
    Run diagnostics on predictions (and optionally ground truth).

    Args
    ----
    preds_path : Path
        Path to predictions CSV (mu_* / sigma_* per bin).
    truth_path : Path, optional
        Path to ground-truth CSV (aligned mu_* columns) if available.
    out_dir : Path
        Directory where diagnostics artifacts will be written.
    report_name : str
        Name of the generated HTML report file ('' to skip HTML).
    fgs1_weight : Optional[float]
        Weight for bin 0 in the weighted GLL. Defaults to
        float(os.getenv('SM_FGS1_WEIGHT', 58.0)).
    physics_thresholds : Optional[Dict[str, float]]
        Optional thresholds forwarded to validators.physics.run_physics_checks, e.g.:
        {
          "sigma_min": 1e-6, "sigma_max": 0.5,
          "tv_rel_thresh": 0.25, "curvature_rel_thresh": 0.15,
          "intervals_k": 1.0, "intervals_min_frac": 0.95
        }

    Returns
    -------
    Dict[str, Any]
        Summary metrics and sanity checks (JSON-serializable).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve FGS1 weight
    if fgs1_weight is None:
        try:
            fgs1_weight = float(os.getenv("SM_FGS1_WEIGHT", "58.0"))
        except Exception:
            fgs1_weight = 58.0

    _print("[bold cyan]Diagnostics[/bold cyan]" if _console else "Diagnostics")
    _print(f"Predictions: {preds_path}")
    if truth_path:
        _print(f"Ground truth: {truth_path}")

    # --- Load predictions ---
    mus, sigmas, mu_cols, _sample_ids = _load_mu_sigma_csv(Path(preds_path))
    n_samples, n_bins = mus.shape
    mus = mus.astype(np.float64, copy=False)
    sigmas = sigmas.astype(np.float64, copy=False)

    # --- Basic health on predictions ---
    mu_finite_mask, mu_finite_stats = _finite_and_counts(mus)
    sigma_finite_mask, sigma_finite_stats = _finite_and_counts(sigmas)

    # Clamp σ for stable downstream metrics (do NOT persist back)
    eps = 1e-12
    sigmas_clamped = np.maximum(sigmas, eps)

    frac_negative_mu = float(np.mean(mus < 0.0))
    frac_mu_gt_one = float(np.mean(mus > 1.0))
    avg_sigma = float(np.mean(sigmas_clamped))
    smoothness = _smoothness_l2_second_diff(mus)

    summary: Dict[str, Any] = {
        "n_samples": int(n_samples),
        "n_bins": int(n_bins),
        "columns_mu": mu_cols,
        "preds_stats": {
            "mu": mu_finite_stats,
            "sigma": sigma_finite_stats,
            "frac_negative_mu": frac_negative_mu,
            "frac_mu_gt_one": frac_mu_gt_one,
            "avg_sigma": avg_sigma,
            "smoothness_l2_second_diff": smoothness,
        },
        "fgs1_weight": float(fgs1_weight),
    }

    # --- Physics checks (optional thresholds) ---
    if run_physics_checks is not None:
        thresholds = physics_thresholds or {}
        phys = run_physics_checks(
            mus, sigmas,
            fgs1_index=0,
            sigma_min=thresholds.get("sigma_min"),
            sigma_max=thresholds.get("sigma_max"),
            tv_rel_thresh=thresholds.get("tv_rel_thresh"),
            curvature_rel_thresh=thresholds.get("curvature_rel_thresh"),
            intervals_k=thresholds.get("intervals_k", 1.0),
            intervals_min_frac=thresholds.get("intervals_min_frac"),
        )
        summary["physics_checks"] = phys
        summary["all_passed"] = bool(phys.get("all_passed", False))
    else:
        summary["physics_checks"] = {"available": False}
        summary["all_passed"] = True  # do not fail when validator module absent

    # --- If ground truth available, compute residual metrics & coverage ---
    if truth_path and Path(truth_path).exists():
        y_true, _, truth_mu_cols, _ = _load_mu_sigma_csv(Path(truth_path))

        # Ensure same bins/columns and sample counts
        if truth_mu_cols != mu_cols:
            raise ValueError("Mismatch in mu_* column order between predictions and truth.")
        if y_true.shape != mus.shape:
            raise ValueError(f"Shape mismatch: truth {y_true.shape} vs preds {mus.shape}")

        resid = mus - y_true
        mse = float(np.mean(resid**2))
        mae = float(np.mean(np.abs(resid)))

        # z-scores
        z = _zscore(y_true, mus, sigmas_clamped)
        z_mean = float(np.mean(z))
        z_std = float(np.std(z))
        z_abs_mean = float(np.mean(np.abs(z)))

        # Likelihoods
        gll = _gll(y_true, mus, sigmas_clamped)
        gll_fgs1 = _gll_fgs1_weighted(y_true, mus, sigmas_clamped, fgs1_weight=float(fgs1_weight))

        # Coverage & deviation from Gaussian ideals
        cov1 = _coverage(y_true, mus, sigmas_clamped, k=1.0)
        cov2 = _coverage(y_true, mus, sigmas_clamped, k=2.0)
        ideal1, ideal2 = 0.682689492, 0.954499736  # 1σ/2σ for Gaussian
        cov1_err = float(cov1 - ideal1)
        cov2_err = float(cov2 - ideal2)

        summary.update(
            {
                "residual_metrics": {
                    "mse": mse,
                    "mae": mae,
                    "resid_mean": float(np.mean(resid)),
                    "resid_std": float(np.std(resid)),
                },
                "likelihood_metrics": {
                    "gll_mean": gll,
                    "gll_fgs1_weighted": gll_fgs1,
                },
                "zscore_stats": {
                    "z_mean": z_mean,
                    "z_std": z_std,
                    "z_abs_mean": z_abs_mean,
                },
                "coverage": {
                    "emp_cov_1sigma": cov1,
                    "emp_cov_2sigma": cov2,
                    "cov1_minus_ideal": cov1_err,
                    "cov2_minus_ideal": cov2_err,
                },
            }
        )

    # --- Persist artifacts (prefer canonical reporter) ---
    summary_json = out_dir / "summary.json"
    if generate_json_and_html is not None:
        # dual write: <out_dir>/summary.{json,html} (+ .sha256)
        generate_json_and_html(summary, out_base=out_dir / "summary", title="SpectraMind V50 — Diagnostics")
        # also keep legacy name if user expects a specific file
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if report_name:
            # generate a second HTML with the requested name for back-compat
            generate_diagnostics_report(  # type: ignore
                summary, out_path=out_dir / report_name, title="SpectraMind V50 — Diagnostics"
            )
    else:
        # plain JSON + minimal HTML fallback
        summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        if report_name:
            _write_html_report_minimal(summary, out_dir / report_name)

    _print(f"[green]✓ Diagnostics complete[/green] → {summary_json}" if _console else f"Diagnostics complete → {summary_json}")
    return summary


__all__ = ["run_diagnostics"]