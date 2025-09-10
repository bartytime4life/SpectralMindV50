# src/spectramind/diagnostics/__init__.py
# =============================================================================
# SpectraMind V50 — Diagnostics Package
# -----------------------------------------------------------------------------
# Provides post-hoc analysis tools for inspecting model outputs, calibration
# artifacts, and reproducibility. Used in the DVC `diagnose` stage and via
# `spectramind diagnose`.
#
# Features:
#   - Metrics: Gaussian Log-Likelihood (GLL), FGS1-weighted GLL, residual stats
#   - Sanity checks: non-negativity, boundedness, smoothness, uncertainty health
#   - Coverage calibration: empirical 1σ / 2σ coverage
#   - Report generation: JSON summary (+ optional minimal HTML)
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

# Optional rich console for CLI integration (fallback to print if missing)
try:
    from rich.console import Console
    _console: Optional[Console] = Console()
except Exception:  # pragma: no cover
    _console = None


def _print(msg: str) -> None:
    if _console:
        _console.print(msg)
    else:
        print(msg)


def _load_mu_sigma_csv(csv_path: Path) -> Tuple[np.ndarray, np.ndarray, list[str], np.ndarray | None]:
    """
    Load mu_*, sigma_* columns (optionally sample_id) from a CSV file.

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

    # Sort by bin index to guarantee alignment if columns are scrambled
    def _key(c: str) -> int:
        # expects names like mu_000, mu_001 ...
        try:
            return int(c.split("_")[-1])
        except Exception:
            return 10**9  # push malformed to end

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


def _smoothness_l2_second_diff(mu: np.ndarray) -> float:
    """
    L2 norm of second-order finite differences averaged over batch.
    Lower is smoother (penalizes rapid oscillations).
    mu: [N, B]
    """
    if mu.shape[1] < 3:
        return 0.0
    d2 = mu[:, 2:] - 2 * mu[:, 1:-1] + mu[:, :-2]
    return float(np.mean(d2**2))


def _gll(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """
    Unweighted Gaussian log-likelihood per-bin (mean over all samples/bins).
    """
    # Clamp sigma for stability
    eps = 1e-12
    s2 = np.maximum(sigma, eps) ** 2
    term = ((y - mu) ** 2) / s2 + np.log(2.0 * np.pi * s2)
    return float(-0.5 * np.mean(term))


def _gll_fgs1_weighted(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, fgs1_weight: float = 58.0) -> float:
    """
    FGS1-weighted Gaussian log-likelihood (~58× on bin 0, others 1×).
    """
    eps = 1e-12
    s2 = np.maximum(sigma, eps) ** 2
    term = ((y - mu) ** 2) / s2 + np.log(2.0 * np.pi * s2)  # shape [N, B]
    w = np.ones_like(term)
    if term.shape[1] > 0:
        w[:, 0] = fgs1_weight
    # mean of weighted NLL per element
    n = term.size
    return float(-0.5 * np.sum(w * term) / np.sum(w))


def _coverage(y: np.ndarray, mu: np.ndarray, sigma: np.ndarray, k: float) -> float:
    """
    Empirical coverage: fraction of bins where |y - mu| <= k * sigma.
    """
    eps = 1e-12
    s = np.maximum(sigma, eps)
    ok = np.abs(y - mu) <= k * s
    return float(np.mean(ok))


def _write_html_report(summary: Dict[str, Any], out_html: Path) -> Path:
    out_html.parent.mkdir(parents=True, exist_ok=True)
    html = [
        "<html><head><meta charset='utf-8'><title>SpectraMind V50 Diagnostics</title>",
        "<style>body{font-family:system-ui,Arial,sans-serif;max-width:900px;margin:40px auto;padding:0 16px}"
        "code,pre{background:#f6f8fa;padding:4px 6px;border-radius:4px}</style></head><body>",
        "<h1>SpectraMind V50 — Diagnostics Report</h1>",
        "<h2>Summary</h2><pre>",
        json.dumps(summary, indent=2),
        "</pre>",
        "</body></html>",
    ]
    out_html.write_text("\n".join(html))
    return out_html


# -----------------------------------------------------------------------------
# Diagnostics entrypoint
# -----------------------------------------------------------------------------
def run_diagnostics(
    preds_path: Path,
    truth_path: Optional[Path] = None,
    out_dir: Path = Path("artifacts/diagnostics"),
    report_name: str = "report.html",
) -> Dict[str, Any]:
    """
    Run diagnostics on predictions (and optionally ground truth).

    Args
    ----
    preds_path : Path
        Path to predictions CSV (mu_* / sigma_* per bin).
    truth_path : Path, optional
        Path to ground-truth CSV (matched mu_* columns) if available.
    out_dir : Path
        Directory where diagnostics artifacts will be written.
    report_name : str
        Name of the generated HTML report file (set to '' to skip HTML).

    Returns
    -------
    Dict[str, Any]
        Summary metrics and sanity checks.
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    _print("[bold cyan]Diagnostics[/bold cyan]" if _console else "Diagnostics")
    _print(f"Predictions: {preds_path}")
    if truth_path:
        _print(f"Ground truth: {truth_path}")

    # --- Load predictions ---
    mus, sigmas, mu_cols, _sample_ids = _load_mu_sigma_csv(preds_path)
    n_samples, n_bins = mus.shape

    # --- Basic health on predictions ---
    mu_finite_mask, mu_finite_stats = _finite_and_counts(mus)
    sigma_finite_mask, sigma_finite_stats = _finite_and_counts(sigmas)

    # Clamp σ to avoid zero/negatives (for metrics downstream)
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
    }

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

        # GLL (unweighted) and FGS1-weighted GLL
        gll = _gll(y_true, mus, sigmas_clamped)
        gll_fgs1 = _gll_fgs1_weighted(y_true, mus, sigmas_clamped, fgs1_weight=58.0)

        # Coverage calibration
        cov_1sigma = _coverage(y_true, mus, sigmas_clamped, k=1.0)
        cov_2sigma = _coverage(y_true, mus, sigmas_clamped, k=2.0)

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
                "coverage": {
                    "emp_cov_1sigma": cov_1sigma,  # ideal ≈ 0.6827 for perfect Gaussian cal
                    "emp_cov_2sigma": cov_2sigma,  # ideal ≈ 0.9545
                },
            }
        )

    # --- Save JSON summary ---
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    # --- Optional minimal HTML report ---
    if report_name:
        report_path = out_dir / report_name
        _write_html_report(summary, report_path)

    _print(f"[green]✓ Diagnostics complete[/green] → {summary_path}" if _console else f"Diagnostics complete → {summary_path}")
    return summary


__all__ = ["run_diagnostics"]
