# src/spectramind/diagnostics/__init__.py
# =============================================================================
# SpectraMind V50 — Diagnostics Package
# -----------------------------------------------------------------------------
# Provides post-hoc analysis tools for inspecting model outputs, calibration
# artifacts, and reproducibility. Used in the DVC `diagnose` stage and via
# `spectramind diagnose`.
#
# Features:
#   - Metrics: Gaussian Log-Likelihood (GLL), residual stats
#   - Visualization: lightcurves, spectra, FFT/UMAP embeddings
#   - Sanity checks: non-negativity, smoothness, uncertainty calibration
#   - Report generation: HTML summaries, plots directory
# =============================================================================

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

# Optional rich console for CLI integration
from rich.console import Console

console = Console()


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
        Path to predictions CSV (mu/sigma per bin).
    truth_path : Path, optional
        Path to ground-truth CSV if available (for offline validation).
    out_dir : Path
        Directory where plots and reports will be written.
    report_name : str
        Name of the generated HTML report file.

    Returns
    -------
    Dict[str, Any]
        Summary metrics (e.g., GLL, coverage, smoothness).
    """
    out_dir.mkdir(parents=True, exist_ok=True)

    console.rule("[bold cyan]Diagnostics[/bold cyan]")
    console.print(f"Predictions: {preds_path}")
    if truth_path:
        console.print(f"Ground truth: {truth_path}")

    # --- Load predictions ---
    import pandas as pd

    preds = pd.read_csv(preds_path)
    mu_cols = [c for c in preds.columns if c.startswith("mu_")]
    sigma_cols = [c for c in preds.columns if c.startswith("sigma_")]

    mus = preds[mu_cols].to_numpy()
    sigmas = preds[sigma_cols].to_numpy()

    # --- Simple sanity checks ---
    nonneg_frac = float(np.mean(mus < 0))
    avg_sigma = float(np.mean(sigmas))

    summary: Dict[str, Any] = {
        "n_samples": len(preds),
        "n_bins": mus.shape[1],
        "frac_negative_mu": nonneg_frac,
        "avg_sigma": avg_sigma,
    }

    # --- If ground truth available, compute residuals & GLL ---
    if truth_path and Path(truth_path).exists():
        truth = pd.read_csv(truth_path)
        y_true = truth[mu_cols].to_numpy()
        resid = mus - y_true
        mse = float(np.mean(resid**2))

        # Gaussian log-likelihood (simplified)
        gll = float(-0.5 * np.mean(((y_true - mus) / sigmas) ** 2 + np.log(sigmas**2)))

        summary.update({"mse": mse, "gll": gll})

    # --- Save JSON summary ---
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))

    console.print(f"[green]✓ Diagnostics complete[/green] → {summary_path}")
    return summary


# -----------------------------------------------------------------------------
# Convenience
# -----------------------------------------------------------------------------
__all__ = ["run_diagnostics"]
