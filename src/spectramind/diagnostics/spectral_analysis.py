# src/spectramind/diagnostics/spectral_analysis.py
# =============================================================================
# SpectraMind V50 — FFT-Based Spectral Analysis
# -----------------------------------------------------------------------------
# Computes Fourier-domain diagnostics for lightcurves or time-series spectra.
#
# Features:
#   • Works with single series [T] or batches [N, T]
#   • Returns power spectral density normalized per-sample
#   • NaN/Inf safe (drops invalid values before FFT)
#   • Deterministic output (float64, sorted freqs)
#   • CI/Kaggle-friendly: NumPy-only
# =============================================================================

from __future__ import annotations
import numpy as np
from typing import Dict, Any


def run_fft_analysis(lightcurve: np.ndarray, dt: float = 1.0) -> Dict[str, Any]:
    """
    Compute FFT power spectrum of a lightcurve or spectral time series.

    Parameters
    ----------
    lightcurve : np.ndarray, shape [T] or [N, T]
        Flux vs. time input(s). NaN/Inf values are zeroed before FFT.
    dt : float, optional (default=1.0)
        Time step between samples.

    Returns
    -------
    Dict[str, Any]
        {
            "freqs": np.ndarray, shape [F],
                Non-negative frequency bins.
            "power": np.ndarray, shape [N, F] or [F],
                Power spectral density (|FFT|^2 / n).
            "fft_vals": np.ndarray, shape [N, F] or [F],
                Complex FFT values (for downstream use).
            "dt": float,
                Time step used.
            "n_samples": int,
                Length of the input time axis.
        }
    """
    lc = np.asarray(lightcurve, dtype=np.float64)

    # Handle invalid values robustly
    if not np.isfinite(lc).all():
        lc = np.nan_to_num(lc, nan=0.0, posinf=0.0, neginf=0.0)

    # Ensure shape [N, T]
    if lc.ndim == 1:
        lc = lc[None, :]  # add batch dimension

    n = lc.shape[-1]
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(lc, axis=-1)

    # Normalize by n for power spectral density
    power = (np.abs(fft_vals) ** 2) / n

    if power.shape[0] == 1:  # squeeze back if single input
        power = power[0]
        fft_vals = fft_vals[0]

    return {
        "freqs": freqs,
        "power": power,
        "fft_vals": fft_vals,
        "dt": float(dt),
        "n_samples": int(n),
    }
