# src/spectramind/diagnostics/spectral_analysis.py
# =============================================================================
# SpectraMind V50 — FFT-Based Spectral Analysis (Upgraded)
# -----------------------------------------------------------------------------
# Computes Fourier-domain diagnostics for lightcurves or time-series spectra.
#
# Key features:
#   • Accepts [T] or [N, T] (auto-batch); float64 processing for determinism
#   • NaN/Inf safe: replaces invalid samples with 0.0 (per-step logged in stats)
#   • Windowing: 'hann' (default), 'ones' (rectangular), or explicit ndarray
#   • Detrending: 'none' | 'mean' | 'linear'
#   • Welch averaging (optional): nperseg/noverlap with 50% default overlap
#   • Normalization: 'per_sample' (default, |FFT|^2 / n) or 'per_hz' (divide by n*dt)
#   • Optional zero-padding: pad_to='none' | 'next_pow2' | int
#   • Returns rich metadata (freq_res, nyquist, n_segments, etc.)
#
# Notes:
#   • We intentionally avoid SciPy for Kaggle/CI portability.
#   • 'per_sample' normalization matches prior behavior (non-density).
#   • 'per_hz' is a simple density proxy: (|FFT|^2 / n) / (n*dt) = |FFT|^2 / (n^2 * dt).
#     This is not a strict periodogram scaling, but is often adequate for comparisons.
# =============================================================================

from __future__ import annotations
from typing import Dict, Any, Tuple, Optional, Union

import numpy as np


WindowSpec = Union[str, np.ndarray, None]


def _as_2d(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim == 1:
        x = x[None, :]
    elif x.ndim != 2:
        raise ValueError(f"Expected [T] or [N,T]; got shape {x.shape}")
    return x


def _detrend(x: np.ndarray, mode: str) -> np.ndarray:
    # x: [N, T]
    if mode == "none":
        return x
    if mode == "mean":
        return x - np.nanmean(x, axis=1, keepdims=True)
    if mode == "linear":
        # Simple least-squares linear detrend per row (NumPy-only)
        n = x.shape[1]
        t = np.arange(n, dtype=np.float64)
        # Fit a + b t; handle NaNs robustly by zeroing invalid samples
        x_clean = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        t0 = (t - t.mean())  # center to improve conditioning
        denom = np.sum(t0**2)
        if denom == 0:
            return x_clean - x_clean.mean(axis=1, keepdims=True)
        b = (x_clean @ t0) / denom                  # [N]
        a = x_clean.mean(axis=1) - b * t.mean()     # [N]
        trend = a[:, None] + b[:, None] * t[None, :]
        return x_clean - trend
    raise ValueError(f"Unknown detrend mode: {mode!r}")


def _make_window(n: int, spec: WindowSpec) -> np.ndarray:
    if spec is None or spec == "hann":
        # Hann window; safe for n>=1
        if n <= 1:
            return np.ones(n, dtype=np.float64)
        k = np.arange(n, dtype=np.float64)
        return 0.5 - 0.5 * np.cos(2.0 * np.pi * k / (n - 1))
    if spec == "ones" or spec == "rect":
        return np.ones(n, dtype=np.float64)
    if isinstance(spec, np.ndarray):
        w = np.asarray(spec, dtype=np.float64)
        if w.ndim != 1 or w.shape[0] != n:
            raise ValueError(f"Window length mismatch: expected {n}, got {w.shape}")
        return w
    raise ValueError(f"Unknown window spec: {spec!r}")


def _next_pow2(n: int) -> int:
    return 1 if n <= 1 else int(2 ** (int(np.ceil(np.log2(n)))))


def _pad_length(n: int, pad_to: Union[str, int]) -> int:
    if pad_to == "none" or pad_to is None:
        return n
    if pad_to == "next_pow2":
        return _next_pow2(n)
    if isinstance(pad_to, int):
        return max(n, int(pad_to))
    raise ValueError(f"Invalid pad_to: {pad_to!r}")


def _apply_invalid_sanitization(x: np.ndarray) -> Tuple[np.ndarray, int]:
    # Replace NaN/Inf with 0.0; return number of replacements
    mask = ~np.isfinite(x)
    n_bad = int(mask.sum())
    if n_bad:
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    return x, n_bad


def _per_sample_power(fft_vals: np.ndarray, n_eff: int) -> np.ndarray:
    # |FFT|^2 / n_eff  (matches original behavior)
    return (np.abs(fft_vals) ** 2) / float(n_eff)


def _per_hz_power(fft_vals: np.ndarray, n_eff: int, dt: float) -> np.ndarray:
    # Simple density proxy: (|FFT|^2 / n) / (n*dt) = |FFT|^2 / (n^2 * dt)
    n = float(n_eff)
    return (np.abs(fft_vals) ** 2) / (n * n * float(dt))


def _rfft_batched(x: np.ndarray, n_fft: int) -> np.ndarray:
    # x: [N, T]; zero-pad/truncate via rfft's n=n_fft
    return np.fft.rfft(x, n=n_fft, axis=1)


def _welch(
    x: np.ndarray,
    dt: float,
    window: WindowSpec,
    nperseg: int,
    noverlap: Optional[int],
    n_fft: Optional[int],
    norm: str,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Welch average over segments.

    Returns
    -------
    freqs : [F]
    power : [N, F]
    n_segments : int
    """
    N, T = x.shape
    if nperseg > T:
        nperseg = T
    if nperseg <= 0:
        raise ValueError("nperseg must be >= 1")

    if noverlap is None:
        noverlap = nperseg // 2  # 50% overlap by default
    noverlap = int(np.clip(noverlap, 0, nperseg - 1))
    step = nperseg - noverlap
    if step <= 0:
        raise ValueError("noverlap must be < nperseg")

    w = _make_window(nperseg, window)
    n_fft = nperseg if (n_fft is None or n_fft < nperseg) else int(n_fft)

    # Segment indices
    starts = np.arange(0, T - nperseg + 1, step, dtype=int)
    n_segments = len(starts)
    if n_segments == 0:
        # fallback to single segment (pad if needed)
        starts = np.array([0], dtype=int)
        n_segments = 1

    # Accumulate power
    acc = None
    for s in starts:
        seg = x[:, s : s + nperseg]  # [N, nperseg]
        seg = _detrend(seg, "none")  # assume detrending applied globally already
        seg = seg * w[None, :]
        fft_vals = _rfft_batched(seg, n_fft=n_fft)  # [N, F]
        if norm == "per_sample":
            p = _per_sample_power(fft_vals, n_eff=nperseg)
        elif norm == "per_hz":
            p = _per_hz_power(fft_vals, n_eff=nperseg, dt=dt)
        else:
            raise ValueError(f"Unknown normalization: {norm!r}")
        acc = p if acc is None else (acc + p)

    power = acc / float(n_segments)
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    return freqs, power, int(n_segments)


def run_fft_analysis(
    lightcurve: np.ndarray,
    dt: float = 1.0,
    *,
    window: WindowSpec = "hann",
    detrend: str = "mean",
    pad_to: Union[str, int, None] = "none",
    normalization: str = "per_sample",  # 'per_sample' (default) or 'per_hz'
    welch_nperseg: Optional[int] = None,
    welch_noverlap: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Compute FFT diagnostics of a lightcurve or batch of lightcurves.

    Parameters
    ----------
    lightcurve : np.ndarray, shape [T] or [N, T]
        Flux vs. time input(s). NaN/Inf values are set to 0.0 before FFT.
    dt : float, default=1.0
        Time step between samples (seconds or arbitrary unit).
    window : {'hann','ones'} or ndarray, default='hann'
        Window applied prior to FFT (global; Welch windows are per segment).
    detrend : {'none','mean','linear'}, default='mean'
        Detrending mode applied before windowing/FFT (global).
    pad_to : {'none','next_pow2', int}, default='none'
        Zero-pad target FFT length. If int, pads/truncates to at least that length.
    normalization : {'per_sample','per_hz'}, default='per_sample'
        Power scaling. 'per_sample' preserves legacy behavior (|FFT|^2 / n).
        'per_hz' divides the per-sample power by (n*dt).
    welch_nperseg : Optional[int], default=None
        If provided, compute Welch-averaged spectrum with segment length nperseg.
    welch_noverlap : Optional[int], default=None
        Overlap for Welch (defaults to 50% if None).

    Returns
    -------
    Dict[str, Any]
        {
            "freqs": [F],           # Non-negative frequency bins (Hz or 1/unit of dt)
            "power": [F] or [N, F], # Scaled power per normalization
            "fft_vals": [F] or [N, F] (complex)  # present only if not Welch
            "dt": float,
            "n_samples": int,       # original time-length (before padding)
            "freq_res": float,      # frequency resolution (1 / (n_fft * dt))
            "nyquist": float,       # 0.5 / dt
            "n_fft": int,           # FFT length used (after padding)
            "n_sanitized": int,     # number of NaN/Inf samples zeroed
            "detrend": str,
            "window": str|ndarray,
            "normalization": str,
            "welch": {
                "enabled": bool,
                "nperseg": Optional[int],
                "noverlap": Optional[int],
                "n_segments": Optional[int],
            }
        }
    """
    if dt <= 0:
        raise ValueError("dt must be positive")

    x = _as_2d(lightcurve)  # [N, T]
    N, T = x.shape

    # Sanitize invalids once, and record count
    x, n_sanitized = _apply_invalid_sanitization(x)

    # Global detrend & window
    x = _detrend(x, detrend)  # [N, T]
    w_global = _make_window(T, window)
    xw = x * w_global[None, :]

    # Welch path
    if welch_nperseg is not None:
        freqs, power, n_segments = _welch(
            xw, dt=dt, window=window, nperseg=int(welch_nperseg),
            noverlap=welch_noverlap, n_fft=None, norm=normalization
        )
        fft_vals = None
        n_fft = (freqs.shape[0] - 1) * 2  # implied for rfft
    else:
        # Single-shot rFFT with optional zero padding
        n_fft = _pad_length(T, pad_to)
        fft_vals = _rfft_batched(xw, n_fft=n_fft)  # [N, F]
        if normalization == "per_sample":
            power = _per_sample_power(fft_vals, n_eff=T)
        elif normalization == "per_hz":
            power = _per_hz_power(fft_vals, n_eff=T, dt=dt)
        else:
            raise ValueError(f"Unknown normalization: {normalization!r}")
        freqs = np.fft.rfftfreq(n_fft, d=dt)

    # Squeeze batch if N==1
    if power.shape[0] == 1:
        power = power[0]
        if fft_vals is not None:
            fft_vals = fft_vals[0]

    out: Dict[str, Any] = {
        "freqs": freqs,
        "power": power,
        "fft_vals": fft_vals,           # may be None when Welch enabled
        "dt": float(dt),
        "n_samples": int(T),
        "freq_res": float(1.0 / (n_fft * dt)),
        "nyquist": float(0.5 / dt),
        "n_fft": int(n_fft),
        "n_sanitized": int(n_sanitized),
        "detrend": detrend,
        "window": "hann" if (isinstance(window, str) and window == "hann") else window,
        "normalization": normalization,
        "welch": {
            "enabled": welch_nperseg is not None,
            "nperseg": int(welch_nperseg) if welch_nperseg is not None else None,
            "noverlap": int(welch_noverlap) if welch_noverlap is not None else None,
            "n_segments": int(n_segments) if welch_nperseg is not None else None,
        },
    }
    return out