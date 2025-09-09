
from __future__ import annotations
import numpy as np
import pandas as pd
from .base import ValidationResult, ValidationError, ok

def _as_arrays(df_or_arr) -> tuple[np.ndarray, np.ndarray]:
    if isinstance(df_or_arr, pd.DataFrame):
        mu_cols = [c for c in df_or_arr.columns if c.startswith("mu_")]
        sg_cols = [c for c in df_or_arr.columns if c.startswith("sigma_")]
        mu = df_or_arr[sorted(mu_cols)].to_numpy(dtype=float, copy=False)
        sg = df_or_arr[sorted(sg_cols)].to_numpy(dtype=float, copy=False)
        return mu, sg
    mu, sg = df_or_arr  # assume (mu, sigma) arrays
    return np.asarray(mu, float), np.asarray(sg, float)

def check_sigma_positive(df_or_arr, eps: float = 0.0) -> ValidationResult:
    mu, sg = _as_arrays(df_or_arr)
    bad = np.where(~(sg > eps))
    if bad[0].size:
        return ValidationResult(False, [ValidationError("sigma not strictly positive", {
            "count": int(bad[0].size), "first_idx": (int(bad[0][0]), int(bad[1][0]))
        })])
    return ok()

def check_mu_nonnegative(df_or_arr, tol: float = -1e-12) -> ValidationResult:
    mu, _ = _as_arrays(df_or_arr)
    bad = np.where(mu < tol)
    if bad[0].size:
        return ValidationResult(False, [ValidationError("mu has negative entries", {
            "count": int(bad[0].size), "first_idx": (int(bad[0][0]), int(bad[1][0])), "tol": tol
        })])
    return ok()

def check_smoothness_tv(df_or_arr, tv_threshold: float = 5.0) -> ValidationResult:
    mu, _ = _as_arrays(df_or_arr)
    # total variation along bin axis
    tv = np.abs(np.diff(mu, axis=1)).sum(axis=1)
    bad = np.where(tv > tv_threshold)[0]
    if bad.size:
        return ValidationResult(False, [ValidationError("excessive total variation", {
            "n_bad": int(bad.size), "max_tv": float(tv[bad].max()), "threshold": tv_threshold
        })])
    return ok()

def check_spike_robust_zscore(df_or_arr, zmax: float = 8.0, window: int = 9) -> ValidationResult:
    mu, _ = _as_arrays(df_or_arr)
    # rolling median & MAD
    pad = window // 2
    med = np.array([np.convolve(row, np.ones(window)/window, mode="same") for row in mu])
    mad = np.median(np.abs(mu - med), axis=1, keepdims=True) + 1e-12
    z = np.abs(mu - med) / mad
    n_spikes = int((z > zmax).sum())
    if n_spikes:
        return ValidationResult(False, [ValidationError("spiky spectrum (robust z-score)", {
            "spikes": n_spikes, "zmax": zmax, "window": window
        })])
    return ok()
