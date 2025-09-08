# =============================================================================
# SpectraMind V50 — Phase-locked systematics modeling & correction
# =============================================================================
# Removes periodic (phase-locked) patterns from time-series cubes [..., T, H, W].
# Supports harmonic regression + phase-binned template, NaN-safe, NumPy/Torch.
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

# ----------------------------------------------------------------------------- #
# Backend shims
# ----------------------------------------------------------------------------- #

def _is_torch(x: BackendArray) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "torch"

def _np():
    import numpy as np; return np

def _torch():
    import torch; return torch

def _to_float(x: BackendArray, dtype: Optional[Union[str, Any]] = None) -> BackendArray:
    if _is_torch(x):
        return x.to(getattr(_torch(), dtype or "float32"))
    np = _np()
    return x.astype(getattr(np, dtype) if isinstance(dtype, str) else (dtype or np.float32), copy=False)

def _nanmean(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum(axis, keepdim=keepdims)
        den = mask.sum(axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        return torch.where(den == 0, torch.tensor(float("nan"), dtype=x.dtype, device=x.device), out)
    return _np().nanmean(x, axis=axis, keepdims=keepdims)

def _clip(x: BackendArray, lo: Optional[float], hi: Optional[float]) -> BackendArray:
    if _is_torch(x):
        return _torch().clamp(x, min=float(lo) if lo is not None else None, max=float(hi) if hi is not None else None)
    return _np().clip(x, lo, hi)

# ----------------------------------------------------------------------------- #
# Phase math + smoothing
# ----------------------------------------------------------------------------- #

def _compute_phase(times: BackendArray, period: float, t0: float, *, wrap=True) -> BackendArray:
    if _is_torch(times):
        phi = 2.0 * _torch().pi * ((times - t0) / period)
        return phi.remainder(2.0 * _torch().pi) if wrap else phi
    np = _np()
    phi = 2.0 * np.pi * ((times - t0) / period)
    return np.mod(phi, 2.0 * np.pi) if wrap else phi

def _savitzky_golay_1d(y, win: int, poly: int):
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=win, polyorder=poly, mode="nearest")
    except Exception:
        import numpy as np
        if win <= 1: return y
        k, pad = win, win // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        return np.convolve(yp, np.ones(k) / k, mode="valid")

def _smooth_template(vec: BackendArray, win: int, poly: int) -> BackendArray:
    if win <= 1: return vec
    if _is_torch(vec):
        t_np = vec.detach().cpu().numpy()
        sm = _savitzky_golay_1d(t_np, win, poly)
        return _torch().from_numpy(sm).to(device=vec.device, dtype=vec.dtype)
    return _savitzky_golay_1d(vec, win, poly)

# ----------------------------------------------------------------------------- #
# Dataclasses
# ----------------------------------------------------------------------------- #

PhaseTemplate = Literal["none", "bins"]

@dataclass
class PhaseParams:
    period: float; t0: float
    time_axis: int = -3
    harmonics: int = 3
    include_dc: bool = True
    template: PhaseTemplate = "none"
    n_bins: int = 32
    smooth_window: int = 0
    smooth_poly: int = 2
    mask_saturated: Optional[BackendArray] = None
    mask_hot: Optional[BackendArray] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class PhaseModel:
    coeffs: Optional[BackendArray]
    basis_info: Dict[str, Any]
    template_bins: Optional[BackendArray]
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PhaseApplyParams:
    subtract_model: bool = True
    subtract_template: bool = False
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class PhaseApplyResult:
    corrected: BackendArray
    model_contrib: Optional[BackendArray]
    template_contrib: Optional[BackendArray]
    meta: Dict[str, Any] = field(default_factory=dict)

# ----------------------------------------------------------------------------- #
# Build & Apply
# ----------------------------------------------------------------------------- #

def build_phase_model(frames: BackendArray, times: BackendArray, params: PhaseParams):
    F = _to_float(frames, dtype=params.dtype)
    F, _ = _move_time_axis(F, params.time_axis)
    T, H, W = F.shape[-3:]
    phi = _compute_phase(times, params.period, params.t0, wrap=True)
    # Harmonic regression
    coeffs, X = None, _design_matrix(phi, params.harmonics, params.include_dc)
    if X is not None:
        Fm = _apply_masks(F, params.mask_saturated, params.mask_hot)
        mu = _nanmean(Fm, axis=-3, keepdims=True)
        y_filled = _where(Fm != Fm, mu, Fm)  # fill NaNs
        coeffs = _solve_least_squares(X, y_filled)
    bins = None
    if params.template == "bins":
        bins = _build_phase_bins(phi, T, H, W, params.n_bins, F)
        if params.smooth_window > 1:
            for h in range(H):
                for w in range(W):
                    bins[:, h, w] = _smooth_template(bins[:, h, w], params.smooth_window, params.smooth_poly)
    return PhaseModel(coeffs, {"harmonics": params.harmonics, "include_dc": params.include_dc}, bins,
                      {"period": params.period, "t0": params.t0, "n_bins": params.n_bins})

def apply_phase_correction(frames, times, model: PhaseModel, pb: PhaseParams, pa: PhaseApplyParams) -> PhaseApplyResult:
    F = _to_float(frames, dtype=pa.dtype)
    F, _ = _move_time_axis(F, pb.time_axis)
    phi = _compute_phase(times, pb.period, pb.t0, wrap=True)
    model_contrib = _predict_from_beta(_design_matrix(phi, pb.harmonics, pb.include_dc), model.coeffs) if pa.subtract_model and model.coeffs is not None else None
    template_contrib = _template_contrib_from_bins(phi, model.template_bins) if pa.subtract_template and model.template_bins is not None else None
    corrected = F - (model_contrib or 0) - (template_contrib or 0)
    if pa.clip_out: corrected = _clip(corrected, *pa.clip_out)
    return PhaseApplyResult(corrected, model_contrib, template_contrib, {"subtract_model": pa.subtract_model})