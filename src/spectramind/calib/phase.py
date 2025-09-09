# src/spectramind/calib/phase.py
# =============================================================================
# SpectraMind V50 — Phase-locked systematics modeling & correction
# -----------------------------------------------------------------------------
# Removes periodic (phase-locked) patterns from time-series cubes [..., T, H, W].
# - Harmonic regression (sin/cos up to K harmonics) with optional DC term
# - Optional phase-binned template (with Savitzky–Golay or mean smoothing)
# - NaN-safe, mask-aware (saturated/hot), backend-agnostic (NumPy/Torch)
# - Vectorized least-squares via pinv(X); dtype/device preserved
# =============================================================================

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal, cast

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

# ----------------------------------------------------------------------------- #
# Backend shims
# ----------------------------------------------------------------------------- #

def _is_torch(x: BackendArray) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "torch"

def _np():
    import numpy as np
    return np

def _torch():
    import torch
    return torch

def _resolve_dtype(backend: str, dtype: Optional[Union[str, Any]]) -> Any:
    if dtype is None:
        return None
    if backend == "torch":
        torch = _torch()
        if isinstance(dtype, str):
            m = {
                "float32": torch.float32, "float": torch.float32,
                "float64": torch.float64, "double": torch.float64,
                "float16": torch.float16, "half": torch.float16, "bfloat16": torch.bfloat16,
            }
            return m.get(dtype.lower(), getattr(torch, dtype))
        return dtype
    np = _np()
    if isinstance(dtype, str):
        return getattr(np, dtype)
    return dtype

def _to_float(x: BackendArray, dtype: Optional[Union[str, Any]] = None) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        target = _resolve_dtype("torch", dtype) or torch.float32
        return x.to(target)
    else:
        np = _np()
        target = _resolve_dtype("numpy", dtype) or np.float32
        return x.astype(target, copy=False)

def _nanmean(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        if hasattr(torch, "nanmean"):
            return torch.nanmean(x, dim=axis, keepdim=keepdims)  # type: ignore[attr-defined]
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, torch.zeros_like(x)).sum(dim=axis, keepdim=keepdims)
        den = mask.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        return torch.where(den == 0, torch.full_like(out, float("nan")), out)
    return _np().nanmean(x, axis=axis, keepdims=keepdims)

def _clip(x: BackendArray, lo: Optional[float], hi: Optional[float]) -> BackendArray:
    if lo is None and hi is None:
        return x
    if _is_torch(x):
        torch = _torch()
        if lo is not None:
            x = torch.clamp(x, min=float(lo))
        if hi is not None:
            x = torch.clamp(x, max=float(hi))
        return x
    return _np().clip(x, lo, hi)

def _where(mask: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(mask) or _is_torch(a) or _is_torch(b):
        return _torch().where(mask, a, b)
    return _np().where(mask, a, b)

def _abs(x: BackendArray) -> BackendArray:
    return x.abs() if _is_torch(x) else _np().abs(x)

# ----------------------------------------------------------------------------- #
# Axis helpers & masks
# ----------------------------------------------------------------------------- #

def _move_time_axis(x: BackendArray, time_axis: int) -> Tuple[BackendArray, int]:
    """
    Return (x', tpos) with time axis moved to position -3 → [..., T, H, W].
    """
    nd = x.ndim
    tpos = nd - 3
    src = time_axis if time_axis >= 0 else nd + time_axis
    if src == tpos:
        return x, tpos
    if _is_torch(x):
        perm = list(range(nd))
        perm[tpos], perm[src] = perm[src], perm[tpos]
        return x.permute(*perm), tpos
    else:
        return x.swapaxes(tpos, src), tpos

def _apply_masks(frames: BackendArray,
                 mask_sat: Optional[BackendArray],
                 mask_hot: Optional[BackendArray]) -> BackendArray:
    if mask_sat is None and mask_hot is None:
        return frames
    out = frames.clone() if _is_torch(frames) else frames.copy()
    if mask_sat is not None:
        if _is_torch(out):
            torch = _torch()
            out = torch.where(mask_sat, torch.full_like(out, float("nan")), out)
        else:
            out[cast("np.ndarray", mask_sat)] = _np().nan  # type: ignore[index]
    if mask_hot is not None:
        if _is_torch(out):
            torch = _torch()
            out = torch.where(mask_hot, torch.full_like(out, float("nan")), out)
        else:
            out[cast("np.ndarray", mask_hot)] = _np().nan  # type: ignore[index]
    return out

# ----------------------------------------------------------------------------- #
# Phase math + smoothing
# ----------------------------------------------------------------------------- #

def _compute_phase(times: BackendArray, period: float, t0: float, *, wrap=True) -> BackendArray:
    if _is_torch(times):
        torch = _torch()
        phi = 2.0 * torch.pi * ((times - t0) / period)
        return phi.remainder(2.0 * torch.pi) if wrap else phi
    np = _np()
    phi = 2.0 * np.pi * ((times - t0) / period)
    return np.mod(phi, 2.0 * np.pi) if wrap else phi

def _savitzky_golay_1d(y, win: int, poly: int):
    try:
        from scipy.signal import savgol_filter  # type: ignore
        return savgol_filter(y, window_length=max(1, int(win) | 1), polyorder=poly, mode="nearest")
    except Exception:
        import numpy as np
        if win <= 1:
            return y
        k = max(1, int(win) | 1)
        pad = k // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        return np.convolve(yp, np.ones(k, dtype=np.float64) / k, mode="valid").astype(y.dtype, copy=False)

def _smooth_template(vec: BackendArray, win: int, poly: int) -> BackendArray:
    if win <= 1:
        return vec
    if _is_torch(vec):
        torch = _torch()
        t_np = vec.detach().cpu().numpy()
        sm = _savitzky_golay_1d(t_np, win, poly)
        return torch.from_numpy(sm).to(device=vec.device, dtype=vec.dtype)
    return _savitzky_golay_1d(vec, win, poly)

# ----------------------------------------------------------------------------- #
# Dataclasses
# ----------------------------------------------------------------------------- #

PhaseTemplate = Literal["none", "bins"]

@dataclass
class PhaseParams:
    period: float
    t0: float
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
# Design matrix / regression helpers
# ----------------------------------------------------------------------------- #

def _design_matrix(phi: BackendArray, K: int, include_dc: bool) -> BackendArray:
    """
    Build design matrix X (T, P) with columns: [1], sin(k*phi), cos(k*phi), k=1..K
    """
    if K < 0:
        raise ValueError("harmonics must be >= 0")
    T = phi.shape[0]
    cols = 0
    if include_dc:
        cols += 1
    cols += 2 * K
    if _is_torch(phi):
        torch = _torch()
        X = torch.empty((T, cols), dtype=phi.dtype, device=phi.device)
        idx = 0
        if include_dc:
            X[:, idx] = 1.0
            idx += 1
        for k in range(1, K + 1):
            X[:, idx] = torch.sin(k * phi); idx += 1
            X[:, idx] = torch.cos(k * phi); idx += 1
        return X
    else:
        np = _np()
        X = np.empty((T, cols), dtype=phi.dtype)
        idx = 0
        if include_dc:
            X[:, idx] = 1.0
            idx += 1
        for k in range(1, K + 1):
            X[:, idx] = np.sin(k * phi); idx += 1
            X[:, idx] = np.cos(k * phi); idx += 1
        return X

def _pinv(mat: BackendArray) -> BackendArray:
    if _is_torch(mat):
        torch = _torch()
        return torch.pinverse(mat)
    else:
        np = _np()
        return np.linalg.pinv(mat)

def _solve_least_squares(X: BackendArray, Y: BackendArray) -> BackendArray:
    """
    Solve beta = pinv(X) @ Y
    X: (T,P), Y: (T,H,W)
    Returns beta: (P,H,W)
    """
    if _is_torch(X) or _is_torch(Y):
        torch = _torch()
        X_t = X if _is_torch(X) else torch.as_tensor(X)
        Y_t = Y if _is_torch(Y) else torch.as_tensor(Y, dtype=X_t.dtype, device=X_t.device)
        pinvX = torch.pinverse(X_t)              # (P,T)
        beta = torch.matmul(pinvX, Y_t)          # (P,H,W)
        return beta
    else:
        np = _np()
        pinvX = np.linalg.pinv(X)                # (P,T)
        beta = pinvX @ Y                         # (P,H,W)
        return beta

def _predict_from_beta(X: BackendArray, beta: BackendArray) -> BackendArray:
    """
    X: (T,P), beta: (P,H,W) -> yhat: (T,H,W)
    """
    if _is_torch(X) or _is_torch(beta):
        torch = _torch()
        X_t = X if _is_torch(X) else torch.as_tensor(X, dtype=beta.dtype, device=beta.device)
        beta_t = beta if _is_torch(beta) else torch.as_tensor(beta, dtype=X_t.dtype, device=X_t.device)
        return torch.matmul(X_t, beta_t)  # (T,H,W)
    else:
        np = _np()
        return X @ beta

# ----------------------------------------------------------------------------- #
# Phase-binned template helpers
# ----------------------------------------------------------------------------- #

def _build_phase_bins(phi: BackendArray, T: int, H: int, W: int, n_bins: int, F: BackendArray) -> BackendArray:
    """
    Create phase-binned template: for each bin b, mean over time indices in bin.
    Returns bins: (n_bins, H, W)
    """
    two_pi = 2.0 * (_torch().pi if _is_torch(phi) else _np().pi)
    if _is_torch(phi):
        torch = _torch()
        edges = torch.linspace(0.0, float(two_pi), steps=n_bins + 1, device=phi.device, dtype=phi.dtype)
        idx = torch.bucketize(phi, edges) - 1
        idx = torch.clamp(idx, 0, n_bins - 1)  # [T]
        bins = torch.full((n_bins, H, W), float("nan"), dtype=F.dtype, device=F.device)
        for b in range(n_bins):
            mask_t = idx == b
            if mask_t.any():
                # mean over selected times (dim=-3)
                sel = F[..., mask_t, :, :]
                m = _nanmean(sel, axis=-3, keepdims=False)
                bins[b] = m
        return bins
    else:
        np = _np()
        edges = np.linspace(0.0, float(two_pi), num=n_bins + 1, dtype=phi.dtype)
        idx = np.digitize(phi, edges) - 1
        idx = np.clip(idx, 0, n_bins - 1)
        bins = np.full((n_bins, H, W), np.nan, dtype=F.dtype)
        for b in range(n_bins):
            mask_t = idx == b
            if mask_t.any():
                sel = F[..., mask_t, :, :]
                m = _nanmean(sel, axis=-3, keepdims=False)
                bins[b] = m
        return bins

def _template_contrib_from_bins(phi: BackendArray, bins: BackendArray) -> BackendArray:
    """
    Map bins (B,H,W) to per-time contribution (T,H,W) by selecting bin value for each time step.
    """
    n_bins = bins.shape[0]
    two_pi = 2.0 * (_torch().pi if _is_torch(phi) else _np().pi)
    if _is_torch(phi) or _is_torch(bins):
        torch = _torch()
        phi_t = phi if _is_torch(phi) else torch.as_tensor(phi, dtype=bins.dtype, device=bins.device)
        idx = torch.floor(phi_t / (two_pi / n_bins)).to(torch.int64).clamp(0, n_bins - 1)  # (T,)
        # Gather per time-step
        # bins: (B,H,W), idx: (T,), we broadcast to (T,H,W) by indexing along 0
        out = bins.index_select(0, idx)  # (T,H,W)
        return out
    else:
        np = _np()
        idx = np.floor(phi / (two_pi / n_bins)).astype(np.int64)
        idx = np.clip(idx, 0, n_bins - 1)
        return bins[idx, ...]  # (T,H,W)

# ----------------------------------------------------------------------------- #
# Build & Apply
# ----------------------------------------------------------------------------- #

def build_phase_model(frames: BackendArray, times: BackendArray, params: PhaseParams) -> PhaseModel:
    """
    Fit harmonic phase-locked model and (optionally) build a phase-binned template.
    frames: [..., T, H, W] (time axis per params.time_axis)
    times : (T,) timestamps (same backend as frames or convertible)
    """
    F = _to_float(frames, dtype=params.dtype)
    F, _ = _move_time_axis(F, params.time_axis)      # [..., T, H, W]
    T, H, W = F.shape[-3], F.shape[-2], F.shape[-1]

    # phase for each time index (T,)
    tvec = _to_float(times, dtype=params.dtype)
    phi = _compute_phase(tvec, params.period, params.t0, wrap=True)

    # Design matrix (T,P)
    X = _design_matrix(phi, params.harmonics, params.include_dc)

    # Apply masks and fill NaNs with time-mean to stabilize LS
    Fm = _apply_masks(F, params.mask_saturated, params.mask_hot)
    mu = _nanmean(Fm, axis=-3, keepdims=True)
    y_filled = _where(_abs(Fm - Fm) > 0, mu, Fm)  # replace NaNs with mean

    # Solve beta = pinv(X) @ y_filled
    coeffs = _solve_least_squares(X, y_filled)  # (P,H,W)

    # Optional phase-binned template
    bins = None
    if params.template == "bins":
        bins = _build_phase_bins(phi, T, H, W, int(params.n_bins), Fm)
        if params.smooth_window and params.smooth_window > 1:
            if _is_torch(bins):
                for h in range(H):
                    for w in range(W):
                        bins[:, h, w] = _smooth_template(bins[:, h, w], params.smooth_window, params.smooth_poly)
            else:
                for h in range(H):
                    for w in range(W):
                        bins[:, h, w] = _smooth_template(bins[:, h, w], params.smooth_window, params.smooth_poly)

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update(
            dict(
                period=float(params.period),
                t0=float(params.t0),
                harmonics=int(params.harmonics),
                include_dc=bool(params.include_dc),
                template=params.template,
                n_bins=int(params.n_bins),
            )
        )

    basis_info = {"harmonics": params.harmonics, "include_dc": params.include_dc}
    return PhaseModel(coeffs=coeffs, basis_info=basis_info, template_bins=bins, meta=meta)


def apply_phase_correction(frames: BackendArray,
                           times: BackendArray,
                           model: PhaseModel,
                           pb: PhaseParams,
                           pa: PhaseApplyParams) -> PhaseApplyResult:
    """
    Apply a previously built phase model/template to correct the data.
    """
    F = _to_float(frames, dtype=pa.dtype)
    F, _ = _move_time_axis(F, pb.time_axis)
    T = F.shape[-3]

    # Phases and basis
    tvec = _to_float(times, dtype=pa.dtype or pb.dtype)
    phi = _compute_phase(tvec, pb.period, pb.t0, wrap=True)
    X = _design_matrix(phi, pb.harmonics, pb.include_dc)

    model_contrib = None
    if pa.subtract_model and (model.coeffs is not None):
        model_contrib = _predict_from_beta(X, model.coeffs)  # (T,H,W)

    template_contrib = None
    if pa.subtract_template and (model.template_bins is not None):
        template_contrib = _template_contrib_from_bins(phi, model.template_bins)  # (T,H,W)

    corrected = F
    if model_contrib is not None:
        corrected = corrected - model_contrib
    if template_contrib is not None:
        corrected = corrected - template_contrib

    if pa.clip_out is not None:
        corrected = _clip(corrected, *pa.clip_out)

    meta: Dict[str, Any] = {}
    if pa.return_intermediate:
        meta.update(
            dict(
                subtract_model=pa.subtract_model,
                subtract_template=pa.subtract_template,
                clip_out=pa.clip_out,
            )
        )

    return PhaseApplyResult(corrected=corrected,
                            model_contrib=model_contrib,
                            template_contrib=template_contrib,
                            meta=meta)

# ----------------------------------------------------------------------------- #
# Minimal self-tests (NumPy)
# ----------------------------------------------------------------------------- #

def _test_phase_basic() -> bool:
    import numpy as np
    rng = np.random.default_rng(0)
    T, H, W = 256, 4, 5
    period, t0 = 10.0, 0.3
    t = np.linspace(0, (T - 1) * 0.5, T).astype(np.float32)  # dt=0.5
    phi = 2*np.pi * (t - t0) / period
    # True model: A*sin(phi)+B*cos(phi)+C
    A, B, C = 2.0, -1.0, 0.5
    model = (A*np.sin(phi) + B*np.cos(phi) + C).astype(np.float32)[:, None, None]
    noise = 0.1 * rng.standard_normal((T, H, W)).astype(np.float32)
    cube = model + noise

    pp = PhaseParams(period=period, t0=t0, harmonics=1, include_dc=True, template="none")
    mdl = build_phase_model(cube, t, pp)
    pa = PhaseApplyParams(subtract_model=True, subtract_template=False)
    out = apply_phase_correction(cube, t, mdl, pp, pa)
    # After removing model, residual should be near zero-mean noise
    m = float(np.nanmean(out.corrected))
    std = float(np.nanstd(out.corrected))
    assert abs(m) < 0.1 and std < 0.3
    return True


if __name__ == "__main__":
    ok = _test_phase_basic()
    print("phase.py self-tests:", ok)