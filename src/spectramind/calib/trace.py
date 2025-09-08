# src/spectramind/calib/trace.py
# =============================================================================
# SpectraMind V50 — Spectral trace detection, modeling & optimal extraction
# -----------------------------------------------------------------------------
# Key upgrades (this revision)
#   • Robust axis normalization to [..., T, Y, X] (works with any prefixes).
#   • Multi-order models: center/width may be [X] or [O, X]; extraction returns
#     [..., T, X] for single-order or [..., T, O, X] for multi-order.
#   • Full NaN-safe math, including nansum helpers for Torch/NumPy.
#   • Background models: per-column median or per-column row-polynomial (deg k).
#   • Horne-style optimal extraction w/ variance propagation; stable denominators.
#   • CPU fallback only when absolutely necessary (e.g., polyfit on Torch).
#   • Self-tests kept light & CPU-safe; no I/O; Kaggle/CI friendly.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

__all__ = [
    "TraceDetectParams",
    "TraceExtractParams",
    "TraceModel",
    "TraceBuildResult",
    "TraceExtractResult",
    "build_trace_model",
    "extract_trace",
]

# -----------------------------------------------------------------------------
# Backend shims (aligned with other calib modules)
# -----------------------------------------------------------------------------

def _is_torch(x: BackendArray) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "torch"

def _np() -> Any:
    import numpy as np
    return np

def _torch() -> Any:
    import torch
    return torch

def _to_float(x: BackendArray, dtype: Optional[Union[str, Any]] = None) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        tdtype = getattr(torch, dtype) if isinstance(dtype, str) else (dtype or torch.float32)
        return x.to(tdtype)
    else:
        np = _np()
        ndtype = getattr(np, dtype) if isinstance(dtype, str) else (dtype or np.float32)
        return x.astype(ndtype, copy=False)

def _zeros_like(x: BackendArray) -> BackendArray:
    return _torch().zeros_like(x) if _is_torch(x) else _np().zeros_like(x)

def _ones_like(x: BackendArray) -> BackendArray:
    return _torch().ones_like(x) if _is_torch(x) else _np().ones_like(x)

def _where(mask: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    return _torch().where(mask, a, b) if _is_torch(mask) else _np().where(mask, a, b)

def _abs(x: BackendArray) -> BackendArray:
    return x.abs() if _is_torch(x) else _np().abs(x)

def _nan_to_num(x: BackendArray, val=0.0) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        # torch.nan_to_num exists on modern Torch
        return torch.nan_to_num(x, nan=val) if hasattr(torch, "nan_to_num") else torch.where(torch.isnan(x), torch.tensor(val, dtype=x.dtype, device=x.device), x)
    else:
        return _np().nan_to_num(x, nan=val)

def _nansum(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        if hasattr(torch, "nansum"):
            return torch.nansum(x, dim=axis, keepdim=keepdims) if axis is not None else torch.nansum(x)
        return _nan_to_num(x, 0.0).sum(dim=axis, keepdim=keepdims) if axis is not None else _nan_to_num(x, 0.0).sum()
    else:
        return _np().nansum(x, axis=axis, keepdims=keepdims)

def _nanmean(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        m = ~torch.isnan(x)
        num = torch.where(m, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims) if axis is not None else torch.where(m, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum()
        den = m.sum(dim=axis, keepdim=keepdims) if axis is not None else m.sum()
        den = den.clamp_min(1)
        out = num / den
        if axis is None:
            return torch.where((m.sum() == 0), torch.tensor(float("nan"), dtype=x.dtype, device=x.device), out)
        return out
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)

def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        if hasattr(torch, "nanmedian"):
            return torch.nanmedian(x, dim=axis, keepdim=keepdims).values if axis is not None else torch.nanmedian(x)
        # CPU fallback
        np = _np()
        xn = x.detach().cpu().numpy()
        out = np.nanmedian(xn, axis=axis, keepdims=keepdims)
        return torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
    else:
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)

def _nanstd(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        mu = _nanmean(x, axis=axis, keepdims=True)
        v = _nanmean((x - mu) ** 2, axis=axis, keepdims=keepdims)
        return torch.sqrt(v)
    else:
        return _np().nanstd(x, axis=axis, keepdims=keepdims)

def _clip(x: BackendArray, low: Optional[float], high: Optional[float]) -> BackendArray:
    if low is None and high is None:
        return x
    if _is_torch(x):
        torch = _torch()
        if low is not None: x = torch.clamp(x, min=float(low))
        if high is not None: x = torch.clamp(x, max=float(high))
        return x
    else:
        return _np().clip(x, low, high)

def _nan_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.full_like(x, float("nan"))
    else:
        import numpy as np
        return np.full_like(x, np.nan)

# -----------------------------------------------------------------------------
# Configuration dataclasses
# -----------------------------------------------------------------------------

PolyFitKind = Literal["poly", "median"]
BkgKind = Literal["none", "column_median", "row_poly"]
ExtractMethod = Literal["optimal", "box"]

@dataclass
class TraceDetectParams:
    """
    Trace detection configuration (per order).
    """
    dispersion_axis: int = -1
    smooth_cols: int = 5
    center_poly_deg: int = 3
    width_poly_deg: int = 2
    fit_kind: PolyFitKind = "poly"
    cr_reject: bool = True
    cr_zmax: float = 6.0
    cr_iter: int = 1
    y_search_half: int = 15
    initial_center: Optional[float] = None
    mask_saturated: Optional[BackendArray] = None
    mask_hot: Optional[BackendArray] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class TraceExtractParams:
    """
    Extraction configuration (per order or multi-order).
    """
    method: ExtractMethod = "optimal"
    ap_half: int = 4
    bkg_kind: BkgKind = "column_median"
    bkg_poly_deg: int = 2
    psf_sigma_y: float = 1.6
    var: Optional[BackendArray] = None
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class TraceModel:
    """
    Trace model:
      center: [X] or [O, X]      (row center vs column)
      width : [X] or [O, X]      (sigma_y vs column)
      *_poly: polynomial coeffs if fit_kind='poly' (descending degree)
    """
    center: BackendArray
    width: BackendArray
    center_poly: Optional[BackendArray]
    width_poly: Optional[BackendArray]
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TraceBuildResult:
    model: TraceModel

@dataclass
class TraceExtractResult:
    flux: BackendArray           # [..., T, X] or [..., T, O, X]
    flux_err: Optional[BackendArray]
    bkg: Optional[BackendArray]  # [..., T, X] (column background) or None
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Axis helpers
# -----------------------------------------------------------------------------

def _normalize_axes(frames: BackendArray, time_axis: int, disp_axis: int) -> Tuple[BackendArray, int, int]:
    """
    Reorder to canonical [..., T, Y, X]; returns (x, T_index=-3, X_index=-1).
    """
    x = frames
    nd = x.ndim
    tgt_T, tgt_X = nd - 3, nd - 1

    if time_axis < 0: time_axis = nd + time_axis
    if disp_axis < 0: disp_axis = nd + disp_axis

    # Move time to tgt_T
    if time_axis != tgt_T:
        perm = list(range(nd))
        perm[tgt_T], perm[time_axis] = perm[time_axis], perm[tgt_T]
        x = x.permute(*perm) if _is_torch(x) else x.transpose(tgt_T, time_axis)
        # adjust disp_axis if it moved
        if disp_axis == tgt_T:
            disp_axis = time_axis
        elif disp_axis == time_axis:
            disp_axis = tgt_T
        time_axis = tgt_T

    # Move dispersion to tgt_X
    if disp_axis != tgt_X:
        perm = list(range(nd))
        perm[tgt_X], perm[disp_axis] = perm[disp_axis], perm[tgt_X]
        x = x.permute(*perm) if _is_torch(x) else x.transpose(tgt_X, disp_axis)
        disp_axis = tgt_X

    return x, -3, -1  # [..., T, Y, X]

# -----------------------------------------------------------------------------
# Detection primitives
# -----------------------------------------------------------------------------

def _sigma_clip_1d(y: BackendArray, lo: float, hi: float, iters: int) -> BackendArray:
    out = y
    for _ in range(max(1, iters)):
        med = _nanmedian(out, axis=-1, keepdims=True)
        std = _nanstd(out, axis=-1, keepdims=True)
        lo_thr = med - lo * std
        hi_thr = med + hi * std
        out = _where((out < lo_thr) | (out > hi_thr), _nan_like(out), out)
    return out

def _column_stats_y(frame: BackendArray, cr_reject: bool, zmax: float, iters: int,
                    y_search_half: int, init_center: Optional[float]) -> Tuple[BackendArray, BackendArray]:
    """
    Single 2D frame [Y, X] → per-column centroid (row) and width (sigma_y).
    Restrict search window around init_center ± y_search_half if given.
    """
    torch_mode = _is_torch(frame)
    Y, X = frame.shape[-2], frame.shape[-1]

    if torch_mode:
        torch = _torch()
        rows = torch.arange(Y, dtype=frame.dtype, device=frame.device)
        yy = rows[:, None].expand(Y, X)
        if init_center is not None:
            y0 = max(0, int(init_center - y_search_half))
            y1 = min(Y, int(init_center + y_search_half) + 1)
            img = frame[y0:y1, :]
            yyw = yy[y0:y1, :]
        else:
            img = frame
            yyw = yy
        if cr_reject:
            img = _sigma_clip_1d(img, lo=zmax, hi=zmax, iters=iters)
        w = _nan_to_num(img, 0.0)
        s0 = w.sum(dim=-2) + 1e-20          # [X]
        s1 = (w * yyw).sum(dim=-2)          # [X]
        mu = s1 / s0
        var = (w * (yyw - mu[None, :]) ** 2).sum(dim=-2) / s0
        sig = (var.clamp_min(1e-10)).sqrt()
        return mu, sig
    else:
        np = _np()
        rows = np.arange(Y, dtype=frame.dtype)
        yy = np.tile(rows[:, None], (1, X))
        if init_center is not None:
            y0 = max(0, int(init_center - y_search_half))
            y1 = min(Y, int(init_center + y_search_half) + 1)
            img = frame[y0:y1, :]
            yyw = yy[y0:y1, :]
        else:
            img = frame
            yyw = yy
        if cr_reject:
            img = _sigma_clip_1d(img, lo=zmax, hi=zmax, iters=iters)
        w = _nan_to_num(img, 0.0)
        s0 = np.sum(w, axis=0) + 1e-20
        s1 = np.sum(w * yyw, axis=0)
        mu = s1 / s0
        var = np.sum(w * (yyw - mu[None, :]) ** 2, axis=0) / s0
        sig = np.sqrt(np.clip(var, 1e-10, None))
        return mu, sig

def _polyfit_1d(x: BackendArray, y: BackendArray, deg: int) -> BackendArray:
    """
    Polynomial coefficients (descending degree). Torch → NumPy fallback for fit.
    """
    if deg <= 0:
        c = _nanmedian(y)
        return c[..., None] if _is_torch(c) else c[..., None]
    if _is_torch(x) or _is_torch(y):
        np = _np(); torch = _torch()
        xv = x.detach().cpu().numpy() if _is_torch(x) else x
        yv = y.detach().cpu().numpy() if _is_torch(y) else y
        yv_f = np.where(np.isnan(yv), np.nanmedian(yv), yv)
        coef = np.polyfit(xv, yv_f, deg=deg)
        device = y.device if _is_torch(y) else (x.device if _is_torch(x) else "cpu")
        dtype = y.dtype if _is_torch(y) else (x.dtype if _is_torch(x) else None)
        return torch.from_numpy(coef).to(device=device, dtype=dtype)
    else:
        np = _np()
        y_f = np.where(np.isnan(y), np.nanmedian(y), y)
        return np.polyfit(x, y_f, deg=deg)

def _polyval_1d(coef: BackendArray, x: BackendArray) -> BackendArray:
    if _is_torch(coef) or _is_torch(x):
        torch = _torch()
        c = coef if _is_torch(coef) else torch.from_numpy(coef).to(device=x.device if _is_torch(x) else "cpu", dtype=(x.dtype if _is_torch(x) else None))
        y = torch.zeros_like(x)
        for a in c:
            y = y * x + a
        return y
    else:
        return _np().polyval(coef, x)

# -----------------------------------------------------------------------------
# Build trace model
# -----------------------------------------------------------------------------

def build_trace_model(
    frames: BackendArray,
    *,
    time_axis: int = -3,
    params: TraceDetectParams,
) -> TraceBuildResult:
    """
    Detect and fit trace (center & width vs column) on a robust reference image.
    Uses time-median if T>1 else the first frame.
    """
    X = _to_float(frames, dtype=params.dtype)
    X, _, _ = _normalize_axes(X, time_axis, params.dispersion_axis)  # [..., T, Y, X]
    T = X.shape[-3]; Y = X.shape[-2]; W = X.shape[-1]

    # Apply masks
    if params.mask_saturated is not None:
        X = _where(params.mask_saturated, _nan_like(X), X)
    if params.mask_hot is not None:
        X = _where(params.mask_hot, _nan_like(X), X)

    # Reference image
    ref = X[..., 0, :, :] if T == 1 else _nanmedian(X, axis=-3)

    # Initial center guess from vertical profile if not provided
    if params.initial_center is None:
        prof = _nanmedian(ref, axis=-1)  # [Y]
        if _is_torch(prof):
            torch = _torch()
            rows = torch.arange(Y, device=ref.device, dtype=ref.dtype)
            c0 = float((rows * _nan_to_num(prof, 0.0)).sum() / (_nan_to_num(prof, 0.0).sum() + 1e-20))
        else:
            np = _np()
            rows = np.arange(Y, dtype=ref.dtype)
            w = _nan_to_num(prof, 0.0)
            c0 = float((rows * w).sum() / (w.sum() + 1e-20))
    else:
        c0 = float(params.initial_center)

    # Per-column centroid & width near c0
    mu, sig = _column_stats_y(ref, params.cr_reject, params.cr_zmax, params.cr_iter,
                              y_search_half=params.y_search_half, init_center=c0)

    # Smooth/fitting across columns
    if _is_torch(ref):
        cols = _torch().arange(W, dtype=ref.dtype, device=ref.device)
    else:
        cols = _np().arange(W, dtype=ref.dtype)

    center_poly = width_poly = None
    if params.fit_kind == "poly":
        center_poly = _polyfit_1d(cols, mu, deg=params.center_poly_deg)
        width_poly  = _polyfit_1d(cols, sig, deg=params.width_poly_deg)
        mu_fit  = _polyval_1d(center_poly, cols)
        sig_fit = _polyval_1d(width_poly, cols)
    else:
        # median/box smoothing in columns
        k = int(params.smooth_cols or 0)
        if k > 1:
            if _is_torch(mu):
                import numpy as _npx
                def _movavg(v, k):
                    pad = k // 2
                    vv = _nan_to_num(v, 0.0).detach().cpu().numpy()
                    vp = _npx.pad(vv, (pad, pad), mode="edge")
                    c = _npx.ones(k, dtype=vp.dtype) / k
                    return _npx.convolve(vp, c, mode="valid")
                mu_fit  = _torch().from_numpy(_movavg(mu, k)).to(device=ref.device, dtype=ref.dtype)
                sig_fit = _torch().from_numpy(_movavg(sig, k)).to(device=ref.device, dtype=ref.dtype)
            else:
                np = _np()
                def _movavg(v, k):
                    pad = k // 2
                    vp = np.pad(v, (pad, pad), mode="edge")
                    c = np.ones(k, dtype=vp.dtype) / k
                    return np.convolve(vp, c, mode="valid")
                mu_fit  = _movavg(mu, k)
                sig_fit = _movavg(sig, k)
        else:
            mu_fit, sig_fit = mu, sig

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update({
            "T": T, "Y": Y, "W": W,
            "initial_center": c0,
            "cr_reject": params.cr_reject, "cr_zmax": params.cr_zmax, "cr_iter": params.cr_iter,
            "fit_kind": params.fit_kind,
        })

    model = TraceModel(center=mu_fit, width=sig_fit, center_poly=center_poly, width_poly=width_poly, meta=meta)
    return TraceBuildResult(model=model)

# -----------------------------------------------------------------------------
# Background modeling
# -----------------------------------------------------------------------------

def _background_column_median(frame: BackendArray) -> BackendArray:
    """Per-column background: median across rows → [X]."""
    return _nanmedian(frame, axis=-2)

def _background_row_poly(frame: BackendArray, deg: int) -> BackendArray:
    """
    Per-column polynomial fit across rows; returns [Y, X] model.
    Torch → NumPy fit fallback (keeps behavior deterministic).
    """
    Y, X = frame.shape[-2], frame.shape[-1]
    torch_mode = _is_torch(frame)
    if torch_mode:
        torch = _torch(); np = _np()
        rows = np.arange(Y)
        out = np.empty((Y, X), dtype=np.float32)
        f = frame.detach().cpu().numpy()
        for j in range(X):
            col = f[:, j]
            col_f = np.where(np.isnan(col), np.nanmedian(col), col)
            coef = np.polyfit(rows, col_f, deg=deg)
            out[:, j] = np.polyval(coef, rows)
        return torch.from_numpy(out).to(device=frame.device, dtype=frame.dtype)
    else:
        np = _np()
        rows = np.arange(Y)
        out = np.empty((Y, X), dtype=frame.dtype)
        for j in range(X):
            col = frame[:, j]
            col_f = np.where(np.isnan(col), np.nanmedian(col), col)
            coef = np.polyfit(rows, col_f, deg=deg)
            out[:, j] = np.polyval(coef, rows)
        return out

# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

def _ensure_order_dims(center: BackendArray, width: BackendArray) -> Tuple[BackendArray, BackendArray, int]:
    """
    Normalize model arrays to shape [O, X] where O is number of orders.
    If 1D [X], return O=1 with leading dim added.
    """
    if center.ndim == 1:
        if _is_torch(center):
            center = center[None, :]
            width  = width[None, :]
        else:
            center = _np().expand_dims(center, 0)
            width  = _np().expand_dims(width, 0)
        O = 1
    else:
        O = center.shape[-2]
    return center, width, O

def extract_trace(
    frames: BackendArray,
    model: TraceModel,
    *,
    time_axis: int = -3,
    dispersion_axis: int = -1,
    params: TraceExtractParams,
) -> TraceExtractResult:
    """
    Extract 1D spectra over time using a trace model.
    Returns:
      flux: [..., T, X] for single-order or [..., T, O, X] for multi-order.
    """
    F = _to_float(frames, dtype=params.dtype)
    F, _, _ = _normalize_axes(F, time_axis, dispersion_axis)  # [..., T, Y, X]
    T = F.shape[-3]; Y = F.shape[-2]; X = F.shape[-1]
    torch_mode = _is_torch(F)

    var = params.var
    if var is not None:
        var = _to_float(var, dtype=params.dtype)
        var, _, _ = _normalize_axes(var, time_axis, dispersion_axis)  # [..., T, Y, X]

    # Prepare background output (per column summary)
    if torch_mode:
        torch = _torch()
        bkg_out = torch.empty((*F.shape[:-3], T, X), dtype=F.dtype, device=F.device) if params.bkg_kind != "none" else None
    else:
        np = _np()
        bkg_out = np.empty((*F.shape[:-3], T, X), dtype=F.dtype) if params.bkg_kind != "none" else None

    # Normalize model to [O, X]
    center, sig_y, O = _ensure_order_dims(model.center, model.width)

    # y coordinates
    y_coords = (_torch().arange(Y, dtype=F.dtype, device=F.device) if torch_mode
                else _np().arange(Y, dtype=F.dtype))

    # Allocate outputs
    out_shape = (*F.shape[:-3], T, X) if O == 1 else (*F.shape[:-3], T, O, X)
    if torch_mode:
        flux = torch.empty(out_shape, dtype=F.dtype, device=F.device)
        flux_err = torch.empty_like(flux) if var is not None else None
    else:
        np = _np()
        flux = np.empty(out_shape, dtype=F.dtype)
        flux_err = np.empty_like(flux) if var is not None else None

    # Iterate time (typically small compared to calib cost)
    for t in range(T):
        fr = F[..., t, :, :]  # [..., Y, X]

        # Background modeling
        if params.bkg_kind == "none":
            bkg_model = None
            bkg_col = None
        elif params.bkg_kind == "column_median":
            bkg_col = _background_column_median(fr)  # [..., X]
            bkg_model = (bkg_col[..., None, :]).repeat(Y, axis=-2) if not torch_mode else bkg_col[..., None, :].expand(*bkg_col.shape[:-1], Y, X)
        else:
            bkg_model = _background_row_poly(fr, deg=params.bkg_poly_deg)  # [..., Y, X]
            bkg_col = _nanmedian(bkg_model, axis=-2)  # summary per column

        if bkg_out is not None:
            bkg_out[..., t, :] = bkg_col if bkg_col is not None else (0.0 if not torch_mode else torch.zeros_like(fr[..., 0, :]))

        fr_bkg = fr - bkg_model if bkg_model is not None else fr  # background-subtracted

        # Handle single vs multi-order
        for o in range(O):
            yc = center[o]          # [X]
            sy = sig_y[o]           # [X]
            # guard sigma
            if torch_mode:
                sy = _where(sy <= 1e-4, _ones_like(sy) * 1.0, sy)
            else:
                sy = _where(sy <= 1e-4, _np().ones_like(sy) * 1.0, sy)

            if params.method == "box":
                # |y - yc[x]| <= ap_half
                if torch_mode:
                    torch = _torch()
                    dy = y_coords[:, None] - yc[None, :]                 # [Y, X]
                    mbox = (dy.abs() <= float(params.ap_half)).to(dtype=fr_bkg.dtype)
                    spec = _nansum(fr_bkg * mbox, axis=-2)               # [..., X]
                    if O == 1:
                        flux[..., t, :] = spec
                    else:
                        flux[..., t, o, :] = spec
                    if flux_err is not None and var is not None:
                        V = var[..., t, :, :]
                        veff = _nansum(_nan_to_num(V * mbox, 0.0), axis=-2)
                        err = (veff.clamp_min(0.0)).sqrt()
                        if O == 1:
                            flux_err[..., t, :] = err
                        else:
                            flux_err[..., t, o, :] = err
                else:
                    np = _np()
                    dy = y_coords[:, None] - yc[None, :]
                    mbox = (np.abs(dy) <= float(params.ap_half)).astype(fr_bkg.dtype)
                    spec = _nansum(fr_bkg * mbox, axis=-2)
                    if O == 1:
                        flux[..., t, :] = spec
                    else:
                        flux[..., t, o, :] = spec
                    if flux_err is not None and var is not None:
                        V = var[..., t, :, :]
                        veff = _nansum(_nan_to_num(V * mbox, 0.0), axis=-2)
                        err = _np().sqrt(_np().clip(veff, 0.0, None))
                        if O == 1:
                            flux_err[..., t, :] = err
                        else:
                            flux_err[..., t, o, :] = err

            else:
                # Horne-style optimal extraction: weights ∝ P / V
                if torch_mode:
                    torch = _torch()
                    P = torch.exp(-0.5 * ((y_coords[:, None] - yc[None, :]) ** 2) / (sy[None, :] ** 2))  # [Y, X]
                    Psum = _nansum(P, axis=-2, keepdims=True)
                    P = P / _where(Psum <= 1e-20, Psum + 1e-20, Psum)
                    if var is not None:
                        V = var[..., t, :, :]
                        W = P / _where(V <= 1e-20, torch.tensor(1e-20, dtype=V.dtype, device=V.device), V)
                    else:
                        W = P
                    num = _nansum(_nan_to_num(W * fr_bkg, 0.0), axis=-2)   # [..., X]
                    den = _nansum(_nan_to_num(W * P, 0.0), axis=-2)
                    spc = num / _where(den <= 1e-20, den + 1e-20, den)
                    if O == 1:
                        flux[..., t, :] = spc
                    else:
                        flux[..., t, o, :] = spc
                    if flux_err is not None and var is not None:
                        denom = (P * P) / _where(V <= 1e-20, torch.tensor(1e-20, dtype=V.dtype, device=V.device), V)
                        deff = _nansum(_nan_to_num(denom, 0.0), axis=-2)
                        err = (1.0 / _where(deff <= 1e-20, deff + 1e-20, deff)).sqrt()
                        if O == 1:
                            flux_err[..., t, :] = err
                        else:
                            flux_err[..., t, o, :] = err
                else:
                    np = _np()
                    P = np.exp(-0.5 * ((y_coords[:, None] - yc[None, :]) ** 2) / (sy[None, :] ** 2))
                    Psum = _nansum(P, axis=-2, keepdims=True)
                    P = P / _where(Psum <= 1e-20, Psum + 1e-20, Psum)
                    if var is not None:
                        V = var[..., t, :, :]
                        W = P / _where(V <= 1e-20, _np().array(1e-20, dtype=V.dtype), V)
                    else:
                        W = P
                    num = _nansum(_nan_to_num(W * fr_bkg, 0.0), axis=-2)
                    den = _nansum(_nan_to_num(W * P, 0.0), axis=-2)
                    spc = num / _where(den <= 1e-20, den + 1e-20, den)
                    if O == 1:
                        flux[..., t, :] = spc
                    else:
                        flux[..., t, o, :] = spc
                    if flux_err is not None and var is not None:
                        denom = (P * P) / _where(V <= 1e-20, _np().array(1e-20, dtype=P.dtype), V)
                        deff = _nansum(_nan_to_num(denom, 0.0), axis=-2)
                        err = (1.0 / _where(deff <= 1e-20, deff + 1e-20, deff)) ** 0.5
                        if O == 1:
                            flux_err[..., t, :] = err
                        else:
                            flux_err[..., t, o, :] = err

    # Post-clip outputs if requested
    if params.clip_out is not None:
        low, high = params.clip_out
        flux = _clip(flux, low, high)
        if flux_err is not None:
            flux_err = _clip(flux_err, 0.0, None)

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update({
            "method": params.method,
            "orders": O,
            "bkg_kind": params.bkg_kind,
            "psf_sigma_y": params.psf_sigma_y,
            "ap_half": params.ap_half,
        })

    return TraceExtractResult(flux=flux, flux_err=flux_err, bkg=bkg_out, meta=meta)

# -----------------------------------------------------------------------------
# Self-tests (light)
# -----------------------------------------------------------------------------

def _make_tilted_stripe(T=3, Y=64, X=128, slope=0.1, center0=32.0, amp=4000.0, sig=2.0, noise=2.0):
    import numpy as np
    yy, xx = np.mgrid[0:Y, 0:X]
    stack = []
    for _ in range(T):
        center_x = center0 + slope * (xx - X/2.0)
        stripe = amp * np.exp(-0.5 * ((yy - center_x)**2) / (sig**2))
        stack.append(stripe + np.random.normal(0, noise, size=(Y, X)))
    return np.stack(stack, axis=0).astype(np.float32)

def _test_build_and_extract_box():
    import numpy as np, math
    T, Y, X = 5, 64, 128
    arr = _make_tilted_stripe(T=T, Y=Y, X=X, slope=0.08, center0=28.0, amp=3000.0, sig=2.0)
    detect = TraceDetectParams(dispersion_axis=-1, center_poly_deg=2, width_poly_deg=1, fit_kind="poly", return_intermediate=True)
    build = build_trace_model(arr, time_axis=-3, params=detect)
    ext_params = TraceExtractParams(method="box", ap_half=5, bkg_kind="column_median", return_intermediate=True)
    res = extract_trace(arr, model=build.model, time_axis=-3, dispersion_axis=-1, params=ext_params)
    expected_peak = 3000.0 * math.sqrt(2.0 * math.pi) * 2.0
    med = float(np.nanmedian(res.flux))
    assert med > 0.4 * expected_peak, f"box extraction too low: med={med}, exp~{expected_peak}"
    return True

def _test_build_and_extract_optimal():
    import numpy as np
    T, Y, X = 4, 64, 96
    arr = _make_tilted_stripe(T=T, Y=Y, X=X, slope=-0.05, center0=34.0, amp=5000.0, sig=1.8, noise=1.2)
    var = np.ones_like(arr) * (1.2**2)
    detect = TraceDetectParams(dispersion_axis=-1, fit_kind="poly", center_poly_deg=2, width_poly_deg=1, return_intermediate=True)
    build = build_trace_model(arr, time_axis=-3, params=detect)
    ext_params = TraceExtractParams(method="optimal", var=var, psf_sigma_y=1.8, bkg_kind="column_median", return_intermediate=True)
    res = extract_trace(arr, model=build.model, time_axis=-3, dispersion_axis=-1, params=ext_params)
    med = float(np.nanmedian(res.flux))
    assert med > 1000.0, f"optimal extraction too low: {med}"
    return True

if __name__ == "__main__":
    ok1 = _test_build_and_extract_box()
    ok2 = _test_build_and_extract_optimal()
    print("trace.py self-tests:", ok1 and ok2)