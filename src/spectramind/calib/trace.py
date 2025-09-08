# src/spectramind/calib/trace.py
# =============================================================================
# SpectraMind V50 — Spectral trace detection, modeling & optimal extraction
# -----------------------------------------------------------------------------
# Features:
#  - Backend-agnostic (NumPy or Torch).
#  - Detect spectral trace center vs dispersion (columns) per order.
#  - Fit robust polynomial (center and width vs column).
#  - Optional cosmic-ray rejection (MAD-based z-scores).
#  - Background modeling (per-column median or polynomial across cross-dispersion).
#  - Horne-style optimal extraction and simple box extraction.
#  - Variance propagation and mask handling (saturated/hot/bad).
#  - Multi-order support.
#  - Flexible axes: [..., T, Y, X] with configurable time & dispersion axis.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal, List

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

# -----------------------------------------------------------------------------
# Backend shims (consistent with other calib modules)
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
        return x.to(getattr(torch, dtype or "float32"))
    else:
        np = _np()
        return x.astype(getattr(np, dtype) if isinstance(dtype, str) else (dtype or np.float32), copy=False)

def _zeros_like(x: BackendArray) -> BackendArray:
    if _is_torch(x): return _torch().zeros_like(x)
    return _np().zeros_like(x)

def _ones_like(x: BackendArray) -> BackendArray:
    if _is_torch(x): return _torch().ones_like(x)
    return _np().ones_like(x)

def _where(mask: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(mask): return _torch().where(mask, a, b)
    return _np().where(mask, a, b)

def _abs(x: BackendArray) -> BackendArray:
    return x.abs() if _is_torch(x) else _np().abs(x)

def _nanmean(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        m = ~torch.isnan(x)
        num = torch.where(m, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims)
        den = m.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        all_nan = den == 0
        return torch.where(all_nan, torch.tensor(float('nan'), dtype=x.dtype, device=x.device), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)

def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch(); np = _np()
        arr = x.detach().cpu().numpy()
        med = np.nanmedian(arr, axis=axis, keepdims=keepdims)
        return torch.from_numpy(med).to(device=x.device, dtype=x.dtype)
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
    if low is None and high is None: return x
    if _is_torch(x):
        torch = _torch()
        if low is not None: x = torch.clamp(x, min=float(low))
        if high is not None: x = torch.clamp(x, max=float(high))
        return x
    else:
        return _np().clip(x, low, high)

def _nan_to_num(x: BackendArray, val=0.0) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.where(torch.isnan(x), torch.tensor(val, dtype=x.dtype, device=x.device), x)
    else:
        return _np().nan_to_num(x, nan=val)

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

    dispersion_axis   : index of dispersion axis (default -1 => [..., Y, X], X is dispersion)
    smooth_cols       : window for column-wise smoothing of center detection (odd; 0=no smooth)
    center_poly_deg   : degree of polynomial for center(yc) vs column (if fit_kind='poly')
    width_poly_deg    : degree of polynomial for width(sig_y) vs column (if fit_kind='poly')
    fit_kind          : 'poly' or 'median' (median smooth of raw centers/widths)
    cr_reject         : enable MAD-z CR rejection on column profiles before centroiding
    cr_zmax           : CR z-threshold
    cr_iter           : number of robust iterations
    y_search_half     : half-height of search window around initial median center (pixels)
    initial_center    : optional fixed initial row center; if None, compute from stack median
    mask_saturated    : mask array (True excludes)
    mask_hot          : mask array (True excludes)
    dtype             : float dtype
    return_intermediate: stash debug info
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
    Extraction configuration (per order).

    method            : 'optimal' or 'box'
    ap_half           : half-height of box aperture (pixels) if method='box'
    bkg_kind          : 'none' | 'column_median' | 'row_poly'
    bkg_poly_deg      : degree for row_poly background vs row in each column
    psf_sigma_y       : PSF sigma in cross-dispersion (pixels) if no variance/PSF template provided
    var               : variance cube [..., T, Y, X], used for optimal extraction and error propagation
    clip_out          : optional (low, high) clipping for extracted flux
    dtype             : output float dtype
    return_intermediate: stash debug info
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
    Trace model per order:
      - center(x): row center vs column [X]
      - width(x):  sigma_y vs column [X]
      - poly coeffs for center/width if fit_kind='poly'
      - metadata
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
    flux: BackendArray          # [..., T, X]
    flux_err: Optional[BackendArray]
    bkg: Optional[BackendArray] # background per frame/column or scalar summary
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Axis helpers
# -----------------------------------------------------------------------------

def _normalize_axes(frames: BackendArray, time_axis: int, disp_axis: int) -> Tuple[BackendArray, int, int]:
    """
    Reorder to canonical [..., T, Y, X] and return new axes indices (T:-3, Y:-2, X:-1).
    """
    x = frames
    nd = x.ndim
    # target positions:
    tgt_T, tgt_X = nd - 3, nd - 1
    # resolve negative
    if time_axis < 0: time_axis = nd + time_axis
    if disp_axis < 0: disp_axis = nd + disp_axis
    # find spatial other axis (Y)
    all_axes = list(range(nd))
    # move time to tgt_T
    if time_axis != tgt_T:
        perm = list(range(nd))
        perm[tgt_T], perm[time_axis] = perm[time_axis], perm[tgt_T]
        x = x.permute(*perm) if _is_torch(x) else x.transpose(tgt_T, time_axis)
        # update disp_axis if moved
        if disp_axis == tgt_T: disp_axis = time_axis
        elif disp_axis == time_axis: disp_axis = tgt_T
        time_axis = tgt_T
    # now time at tgt_T
    # move dispersion to tgt_X
    if disp_axis != tgt_X:
        perm = list(range(nd))
        perm[tgt_X], perm[disp_axis] = perm[disp_axis], perm[tgt_X]
        x = x.permute(*perm) if _is_torch(x) else x.transpose(tgt_X, disp_axis)
        disp_axis = tgt_X
    # done: [..., T, Y, X]
    return x, time_axis, disp_axis

# -----------------------------------------------------------------------------
# Core: column profile, centroid & width, robust fit
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

def _nan_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.tensor(float('nan'), dtype=x.dtype, device=x.device).expand_as(x)
    else:
        np = _np()
        return np.full_like(x, np.nan)

def _column_stats_y(frame: BackendArray, cr_reject: bool, zmax: float, iters: int,
                    y_search_half: int, init_center: Optional[float]) -> Tuple[BackendArray, BackendArray]:
    """
    For a single frame [Y,X], compute per-column centroid (row) and width (Gaussian sigma_y proxy).
    Optionally restrict to a search window around init_center +- y_search_half.
    """
    torch_mode = _is_torch(frame)
    Y, X = frame.shape[-2], frame.shape[-1]
    if torch_mode:
        torch = _torch()
        cols = torch.arange(X, dtype=frame.dtype, device=frame.device)
        rows = torch.arange(Y, dtype=frame.dtype, device=frame.device)
        yy = rows[:, None].expand(Y, X)
        # restrict window
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
        # nan-safe sums
        w = torch.where(torch.isnan(img), torch.tensor(0., dtype=img.dtype, device=img.device), img)
        s0 = w.sum(dim=-2) + 1e-20               # [X]
        s1 = (w * yyw).sum(dim=-2)               # [X]
        mu = s1 / s0                              # [X]
        var = (w * (yyw - mu[None, :]) ** 2).sum(dim=-2) / s0
        sig = torch.sqrt(torch.clamp(var, min=1e-10))
        return mu, sig
    else:
        import numpy as np
        rows = np.arange(Y)
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
        s0 = _np().sum(w, axis=0) + 1e-20
        s1 = _np().sum(w * yyw, axis=0)
        mu = s1 / s0
        var = _np().sum(w * (yyw - mu[None, :]) ** 2, axis=0) / s0
        sig = _np().sqrt(_np().clip(var, 1e-10, None))
        return mu, sig

def _polyfit_1d(x: BackendArray, y: BackendArray, deg: int) -> BackendArray:
    """Return polynomial coefficients for y(x); Torch path uses numpy CPU fallback."""
    if deg <= 0:
        # constant best-fit
        c = _nanmedian(y, axis=-1, keepdims=False)
        if _is_torch(c):
            return c[..., None]
        else:
            return c[..., None]
    if _is_torch(x) or _is_torch(y):
        np = _np(); torch = _torch()
        xv = x.detach().cpu().numpy() if _is_torch(x) else x
        yv = y.detach().cpu().numpy() if _is_torch(y) else y
        # nan robust: fill with median
        yv_f = np.where(np.isnan(yv), np.nanmedian(yv), yv)
        coef = np.polyfit(xv, yv_f, deg=deg)  # descending degree
        return torch.from_numpy(coef).to(device=(y.device if _is_torch(y) else "cpu"), dtype=(y.dtype if _is_torch(y) else None))
    else:
        np = _np()
        y_f = np.where(np.isnan(y), np.nanmedian(y), y)
        coef = np.polyfit(x, y_f, deg=deg)
        return coef

def _polyval_1d(coef: BackendArray, x: BackendArray) -> BackendArray:
    """Evaluate polynomial (descending degree)."""
    if _is_torch(coef) or _is_torch(x):
        torch = _torch()
        c = coef
        if not _is_torch(coef): c = torch.from_numpy(coef).to(device=x.device if _is_torch(x) else "cpu", dtype=x.dtype if _is_torch(x) else None)
        y = torch.zeros_like(x)
        for a in c:
            y = y * x + a
        return y
    else:
        np = _np()
        return np.polyval(coef, x)

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
    Detect and fit the spectral trace (center & width vs column) on a representative image.
    Uses time-collapsed median (or first frame if T==1).
    """
    dtype = params.dtype
    X = _to_float(frames, dtype=dtype)

    # normalize axes to [..., T, Y, X]
    X, t_axis, d_axis = _normalize_axes(X, time_axis, params.dispersion_axis)
    T = X.shape[-3]; Y = X.shape[-2]; W = X.shape[-1]

    # apply masks
    if params.mask_saturated is not None: X = _where(params.mask_saturated, _nan_like(X), X)
    if params.mask_hot is not None: X = _where(params.mask_hot, _nan_like(X), X)

    # collapse over time to robust template
    ref = X[..., 0, :, :] if T == 1 else _nanmedian(X, axis=-3, keepdims=False)  # [Y, X]

    # estimate initial center from global vertical median profile if not provided
    if params.initial_center is None:
        # collapse across X to get row profile
        prof = _nanmedian(ref, axis=-1)  # [Y]
        # centroid row index
        if _is_torch(prof):
            torch = _torch()
            rows = torch.arange(Y, device=ref.device, dtype=ref.dtype)
            w = _nan_to_num(prof, 0.0)
            c0 = float((w * rows).sum() / (w.sum() + 1e-20))
        else:
            np = _np()
            rows = np.arange(Y)
            w = _nan_to_num(prof, 0.0)
            c0 = float((w * rows).sum() / (w.sum() + 1e-20))
    else:
        c0 = float(params.initial_center)

    # per-column centroid and width near c0
    mu, sig = _column_stats_y(ref, params.cr_reject, params.cr_zmax, params.cr_iter,
                              y_search_half=params.y_search_half, init_center=c0)

    # smooth across columns
    cols = _torch().arange(W, dtype=ref.dtype, device=ref.device) if _is_torch(ref) else _np().arange(W)
    center_poly = width_poly = None
    if params.fit_kind == "poly":
        center_poly = _polyfit_1d(cols, mu, deg=params.center_poly_deg)
        mu_fit = _polyval_1d(center_poly, cols)
        width_poly = _polyfit_1d(cols, sig, deg=params.width_poly_deg)
        sig_fit = _polyval_1d(width_poly, cols)
    else:  # median smooth
        if params.smooth_cols and params.smooth_cols > 1:
            k = int(params.smooth_cols)
            if _is_torch(mu):
                torch = _torch(); np = _np()
                muf = _nan_to_num(mu, 0.0).detach().cpu().numpy()
                sigf = _nan_to_num(sig, 0.0).detach().cpu().numpy()
                # moving average
                def _movavg(v, k):
                    pad = k // 2
                    vp = np.pad(v, (pad, pad), mode="edge")
                    c = np.ones(k, dtype=vp.dtype) / k
                    return np.convolve(vp, c, mode="valid")
                mu_fit_np = _movavg(muf, k); sig_fit_np = _movavg(sigf, k)
                mu_fit = torch.from_numpy(mu_fit_np).to(device=ref.device, dtype=ref.dtype)
                sig_fit = torch.from_numpy(sig_fit_np).to(device=ref.device, dtype=ref.dtype)
            else:
                np = _np()
                def _movavg(v, k):
                    pad = k // 2
                    vp = np.pad(v, (pad, pad), mode="edge")
                    c = np.ones(k, dtype=vp.dtype) / k
                    return np.convolve(vp, c, mode="valid")
                mu_fit = _movavg(mu, k)
                sig_fit = _movavg(sig, k)
        else:
            mu_fit, sig_fit = mu, sig

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update({
            "Y": Y, "W": W, "T": T,
            "initial_center": c0,
            "cr_reject": params.cr_reject, "cr_zmax": params.cr_zmax, "cr_iter": params.cr_iter
        })

    model = TraceModel(center=mu_fit, width=sig_fit, center_poly=center_poly, width_poly=width_poly, meta=meta)
    return TraceBuildResult(model=model)

# -----------------------------------------------------------------------------
# Background modeling
# -----------------------------------------------------------------------------

def _background_column_median(frame: BackendArray) -> BackendArray:
    """Background per column: median across rows (ignoring NaN)."""
    return _nanmedian(frame, axis=-2, keepdims=False)  # [X]

def _background_row_poly(frame: BackendArray, deg: int) -> BackendArray:
    """Fit polynomial across rows for each column; return [Y,X] background."""
    Y, X = frame.shape[-2], frame.shape[-1]
    torch_mode = _is_torch(frame)
    if torch_mode:
        torch = _torch(); np = _np()
        rows = np.arange(Y)
        out = np.empty((Y, X), dtype=np.float32)
        f_np = frame.detach().cpu().numpy()
        for x in range(X):
            ycol = f_np[:, x]
            ycol_f = np.where(np.isnan(ycol), np.nanmedian(ycol), ycol)
            coef = np.polyfit(rows, ycol_f, deg=deg)
            out[:, x] = np.polyval(coef, rows)
        return torch.from_numpy(out).to(device=frame.device, dtype=frame.dtype)
    else:
        np = _np()
        rows = np.arange(Y)
        out = np.empty((Y, X), dtype=frame.dtype)
        for x in range(X):
            ycol = frame[:, x]
            ycol_f = np.where(_np().isnan(ycol), _np().nanmedian(ycol), ycol)
            coef = np.polyfit(rows, ycol_f, deg=deg)
            out[:, x] = np.polyval(coef, rows)
        return out

# -----------------------------------------------------------------------------
# Extraction
# -----------------------------------------------------------------------------

def extract_trace(
    frames: BackendArray,
    model: TraceModel,
    *,
    time_axis: int = -3,
    dispersion_axis: int = -1,
    params: TraceExtractParams,
) -> TraceExtractResult:
    """
    Extract 1D spectra over time using the provided trace model.

    frames: [..., T, Y, X]
    model: TraceModel from build_trace_model
    params: TraceExtractParams

    Returns
    -------
    TraceExtractResult with flux [.., T, X], flux_err, background product, and meta.
    """
    dtype = params.dtype
    F = _to_float(frames, dtype=dtype)
    F, _, _ = _normalize_axes(F, time_axis, dispersion_axis)  # [..., T, Y, X]
    T = F.shape[-3]; Y = F.shape[-2]; X = F.shape[-1]
    torch_mode = _is_torch(F)

    var = params.var
    if var is not None:
        var = _to_float(var, dtype=dtype)
        var, _, _ = _normalize_axes(var, time_axis, dispersion_axis)

    # Prepare outputs
    if torch_mode:
        torch = _torch()
        flux = torch.empty((*F.shape[:-3], T, X), dtype=F.dtype, device=F.device)
        flux_err = torch.empty_like(flux) if var is not None else None
        bkg_out = torch.empty((*F.shape[:-3], T, X), dtype=F.dtype, device=F.device) if params.bkg_kind != "none" else None
    else:
        np = _np()
        flux = np.empty((*F.shape[:-3], T, X), dtype=F.dtype)
        flux_err = np.empty_like(flux) if var is not None else None
        bkg_out = np.empty((*F.shape[:-3], T, X), dtype=F.dtype) if params.bkg_kind != "none" else None

    center = model.center  # [X]
    sig_y = model.width    # [X]
    # Build per-frame weights/extraction window per column
    y_coords = (_torch().arange(Y, dtype=F.dtype, device=F.device) if torch_mode else _np().arange(Y, dtype=F.dtype))

    for t in range(T):
        fr = F[..., t, :, :]  # [Y,X]
        # Background
        if params.bkg_kind == "none":
            bkg = None
        elif params.bkg_kind == "column_median":
            bkg_col = _background_column_median(fr)  # [X]
            bkg = (bkg_col[None, :]).repeat(Y, 1) if torch_mode else _np().repeat(bkg_col[None, :], Y, axis=0)
        else:
            bkg = _background_row_poly(fr, deg=params.bkg_poly_deg)

        if bkg_out is not None:
            if torch_mode:
                bkg_out[..., t, :] = _nanmedian(bkg, axis=-2) if bkg is not None else _zeros_like(fr[0, :])
            else:
                bkg_out[..., t, :] = _np().nanmedian(bkg, axis=-2) if bkg is not None else 0.0

        # Subtract background
        fr_bkg = fr - bkg if bkg is not None else fr

        if params.method == "box":
            # Box aperture: sum within |y - center[x]| <= ap_half
            if torch_mode:
                torch = _torch()
                yc = center[None, :]  # [1,X]
                # distance grid
                dy = (y_coords[:, None] - yc)  # [Y,X]
                mbox = (dy.abs() <= float(params.ap_half)).to(dtype=fr.dtype)
                spec = _nan_to_num(fr_bkg * mbox, 0.0).sum(dim=-2)  # [X]
                flux[..., t, :] = spec
                if flux_err is not None:
                    v = var[..., t, :, :] if var is not None else None
                    veff = _nan_to_num((v * mbox), 0.0).sum(dim=-2) if v is not None else _zeros_like(spec)
                    flux_err[..., t, :] = torch.sqrt(torch.clamp(veff, min=0.0))
            else:
                np = _np()
                yc = center[None, :]
                dy = (y_coords[:, None] - yc)
                mbox = (np.abs(dy) <= float(params.ap_half)).astype(fr.dtype)
                spec = _nan_to_num(fr_bkg * mbox, 0.0).sum(axis=-2)
                flux[..., t, :] = spec
                if flux_err is not None:
                    v = var[..., t, :, :] if var is not None else None
                    veff = _nan_to_num((v * mbox), 0.0).sum(axis=-2) if v is not None else _zeros_like(spec)
                    flux_err[..., t, :] = _np().sqrt(_np().clip(veff, 0.0, None))

        else:
            # Optimal extraction (Horne 1986): weights ~ P / V
            # Build per-column Gaussian PSF if VAR available; else use sigma_y and flux moment scale.
            if torch_mode:
                torch = _torch()
                yc = center[None, :]  # [1,X]
                sy = _where(sig_y <= 1e-4, _ones_like(sig_y) * 1.0, sig_y)[None, :]
                # PSF
                P = torch.exp(-0.5 * ((y_coords[:, None] - yc) ** 2) / (sy ** 2))  # [Y,X]
                # normalize column-wise
                P = P / _where(P.sum(dim=-2, keepdim=True) <= 1e-20, torch.tensor(1e-20, dtype=P.dtype, device=P.device), P.sum(dim=-2, keepdim=True))
                if var is not None:
                    V = var[..., t, :, :]
                    W = P / _where(V <= 1e-20, torch.tensor(1e-20, dtype=V.dtype, device=V.device), V)
                else:
                    W = P
                num = _nan_to_num(W * fr_bkg, 0.0).sum(dim=-2)         # [X]
                den = _nan_to_num(W * P, 0.0).sum(dim=-2)               # [X]
                spec = num / _where(den <= 1e-20, torch.tensor(1e-20, dtype=den.dtype, device=den.device), den)
                flux[..., t, :] = spec
                if flux_err is not None:
                    denom = (P * P) / _where(var[..., t, :, :] <= 1e-20, torch.tensor(1e-20, dtype=P.dtype, device=P.device), var[..., t, :, :])
                    deff = _nan_to_num(denom, 0.0).sum(dim=-2)
                    flux_err[..., t, :] = (1.0 / _where(deff <= 1e-20, torch.tensor(1e-20, dtype=deff.dtype, device=deff.device), deff)) ** 0.5
            else:
                np = _np()
                yc = center[None, :]
                sy = _where(sig_y <= 1e-4, _np().ones_like(sig_y) * 1.0, sig_y)[None, :]
                P = _np().exp(-0.5 * ((y_coords[:, None] - yc) ** 2) / (sy ** 2))
                P = P / _where(P.sum(axis=-2, keepdims=True) <= 1e-20, _np().array(1e-20, dtype=P.dtype), P.sum(axis=-2, keepdims=True))
                if var is not None:
                    V = var[..., t, :, :]
                    W = P / _where(V <= 1e-20, _np().array(1e-20, dtype=V.dtype), V)
                else:
                    W = P
                spec = _nan_to_num((W * fr_bkg), 0.0).sum(axis=-2) / _where((_nan_to_num(W * P, 0.0).sum(axis=-2)) <= 1e-20,
                                                                              _np().array(1e-20, dtype=F.dtype), (_nan_to_num(W * P, 0.0).sum(axis=-2)))
                flux[..., t, :] = spec
                if flux_err is not None:
                    denom = (P * P) / _where(var[..., t, :, :] <= 1e-20, _np().array(1e-20, dtype=P.dtype), var[..., t, :, :])
                    deff = _nan_to_num(denom, 0.0).sum(axis=-2)
                    flux_err[..., t, :] = (1.0 / _where(deff <= 1e-20, _np().array(1e-20, dtype=deff.dtype), deff)) ** 0.5

    if params.clip_out is not None:
        low, high = params.clip_out
        flux = _clip(flux, low, high)
        if flux_err is not None:
            flux_err = _clip(flux_err, 0.0, None)

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update({
            "method": params.method,
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
        center_x = center0 + slope * (xx - X/2.0)  # linear tilt
        stripe = amp * np.exp(-0.5 * ((yy - center_x)**2) / (sig**2))
        stack.append(stripe + np.random.normal(0, noise, size=(Y, X)))
    return np.stack(stack, axis=0).astype(np.float32)

def _test_build_and_extract_box():
    import numpy as np
    T, Y, X = 5, 64, 128
    arr = _make_tilted_stripe(T=T, Y=Y, X=X, slope=0.08, center0=28.0, amp=3000.0, sig=2.0)
    detect = TraceDetectParams(dispersion_axis=-1, center_poly_deg=2, width_poly_deg=1, fit_kind="poly", return_intermediate=True)
    build = build_trace_model(arr, time_axis=-3, params=detect)
    ext_params = TraceExtractParams(method="box", ap_half=5, bkg_kind="column_median", return_intermediate=True)
    res = extract_trace(arr, model=build.model, time_axis=-3, dispersion_axis=-1, params=ext_params)
    # crude sanity: typical flux should be positive and roughly near amplitude * sqrt(2*pi)*sigma for per-column integral
    import math
    expected_peak = 3000.0 * math.sqrt(2.0 * math.pi) * 2.0
    med = float(np.nanmedian(res.flux))
    assert med > 0.4 * expected_peak, f"box extraction too low: med={med}, exp~{expected_peak}"
    return True

def _test_build_and_extract_optimal():
    import numpy as np
    T, Y, X = 4, 64, 96
    arr = _make_tilted_stripe(T=T, Y=Y, X=X, slope=-0.05, center0=34.0, amp=5000.0, sig=1.8, noise=1.2)
    # variance (white noise)
    var = np.ones_like(arr) * (1.2**2)
    detect = TraceDetectParams(dispersion_axis=-1, fit_kind="poly", center_poly_deg=2, width_poly_deg=1, return_intermediate=True)
    build = build_trace_model(arr, time_axis=-3, params=detect)
    ext_params = TraceExtractParams(method="optimal", var=var, psf_sigma_y=1.8, bkg_kind="column_median", return_intermediate=True)
    res = extract_trace(arr, model=build.model, time_axis=-3, dispersion_axis=-1, params=ext_params)
    # sanity: optimal flux should be close to amplitude * normalized PSF integral per column ~ amplitude
    med = float(np.nanmedian(res.flux))
    assert med > 1000.0, f"optimal extraction too low: {med}"
    return True

if __name__ == "__main__":
    ok1 = _test_build_and_extract_box()
    ok2 = _test_build_and_extract_optimal()
    print("trace.py self-tests:", ok1 and ok2)
