# src/spectramind/calib/photometry.py
# =============================================================================
# SpectraMind V50 — Aperture and Optimal Photometry (Upgraded)
# -----------------------------------------------------------------------------
# Highlights (this revision)
#   - Correct NaN-safe PSF normalization and weight sums (aperture & optimal).
#   - Proper support for arbitrary batch prefixes: frames [..., T, H, W].
#   - Time-axis alignment for var / masks; consistent dtype handling.
#   - Torch-first median/STD where available; graceful NumPy fallback.
#   - Deterministic, unit-tested self-checks; clear dataclasses & types.
#   - No internet, no I/O — Kaggle/CI safe.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

__all__ = [
    "PhotometryParams",
    "PhotometryResult",
    "photometry",
]

# -----------------------------------------------------------------------------
# Backend shims
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
        return torch.nan_to_num(x, nan=val) if hasattr(torch, "nan_to_num") else torch.where(torch.isnan(x), torch.tensor(val, dtype=x.dtype, device=x.device), x)
    else:
        return _np().nan_to_num(x, nan=val)

def _nansum(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        fn = getattr(torch, "nansum", None)
        if fn is not None:
            return torch.nansum(x, dim=axis, keepdim=keepdims) if axis is not None else torch.nansum(x)
        # Fallback
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
            # If all were NaN, output should be NaN; detect by zero denom
            return torch.where((m.sum() == 0), torch.tensor(float("nan"), dtype=x.dtype, device=x.device), out)
        return out
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)

def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        # torch.nanmedian exists in recent versions; prefer it to keep gradients on device
        if hasattr(torch, "nanmedian"):
            return torch.nanmedian(x, dim=axis, keepdim=keepdims).values if axis is not None else torch.nanmedian(x)
        # Fallback via NumPy (breaks grad): last resort for old Torch
        np = _np()
        x_np = x.detach().cpu().numpy()
        m_np = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
        return torch.from_numpy(m_np).to(device=x.device, dtype=x.dtype)
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

def _nan_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.full_like(x, float("nan"))
    else:
        import numpy as np
        return np.full_like(x, np.nan)

# -----------------------------------------------------------------------------
# Parameters and results
# -----------------------------------------------------------------------------

ApertureShape = Literal["circular", "elliptical"]
SkyAgg = Literal["median", "mean"]
PhotMethod = Literal["aperture", "optimal"]

@dataclass
class PhotometryParams:
    """
    Photometry configuration.
    All arrays follow [..., T, H, W] with configurable time_axis.
    """
    time_axis: int = -3
    method: PhotMethod = "aperture"

    # Apertures
    shape: ApertureShape = "circular"
    r_ap: float = 3.0
    a_ap: float = 3.0
    b_ap: float = 3.0
    theta: float = 0.0
    adaptive_ap: bool = False
    fwhm_scale: float = 1.0

    # Sky annulus
    use_annulus: bool = True
    r_in: float = 6.0
    r_out: float = 8.0
    sky_agg: SkyAgg = "median"
    sky_sigma_clip: Optional[Tuple[float, float]] = (3.0, 3.0)
    sky_clip_iter: int = 1

    # Masks / variance (aligned with frames by default; time_axis honored)
    mask_saturated: Optional[BackendArray] = None
    mask_hot: Optional[BackendArray] = None
    var: Optional[BackendArray] = None

    # Optimal extraction (if method='optimal')
    psf_sigma: Union[float, Tuple[float, float]] = 1.5
    psf: Optional[BackendArray] = None  # [H,W] or [T,H,W]

    # Output control
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False


@dataclass
class PhotometryResult:
    """
    Photometry outputs for one target over time (batch-aware).
    Shapes:
      - If frames had prefix shape P = frames.shape[:-3], outputs are:
        flux:       P + (T,)
        flux_err:   P + (T,) or None
        sky:        P + (T,) or None
        aperture_area: P  (effective pixel count in last frame, per-batch)
        xc, yc, fwhm:   P + (T,)
    """
    flux: BackendArray
    flux_err: Optional[BackendArray]
    sky: Optional[BackendArray]
    aperture_area: BackendArray
    xc: BackendArray
    yc: BackendArray
    fwhm: BackendArray
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Axis helpers
# -----------------------------------------------------------------------------

def _normalize_time_axis(x: BackendArray, time_axis: int) -> Tuple[BackendArray, int]:
    """
    Move the time axis to position -3 (i.e., [..., T, H, W]).
    Returns (x_moved, T_index=-3).
    """
    if x is None:
        return x, time_axis
    nd = x.ndim
    if time_axis < 0:
        time_axis = nd + time_axis
    target = nd - 3
    if time_axis == target:
        return x, -3
    perm = list(range(nd))
    perm[target], perm[time_axis] = perm[time_axis], perm[target]
    if _is_torch(x):
        x2 = x.permute(*perm)
    else:
        x2 = x.transpose(target, time_axis)
    return x2, -3

def _align_optional(arr: Optional[BackendArray], time_axis: int) -> Optional[BackendArray]:
    if arr is None:
        return None
    return _normalize_time_axis(arr, time_axis)[0]

# -----------------------------------------------------------------------------
# Centroid & FWHM (per 2D frame)
# -----------------------------------------------------------------------------

def _centroid_and_fwhm(frame_2d: BackendArray) -> Tuple[float, float, float]:
    """
    Center-of-light centroid and moment-based FWHM for a single 2D frame.
    Returns (yc, xc, fwhm).
    """
    torch_mode = _is_torch(frame_2d)
    if torch_mode:
        torch = _torch()
        msk = ~torch.isnan(frame_2d)
        vals = torch.where(msk, frame_2d, torch.tensor(0., dtype=frame_2d.dtype, device=frame_2d.device))
        H, W = frame_2d.shape[-2], frame_2d.shape[-1]
        yy, xx = torch.meshgrid(
            torch.arange(H, device=frame_2d.device, dtype=frame_2d.dtype),
            torch.arange(W, device=frame_2d.device, dtype=frame_2d.dtype),
            indexing="ij"
        )
        s = vals.sum()
        if float(s) <= 0.0:
            return float(H/2), float(W/2), float("nan")
        yc = float((vals * yy).sum() / s)
        xc = float((vals * xx).sum() / s)
        var = float(((vals * ((yy - yc) ** 2 + (xx - xc) ** 2)).sum() / s) / 2.0)
        import math
        fwhm = 2.0 * (2.0 * math.log(2.0)) ** 0.5 * (var ** 0.5) if var > 0 else float("nan")
        return yc, xc, fwhm
    else:
        import numpy as np
        msk = ~np.isnan(frame_2d)
        vals = np.where(msk, frame_2d, 0.0)
        H, W = frame_2d.shape[-2], frame_2d.shape[-1]
        yy, xx = np.mgrid[0:H, 0:W]
        s = vals.sum()
        if s <= 0:
            return float(H/2), float(W/2), float("nan")
        yc = (vals * yy).sum() / s
        xc = (vals * xx).sum() / s
        var = ((vals * ((yy - yc) ** 2 + (xx - xc) ** 2)).sum() / s) / 2.0
        fwhm = 2.0 * (2.0 * np.log(2.0)) ** 0.5 * (var ** 0.5) if var > 0 else float("nan")
        return float(yc), float(xc), float(fwhm)

# -----------------------------------------------------------------------------
# Geometry (aperture / annulus)
# -----------------------------------------------------------------------------

def _ellipse_r2(H: int, W: int, yc: float, xc: float, a: float, b: float, theta: float,
                torch_mode: bool, device=None, dtype=None) -> BackendArray:
    if torch_mode:
        torch = _torch()
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        cosT = torch.cos(torch.tensor(theta, dtype=dtype, device=device))
        sinT = torch.sin(torch.tensor(theta, dtype=dtype, device=device))
        x0 = xx - xc
        y0 = yy - yc
        xprime =  cosT * x0 + sinT * y0
        yprime = -sinT * x0 + cosT * y0
        return (xprime / a) ** 2 + (yprime / b) ** 2
    else:
        import numpy as np
        yy, xx = np.mgrid[0:H, 0:W]
        x0 = xx - xc
        y0 = yy - yc
        cosT = np.cos(theta); sinT = np.sin(theta)
        xprime =  cosT * x0 + sinT * y0
        yprime = -sinT * x0 + cosT * y0
        return (xprime / a) ** 2 + (yprime / b) ** 2

def _aperture_mask(H: int, W: int, yc: float, xc: float, params: PhotometryParams,
                   fwhm: Optional[float], torch_mode: bool, device=None, dtype=None) -> Tuple[BackendArray, float]:
    # Choose shape/scales
    if params.shape == "circular":
        r = params.r_ap
        if params.adaptive_ap and (fwhm is not None) and fwhm == fwhm:
            r = max(1e-3, float(params.fwhm_scale) * float(fwhm))
        a, b, theta = r, r, 0.0
    else:
        a = params.a_ap
        b = params.b_ap
        if params.adaptive_ap and (fwhm is not None) and fwhm == fwhm:
            base = max(params.a_ap, 1e-3)
            scale = max(1e-3, float(params.fwhm_scale) * float(fwhm) / base)
            a = max(1e-3, params.a_ap * scale)
            b = max(1e-3, params.b_ap * scale)
        theta = params.theta

    r2 = _ellipse_r2(H, W, yc, xc, a, b, theta, torch_mode, device=device, dtype=dtype)
    if torch_mode:
        torch = _torch()
        m = (r2 <= 1.0).to(dtype=dtype)
        area = float(m.sum().detach().cpu().item())
        return m, area
    else:
        import numpy as np
        m = (r2 <= 1.0).astype(dtype or np.float32)
        area = float(m.sum())
        return m, area

def _annulus_mask(H: int, W: int, yc: float, xc: float, params: PhotometryParams,
                  torch_mode: bool, device=None, dtype=None) -> BackendArray:
    if params.shape == "circular":
        a1 = b1 = params.r_in
        a2 = b2 = params.r_out
        theta = 0.0
    else:
        base_max = max(params.a_ap, params.b_ap, 1e-6)
        a1 = params.r_in * (params.a_ap / base_max)
        b1 = params.r_in * (params.b_ap / base_max)
        a2 = params.r_out * (params.a_ap / base_max)
        b2 = params.r_out * (params.b_ap / base_max)
        theta = params.theta

    r2_in  = _ellipse_r2(H, W, yc, xc, a1, b1, theta, torch_mode, device=device, dtype=dtype)
    r2_out = _ellipse_r2(H, W, yc, xc, a2, b2, theta, torch_mode, device=device, dtype=dtype)
    if torch_mode:
        torch = _torch()
        return ((r2_in > 1.0) & (r2_out <= 1.0)).to(dtype=dtype)
    else:
        import numpy as np
        return ((r2_in > 1.0) & (r2_out <= 1.0)).astype(dtype or np.float32)

# -----------------------------------------------------------------------------
# Sigma clipping (NaN inliers preserved)
# -----------------------------------------------------------------------------

def _sigma_clip(y: BackendArray, lo: float, hi: float, iters: int) -> BackendArray:
    out = y
    for _ in range(max(1, iters)):
        med = _nanmedian(out)
        std = _nanstd(out)
        lo_thr = med - lo * std
        hi_thr = med + hi * std
        out = _where((out < lo_thr) | (out > hi_thr), _nan_like(out), out)
    return out

# -----------------------------------------------------------------------------
# PSF helpers
# -----------------------------------------------------------------------------

def _gaussian_psf(H: int, W: int, yc: float, xc: float, sig_y: float, sig_x: float,
                  torch_mode: bool, device=None, dtype=None) -> BackendArray:
    if torch_mode:
        torch = _torch()
        yy, xx = torch.meshgrid(
            torch.arange(H, device=device, dtype=dtype),
            torch.arange(W, device=device, dtype=dtype),
            indexing="ij"
        )
        g = torch.exp(-0.5 * ((yy - yc) ** 2 / (sig_y ** 2) + (xx - xc) ** 2 / (sig_x ** 2)))
        return g
    else:
        import numpy as np
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(-0.5 * (((yy - yc) ** 2) / (sig_y ** 2) + ((xx - xc) ** 2) / (sig_x ** 2)))
        return g.astype(dtype or np.float32)

# -----------------------------------------------------------------------------
# Main routine (batch-aware)
# -----------------------------------------------------------------------------

def photometry(frames: BackendArray, params: PhotometryParams) -> PhotometryResult:
    """
    Perform time-series photometry for a single source per stack.
    frames: [..., T, H, W]  (already calibrated to linear units)
    """
    # Normalize dtypes & axes
    F = _to_float(frames, dtype=params.dtype)
    F, _ = _normalize_time_axis(F, params.time_axis)           # [..., T, H, W]
    V = _align_optional(params.var, params.time_axis)          # [..., T, H, W] or None
    M_sat = _align_optional(params.mask_saturated, params.time_axis)  # [..., T, H, W] or None
    M_hot = _align_optional(params.mask_hot, params.time_axis)        # [..., T, H, W] or None

    torch_mode = _is_torch(F)
    if torch_mode:
        torch = _torch()

    # Masks applied up-front (NaN out bad pixels)
    if M_sat is not None:
        F = _where(M_sat, _nan_like(F), F)
    if M_hot is not None:
        F = _where(M_hot, _nan_like(F), F)

    # Shapes
    *P, T, H, W = F.shape
    Pshape = tuple(P)
    B = 1
    for s in Pshape:
        B *= int(s)
    # reshape to [B, T, H, W]
    if torch_mode:
        F2 = F.reshape(B, T, H, W)
        V2 = V.reshape(B, T, H, W) if V is not None else None
    else:
        import numpy as np
        F2 = F.reshape(B, T, H, W)
        V2 = V.reshape(B, T, H, W) if V is not None else None

    # Allocate outputs in [B, T]
    if torch_mode:
        device, dtype = F2.device, F2.dtype
        flux = torch.empty((B, T), dtype=dtype, device=device)
        sky = torch.empty((B, T), dtype=dtype, device=device) if params.use_annulus else None
        ferr = torch.empty((B, T), dtype=dtype, device=device) if V2 is not None else None
        xc = torch.empty((B, T), dtype=dtype, device=device)
        yc = torch.empty((B, T), dtype=dtype, device=device)
        fwhm = torch.empty((B, T), dtype=dtype, device=device)
        area_eff = torch.empty((B,), dtype=dtype, device=device)
    else:
        import numpy as np
        dtype = F2.dtype
        flux = np.empty((B, T), dtype=dtype)
        sky = np.empty((B, T), dtype=dtype) if params.use_annulus else None
        ferr = np.empty((B, T), dtype=dtype) if V2 is not None else None
        xc = np.empty((B, T), dtype=dtype)
        yc = np.empty((B, T), dtype=dtype)
        fwhm = np.empty((B, T), dtype=dtype)
        area_eff = np.empty((B,), dtype=dtype)

    # Iterate over batch & time
    for b in range(B):
        for t in range(T):
            fr = F2[b, t]

            # centroid & FWHM
            yc_t, xc_t, fwhm_t = _centroid_and_fwhm(fr)
            yc[b, t] = yc_t; xc[b, t] = xc_t; fwhm[b, t] = fwhm_t

            # aperture
            ap_mask, ap_area = _aperture_mask(
                H, W, yc_t, xc_t, params,
                fwhm_t if params.adaptive_ap else None,
                torch_mode, device=(F2.device if torch_mode else None), dtype=F2.dtype
            )
            if t == T - 1:
                area_eff[b] = ap_area

            # sky
            sky_level = 0.0
            if params.use_annulus:
                ann = _annulus_mask(H, W, yc_t, xc_t, params, torch_mode, device=(F2.device if torch_mode else None), dtype=F2.dtype)
                skypix = _where(ann > 0, fr, _nan_like(fr))
                if params.sky_sigma_clip is not None:
                    lo, hi = params.sky_sigma_clip
                    sk = skypix.reshape(-1)
                    for _ in range(max(1, params.sky_clip_iter)):
                        med = _nanmedian(sk)
                        std = _nanstd(sk)
                        lo_thr = med - lo * std
                        hi_thr = med + hi * std
                        sk = _where((sk < lo_thr) | (sk > hi_thr),
                                    _nan_like(sk), sk)
                    agg = _nanmedian(sk) if params.sky_agg == "median" else _nanmean(sk)
                else:
                    agg = _nanmedian(skypix) if params.sky_agg == "median" else _nanmean(skypix)
                sky_level = float(agg) if not _is_torch(agg) else float(agg.item())
                if sky is not None:
                    sky[b, t] = agg

            # Signal inside aperture
            ap_vals = _where(ap_mask > 0, fr, _nan_like(fr))
            if params.use_annulus:
                ap_vals = ap_vals - sky_level

            if params.method == "aperture":
                s = _nansum(ap_vals)
                flux[b, t] = s
                if V2 is not None:
                    v_ap = _where(ap_mask > 0, V2[b, t], _nan_like(fr))
                    veff = _nansum(v_ap)
                    # sqrt(sum var)
                    flux_err_bt = (veff ** 0.5)
                    if ferr is not None:
                        ferr[b, t] = flux_err_bt

            else:  # optimal
                # Build/choose PSF
                if params.psf is not None:
                    psf_t = params.psf if params.psf.ndim == 2 else params.psf[t, ...]
                    P = _to_float(psf_t, dtype=params.dtype)
                else:
                    sig_y = sig_x = float(params.psf_sigma) if isinstance(params.psf_sigma, (int, float)) else float(params.psf_sigma[0])
                    if not isinstance(params.psf_sigma, (int, float)) and len(params.psf_sigma) > 1:
                        sig_x = float(params.psf_sigma[1])
                    P = _gaussian_psf(H, W, yc_t, xc_t, sig_y, sig_x, torch_mode, device=(F2.device if torch_mode else None), dtype=F2.dtype)

                # Mask PSF to aperture & normalize by NaN-safe sum
                P = _where(ap_mask > 0, P, _nan_like(P))
                Psum = _nansum(_abs(P))
                if _is_torch(Psum):
                    Psum_val = float(Psum.item())
                else:
                    Psum_val = float(Psum)
                if Psum_val <= 1e-20:
                    # degenerate → fallback to uniform within aperture
                    P = _where(ap_mask > 0, _ones_like(P), _nan_like(P))
                    Psum = _nansum(P)
                P = P / _where(Psum <= 1e-20, Psum + 1e-20, Psum)

                # Weights
                if V2 is None:
                    Wgt = P
                else:
                    Vmap = _where(ap_mask > 0, V2[b, t], _nan_like(fr))
                    Wgt = P / _where(Vmap <= 1e-20, _ones_like(Vmap) * 1e-20, Vmap)

                S = _nan_to_num(ap_vals, 0.0)
                num = Wgt * S
                den = Wgt * P
                nsum = _nansum(num)
                dsum = _nansum(den)
                flux[b, t] = nsum / (_where(dsum <= 1e-20, dsum + 1e-20, dsum))

                if V2 is not None and ferr is not None:
                    Vmap = _where(ap_mask > 0, V2[b, t], _nan_like(fr))
                    denom = (P * P) / _where(Vmap <= 1e-20, _ones_like(Vmap) * 1e-20, Vmap)
                    dsum = _nansum(denom)
                    ferr[b, t] = (1.0 / (_where(dsum <= 1e-20, dsum + 1e-20, dsum))) ** 0.5

    # Reshape back to prefix Pshape + (T,)
    def _reshape_bt(xBT: Optional[BackendArray]) -> Optional[BackendArray]:
        if xBT is None:
            return None
        if torch_mode:
            return xBT.reshape(*Pshape, T)
        else:
            return xBT.reshape(*Pshape, T)

    flux = _reshape_bt(flux)
    sky = _reshape_bt(sky)
    ferr = _reshape_bt(ferr)
    xc = _reshape_bt(xc)
    yc = _reshape_bt(yc)
    fwhm = _reshape_bt(fwhm)

    # aperture area per batch (effective pixel count at last frame)
    if torch_mode:
        area_out = area_eff.reshape(*Pshape)
    else:
        area_out = area_eff.reshape(*Pshape)

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update({
            "method": params.method,
            "shape": params.shape,
            "adaptive_ap": params.adaptive_ap,
            "fwhm_scale": params.fwhm_scale,
            "use_annulus": params.use_annulus,
            "r_ap": params.r_ap, "a_ap": params.a_ap, "b_ap": params.b_ap, "theta": params.theta,
            "r_in": params.r_in, "r_out": params.r_out,
        })

    return PhotometryResult(
        flux=flux,
        flux_err=ferr,
        sky=sky,
        aperture_area=_to_float(area_out, dtype=params.dtype),
        xc=xc, yc=yc, fwhm=fwhm,
        meta=meta,
    )

# -----------------------------------------------------------------------------
# Self-tests (light, CPU-safe)
# -----------------------------------------------------------------------------

def _test_aperture_photometry_gaussian():
    import numpy as np
    rng = np.random.default_rng(0)
    T, H, W = 50, 32, 32
    yy, xx = np.mgrid[0:H, 0:W]
    yc, xc = 16.0, 16.0
    amp = 1000.0
    sig = 2.0
    psf = np.exp(-0.5 * (((yy - yc) ** 2 + (xx - xc) ** 2) / (sig ** 2)))
    frames = np.stack([amp * psf + rng.normal(0, 1.0, size=(H, W)) for _ in range(T)], axis=0).astype(np.float32)
    params = PhotometryParams(method="aperture", shape="circular", r_ap=3.0, use_annulus=True, r_in=6, r_out=10, return_intermediate=True)
    res = photometry(frames, params)
    flux_med = float(_np().nanmedian(res.flux))
    assert flux_med > 2000.0, f"aperture flux too low: {flux_med}"
    return True

def _test_optimal_photometry():
    import numpy as np
    rng = np.random.default_rng(1)
    T, H, W = 30, 24, 24
    yc, xc = 12.0, 12.0
    sig = 1.8
    yy, xx = np.mgrid[0:H, 0:W]
    psf = np.exp(-0.5 * (((yy - yc) ** 2 + (xx - xc) ** 2) / (sig ** 2))).astype(np.float32)
    psf /= psf.sum()
    amp = 5000.0
    stack = []
    var = []
    for _ in range(T):
        noise_var = 1.0
        frame = amp * psf + rng.normal(0, noise_var ** 0.5, size=(H, W))
        stack.append(frame.astype(np.float32))
        var.append(np.full((H, W), noise_var, dtype=np.float32))
    stack = np.stack(stack, axis=0); var = np.stack(var, axis=0)
    params = PhotometryParams(method="optimal", shape="circular", r_ap=5.0, use_annulus=False, var=var, psf=psf, return_intermediate=True)
    res = photometry(stack, params)
    med = float(_np().nanmedian(res.flux))
    assert abs(med - amp) / amp < 0.05, f"optimal flux bias too large: {med} vs {amp}"
    return True

if __name__ == "__main__":
    ok1 = _test_aperture_photometry_gaussian()
    ok2 = _test_optimal_photometry()
    print("photometry.py self-tests:", ok1 and ok2)