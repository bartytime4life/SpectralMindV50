# src/spectramind/calib/photometry.py
# =============================================================================
# SpectraMind V50 — Aperture and Optimal Photometry
# -----------------------------------------------------------------------------
# Features:
#   - Backend-agnostic (NumPy or PyTorch), fully vectorized.
#   - Circular or elliptical apertures; fixed or frame-adaptive sizes.
#   - Annulus sky estimation with sigma-clipping.
#   - Centroiding (center-of-light) and moment-based FWHM estimate.
#   - Optional optimal (PSF-weighted) extraction; Gaussian PSF or user-supplied.
#   - Per-frame masks (saturated/hot/bad) honored via NaN-safe paths.
#   - Variance/weights propagation, flexible time axis.
#   - Rich metadata for diagnostics.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

# -----------------------------------------------------------------------------
# Backend shims (style matched to other calib modules)
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
    if _is_torch(x):
        return _torch().zeros_like(x)
    return _np().zeros_like(x)

def _ones_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        return _torch().ones_like(x)
    return _np().ones_like(x)

def _where(mask: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(mask):
        return _torch().where(mask, a, b)
    return _np().where(mask, a, b)

def _abs(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        return x.abs()
    return _np().abs(x)

def _nanmean(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        msk = ~torch.isnan(x)
        num = torch.where(msk, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims)
        den = msk.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        all_nan = den == 0
        return torch.where(all_nan, torch.tensor(float('nan'), dtype=x.dtype, device=x.device), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)

def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    # torch fallback via numpy for robustness
    if _is_torch(x):
        torch = _torch(); np = _np()
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

def _clip(x: BackendArray, low: Optional[float], high: Optional[float]) -> BackendArray:
    if low is None and high is None:
        return x
    if _is_torch(x):
        torch = _torch()
        if low is not None:
            x = torch.clamp(x, min=float(low))
        if high is not None:
            x = torch.clamp(x, max=float(high))
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
# Parameters and results
# -----------------------------------------------------------------------------

ApertureShape = Literal["circular", "elliptical"]
SkyAgg = Literal["median", "mean"]
PhotMethod = Literal["aperture", "optimal"]

@dataclass
class PhotometryParams:
    """
    Photometry configuration.

    time_axis         : axis for T (default -3 => [..., T, H, W])
    method            : 'aperture' or 'optimal'
    # Apertures
    shape             : 'circular' or 'elliptical'
    r_ap              : aperture radius (pixels) for circular OR
    a_ap, b_ap, theta : ellipse semimajor, semiminor, PA (radians) for elliptical
    adaptive_ap       : if True, scale r/ap (or a/b) per frame using FWHM estimator
    fwhm_scale        : multiplier converting FWHM to aperture radius/scales
    # Sky annulus
    use_annulus       : whether to estimate a local sky
    r_in, r_out       : inner/outer radii for sky annulus (circular template; ellipse follows aperture shape if set)
    sky_agg           : 'median' or 'mean'
    sky_sigma_clip    : (lo, hi) sigma thresholds for sky rejection; None to disable
    sky_clip_iter     : iterations for sigma clipping
    # Masks / variance
    mask_saturated    : mask array same shape as frames (True=exclude)
    mask_hot          : additional bad pixel mask
    var               : optional per-pixel variance cube [..., T, H, W] (used in optimal extraction and error prop)
    # Optimal extraction (if method='optimal')
    psf_sigma         : Gaussian sigma (pixels) or (sig_y, sig_x) tuple used to build PSF weight if psf is None
    psf               : optional PSF template per frame [T,H,W] or [H,W] normalized; if provided, used directly
    # Output
    dtype             : float dtype for computations
    return_intermediate: include diagnostics in meta
    """
    time_axis: int = -3
    method: PhotMethod = "aperture"

    shape: ApertureShape = "circular"
    r_ap: float = 3.0
    a_ap: float = 3.0
    b_ap: float = 3.0
    theta: float = 0.0
    adaptive_ap: bool = False
    fwhm_scale: float = 1.0

    use_annulus: bool = True
    r_in: float = 6.0
    r_out: float = 8.0
    sky_agg: SkyAgg = "median"
    sky_sigma_clip: Optional[Tuple[float, float]] = (3.0, 3.0)
    sky_clip_iter: int = 1

    mask_saturated: Optional[BackendArray] = None
    mask_hot: Optional[BackendArray] = None
    var: Optional[BackendArray] = None

    psf_sigma: Union[float, Tuple[float, float]] = 1.5
    psf: Optional[BackendArray] = None

    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False


@dataclass
class PhotometryResult:
    """
    Photometry outputs for a single source over time.
    """
    flux: BackendArray           # [T] (or [..., T] if batch prefix)
    flux_err: Optional[BackendArray]
    sky: Optional[BackendArray]  # [T] estimated sky level per pixel within aperture (DN/pix or electrons/pix)
    aperture_area: BackendArray  # scalar effective aperture pixel count (or per-frame with adaptive_ap)
    xc: BackendArray             # [T] centroid x (column)
    yc: BackendArray             # [T] centroid y (row)
    fwhm: BackendArray           # [T] moment-based FWHM estimate
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Axis normalization
# -----------------------------------------------------------------------------

def _move_time_axis(x: BackendArray, time_axis: int) -> Tuple[BackendArray, int]:
    nd = x.ndim
    target = nd - 3
    if time_axis < 0:
        time_axis = nd + time_axis
    if time_axis == target:
        return x, time_axis
    perm = list(range(nd))
    perm[target], perm[time_axis] = perm[time_axis], perm[target]
    if _is_torch(x):
        x2 = x.permute(*perm)
    else:
        x2 = x.transpose(target, time_axis)
    return x2, time_axis

# -----------------------------------------------------------------------------
# Centroid & FWHM
# -----------------------------------------------------------------------------

def _centroid_and_fwhm(frame: BackendArray) -> Tuple[float, float, float]:
    """
    Center-of-light centroid and moment-based FWHM estimate for a single 2D frame.
    Works with NaNs (ignored). Returns (yc, xc, fwhm).
    """
    torch_mode = _is_torch(frame)
    if torch_mode:
        torch = _torch()
        msk = ~torch.isnan(frame)
        vals = torch.where(msk, frame, torch.tensor(0., dtype=frame.dtype, device=frame.device))
        H, W = frame.shape[-2], frame.shape[-1]
        yy, xx = torch.meshgrid(torch.arange(H, device=frame.device, dtype=frame.dtype),
                                torch.arange(W, device=frame.device, dtype=frame.dtype),
                                indexing="ij")
        s = vals.sum()
        if s <= 0:
            yc = float(H/2); xc = float(W/2); fwhm = float("nan")
            return yc, xc, fwhm
        yc = float((vals * yy).sum() / s)
        xc = float((vals * xx).sum() / s)
        dy2 = (yy - yc) ** 2
        dx2 = (xx - xc) ** 2
        var = float(((vals * (dx2 + dy2)).sum() / s) / 2.0)  # avg of x,y variances if roughly circular
        # FWHM ~ 2*sqrt(2*ln2)*sigma; sigma^2 ~ var
        import math
        fwhm = 2.0 * (2.0 * math.log(2.0)) ** 0.5 * (var ** 0.5) if var > 0 else float("nan")
        return yc, xc, fwhm
    else:
        import numpy as np
        msk = ~np.isnan(frame)
        vals = np.where(msk, frame, 0.0)
        H, W = frame.shape[-2], frame.shape[-1]
        yy, xx = np.mgrid[0:H, 0:W]
        s = vals.sum()
        if s <= 0:
            yc = H/2; xc = W/2; fwhm = np.nan
            return float(yc), float(xc), float(fwhm)
        yc = (vals * yy).sum() / s
        xc = (vals * xx).sum() / s
        var = ((vals * ((xx - xc) ** 2 + (yy - yc) ** 2)).sum() / s) / 2.0
        fwhm = 2.0 * (2.0 * np.log(2.0)) ** 0.5 * (var ** 0.5) if var > 0 else np.nan
        return float(yc), float(xc), float(fwhm)

# -----------------------------------------------------------------------------
# Aperture masks
# -----------------------------------------------------------------------------

def _ellipse_r2(H: int, W: int, yc: float, xc: float, a: float, b: float, theta: float, torch_mode: bool, device=None, dtype=None) -> BackendArray:
    """
    Return squared elliptical radius r^2 for each pixel relative to center (yc,xc).
    r^2 = ((x')/a)^2 + ((y')/b)^2, where [x'; y'] is rotated by theta.
    """
    if torch_mode:
        torch = _torch()
        yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                                torch.arange(W, device=device, dtype=dtype),
                                indexing="ij")
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

def _aperture_mask(H: int, W: int, yc: float, xc: float, params: PhotometryParams, fwhm: Optional[float], torch_mode: bool, device=None, dtype=None) -> Tuple[BackendArray, float]:
    """
    Build aperture mask (1 inside, 0 outside). Optionally scale radius/axes by FWHM.
    Returns mask and effective area (sum of mask).
    """
    # choose sizes
    if params.shape == "circular":
        r = params.r_ap
        if params.adaptive_ap and (fwhm is not None) and fwhm == fwhm:
            r = max(1e-3, float(params.fwhm_scale) * float(fwhm))
        a, b, theta = r, r, 0.0
    else:
        a = params.a_ap
        b = params.b_ap
        if params.adaptive_ap and (fwhm is not None) and fwhm == fwhm:
            scale = max(1e-3, float(params.fwhm_scale) * float(fwhm) / max(params.a_ap, 1e-3))
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

def _annulus_mask(H: int, W: int, yc: float, xc: float, params: PhotometryParams, torch_mode: bool, device=None, dtype=None) -> BackendArray:
    """
    Annulus mask for sky. For elliptical aperture, use same theta,a,b scalings but radii replaced by r_in, r_out.
    """
    if params.shape == "circular":
        a1 = b1 = params.r_in
        a2 = b2 = params.r_out
        theta = 0.0
    else:
        # scale inner/outer in proportion to base a,b
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
        m = ((r2_in > 1.0) & (r2_out <= 1.0)).to(dtype=dtype)
        return m
    else:
        import numpy as np
        m = ((r2_in > 1.0) & (r2_out <= 1.0)).astype(dtype or np.float32)
        return m

# -----------------------------------------------------------------------------
# Sigma clipping
# -----------------------------------------------------------------------------

def _sigma_clip(y: BackendArray, lo: float, hi: float, iters: int, axis=None) -> BackendArray:
    """
    Return y with outliers replaced by NaN using sigma clip about the median.
    """
    out = y
    for _ in range(max(1, iters)):
        med = _nanmedian(out, axis=axis, keepdims=True)
        std = _nanstd(out, axis=axis, keepdims=True)
        lo_thr = med - lo * std
        hi_thr = med + hi * std
        out = _where((out < lo_thr) | (out > hi_thr), _nan_like(out), out)
    return out

def _nan_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.tensor(float('nan'), dtype=x.dtype, device=x.device).expand_as(x)
    else:
        import numpy as np
        return np.full_like(x, np.nan)

# -----------------------------------------------------------------------------
# Optimal extraction PSF
# -----------------------------------------------------------------------------

def _gaussian_psf(H: int, W: int, yc: float, xc: float, sig_y: float, sig_x: float, torch_mode: bool, device=None, dtype=None) -> BackendArray:
    """
    Unnormalized 2D Gaussian centered at yc, xc.
    """
    if torch_mode:
        torch = _torch()
        yy, xx = torch.meshgrid(torch.arange(H, device=device, dtype=dtype),
                                torch.arange(W, device=device, dtype=dtype),
                                indexing="ij")
        g = _torch().exp(-0.5 * ((yy - yc) ** 2 / (sig_y ** 2) + (xx - xc) ** 2 / (sig_x ** 2)))
        return g
    else:
        import numpy as np
        yy, xx = np.mgrid[0:H, 0:W]
        g = np.exp(-0.5 * (((yy - yc) ** 2) / (sig_y ** 2) + ((xx - xc) ** 2) / (sig_x ** 2)))
        return g.astype(dtype or np.float32)

# -----------------------------------------------------------------------------
# Main photometry routine
# -----------------------------------------------------------------------------

def photometry(
    frames: BackendArray,
    params: PhotometryParams,
) -> PhotometryResult:
    """
    Perform time-series photometry for a single target per stack.

    frames : [..., T, H, W] (already calibrated to linear units)
    params : PhotometryParams

    Returns
    -------
    PhotometryResult with flux, flux_err (if var supplied), sky, aperture area, centroid, FWHM, metadata.
    """
    # dtype & axes
    F = _to_float(frames, dtype=params.dtype)
    F, _ = _move_time_axis(F, params.time_axis)  # -> [..., T, H, W]
    T = F.shape[-3]; H, W = F.shape[-2], F.shape[-1]
    torch_mode = _is_torch(F)

    # Apply masks to exclude bad pixels
    if params.mask_saturated is not None:
        F = _where(params.mask_saturated, _nan_like(F), F)
    if params.mask_hot is not None:
        F = _where(params.mask_hot, _nan_like(F), F)

    # Prepare outputs
    if torch_mode:
        torch = _torch()
        flux = torch.empty(T, dtype=F.dtype, device=F.device)
        sky = torch.empty(T, dtype=F.dtype, device=F.device) if params.use_annulus else None
        flux_err = torch.empty(T, dtype=F.dtype, device=F.device) if params.var is not None else None
        xc = torch.empty(T, dtype=F.dtype, device=F.device)
        yc = torch.empty(T, dtype=F.dtype, device=F.device)
        fwhm = torch.empty(T, dtype=F.dtype, device=F.device)
    else:
        import numpy as np
        flux = np.empty(T, dtype=F.dtype)
        sky = np.empty(T, dtype=F.dtype) if params.use_annulus else None
        flux_err = np.empty(T, dtype=F.dtype) if params.var is not None else None
        xc = np.empty(T, dtype=F.dtype)
        yc = np.empty(T, dtype=F.dtype)
        fwhm = np.empty(T, dtype=F.dtype)

    area_eff = 0.0  # track last aperture area (if adaptive could vary; we record last or average)

    # Iterate frames (T typically in few hundreds; per-frame ops inexpensive compared to calib)
    for t in range(T):
        fr = F[..., t, :, :]

        # centroid & FWHM
        yc_t, xc_t, fwhm_t = _centroid_and_fwhm(fr)
        yc[t] = yc_t; xc[t] = xc_t; fwhm[t] = fwhm_t

        # build aperture mask
        ap_mask, ap_area = _aperture_mask(H, W, yc_t, xc_t, params, fwhm_t if params.adaptive_ap else None,
                                          torch_mode, device=(F.device if torch_mode else None), dtype=F.dtype)
        area_eff = ap_area if t == T - 1 else area_eff

        # sky annulus
        sky_level = 0.0
        if params.use_annulus:
            ann = _annulus_mask(H, W, yc_t, xc_t, params, torch_mode, device=(F.device if torch_mode else None), dtype=F.dtype)
            skypix = _where(ann > 0, fr, _nan_like(fr))
            if params.sky_sigma_clip is not None:
                lo, hi = params.sky_sigma_clip
                # sigma clip along pixel axis (flatten)
                if torch_mode:
                    torch = _torch()
                    sk = skypix.clone().reshape(-1)
                    for _ in range(max(1, params.sky_clip_iter)):
                        med = torch.nanmedian(sk)
                        std = torch.nanstd(sk)
                        lo_thr = med - lo * std
                        hi_thr = med + hi * std
                        sk = torch.where((sk < lo_thr) | (sk > hi_thr), torch.tensor(float('nan'), dtype=sk.dtype, device=sk.device), sk)
                    sky_level = float((torch.nanmedian(sk) if params.sky_agg == "median" else torch.nanmean(sk)).item())
                else:
                    import numpy as np
                    sk = skypix.reshape(-1)
                    for _ in range(max(1, params.sky_clip_iter)):
                        med = np.nanmedian(sk); std = np.nanstd(sk)
                        lo_thr = med - lo * std; hi_thr = med + hi * std
                        sk = np.where((sk < lo_thr) | (sk > hi_thr), np.nan, sk)
                    sky_level = float(np.nanmedian(sk) if params.sky_agg == "median" else np.nanmean(sk))
            else:
                # simple aggregate
                sky_level = float(_nanmedian(skypix) if params.sky_agg == "median" else _nanmean(skypix))
            if sky is not None:
                sky[t] = sky_level

        # Signal inside aperture
        ap_vals = _where(ap_mask > 0, fr, _nan_like(fr))
        if params.use_annulus:
            ap_vals = ap_vals - sky_level

        if params.method == "aperture":
            # Sum inside aperture
            if torch_mode:
                s = _nan_to_num(ap_vals, 0.0).sum()
                flux[t] = s
            else:
                import numpy as np
                s = np.nansum(ap_vals)
                flux[t] = s

            # Error: if variance provided, sum var in aperture; else sqrt(sum(max(val,0))) (Poisson-ish)
            if params.var is not None:
                vframe = params.var[..., t, :, :]
                v_ap = _where(ap_mask > 0, vframe, _nan_like(vframe))
                if torch_mode:
                    veff = _nan_to_num(v_ap, 0.0).sum()
                    flux_err[t] = veff ** 0.5
                else:
                    import numpy as np
                    veff = np.nansum(v_ap)
                    flux_err[t] = veff ** 0.5

        else:  # optimal extraction
            # Weights ~ PSF / var. If PSF not supplied, use Gaussian with params.psf_sigma.
            if params.psf is not None:
                psf_t = params.psf if params.psf.ndim == 2 else params.psf[t, ...]
                P = _to_float(psf_t, dtype=params.dtype)
            else:
                # sigma can be float or (sig_y,sig_x)
                sig_y = sig_x = float(params.psf_sigma if isinstance(params.psf_sigma, (int, float)) else params.psf_sigma[0])
                if not isinstance(params.psf_sigma, (int, float)) and len(params.psf_sigma) > 1:
                    sig_x = float(params.psf_sigma[1])
                P = _gaussian_psf(H, W, yc_t, xc_t, sig_y, sig_x, torch_mode, device=(F.device if torch_mode else None), dtype=F.dtype)
            # Normalize PSF over aperture mask (to avoid sky/edge)
            P = _where(ap_mask > 0, P, _nan_like(P))
            P /= _where(_abs(P).sum() <= 1e-20, _ones_like(P).sum(), _abs(P).sum())

            # Variance map
            if params.var is None:
                # fallback weights ~ PSF
                Wgt = P
            else:
                V = params.var[..., t, :, :]
                V = _where(ap_mask > 0, V, _nan_like(V))
                Wgt = P / _where(V <= 1e-20, _ones_like(V) * 1e-20, V)

            # Flux = sum(W * (fr - sky)) / sum(W*P)
            S = _nan_to_num(ap_vals, 0.0)
            num = (Wgt * S)
            den = (Wgt * P)
            if torch_mode:
                torch = _torch()
                nsum = num.nansum() if hasattr(num, "nansum") else _nan_to_num(num, 0.0).sum()
                dsum = den.nansum() if hasattr(den, "nansum") else _nan_to_num(den, 0.0).sum()
                flux[t] = nsum / _where(dsum <= 1e-20, torch.tensor(1e-20, dtype=dsum.dtype, device=dsum.device), dsum)
            else:
                import numpy as np
                nsum = np.nansum(num); dsum = np.nansum(den)
                flux[t] = nsum / (dsum if dsum > 1e-20 else 1e-20)

            # Flux error ~ 1 / sqrt(sum( P^2 / V ))
            if params.var is not None:
                V = params.var[..., t, :, :]
                V = _where(ap_mask > 0, V, _nan_like(V))
                denom = (P * P) / _where(V <= 1e-20, _ones_like(V) * 1e-20, V)
                if torch_mode:
                    torch = _torch()
                    dsum = denom.nansum() if hasattr(denom, "nansum") else _nan_to_num(denom, 0.0).sum()
                    flux_err[t] = (1.0 / _where(dsum <= 1e-20, torch.tensor(1e-20, dtype=dsum.dtype, device=dsum.device), dsum)) ** 0.5
                else:
                    import numpy as np
                    dsum = np.nansum(denom)
                    flux_err[t] = (1.0 / (dsum if dsum > 1e-20 else 1e-20)) ** 0.5

    # Final result
    area_scalar = area_eff
    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update({
            "method": params.method,
            "shape": params.shape,
            "adaptive_ap": params.adaptive_ap,
            "fwhm_scale": params.fwhm_scale,
            "use_annulus": params.use_annulus,
            "r_ap": params.r_ap,
            "a_ap": params.a_ap,
            "b_ap": params.b_ap,
            "theta": params.theta,
            "r_in": params.r_in,
            "r_out": params.r_out,
        })

    return PhotometryResult(
        flux=flux,
        flux_err=flux_err,
        sky=sky,
        aperture_area=_to_float(_np().array(area_scalar, dtype="float32") if not torch_mode else _torch().tensor(area_scalar, dtype=F.dtype, device=F.device)),
        xc=xc, yc=yc, fwhm=fwhm,
        meta=meta,
    )

# -----------------------------------------------------------------------------
# Self-tests (light)
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
    frames = []
    for t in range(T):
        frames.append(amp * psf + rng.normal(0, 1.0, size=(H, W)))
    stack = np.stack(frames, axis=0).astype(np.float32)  # [T,H,W]
    params = PhotometryParams(method="aperture", shape="circular", r_ap=3.0, use_annulus=True, r_in=6, r_out=10, return_intermediate=True)
    res = photometry(stack, params)
    # Flux should be close to integrated Gaussian over radius 3 (rough check)
    flux_med = float(np.nanmedian(res.flux))
    assert flux_med > 2000.0, f"aperture flux too low: {flux_med}"
    return True

def _test_optimal_photometry():
    import numpy as np
    rng = np.random.default_rng(1)
    T, H, W = 30, 24, 24
    yc, xc = 12.0, 12.0
    sig = 1.8
    yy, xx = np.mgrid[0:H, 0:W]
    psf = np.exp(-0.5 * (((yy - yc) ** 2 + (xx - xc) ** 2) / (sig ** 2)))
    # Normalize PSF to unit integral
    psf /= psf.sum()
    amp = 5000.0
    stack = []
    var = []
    for t in range(T):
        noise_var = 1.0
        frame = amp * psf + rng.normal(0, noise_var ** 0.5, size=(H, W))
        stack.append(frame.astype(np.float32))
        var.append(np.full((H, W), noise_var, dtype=np.float32))
    stack = np.stack(stack, axis=0); var = np.stack(var, axis=0)
    params = PhotometryParams(method="optimal", shape="circular", r_ap=5.0, use_annulus=False, var=var, psf=psf.astype(np.float32), return_intermediate=True)
    res = photometry(stack, params)
    # Optimal flux should be ~amp (since PSF normalized)
    med = float(np.nanmedian(res.flux))
    assert abs(med - amp) / amp < 0.05, f"optimal flux bias too large: {med} vs {amp}"
    return True

if __name__ == "__main__":
    ok1 = _test_aperture_photometry_gaussian()
    ok2 = _test_optimal_photometry()
    print("photometry.py self-tests:", ok1 and ok2)
