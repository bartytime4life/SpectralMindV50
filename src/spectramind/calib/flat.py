# src/spectramind/calib/flat.py
# =============================================================================
# SpectraMind V50 — Flat-field modeling & application (PRNU + illumination)
# -----------------------------------------------------------------------------
# Build a master flat from stacks of flat frames:
#   - robust temporal aggregation (mean/median)
#   - optional cosmic-ray rejection (MAD z-score)
#   - large-scale illumination estimation: gaussian low-pass or polynomial
#   - PRNU map normalized to mean 1 (illumination removed)
#   - hot/bad pixel masks (from PRNU and temporal var outliers)
#   - per-pixel variance & weights
#
# Apply the master flat to science frames (divide), with optional illumination
# reinjection/removal and variance propagation. Backend agnostic (NumPy/Torch).
# Canonical image layout [..., H, W] and stacks [..., N, H, W] (configurable axis).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

# -----------------------------------------------------------------------------
# Backend shims (kept consistent with adc.py / cds.py / dark.py)
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
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims)
        den = mask.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        all_nan = den == 0
        return torch.where(all_nan, torch.tensor(float('nan'), dtype=x.dtype, device=x.device), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)

def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        # robust fallback via numpy (stable and simple)
        torch = _torch(); np = _np()
        x_np = x.detach().cpu().numpy()
        m_np = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
        return torch.from_numpy(m_np).to(device=x.device, dtype=x.dtype)
    else:
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)

def _nanstd(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        m = _nanmean(x, axis=axis, keepdims=True)
        v = _nanmean((x - m) ** 2, axis=axis, keepdims=keepdims)
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

# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------

AggMode = Literal["mean", "median"]
IllumModel = Literal["none", "gaussian", "poly2"]

@dataclass
class FlatBuildParams:
    """
    Controls building of master flat (PRNU + Illumination).

    time_axis         : axis of stack (default -3 => [..., N, H, W])
    agg               : temporal aggregator ("mean"|"median")
    cosmic_reject     : enable MAD z-score rejection across time
    cr_zmax           : CR rejection threshold
    cr_iter           : CR rejection iterations
    illum_model       : 'none' | 'gaussian' | 'poly2'
        - gaussian: low-pass blur with sigma (pixels)
        - poly2  : quadratic surface fit (x,y,xy,x^2,y^2)
    illum_sigma       : gaussian sigma (pixels) if illum_model='gaussian'
    hot_sigma         : PRNU > mean + hot_sigma*std -> hot
    bad_sigma         : temporal var > mean + bad_sigma*std -> bad
    clip_prnu         : (low, high) clipping on PRNU map (e.g., (0.2, 5.0))
    dtype             : output dtype
    return_intermediate: stash debug info
    """
    time_axis: int = -3
    agg: AggMode = "median"
    cosmic_reject: bool = True
    cr_zmax: float = 6.0
    cr_iter: int = 1
    illum_model: IllumModel = "gaussian"
    illum_sigma: float = 32.0
    hot_sigma: Optional[float] = 6.0
    bad_sigma: Optional[float] = 6.0
    clip_prnu: Optional[Tuple[Optional[float], Optional[float]]] = (0.1, 10.0)
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class FlatModel:
    """
    Master flat product: PRNU map normalized to mean 1 (illumination removed),
    illumination field (if modeled), per-pixel variance/weights, masks, and meta.
    """
    prnu: BackendArray               # unity-mean PRNU, illumination removed
    illum: Optional[BackendArray]    # large-scale illumination field (>=0), or None
    var: Optional[BackendArray]      # per-pixel temporal var of master (pre-normalize)
    weights: Optional[BackendArray]  # effective sample count
    mask_hot: Optional[BackendArray] # hot pixel mask from PRNU outliers
    mask_bad: Optional[BackendArray] # bad noisy pixel mask from var outliers
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class FlatBuildResult:
    master: FlatModel

@dataclass
class FlatApplyParams:
    """
    How to apply the flat to science frames.

    use_illum          : if True and illum present, divide science by illum first
    mask_out_hot_bad   : if True, keep hot/bad pixels masked (NaN) in output
    propagate_var      : if True and var provided, propagate variance through division
    clip_out           : optional (low, high) clip on corrected image
    dtype              : output dtype
    return_intermediate: stash debug info
    """
    use_illum: bool = True
    mask_out_hot_bad: bool = True
    propagate_var: bool = True
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class FlatApplyResult:
    corrected: BackendArray
    var: Optional[BackendArray]
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Internals: axis helpers, robust temporal, illumination fits
# -----------------------------------------------------------------------------

def _move_stack_axis(x: BackendArray, time_axis: int) -> Tuple[BackendArray, int]:
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

def _robust_temporal(x: BackendArray, axis: int, agg: AggMode,
                     cosmic_reject: bool, zmax: float, iters: int) -> Tuple[BackendArray, BackendArray, BackendArray]:
    cur = x
    if not cosmic_reject:
        if agg == "mean":
            val = _nanmean(cur, axis=axis, keepdims=False)
        else:
            val = _nanmedian(cur, axis=axis, keepdims=False)
        # weights, var
        if _is_torch(x):
            torch = _torch()
            w = (~torch.isnan(cur)).sum(dim=axis)
            v = _nanmean((cur - val.unsqueeze(axis)) ** 2, axis=axis, keepdims=False)
        else:
            np = _np()
            w = np.sum(~np.isnan(cur), axis=axis)
            # broadcast val
            v = _nanmean((cur - val[(...,) + (None,)* (cur.ndim - val.ndim)]) ** 2, axis=axis, keepdims=False)
        return val, v, w

    torch_mode = _is_torch(x)
    for _ in range(max(1, iters)):
        med = _nanmedian(cur, axis=axis, keepdims=True)
        mad = _nanmedian(_abs(cur - med), axis=axis, keepdims=True)
        if torch_mode:
            torch = _torch()
            eps = torch.tensor(1e-12, dtype=mad.dtype, device=mad.device)
            mad = _where(mad <= eps, eps, mad)
            z = 0.6745 * (cur - med) / mad
            rej = _abs(z) > zmax
            cur = _where(rej, torch.tensor(float('nan'), dtype=cur.dtype, device=cur.device), cur)
        else:
            np = _np()
            mad = _where(mad <= 1e-12, np.array(1e-12, dtype=mad.dtype), mad)
            z = 0.6745 * (cur - med) / mad
            rej = _abs(z) > zmax
            cur = cur.copy()
            cur[rej] = np.nan

    if agg == "mean":
        val = _nanmean(cur, axis=axis, keepdims=False)
    else:
        val = _nanmedian(cur, axis=axis, keepdims=False)

    if torch_mode:
        torch = _torch()
        w = (~torch.isnan(cur)).sum(dim=axis)
        v = _nanmean((cur - val.unsqueeze(axis)) ** 2, axis=axis, keepdims=False)
    else:
        np = _np()
        w = np.sum(~np.isnan(cur), axis=axis)
        v = _nanmean((cur - val[(...,) + (None,)* (cur.ndim - val.ndim)]) ** 2, axis=axis, keepdims=False)

    return val, v, w

def _gaussian_blur(image: BackendArray, sigma: float) -> BackendArray:
    """
    Simple separable gaussian blur (NumPy) for large-scale illumination.
    Torch path falls back to CPU numpy to keep deps minimal; cost is acceptable
    since this runs once per build.
    """
    np = _np()
    from math import ceil
    if sigma <= 0:
        return image

    def _blur_np(img_np: "np.ndarray", sigma: float) -> "np.ndarray":
        from scipy.ndimage import gaussian_filter
        # Use scipy if available (fast, robust)
        return gaussian_filter(img_np, sigma=sigma, mode="nearest")

    # Try scipy; if not available, implement a simple separable kernel
    try:
        if _is_torch(image):
            x_np = image.detach().cpu().numpy()
            out_np = _blur_np(x_np, sigma)
            torch = _torch()
            return torch.from_numpy(out_np).to(device=image.device, dtype=image.dtype)
        else:
            return _blur_np(image, sigma)
    except Exception:
        # Lightweight fallback: build 1D gaussian kernel and convolve separably
        def _kernel(sig: float) -> "np.ndarray":
            radius = max(1, int(ceil(3 * sig)))
            xs = np.arange(-radius, radius + 1, dtype=np.float64)
            k = np.exp(-(xs ** 2) / (2.0 * sig * sig))
            k /= k.sum()
            return k

        def _sep_conv(img_np: "np.ndarray", k: "np.ndarray") -> "np.ndarray":
            # convolve H then W per channel
            from numpy.lib.stride_tricks import sliding_window_view as swv
            H, W = img_np.shape[-2], img_np.shape[-1]
            r = k.shape[0] // 2
            # pad
            pad = ((0,0),) * (img_np.ndim - 2) + ((r,r), (r,r))
            imgp = np.pad(img_np, pad, mode="edge")
            # H conv
            v = swv(imgp, window_shape=(1, k.size))  # not trivially aligned across last dims; fallback to for-loops:
            # Simpler and still OK: do 1D conv along each axis using fftconvolve
            from scipy.signal import fftconvolve
            tmp = fftconvolve(imgp, k.reshape(1, -1), mode="same")
            tmp = fftconvolve(tmp, k.reshape(-1, 1), mode="same")
            return tmp[..., r:-r, r:-r]

        if _is_torch(image):
            torch = _torch()
            x_np = image.detach().cpu().numpy()
            k = _kernel(float(sigma))
            out_np = _sep_conv(x_np, k)
            return torch.from_numpy(out_np).to(device=image.device, dtype=image.dtype)
        else:
            k = _kernel(float(sigma))
            return _sep_conv(image, k)

def _fit_poly2(image: BackendArray) -> BackendArray:
    """
    Fit a 2D quadratic surface: a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2.
    Torch path falls back to CPU numpy solve.
    """
    np = _np()
    H, W = image.shape[-2], image.shape[-1]
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.stack([np.ones_like(xx), xx, yy, xx*yy, xx*xx, yy*yy], axis=-1)  # [H,W,6]
    if _is_torch(image):
        x_np = image.detach().cpu().numpy()
        A2 = A.reshape(-1, 6)
        b = x_np.reshape(-1)
        coef, *_ = np.linalg.lstsq(A2, b, rcond=None)
        fit_np = (A2 @ coef).reshape(H, W)
        torch = _torch()
        return torch.from_numpy(fit_np).to(device=image.device, dtype=image.dtype)
    else:
        A2 = A.reshape(-1, 6)
        b = image.reshape(-1)
        coef, *_ = np.linalg.lstsq(A2, b, rcond=None)
        fit = (A2 @ coef).reshape(H, W)
        return fit

def _make_masks_from_prnu_and_var(prnu: BackendArray, var: BackendArray,
                                  hot_sigma: Optional[float], bad_sigma: Optional[float]) -> Tuple[Optional[BackendArray], Optional[BackendArray], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if hot_sigma is None and bad_sigma is None:
        return None, None, meta

    if _is_torch(prnu) or _is_torch(var):
        torch = _torch()
        # PRNU stats around unity
        m = torch.nanmean(prnu)
        s = torch.nanstd(prnu)
        hot = (prnu > (m + (hot_sigma or 0.0) * s)) if hot_sigma is not None else None

        vm = torch.nanmean(var)
        vs = torch.nanstd(var)
        bad = (var > (vm + (bad_sigma or 0.0) * vs)) if bad_sigma is not None else None

        meta.update(dict(prnu_mean=float(m), prnu_std=float(s), var_mean=float(vm), var_std=float(vs)))
        return hot, bad, meta
    else:
        np = _np()
        m = np.nanmean(prnu); s = np.nanstd(prnu)
        hot = (prnu > (m + (hot_sigma or 0.0) * s)) if hot_sigma is not None else None

        vm = np.nanmean(var); vs = np.nanstd(var)
        bad = (var > (vm + (bad_sigma or 0.0) * vs)) if bad_sigma is not None else None

        meta.update(dict(prnu_mean=float(m), prnu_std=float(s), var_mean=float(vm), var_std=float(vs)))
        return hot, bad, meta

# -----------------------------------------------------------------------------
# Build master flat
# -----------------------------------------------------------------------------

def build_master_flat(
    flat_stack: BackendArray,
    build: FlatBuildParams,
) -> FlatBuildResult:
    """
    Build a master flat (PRNU + Illumination) from a stack of flat-field frames.

    flat_stack : [..., N, H, W] (or time axis anywhere; see build.time_axis)
    build      : FlatBuildParams

    Returns
    -------
    FlatBuildResult with FlatModel:
      - prnu : unity-mean PRNU (illumination removed)
      - illum: large-scale illumination (>=0) if modeled
      - var  : temporal var (pre-normalization)
      - weights: effective frame counts
      - mask_hot/bad: boolean masks from PRNU/var outliers
    """
    x = _to_float(flat_stack, dtype=build.dtype)
    x, _ = _move_stack_axis(x, build.time_axis)  # -> [..., N, H, W]
    N = x.shape[-3]
    if N < 1:
        raise ValueError("build_master_flat: need at least one flat frame")

    # Robust temporal aggregation
    master_raw, var, weights = _robust_temporal(
        x, axis=-3, agg=build.agg,
        cosmic_reject=build.cosmic_reject,
        zmax=build.cr_zmax, iters=build.cr_iter
    )
    # Large-scale illumination
    illum = None
    if build.illum_model == "gaussian":
        illum = _gaussian_blur(master_raw, float(build.illum_sigma))
        # ensure positive (avoid division by near-zero)
        if _is_torch(illum):
            torch = _torch()
            illum = _where(illum <= 1e-12, torch.tensor(1e-12, dtype=illum.dtype, device=illum.device), illum)
        else:
            np = _np()
            illum = _where(illum <= 1e-12, np.array(1e-12, dtype=illum.dtype), illum)
    elif build.illum_model == "poly2":
        illum = _fit_poly2(master_raw)
        if _is_torch(illum):
            torch = _torch()
            illum = _where(illum <= 1e-12, torch.tensor(1e-12, dtype=illum.dtype, device=illum.device), illum)
        else:
            np = _np()
            illum = _where(illum <= 1e-12, np.array(1e-12, dtype=illum.dtype), illum)
    elif build.illum_model == "none":
        illum = None
    else:
        raise ValueError(f"Unknown illum_model: {build.illum_model}")

    # Remove illumination => PRNU
    if illum is not None:
        prnu = master_raw / illum
    else:
        prnu = master_raw

    # Normalize PRNU to unity mean
    m = _nanmean(prnu, axis=(-2, -1), keepdims=True)
    prnu = prnu / _where(_abs(m) < 1e-12, m + 1e-12, m)

    # Optional clipping of PRNU
    if build.clip_prnu is not None:
        low, high = build.clip_prnu
        prnu = _clip(prnu, low, high)

    # Create masks from PRNU and var outliers
    mask_hot, mask_bad, stats_meta = _make_masks_from_prnu_and_var(prnu, var, build.hot_sigma, build.bad_sigma)

    meta: Dict[str, Any] = {}
    if build.return_intermediate:
        meta.update({
            "N_stack": N,
            "agg": build.agg,
            "cosmic_reject": build.cosmic_reject,
            "cr_zmax": build.cr_zmax,
            "cr_iter": build.cr_iter,
            "illum_model": build.illum_model,
            "illum_sigma": build.illum_sigma if build.illum_model == "gaussian" else None,
            **stats_meta
        })

    model = FlatModel(
        prnu=prnu,
        illum=illum,
        var=var,
        weights=weights,
        mask_hot=mask_hot,
        mask_bad=mask_bad,
        meta=meta,
    )
    return FlatBuildResult(master=model)

# -----------------------------------------------------------------------------
# Apply flat to science
# -----------------------------------------------------------------------------

def apply_flat(
    science: BackendArray,
    model: FlatModel,
    apply: FlatApplyParams,
) -> FlatApplyResult:
    """
    Divide science frames by illumination (optional) and PRNU.

    Args
    ----
    science : [..., H, W] or batched [..., ..., H, W]
    model   : FlatModel
    apply   : FlatApplyParams

    Returns
    -------
    FlatApplyResult
      - corrected: (science / illum?) / prnu
      - var: variance propagated through division if requested and model.var present
    """
    s = _to_float(science, dtype=apply.dtype)

    # Optional illumination correction
    if apply.use_illum and (model.illum is not None):
        illum = model.illum
        if _is_torch(illum):
            torch = _torch()
            illum = _where(illum <= 1e-12, torch.tensor(1e-12, dtype=illum.dtype, device=illum.device), illum)
        else:
            np = _np()
            illum = _where(illum <= 1e-12, np.array(1e-12, dtype=illum.dtype), illum)
        s = s / illum

    # PRNU division
    prnu = model.prnu
    if _is_torch(prnu):
        torch = _torch()
        prnu = _where(prnu <= 1e-12, torch.tensor(1e-12, dtype=prnu.dtype, device=prnu.device), prnu)
    else:
        np = _np()
        prnu = _where(prnu <= 1e-12, np.array(1e-12, dtype=prnu.dtype), prnu)

    # Mask hot/bad if requested
    if apply.mask_out_hot_bad and (model.mask_hot is not None or model.mask_bad is not None):
        if _is_torch(s) or _is_torch(prnu):
            torch = _torch()
            bad = None
            if model.mask_hot is not None:
                bad = model.mask_hot if bad is None else (bad | model.mask_hot)
            if model.mask_bad is not None:
                bad = model.mask_bad if bad is None else (bad | model.mask_bad)
            if bad is not None:
                s = _where(bad, torch.tensor(float('nan'), dtype=s.dtype, device=s.device), s)
                prnu = _where(bad, torch.tensor(float('nan'), dtype=prnu.dtype, device=s.device), prnu)
        else:
            np = _np()
            bad = None
            if model.mask_hot is not None:
                bad = model.mask_hot if bad is None else (bad | model.mask_hot)
            if model.mask_bad is not None:
                bad = model.mask_bad if bad is None else (bad | model.mask_bad)
            if bad is not None:
                s = s.copy(); prnu = prnu.copy()
                s[bad] = np.nan
                prnu[bad] = np.nan

    corrected = s / prnu

    # Variance propagation (approx): if y = s / p, var(y) ≈ var(s)/p^2 + s^2*var(p)/p^4.
    var_out: Optional[BackendArray] = None
    if apply.propagate_var and (model.var is not None):
        # We don't know var(science) here; we propagate only the flat's contribution term: s^2 * var(prnu) / prnu^4.
        var_p = model.var
        if _is_torch(var_p) or _is_torch(prnu) or _is_torch(s):
            torch = _torch()
            prnu2 = prnu * prnu
            prnu4 = prnu2 * prnu2
            var_out = (s * s) * var_p / _where(prnu4 <= 1e-18, torch.tensor(1e-18, dtype=prnu4.dtype, device=prnu4.device), prnu4)
        else:
            np = _np()
            prnu2 = prnu * prnu
            prnu4 = prnu2 * prnu2
            den = _where(prnu4 <= 1e-18, np.array(1e-18, dtype=prnu4.dtype), prnu4)
            var_out = (s * s) * var_p / den

    # Optional clipping
    if apply.clip_out is not None:
        low, high = apply.clip_out
        corrected = _clip(corrected, low, high)

    meta: Dict[str, Any] = {}
    if apply.return_intermediate:
        meta.update({
            "use_illum": apply.use_illum,
            "mask_out_hot_bad": apply.mask_out_hot_bad,
            "propagate_var": apply.propagate_var,
            "clip_out": apply.clip_out,
        })

    return FlatApplyResult(corrected=corrected, var=var_out, meta=meta)

# -----------------------------------------------------------------------------
# Light self-tests
# -----------------------------------------------------------------------------

def _test_build_and_apply():
    import numpy as np
    rng = np.random.default_rng(0)
    N, H, W = 8, 16, 16
    # Synthetic PRNU: gentle pixel-to-pixel ripple
    yy, xx = np.mgrid[0:H, 0:W]
    prnu_true = 1.0 + 0.02 * np.sin(2*np.pi*xx/W) * np.cos(2*np.pi*yy/H)
    illum_true = 1000.0 * (1.0 + 0.1*np.cos(2*np.pi*xx/W))
    stack = (prnu_true * illum_true)[None, ...] + rng.normal(0, 2.0, size=(N, H, W))
    # build
    build = FlatBuildParams(agg="median", cosmic_reject=True, illum_model="gaussian", illum_sigma=8.0, return_intermediate=True)
    res = build_master_flat(stack.astype(np.float32), build)
    # apply to science with same illum/prnu -> should flatten to ~illum amplitude
    sci = (prnu_true * illum_true) + rng.normal(0, 1.0, size=(H, W))
    out = apply_flat(sci.astype(np.float32), res.master, FlatApplyParams(use_illum=True))
    # After dividing by illum then PRNU, corrected should be ~1 (up to noise)
    med = float(np.nanmedian(out.corrected))
    assert abs(med - 1.0) < 0.05, f"median after flat too far from unity: {med}"
    return True

if __name__ == "__main__":
    ok = _test_build_and_apply()
    print("flat.py self-tests:", ok)
