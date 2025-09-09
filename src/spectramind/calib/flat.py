# src/spectramind/calib/flat.py
# =============================================================================
# SpectraMind V50 — Flat-field modeling & application (PRNU + illumination)
# -----------------------------------------------------------------------------
# Build a master flat from stacks of flat frames:
#   - robust temporal aggregation (mean/median)
#   - optional cosmic-ray rejection (MAD z-score, iterated)
#   - large-scale illumination estimation: gaussian low-pass or quadratic poly
#   - PRNU map normalized to mean 1 (illumination removed)
#   - hot/bad pixel masks (from PRNU and temporal var outliers)
#   - per-pixel variance & weights
#
# Apply the master flat to science frames (divide), with optional illumination
# correction, outlier masking, and variance propagation. Backend-agnostic
# (NumPy or Torch), canonical image layout [..., H, W], stacks [..., N, H, W].
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal, cast

__all__ = [
    "FlatBuildParams",
    "FlatModel",
    "FlatBuildResult",
    "FlatApplyParams",
    "FlatApplyResult",
    "build_master_flat",
    "apply_flat",
]

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821
AggMode = Literal["mean", "median"]
IllumModel = Literal["none", "gaussian", "poly2"]


# -----------------------------------------------------------------------------
# Backend shims (alignment with adc.py / cds.py / dark.py)
# -----------------------------------------------------------------------------

def _is_torch(x: BackendArray) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "torch"


def _np() -> Any:
    import numpy as np
    return np


def _torch() -> Any:
    import torch
    return torch


def _resolve_dtype(backend: str, dtype: Optional[Union[str, Any]]) -> Any:
    if dtype is None:
        return None
    if backend == "torch":
        torch = _torch()
        if isinstance(dtype, str):
            m = {
                "float32": torch.float32,
                "float": torch.float32,
                "float64": torch.float64,
                "double": torch.float64,
                "float16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            return m.get(dtype.lower(), getattr(torch, dtype))
        return dtype
    # numpy
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


def _zeros_like(x: BackendArray) -> BackendArray:
    return _torch().zeros_like(x) if _is_torch(x) else _np().zeros_like(x)


def _full_like(x: BackendArray, fill: float) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.full_like(x, fill)
    else:
        np = _np()
        out = np.empty_like(x, dtype=x.dtype)
        out.fill(fill)
        return out


def _where(mask: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    return _torch().where(mask, a, b) if _is_torch(mask) else _np().where(mask, a, b)


def _abs(x: BackendArray) -> BackendArray:
    return x.abs() if _is_torch(x) else _np().abs(x)


def _nanmean(x: BackendArray, axis=None, keepdims: bool = False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        if hasattr(torch, "nanmean"):
            return torch.nanmean(x, dim=axis, keepdim=keepdims)  # type: ignore[attr-defined]
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims)
        den = mask.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        all_nan = den == 0
        return torch.where(all_nan, torch.tensor(float("nan"), dtype=x.dtype, device=x.device), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)


def _torch_nanmedian(x, axis=None, keepdims=False):
    torch = _torch()
    # Prefer native nanmedian when present
    if hasattr(torch, "nanmedian"):
        return torch.nanmedian(x, dim=axis, keepdim=keepdims).values  # type: ignore[attr-defined]
    # Fallback: +inf masking + sort + take_along_dim
    isnan = torch.isnan(x)
    xf = torch.where(isnan, torch.full_like(x, float("inf")), x)
    sorted_xf, _ = torch.sort(xf, dim=axis)
    valid = (~isnan).sum(dim=axis, keepdim=True)
    low_idx = (valid - 1) // 2
    high_idx = valid // 2
    if hasattr(torch, "take_along_dim"):
        tl = torch.take_along_dim(sorted_xf, low_idx.long(), dim=axis)
        th = torch.take_along_dim(sorted_xf, high_idx.long(), dim=axis)
        med = 0.5 * (tl + th)
        if not keepdims:
            med = med.squeeze(dim=axis)  # type: ignore[arg-type]
        all_nan = (valid == 0)
        if all_nan.any():
            med = torch.where(all_nan.squeeze(dim=axis), torch.full_like(med, float("nan")), med)
        return med
    return _to_numpy_and_back_nanmedian(x, axis=axis, keepdims=keepdims)


def _nanmedian(x: BackendArray, axis=None, keepdims: bool = False) -> BackendArray:
    if _is_torch(x):
        return _torch_nanmedian(x, axis=axis, keepdims=keepdims)
    else:
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)


def _to_numpy_and_back_nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if not _is_torch(x):
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)
    torch = _torch()
    np = _np()
    x_np = x.detach().cpu().numpy()
    m_np = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
    return torch.from_numpy(m_np).to(device=x.device, dtype=x.dtype)


def _nanstd(x: BackendArray, axis=None, keepdims: bool = False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        if hasattr(torch, "nanstd"):
            return torch.nanstd(x, dim=axis, keepdim=keepdims)  # type: ignore[attr-defined]
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


def _safe_div(n: BackendArray, d: BackendArray, eps: float) -> BackendArray:
    """
    n / max(|d|, eps) with dtype/device consistency; preserves NaNs.
    """
    if _is_torch(n) or _is_torch(d):
        torch = _torch()
        d = d if _is_torch(d) else torch.as_tensor(d, device=getattr(n, "device", None), dtype=getattr(n, "dtype", None))
        n = n if _is_torch(n) else torch.as_tensor(n, device=getattr(d, "device", None), dtype=getattr(d, "dtype", None))
        d2 = torch.where(d.abs() <= eps, torch.full_like(d, eps), d)
        return n / d2
    else:
        np = _np()
        d2 = np.where(np.abs(d) <= eps, np.array(eps, dtype=d.dtype), d)
        return n / d2


def _broadcast_to_hw(x: BackendArray, like_hw: BackendArray) -> BackendArray:
    """
    Ensure x has shape broadcastable to like_hw's [..., H, W] (e.g. expand [H,W] to leading batch dims).
    """
    if _is_torch(like_hw) or _is_torch(x):
        torch = _torch()
        return x if _is_torch(x) else torch.as_tensor(x, device=getattr(like_hw, "device", None), dtype=getattr(like_hw, "dtype", None))
    return x


# -----------------------------------------------------------------------------
# Config dataclasses
# -----------------------------------------------------------------------------

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
    hot_sigma         : PRNU > mean + hot_sigma*std -> hot pixel
    bad_sigma         : temporal var > mean + bad_sigma*std -> bad pixel
    clip_prnu         : (low, high) clipping on PRNU map
    dtype             : output dtype for intermediates
    return_intermediate: stash debug/meta info in model.meta
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
    mask_hot: Optional[BackendArray] # hot pixel mask from PRNU outliers (bool)
    mask_bad: Optional[BackendArray] # bad noisy pixel mask from var outliers (bool)
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FlatBuildResult:
    master: FlatModel


@dataclass
class FlatApplyParams:
    """
    How to apply the flat to science frames.

    use_illum          : if True and illum present, divide science by illum first
    mask_out_hot_bad   : if True, set hot/bad pixels to NaN before division
    propagate_var      : if True and var provided, propagate variance through division
    clip_out           : optional (low, high) clip on corrected image
    dtype              : output dtype for corrected/var
    return_intermediate: stash debug info in result.meta
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

def _move_stack_axis(x: BackendArray, time_axis: int) -> BackendArray:
    """
    Return x with the stack/time axis moved to position -3 (i.e. [..., N, H, W]).
    """
    nd = x.ndim
    target = nd - 3
    src = time_axis if time_axis >= 0 else nd + time_axis
    if src == target:
        return x
    perm = list(range(nd))
    perm[target], perm[src] = perm[src], perm[target]
    if _is_torch(x):
        return x.permute(*perm)
    else:
        return x.transpose(target, src)


def _robust_temporal(
    x: BackendArray,
    axis: int,
    agg: AggMode,
    cosmic_reject: bool,
    zmax: float,
    iters: int,
) -> Tuple[BackendArray, BackendArray, BackendArray]:
    """
    Aggregate across time with optional MAD-based CR rejection.
    Returns (value, variance, weights).
    """
    cur = x

    if not cosmic_reject:
        if agg == "mean":
            val = _nanmean(cur, axis=axis, keepdims=False)
        else:
            val = _nanmedian(cur, axis=axis, keepdims=False)

        if _is_torch(cur):
            torch = _torch()
            w = (~torch.isnan(cur)).sum(dim=axis)
            v = _nanmean((cur - val.unsqueeze(axis)) ** 2, axis=axis, keepdims=False)
        else:
            np = _np()
            w = np.sum(~np.isnan(cur), axis=axis)
            v = _nanmean((cur - _np().expand_dims(val, axis=axis)) ** 2, axis=axis, keepdims=False)
        return val, v, w

    # Iterative MAD-based outlier rejection
    for _ in range(max(1, iters)):
        med = _nanmedian(cur, axis=axis, keepdims=True)
        mad = _nanmedian(_abs(cur - med), axis=axis, keepdims=True)
        if _is_torch(cur):
            torch = _torch()
            eps = torch.tensor(1e-12, dtype=mad.dtype, device=mad.device)
            mad = _where(mad <= eps, eps, mad)
            z = 0.6745 * (cur - med) / mad
            rej = _abs(z) > zmax
            cur = _where(rej, torch.tensor(float("nan"), dtype=cur.dtype, device=cur.device), cur)
        else:
            np = _np()
            mad = _where(mad <= 1e-12, np.array(1e-12, dtype=mad.dtype), mad)
            z = 0.6745 * (cur - med) / mad
            rej = _abs(z) > zmax
            cur = cur.copy()
            cur[rej] = np.nan

    # Final aggregation
    if agg == "mean":
        val = _nanmean(cur, axis=axis, keepdims=False)
    else:
        val = _nanmedian(cur, axis=axis, keepdims=False)

    if _is_torch(cur):
        torch = _torch()
        w = (~torch.isnan(cur)).sum(dim=axis)
        v = _nanmean((cur - val.unsqueeze(axis)) ** 2, axis=axis, keepdims=False)
    else:
        np = _np()
        w = np.sum(~np.isnan(cur), axis=axis)
        v = _nanmean((cur - _np().expand_dims(val, axis=axis)) ** 2, axis=axis, keepdims=False)

    return val, v, w


def _gaussian_blur(image: BackendArray, sigma: float) -> BackendArray:
    """
    Gaussian low-pass for large-scale illumination.
    Tries SciPy (fast). If unavailable, uses a pure-NumPy separable kernel.

    NOTE: This runs once per build; the NumPy fallback favors simplicity over peak speed.
    """
    if sigma <= 0:
        return image

    # Try SciPy if available
    try:
        from scipy.ndimage import gaussian_filter  # type: ignore
        if _is_torch(image):
            torch = _torch()
            x_np = image.detach().cpu().numpy()
            out_np = gaussian_filter(x_np, sigma=float(sigma), mode="nearest")
            return torch.from_numpy(out_np).to(device=image.device, dtype=image.dtype)
        else:
            return gaussian_filter(image, sigma=float(sigma), mode="nearest")
    except Exception:
        pass  # fall through to NumPy-only separable implementation

    # NumPy-only separable blur
    np = _np()
    from math import ceil

    radius = max(1, int(ceil(3.0 * float(sigma))))
    xs = np.arange(-radius, radius + 1, dtype=np.float64)
    k = np.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    k = (k / k.sum()).astype(np.float64)  # [K]

    def _sep_conv_last2(img_np: "np.ndarray", k1d: "np.ndarray") -> "np.ndarray":
        H, W = img_np.shape[-2], img_np.shape[-1]
        r = k1d.size // 2
        pad_cfg = ((0, 0),) * (img_np.ndim - 2) + ((r, r), (r, r))
        p = np.pad(img_np, pad_cfg, mode="edge")
        # Convolve along W (last axis)
        tmp = np.apply_along_axis(lambda v: np.convolve(v, k1d, mode="same"), axis=-1, arr=p)
        # Convolve along H (second-last axis)
        tmp = np.apply_along_axis(lambda v: np.convolve(v, k1d, mode="same"), axis=-2, arr=tmp)
        return tmp[..., r:-r, r:-r].astype(img_np.dtype, copy=False)

    if _is_torch(image):
        torch = _torch()
        x_np = image.detach().cpu().numpy()
        out_np = _sep_conv_last2(x_np, k)
        return torch.from_numpy(out_np).to(device=image.device, dtype=image.dtype)
    else:
        return _sep_conv_last2(image, k)


def _fit_poly2(image: BackendArray) -> BackendArray:
    """
    Fit a 2D quadratic surface: a0 + a1*x + a2*y + a3*x*y + a4*x^2 + a5*y^2.
    Torch path falls back to CPU NumPy solve (robust & dependency-light).
    """
    np = _np()
    H, W = image.shape[-2], image.shape[-1]
    yy, xx = np.mgrid[0:H, 0:W]
    A = np.stack([np.ones_like(xx), xx, yy, xx * yy, xx * xx, yy * yy], axis=-1)  # [H,W,6]
    A2 = A.reshape(-1, 6)
    if _is_torch(image):
        torch = _torch()
        b = image.detach().cpu().numpy().reshape(-1)
        coef, *_ = np.linalg.lstsq(A2, b, rcond=None)
        fit_np = (A2 @ coef).reshape(H, W)
        return torch.from_numpy(fit_np).to(device=image.device, dtype=image.dtype)
    else:
        b = image.reshape(-1)
        coef, *_ = np.linalg.lstsq(A2, b, rcond=None)
        fit = (A2 @ coef).reshape(H, W)
        return fit


def _make_masks_from_prnu_and_var(
    prnu: BackendArray,
    var: BackendArray,
    hot_sigma: Optional[float],
    bad_sigma: Optional[float],
) -> Tuple[Optional[BackendArray], Optional[BackendArray], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if hot_sigma is None and bad_sigma is None:
        return None, None, meta

    if _is_torch(prnu) or _is_torch(var):
        torch = _torch()
        m = torch.nanmean(prnu); s = torch.nanstd(prnu)
        hot = (prnu > (m + (hot_sigma or 0.0) * s)) if hot_sigma is not None else None

        vm = torch.nanmean(var); vs = torch.nanstd(var)
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

def _normalize_unity(x: BackendArray, eps: float = 1e-12) -> BackendArray:
    m = _nanmean(x, axis=(-2, -1), keepdims=True)
    return _safe_div(x, _where(_abs(m) < eps, m + eps, m), eps)


def build_master_flat(flat_stack: BackendArray, build: FlatBuildParams) -> FlatBuildResult:
    """
    Build a master flat (PRNU + Illumination) from a stack of flat-field frames.

    Args
    ----
    flat_stack : [..., N, H, W] (or time axis anywhere; see build.time_axis)
    build      : FlatBuildParams

    Returns
    -------
    FlatBuildResult with FlatModel fields:
      - prnu : unity-mean PRNU (illumination removed)
      - illum: large-scale illumination (>=0) if modeled; None otherwise
      - var  : temporal variance (pre-normalization)
      - weights: effective frame counts
      - mask_hot/bad: boolean masks from PRNU/var outliers
    """
    x = _to_float(flat_stack, dtype=build.dtype)
    x = _move_stack_axis(x, build.time_axis)  # -> [..., N, H, W]
    N = x.shape[-3]
    if N < 1:
        raise ValueError("build_master_flat: need at least one flat frame")

    # Robust temporal aggregation
    master_raw, var, weights = _robust_temporal(
        x, axis=-3, agg=build.agg, cosmic_reject=build.cosmic_reject, zmax=build.cr_zmax, iters=build.cr_iter
    )

    # Large-scale illumination
    if build.illum_model == "gaussian":
        illum = _gaussian_blur(master_raw, float(build.illum_sigma))
        illum = _where(_abs(illum) <= 1e-12, _full_like(illum, 1e-12), illum)
    elif build.illum_model == "poly2":
        illum = _fit_poly2(master_raw)
        illum = _where(_abs(illum) <= 1e-12, _full_like(illum, 1e-12), illum)
    elif build.illum_model == "none":
        illum = None
    else:
        raise ValueError(f"Unknown illum_model: {build.illum_model!r}")

    # Remove illumination => PRNU
    prnu = _safe_div(master_raw, illum, 1e-12) if illum is not None else master_raw

    # Normalize PRNU to unity mean
    prnu = _normalize_unity(prnu, eps=1e-12)

    # Optional clipping of PRNU
    if build.clip_prnu is not None:
        low, high = build.clip_prnu
        prnu = _clip(prnu, low, high)

    # Create masks from PRNU and var outliers
    mask_hot, mask_bad, stats_meta = _make_masks_from_prnu_and_var(prnu, var, build.hot_sigma, build.bad_sigma)

    meta: Dict[str, Any] = {}
    if build.return_intermediate:
        meta.update(
            {
                "N_stack": int(N),
                "agg": build.agg,
                "cosmic_reject": build.cosmic_reject,
                "cr_zmax": float(build.cr_zmax),
                "cr_iter": int(build.cr_iter),
                "illum_model": build.illum_model,
                "illum_sigma": float(build.illum_sigma) if build.illum_model == "gaussian" else None,
                **stats_meta,
            }
        )

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

def apply_flat(science: BackendArray, model: FlatModel, apply: FlatApplyParams) -> FlatApplyResult:
    """
    Divide science frames by illumination (optional) and PRNU.

    Args
    ----
    science : [..., H, W] or batched [..., ..., H, W]
    model   : FlatModel
    apply   : FlatApplyParams

    Returns
    -------
    FlatApplyResult:
      - corrected: (science / illum?) / prnu
      - var: variance propagated through division if requested and model.var present
    """
    s = _to_float(science, dtype=apply.dtype)

    # Optional illumination correction
    if apply.use_illum and (model.illum is not None):
        illum = _broadcast_to_hw(model.illum, s)
        illum = _where(_abs(illum) <= 1e-12, _full_like(illum, 1e-12), illum)
        s = _safe_div(s, illum, 1e-12)

    # Aggregate mask for hot/bad pixels
    bad_mask: Optional[BackendArray] = None
    if apply.mask_out_hot_bad and (model.mask_hot is not None or model.mask_bad is not None):
        if model.mask_hot is not None:
            bad_mask = model.mask_hot if bad_mask is None else (bad_mask | model.mask_hot)
        if model.mask_bad is not None:
            bad_mask = model.mask_bad if bad_mask is None else (bad_mask | model.mask_bad)
        if bad_mask is not None:
            if _is_torch(s):
                torch = _torch()
                s = _where(bad_mask, torch.tensor(float("nan"), dtype=s.dtype, device=s.device), s)
            else:
                np = _np()
                s = s.copy()
                s[cast("np.ndarray", bad_mask)] = np.nan  # type: ignore[index]

    # PRNU division
    prnu = _broadcast_to_hw(model.prnu, s)
    prnu = _where(_abs(prnu) <= 1e-12, _full_like(prnu, 1e-12), prnu)
    corrected = _safe_div(s, prnu, 1e-12)

    # Variance propagation (approx): if y = s / p, var(y) ≈ var(s)/p^2 + s^2*var(p)/p^4.
    # We don't know var(science) here; we propagate only the flat's contribution term: s^2 * var(p) / p^4.
    var_out: Optional[BackendArray] = None
    if apply.propagate_var and (model.var is not None):
        var_p = _broadcast_to_hw(model.var, prnu)
        prnu2 = prnu * prnu
        prnu4 = prnu2 * prnu2
        var_out = _safe_div((s * s) * var_p, prnu4, 1e-18)

    # Optional clipping
    if apply.clip_out is not None:
        low, high = apply.clip_out
        corrected = _clip(corrected, low, high)

    meta: Dict[str, Any] = {}
    if apply.return_intermediate:
        meta.update(
            {
                "use_illum": apply.use_illum,
                "mask_out_hot_bad": apply.mask_out_hot_bad,
                "propagate_var": apply.propagate_var,
                "clip_out": apply.clip_out,
            }
        )

    return FlatApplyResult(corrected=corrected, var=var_out, meta=meta)


# -----------------------------------------------------------------------------
# Light self-tests (quick NumPy sanity)
# -----------------------------------------------------------------------------

def _test_build_and_apply() -> bool:
    import numpy as np
    rng = np.random.default_rng(0)
    N, H, W = 8, 16, 16
    # Synthetic PRNU: gentle pixel-to-pixel ripple
    yy, xx = np.mgrid[0:H, 0:W]
    prnu_true = 1.0 + 0.02 * np.sin(2 * np.pi * xx / W) * np.cos(2 * np.pi * yy / H)
    illum_true = 1000.0 * (1.0 + 0.1 * np.cos(2 * np.pi * xx / W))

    stack = (prnu_true * illum_true)[None, ...] + rng.normal(0, 2.0, size=(N, H, W))
    build = FlatBuildParams(
        agg="median",
        cosmic_reject=True,
        illum_model="gaussian",
        illum_sigma=8.0,
        return_intermediate=True,
    )
    res = build_master_flat(stack.astype(np.float32), build)

    # Apply to science made from same ground truth (should flatten to ~1)
    sci = (prnu_true * illum_true) + rng.normal(0, 1.0, size=(H, W))
    out = apply_flat(sci.astype(np.float32), res.master, FlatApplyParams(use_illum=True))
    med = float(np.nanmedian(out.corrected))
    assert abs(med - 1.0) < 0.06, f"median after flat too far from unity: {med}"
    return True


if __name__ == "__main__":
    ok = _test_build_and_apply()
    print("flat.py self-tests:", ok)