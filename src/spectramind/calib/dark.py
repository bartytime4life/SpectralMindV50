# src/spectramind/calib/dark.py
# =============================================================================
# SpectraMind V50 — Dark frame modeling & subtraction
# -----------------------------------------------------------------------------
# Build a master dark from stacks of dark frames:
#   - temporal aggregation: mean or median
#   - robust cosmic-ray rejection: MAD-based z-score iterations
#   - hot/bad pixel maps from temporal stats
#   - variance/weights propagation
#   - reference exposure & temperature stored for later scaling
#
# Apply the master dark to science data with optional variance propagation,
# hot/bad masking, clipping, and exposure/temperature scaling.
#
# Backend-agnostic: NumPy ndarray or Torch tensor inputs.
# Canonical image layout [..., H, W] and dark stacks [..., N, H, W] (configurable axis).
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal, cast

__all__ = [
    "DarkBuildParams",
    "DarkScaleParams",
    "DarkModel",
    "ApplyDarkParams",
    "DarkBuildResult",
    "DarkApplyResult",
    "build_master_dark",
    "apply_dark",
]

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821
AggMode = Literal["mean", "median"]
ScaleModel = Literal["none", "linear", "arrhenius"]


# -----------------------------------------------------------------------------
# Backend shims (aligned with adc.py / cds.py / flat.py)
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
        target = getattr(torch, dtype) if isinstance(dtype, str) else (dtype or torch.float32)
        return x.to(target)
    else:
        np = _np()
        target = getattr(np, dtype) if isinstance(dtype, str) else (dtype or np.float32)
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
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, torch.tensor(0.0, dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims)
        den = mask.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        all_nan = den == 0
        return torch.where(all_nan, torch.tensor(float("nan"), dtype=x.dtype, device=x.device), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)


def _nanmedian(x: BackendArray, axis=None, keepdims: bool = False) -> BackendArray:
    if _is_torch(x):
        torch = _torch(); np = _np()
        x_np = x.detach().cpu().numpy()
        m_np = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
        return torch.from_numpy(m_np).to(device=x.device, dtype=x.dtype)
    else:
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)


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
    """Elementwise n / max(|d|, eps) with dtype/device consistency; preserves NaNs."""
    if _is_torch(n) or _is_torch(d):
        torch = _torch()
        n_t = n if _is_torch(n) else torch.as_tensor(n, dtype=getattr(d, "dtype", None), device=getattr(d, "device", None))
        d_t = d if _is_torch(d) else torch.as_tensor(d, dtype=getattr(n_t, "dtype", None), device=getattr(n_t, "device", None))
        d2 = torch.where(d_t.abs() <= eps, torch.full_like(d_t, eps), d_t)
        return n_t / d2
    else:
        np = _np()
        d2 = np.where(np.abs(d) <= eps, np.array(eps, dtype=getattr(d, "dtype", None) or n.dtype), d)
        return n / d2


def _broadcast_to_hw(x: BackendArray, like_hw: BackendArray) -> BackendArray:
    """Ensure x broadcasts over like_hw's [..., H, W]; returns appropriate backend tensor/array."""
    if _is_torch(like_hw) or _is_torch(x):
        torch = _torch()
        return x if _is_torch(x) else torch.as_tensor(x, device=getattr(like_hw, "device", None), dtype=getattr(like_hw, "dtype", None))
    return x


# -----------------------------------------------------------------------------
# Models & Params
# -----------------------------------------------------------------------------

@dataclass
class DarkBuildParams:
    """
    Controls building of a master dark from dark stacks.

    time_axis         : axis of the stack dimension (default -3 => [..., N, H, W])
    agg               : 'mean' or 'median' temporal aggregation
    cosmic_reject     : enable MAD-zscore CR rejection
    cr_zmax           : rejection threshold
    cr_iter           : robust iterations
    hot_sigma         : pixels with temporal mean > mean + hot_sigma*std flagged hot (None to disable)
    bad_sigma         : pixels with temporal var > mean + bad_sigma*std flagged bad (None to disable)
    clip_out          : optional (low, high) to clip master dark
    dtype             : output float dtype
    return_intermediate: include debug info in model.meta
    """
    time_axis: int = -3
    agg: AggMode = "median"
    cosmic_reject: bool = True
    cr_zmax: float = 6.0
    cr_iter: int = 1
    hot_sigma: Optional[float] = 8.0
    bad_sigma: Optional[float] = 8.0
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False


@dataclass
class DarkScaleParams:
    """
    Reference scaling conditions for the built master.

    exposure_ref      : exposure time used for master dark (seconds)
    temperature_ref   : sensor temperature for master dark (Kelvin)
    scale_exposure    : if True, scale linearly with exposure time (multiplicative)
    scale_temperature : 'none' | 'linear' | 'arrhenius'
        - 'linear'    : add offset = temp_coeff * (T - T_ref)
        - 'arrhenius' : multiply by exp(Ea_over_k * (1/T - 1/T_ref))
    temp_coeff        : slope for linear model (dark DN per Kelvin)
    Ea_over_k         : activation energy over Boltzmann (Kelvin) for Arrhenius
    """
    exposure_ref: float = 1.0
    temperature_ref: Optional[float] = None
    scale_exposure: bool = True
    scale_temperature: ScaleModel = "none"
    temp_coeff: float = 0.0
    Ea_over_k: float = 0.0


@dataclass
class DarkModel:
    """Master dark product ready for application and scaling."""
    master: BackendArray                 # [..., H, W] in DN (or e-) per exposure_ref
    var: Optional[BackendArray]          # per-pixel variance estimate of master (at ref conditions)
    weights: Optional[BackendArray]      # effective sample counts
    mask_hot: Optional[BackendArray]     # boolean mask of hot pixels
    mask_bad: Optional[BackendArray]     # boolean mask of noisy/bad pixels
    scale: DarkScaleParams               # reference scaling params
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplyDarkParams:
    """
    Parameters for applying (subtracting) the dark model to science frames.

    exposure_target   : science exposure time (seconds)
    temperature_target: science temperature (Kelvin), optional
    propagate_var     : if True and var provided, compute output variance
    mask_out_hot      : if True, set hot/bad pixels to NaN before subtraction
    clip_out          : optional (low, high)
    dtype             : output float dtype
    return_intermediate: include debug info in result.meta
    """
    exposure_target: float = 1.0
    temperature_target: Optional[float] = None
    propagate_var: bool = True
    mask_out_hot: bool = True
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False


@dataclass
class DarkBuildResult:
    master: DarkModel


@dataclass
class DarkApplyResult:
    corrected: BackendArray
    var: Optional[BackendArray]
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Helpers
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


def _expand_like(val: BackendArray, ref: BackendArray, axis: int) -> BackendArray:
    """Expand val to broadcast shape of ref along `axis`."""
    if _is_torch(val) or _is_torch(ref):
        torch = _torch()
        shape = list(ref.shape)
        shape[axis] = 1
        return val.reshape(*shape)
    else:
        np = _np()
        shape = list(ref.shape)
        shape[axis] = 1
        return val.reshape(shape)


def _robust_temporal(
    x: BackendArray,
    axis: int,
    agg: AggMode,
    cosmic_reject: bool,
    zmax: float,
    iters: int,
) -> Tuple[BackendArray, BackendArray, BackendArray]:
    """
    Robust temporal aggregation with optional MAD-based CR rejection.
    Returns: (mean_or_median, variance, weights)
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
            v = _nanmean((cur - _expand_like(val, cur, axis)) ** 2, axis=axis, keepdims=False)
        return val, v, w

    # Iterative MAD rejection
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
        v = _nanmean((cur - _expand_like(val, cur, axis)) ** 2, axis=axis, keepdims=False)

    return val, v, w


def _make_hot_bad_masks(
    master: BackendArray,
    var: BackendArray,
    hot_sigma: Optional[float],
    bad_sigma: Optional[float],
) -> Tuple[Optional[BackendArray], Optional[BackendArray], Dict[str, Any]]:
    meta: Dict[str, Any] = {}
    if hot_sigma is None and bad_sigma is None:
        return None, None, meta

    if _is_torch(master) or _is_torch(var):
        torch = _torch()
        m_mean = torch.nanmean(master)
        m_std = torch.nanstd(master)
        v_mean = torch.nanmean(var)
        v_std = torch.nanstd(var)
        hot = (master > (m_mean + (hot_sigma or 0.0) * m_std)) if hot_sigma is not None else None
        bad = (var > (v_mean + (bad_sigma or 0.0) * v_std)) if bad_sigma is not None else None
    else:
        np = _np()
        m_mean = np.nanmean(master)
        m_std = np.nanstd(master)
        v_mean = np.nanmean(var)
        v_std = np.nanstd(var)
        hot = (master > (m_mean + (hot_sigma or 0.0) * m_std)) if hot_sigma is not None else None
        bad = (var > (v_mean + (bad_sigma or 0.0) * v_std)) if bad_sigma is not None else None

    meta.update(dict(global_mean=float(m_mean), global_std=float(m_std), var_mean=float(v_mean), var_std=float(v_std)))
    return hot, bad, meta


def _scaling_factors(
    exposure_ref: float,
    exposure_target: float,
    temperature_ref: Optional[float],
    temperature_target: Optional[float],
    scale_exposure: bool,
    temp_model: ScaleModel,
    temp_coeff: float,
    Ea_over_k: float,
    backend_like: BackendArray,
) -> Tuple[float, BackendArray]:
    """
    Compute multiplicative k and additive offset b so that:
        scaled_dark = k * master + b

    - Exposure scaling is multiplicative (k *= exposure_target / exposure_ref) if enabled.
    - Temperature scaling:
        * 'none'     → b += 0
        * 'linear'   → b += temp_coeff * (T_target - T_ref)
        * 'arrhenius'→ k *= exp(Ea_over_k * (1/T_target - 1/T_ref))
    """
    k = 1.0
    b: BackendArray
    if _is_torch(backend_like):
        torch = _torch()
        b = torch.tensor(0.0, dtype=backend_like.dtype, device=backend_like.device)
    else:
        np = _np()
        b = np.array(0.0, dtype=backend_like.dtype)

    # Exposure factor
    if scale_exposure and (exposure_ref is not None) and (exposure_target is not None) and exposure_ref > 0:
        k *= float(exposure_target) / float(exposure_ref)

    # Temperature factor
    if temp_model != "none" and (temperature_ref is not None) and (temperature_target is not None):
        Tref = float(temperature_ref)
        Ttar = float(temperature_target)
        if temp_model == "linear":
            delta = temp_coeff * (Ttar - Tref)
            if _is_torch(b):
                torch = _torch()
                b = b + torch.tensor(delta, dtype=b.dtype, device=b.device)
            else:
                b = b + delta
        elif temp_model == "arrhenius":
            coeff = Ea_over_k * (1.0 / Ttar - 1.0 / Tref)
            if _is_torch(backend_like):
                torch = _torch()
                k = k * float(torch.exp(torch.tensor(coeff, dtype=backend_like.dtype, device=backend_like.device)).item())
            else:
                np = _np()
                k = k * float(np.exp(coeff))
        else:
            raise ValueError(f"Unknown temperature model: {temp_model}")

    return k, b


def _scale_dark(master: BackendArray, scale: DarkScaleParams, exposure_target: float,
                temperature_target: Optional[float]) -> Tuple[BackendArray, float]:
    """
    Return scaled dark frame and multiplicative factor k used (for variance scaling).
    Uses scaled = k * master + b ; returns both scaled and k.
    """
    k, b = _scaling_factors(
        exposure_ref=scale.exposure_ref,
        exposure_target=exposure_target,
        temperature_ref=scale.temperature_ref,
        temperature_target=temperature_target,
        scale_exposure=scale.scale_exposure,
        temp_model=scale.scale_temperature,
        temp_coeff=scale.temp_coeff,
        Ea_over_k=scale.Ea_over_k,
        backend_like=master,
    )
    # k * master + b (broadcast b safely)
    scaled = master * (k if isinstance(k, float) else k)  # k is float
    b_cast = _broadcast_to_hw(b, master)
    if _is_torch(scaled) or _is_torch(b_cast):
        scaled = scaled + (b_cast if _is_torch(b_cast) else _torch().as_tensor(b_cast, dtype=scaled.dtype, device=scaled.device))
    else:
        scaled = scaled + b_cast
    return scaled, k


# -----------------------------------------------------------------------------
# Build master dark
# -----------------------------------------------------------------------------

def build_master_dark(dark_stack: BackendArray, build: DarkBuildParams, scale: DarkScaleParams) -> DarkBuildResult:
    """
    Build a master dark frame from a stack of dark frames.

    Args
    ----
    dark_stack : [..., N, H, W] or with time-axis anywhere (see build.time_axis)
    build      : DarkBuildParams
    scale      : DarkScaleParams (stored; no scaling performed at build)

    Returns
    -------
    DarkBuildResult with DarkModel (master, var, weights, masks, scale).
    """
    x = _to_float(dark_stack, dtype=build.dtype)
    x = _move_stack_axis(x, build.time_axis)  # -> [..., N, H, W]
    N = x.shape[-3]
    if N < 1:
        raise ValueError("build_master_dark: need at least one dark frame")

    # Robust aggregation across stack
    master, var, weights = _robust_temporal(
        x, axis=-3, agg=build.agg, cosmic_reject=build.cosmic_reject, zmax=build.cr_zmax, iters=build.cr_iter
    )

    # Optional clipping
    if build.clip_out is not None:
        low, high = build.clip_out
        master = _clip(master, low, high)

    # Hot/bad masks from temporal stats
    mask_hot, mask_bad, stats_meta = _make_hot_bad_masks(master, var, build.hot_sigma, build.bad_sigma)

    meta: Dict[str, Any] = {}
    if build.return_intermediate:
        meta.update(
            {
                "input_shape": tuple(dark_stack.shape),
                "stack_N": int(N),
                "agg": build.agg,
                "cosmic_reject": build.cosmic_reject,
                "cr_zmax": float(build.cr_zmax),
                "cr_iter": int(build.cr_iter),
                **stats_meta,
            }
        )

    model = DarkModel(
        master=master,
        var=var,
        weights=weights,
        mask_hot=mask_hot,
        mask_bad=mask_bad,
        scale=scale,
        meta=meta,
    )
    return DarkBuildResult(master=model)


# -----------------------------------------------------------------------------
# Apply (subtract) dark
# -----------------------------------------------------------------------------

def apply_dark(science: BackendArray, model: DarkModel, apply: ApplyDarkParams) -> DarkApplyResult:
    """
    Subtract a (scaled) master dark from science frames.

    Args
    ----
    science : [..., H, W] or batch [..., ..., H, W]
    model   : DarkModel built at reference exposure/temperature
    apply   : ApplyDarkParams

    Returns
    -------
    DarkApplyResult
      - corrected: science - (k * master + b)
      - var: propagated variance (k^2 * var_master) if available and requested, else None
    """
    s = _to_float(science, dtype=apply.dtype)

    # Build hot/bad aggregate mask if requested (before subtraction)
    if apply.mask_out_hot and (model.mask_hot is not None or model.mask_bad is not None):
        bad = None
        if model.mask_hot is not None:
            bad = model.mask_hot if bad is None else (bad | model.mask_hot)
        if model.mask_bad is not None:
            bad = model.mask_bad if bad is None else (bad | model.mask_bad)
        if bad is not None:
            if _is_torch(s):
                torch = _torch()
                s = _where(bad, torch.tensor(float("nan"), dtype=s.dtype, device=s.device), s)
            else:
                np = _np()
                s = s.copy()
                s[cast("np.ndarray", bad)] = np.nan  # type: ignore[index]

    # Scale master to target conditions
    scaled_dark, k = _scale_dark(
        model.master, model.scale, exposure_target=apply.exposure_target, temperature_target=apply.temperature_target
    )

    # Subtract (broadcast handles leading batch dims)
    corrected = s - _broadcast_to_hw(scaled_dark, s)

    # Optional clipping
    if apply.clip_out is not None:
        low, high = apply.clip_out
        corrected = _clip(corrected, low, high)

    # Variance propagation: var_out = var_science + k^2 * var_master  (we don't know var_science → only dark term)
    var_out: Optional[BackendArray] = None
    if apply.propagate_var and (model.var is not None):
        if _is_torch(model.var):
            torch = _torch()
            var_out = model.var * (k ** 2)
        else:
            var_out = model.var * (k ** 2)

    meta: Dict[str, Any] = {}
    if apply.return_intermediate:
        meta.update(
            {
                "exposure_target": apply.exposure_target,
                "temperature_target": apply.temperature_target,
                "mask_out_hot": apply.mask_out_hot,
                "propagate_var": apply.propagate_var,
                "clip_out": apply.clip_out,
                "scale_exposure": model.scale.scale_exposure,
                "scale_temperature": model.scale.scale_temperature,
                "exposure_ref": model.scale.exposure_ref,
                "temperature_ref": model.scale.temperature_ref,
                "k_multiplicative": k,
            }
        )

    return DarkApplyResult(corrected=corrected, var=var_out, meta=meta)


# -----------------------------------------------------------------------------
# Self-tests (quick NumPy sanity)
# -----------------------------------------------------------------------------

def _test_build_and_apply_mean() -> bool:
    import numpy as np
    rng = np.random.default_rng(0)
    N, H, W = 8, 8, 8
    true_dark = 12.0  # DN
    stack = true_dark + rng.normal(0, 0.3, size=(N, H, W)).astype(np.float32)

    build = DarkBuildParams(agg="mean", cosmic_reject=False, return_intermediate=True)
    scale = DarkScaleParams(exposure_ref=10.0, scale_exposure=True)
    res = build_master_dark(stack, build, scale)

    m = float(np.nanmedian(res.master.master))
    assert abs(m - true_dark) < 0.5, f"master too far: {m}"

    # Apply to a science frame with exposure 20s (expect ~ double subtraction)
    sci = (true_dark * 2.0) + rng.normal(0, 0.2, size=(H, W)).astype(np.float32)
    out = apply_dark(sci, res.master, ApplyDarkParams(exposure_target=20.0))
    med = float(np.nanmedian(out.corrected))
    # corrected should be ~0 (noise)
    assert abs(med) < 1.0
    return True


def _test_cr_reject_and_masks() -> bool:
    import numpy as np
    rng = np.random.default_rng(1)
    N, H, W = 6, 6, 6
    base = 5.0
    stack = base + rng.normal(0, 0.1, size=(N, H, W)).astype(np.float32)
    # inject CR spikes into middle frames
    stack[2, 2, 2] += 100.0
    stack[3, 1, 4] += 50.0

    build = DarkBuildParams(
        agg="median", cosmic_reject=True, cr_zmax=5.0, cr_iter=1, hot_sigma=6.0, bad_sigma=6.0, return_intermediate=True
    )
    scale = DarkScaleParams(exposure_ref=1.0)
    res = build_master_dark(stack, build, scale)

    m = float(np.nanmedian(res.master.master))
    assert abs(m - base) < 0.5, f"CR reject failed: {m} vs {base}"
    return True


if __name__ == "__main__":
    ok1 = _test_build_and_apply_mean()
    ok2 = _test_cr_reject_and_masks()
    print("dark.py self-tests:", ok1 and ok2)
