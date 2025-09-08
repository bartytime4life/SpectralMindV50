# src/spectramind/calib/dark.py
# =============================================================================
# SpectraMind V50 — Dark frame modeling & subtraction
# -----------------------------------------------------------------------------
# Build a master dark from stacks of dark frames:
#   - temporal aggregation: mean or median
#   - robust cosmic-ray rejection: MAD-based zscore iterations
#   - bad/hot pixel map creation (thresholded statistics)
#   - variance/weights propagation
#   - exposure-time & temperature scaling to target conditions
#
# Apply the master dark to science data with optional variance propagation
# and in-flight masking of hot/bad pixels.
#
# Backend agnostic: NumPy ndarray or Torch tensor inputs.
# Canonical image layout [..., H, W] and dark stacks [..., N, H, W]; configurable time axis.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821


# -----------------------------------------------------------------------------
# Backend shims (aligned with adc.py / cds.py style)
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
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims)
        den = mask.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        all_nan = den == 0
        return torch.where(all_nan, torch.tensor(float('nan'), dtype=x.dtype, device=x.device), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)


def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    # Torch fallback via numpy to guarantee numerical stability on older torch
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


def _nan_to_num(x: BackendArray, val=0.0) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.where(torch.isnan(x), torch.tensor(val, dtype=x.dtype, device=x.device), x)
    else:
        return _np().nan_to_num(x, nan=val)


# -----------------------------------------------------------------------------
# Models & Params
# -----------------------------------------------------------------------------

AggMode = Literal["mean", "median"]
ScaleModel = Literal["none", "linear", "arrhenius"]

@dataclass
class DarkBuildParams:
    """
    Controls building of a master dark from dark stacks.

    time_axis        : axis of the stack dimension (default -3 => [..., N, H, W])
    agg              : 'mean' or 'median' temporal aggregation
    cosmic_reject    : enable MAD-zscore CR rejection
    cr_zmax          : rejection threshold
    cr_iter          : robust iterations
    hot_sigma        : pixels with temporal mean > (global_mean + hot_sigma*global_std) flagged hot
                       (set None to disable)
    bad_sigma        : pixels with temporal std > (global_std + bad_sigma*global_std) flagged bad
                       (set None to disable)
    clip_out         : optional (low, high) to clip master dark
    dtype            : output float dtype
    return_intermediate: include debug info in meta
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
    Scaling model for applying dark to target conditions.

    exposure_ref     : exposure time used for master dark (seconds)
    temperature_ref  : sensor temperature for master dark (Kelvin)
    scale_exposure   : if True, scale linearly with exposure time (dark current)
    scale_temperature: 'none' | 'linear' | 'arrhenius'
        - 'linear'    : dark ~ a + b*(T - T_ref)   (b provided by temp_coeff)
        - 'arrhenius' : dark ~ dark * exp( Ea/k * (1/T - 1/T_ref) ); provide Ea_over_k
    temp_coeff       : linear slope (units: dark per Kelvin); used if scale_temperature='linear'
    Ea_over_k        : activation over Boltzmann (units: Kelvin); used if 'arrhenius'
    """
    exposure_ref: float = 1.0
    temperature_ref: Optional[float] = None
    scale_exposure: bool = True
    scale_temperature: ScaleModel = "none"
    temp_coeff: float = 0.0
    Ea_over_k: float = 0.0


@dataclass
class DarkModel:
    """
    Master dark product ready for application and scaling.
    """
    master: BackendArray                 # master dark frame [..., H, W] in DN (or electrons) per exposure_ref
    var: Optional[BackendArray]          # per-pixel variance estimate of master
    weights: Optional[BackendArray]      # effective sample counts used
    mask_hot: Optional[BackendArray]     # boolean mask of hot pixels
    mask_bad: Optional[BackendArray]     # boolean mask of noisy/bad pixels
    scale: DarkScaleParams               # scaling parameters
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ApplyDarkParams:
    """
    Parameters for applying (subtracting) the dark model to science frames.

    exposure_target  : science exposure time (seconds)
    temperature_target: science temperature (Kelvin), optional
    propagate_var    : if True and var is provided, compute output variance
    mask_out_hot     : if True, keep hot/bad pixels masked (NaN) in output
    clip_out         : optional (low, high)
    dtype            : output float dtype
    return_intermediate: include debug info
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

def _move_stack_axis(x: BackendArray, time_axis: int) -> Tuple[BackendArray, int]:
    """
    Move the time/stack axis to canonical position -3 (i.e., [..., N, H, W]).
    """
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
    """
    Robust temporal aggregation with optional MAD-based CR rejection.

    Returns:
      mean_or_median, variance (nanvar), weights (# non-NaN)
    """
    # initial view
    cur = x
    if not cosmic_reject:
        if agg == "mean":
            val = _nanmean(cur, axis=axis, keepdims=False)
        else:
            val = _nanmedian(cur, axis=axis, keepdims=False)
        if _is_torch(x):
            torch = _torch()
            w = (~torch.isnan(cur)).sum(dim=axis)
            v = _nanmean((cur - val.unsqueeze(axis)) ** 2, axis=axis, keepdims=False)
        else:
            np = _np()
            w = np.sum(~np.isnan(cur), axis=axis)
            v = _nanmean((cur - _expand_like(val, cur, axis)) ** 2, axis=axis, keepdims=False)
        return val, v, w

    # rejection loop
    torch_mode = _is_torch(x)
    rej_mask = None
    for _ in range(max(1, iters)):
        med = _nanmedian(cur, axis=axis, keepdims=True)
        mad = _nanmedian(_abs(cur - med), axis=axis, keepdims=True)
        # robust z = 0.6745*(x - med)/MAD, avoid div 0
        if torch_mode:
            torch = _torch()
            eps = torch.tensor(1e-12, dtype=mad.dtype, device=mad.device)
            mad = _where(mad <= eps, eps, mad)
            z = 0.6745 * (cur - med) / mad
            rej = _abs(z) > zmax
            cur = _where(rej, torch.tensor(float('nan'), dtype=cur.dtype, device=cur.device), cur)
            rej_mask = rej if rej_mask is None else (rej_mask | rej)
        else:
            np = _np()
            mad = _where(mad <= 1e-12, np.array(1e-12, dtype=mad.dtype), mad)
            z = 0.6745 * (cur - med) / mad
            rej = _abs(z) > zmax
            cur = cur.copy()
            cur[rej] = np.nan
            rej_mask = rej if rej_mask is None else (rej_mask | rej)

    # final aggregation
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
        v = _nanmean((cur - _expand_like(val, cur, axis)) ** 2, axis=axis, keepdims=False)

    return val, v, w


def _expand_like(val: BackendArray, ref: BackendArray, axis: int) -> BackendArray:
    """
    Expand val to broadcast shape of ref along axis.
    """
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


def _make_hot_bad_masks(master: BackendArray, var: BackendArray,
                        hot_sigma: Optional[float], bad_sigma: Optional[float]) -> Tuple[Optional[BackendArray], Optional[BackendArray], Dict[str, Any]]:
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

    meta.update(dict(global_mean=float(m_mean), global_std=float(m_std),
                     var_mean=float(v_mean), var_std=float(v_std)))
    return hot, bad, meta


def _scale_dark(master: BackendArray,
                exposure_ref: float, exposure_target: float,
                temperature_ref: Optional[float], temperature_target: Optional[float],
                temp_model: ScaleModel, temp_coeff: float, Ea_over_k: float,
                ) -> BackendArray:
    """
    Scale a master dark to target exposure and temperature.
    """
    out = master
    # Exposure scaling (linear with time)
    if exposure_ref is not None and exposure_target is not None and exposure_ref > 0:
        out = out * (float(exposure_target) / float(exposure_ref))

    # Temperature scaling
    if temp_model != "none" and temperature_ref is not None and temperature_target is not None:
        Tref = float(temperature_ref)
        Ttar = float(temperature_target)
        if temp_model == "linear":
            # additive linear: out += slope * (T - Tref)
            if _is_torch(out):
                torch = _torch()
                out = out + torch.tensor(temp_coeff * (Ttar - Tref), dtype=out.dtype, device=out.device)
            else:
                out = out + temp_coeff * (Ttar - Tref)
        elif temp_model == "arrhenius":
            # multiplicative: out *= exp(Ea/k * (1/T - 1/Tref))
            coeff = Ea_over_k * (1.0 / Ttar - 1.0 / Tref)
            if _is_torch(out):
                torch = _torch()
                out = out * torch.exp(torch.tensor(coeff, dtype=out.dtype, device=out.device))
            else:
                out = out * _np().exp(coeff)
        else:
            raise ValueError(f"Unknown temperature model: {temp_model}")
    return out


# -----------------------------------------------------------------------------
# Build master dark
# -----------------------------------------------------------------------------

def build_master_dark(
    dark_stack: BackendArray,
    build: DarkBuildParams,
    scale: DarkScaleParams,
) -> DarkBuildResult:
    """
    Build a master dark frame from a stack of dark frames.

    dark_stack : [..., N, H, W] or with time-axis anywhere (see build.time_axis)
    build      : DarkBuildParams
    scale      : DarkScaleParams (only stores ref conditions for later apply; no scaling here)

    Returns
    -------
    DarkBuildResult with DarkModel (master, var, weights, masks, scale).
    """
    x = _to_float(dark_stack, dtype=build.dtype)
    x, orig_axis = _move_stack_axis(x, build.time_axis)   # -> [..., N, H, W]
    N = x.shape[-3]
    if N < 1:
        raise ValueError("build_master_dark: need at least one dark frame")

    # Robust aggregation across stack
    master, var, weights = _robust_temporal(
        x, axis=-3, agg=build.agg,
        cosmic_reject=build.cosmic_reject,
        zmax=build.cr_zmax, iters=build.cr_iter
    )

    # Optional clipping
    if build.clip_out is not None:
        low, high = build.clip_out
        master = _clip(master, low, high)

    # Create hot/bad masks from temporal stats
    mask_hot, mask_bad, stats_meta = _make_hot_bad_masks(master, var, build.hot_sigma, build.bad_sigma)

    meta: Dict[str, Any] = {}
    if build.return_intermediate:
        meta.update({
            "input_shape": tuple(dark_stack.shape),
            "stack_N": N,
            "agg": build.agg,
            "cosmic_reject": build.cosmic_reject,
            "cr_zmax": build.cr_zmax,
            "cr_iter": build.cr_iter,
            **stats_meta
        })

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

def apply_dark(
    science: BackendArray,
    model: DarkModel,
    apply: ApplyDarkParams,
) -> DarkApplyResult:
    """
    Subtract a (scaled) master dark from science frames.

    science : [..., H, W] or batch [..., ..., H, W]
    model   : DarkModel built at reference exposure/temperature
    apply   : ApplyDarkParams

    Returns
    -------
    DarkApplyResult
      - corrected: science - scaled_dark
      - var: propagated variance if available and requested, else None
      - meta: details
    """
    s = _to_float(science, dtype=apply.dtype)

    # Scale dark to target exposure/temperature
    d = _scale_dark(
        model.master,
        exposure_ref=model.scale.exposure_ref,
        exposure_target=apply.exposure_target,
        temperature_ref=model.scale.temperature_ref,
        temperature_target=apply.temperature_target,
        temp_model=model.scale.scale_temperature,
        temp_coeff=model.scale.temp_coeff,
        Ea_over_k=model.scale.Ea_over_k,
    )

    # Broadcast to science shape (batch-safe)
    # Both s and d are [..., H, W]. Let broadcasting handle leading dims.
    # Apply masks if requested
    if apply.mask_out_hot and (model.mask_hot is not None or model.mask_bad is not None):
        if _is_torch(s) or _is_torch(d):
            torch = _torch()
            bad = None
            if model.mask_hot is not None:
                bad = model.mask_hot if bad is None else (bad | model.mask_hot)
            if model.mask_bad is not None:
                bad = model.mask_bad if bad is None else (bad | model.mask_bad)
            if bad is not None:
                # expand bad to match leading dims by broadcasting
                nanv = torch.tensor(float('nan'), dtype=s.dtype, device=s.device)
                s = _where(bad, nanv, s)
                d = _where(bad, nanv, d)
        else:
            np = _np()
            bad = None
            if model.mask_hot is not None:
                bad = model.mask_hot if bad is None else (bad | model.mask_hot)
            if model.mask_bad is not None:
                bad = model.mask_bad if bad is None else (bad | model.mask_bad)
            if bad is not None:
                s = s.copy()
                d = d.copy()
                s[bad] = np.nan
                d[bad] = np.nan

    corrected = s - d

    # Optional clipping
    if apply.clip_out is not None:
        low, high = apply.clip_out
        corrected = _clip(corrected, low, high)

    # Variance propagation: var_out = var_science + var_dark (if provided)
    var_out: Optional[BackendArray] = None
    if apply.propagate_var and model.var is not None:
        # Scale var by same exposure/temperature factors: if scaling dark multiplicatively by k, var scales by k^2.
        # Derive effective multiplicative factor between 'd' and 'model.master' (handle NaNs robustly):
        if _is_torch(d) or _is_torch(model.master):
            torch = _torch()
            # avoid division by zero; compute k as median of d/master over valid pixels
            ratio = d / _where(_abs(model.master) < 1e-20,
                               torch.tensor(1e-20, dtype=d.dtype, device=d.device),
                               model.master)
            # Use scalar-ish robust factor: median of finite ratios
            k = torch.nanmedian(ratio).item()
            var_out = model.var * (k ** 2)
        else:
            np = _np()
            denom = _where(_abs(model.master) < 1e-20, np.array(1e-20, dtype=d.dtype), model.master)
            ratio = d / denom
            k = float(np.nanmedian(ratio))
            var_out = model.var * (k ** 2)

    meta: Dict[str, Any] = {}
    if apply.return_intermediate:
        meta.update({
            "exposure_target": apply.exposure_target,
            "temperature_target": apply.temperature_target,
            "mask_out_hot": apply.mask_out_hot,
            "propagate_var": apply.propagate_var,
            "clip_out": apply.clip_out,
            "scale_model": model.scale.scale_temperature,
            "exposure_ref": model.scale.exposure_ref,
            "temperature_ref": model.scale.temperature_ref,
        })

    return DarkApplyResult(corrected=corrected, var=var_out, meta=meta)


# -----------------------------------------------------------------------------
# Self-tests (light)
# -----------------------------------------------------------------------------

def _test_build_and_apply_mean():
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
    # corrected should be ~0
    assert abs(med) < 1.0
    return True


def _test_cr_reject_and_masks():
    import numpy as np
    rng = np.random.default_rng(1)
    N, H, W = 6, 6, 6
    base = 5.0
    stack = base + rng.normal(0, 0.1, size=(N, H, W)).astype(np.float32)
    # inject CR spikes into middle frames
    stack[2, 2, 2] += 100.0
    stack[3, 1, 4] += 50.0
    build = DarkBuildParams(agg="median", cosmic_reject=True, cr_zmax=5.0, cr_iter=1,
                            hot_sigma=6.0, bad_sigma=6.0, return_intermediate=True)
    scale = DarkScaleParams(exposure_ref=1.0)
    res = build_master_dark(stack, build, scale)
    # master median should be close to base after CR reject
    import numpy as np
    m = float(np.nanmedian(res.master.master))
    assert abs(m - base) < 0.5
    return True


if __name__ == "__main__":
    ok1 = _test_build_and_apply_mean()
    ok2 = _test_cr_reject_and_masks()
    print("dark.py self-tests:", ok1 and ok2)
