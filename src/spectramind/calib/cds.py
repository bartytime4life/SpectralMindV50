# src/spectramind/calib/cds.py
# =============================================================================
# SpectraMind V50 — Correlated Double Sampling (CDS) utilities
# -----------------------------------------------------------------------------
# Supports:
#   - simple two-sample CDS (pairwise difference)
#   - Fowler-N sampling (avg N ref and N sig frames, difference)
#   - rolling/windowed CDS (sliding average)
#   - robust temporal aggregation (mean/median)
#   - optional cosmic-ray rejection via MAD-based z-scores
#   - hot/saturated pixel masking and weight/variance propagation
#
# Backend-agnostic API: works with NumPy ndarray or Torch tensor inputs.
# Expected shapes: time-series frames as [..., T, H, W] (configurable time_axis)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

__all__ = [
    "AggMode",
    "CDSMode",
    "WeightMode",
    "CDSParams",
    "CDSResult",
    "cds",
]

# -----------------------------------------------------------------------------
# Minimal backend shims (mirroring adc.py style for consistency)
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


def _as_tensor_like(x: BackendArray, val: float) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.tensor(val, dtype=x.dtype, device=x.device)
    else:
        np = _np()
        return np.array(val, dtype=x.dtype)


def _to_float(x: BackendArray, dtype: Optional[Union[str, Any]] = None) -> BackendArray:
    """
    Convert to float dtype without extra copy when possible.
    torch: dtype is like "float32" | "float64" | torch.float32 | torch.float64
    numpy: dtype can be np.float32, "float32", etc.
    """
    if _is_torch(x):
        torch = _torch()
        target = _resolve_dtype("torch", dtype) or torch.float32
        return x.to(target)
    else:
        np = _np()
        target = _resolve_dtype("numpy", dtype) or np.float32
        return x.astype(target, copy=False)


def _zeros_like(x: BackendArray, shape: Optional[Tuple[int, ...]] = None) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.zeros(shape or x.shape, dtype=x.dtype, device=x.device)
    np = _np()
    return np.zeros(shape or x.shape, dtype=x.dtype)


def _empty_like(x: BackendArray, shape: Tuple[int, ...]) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.empty(shape, dtype=x.dtype, device=x.device)
    return _np().empty(shape, dtype=x.dtype)


def _where(mask: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(mask) or _is_torch(a) or _is_torch(b):
        return _torch().where(mask, a, b)
    return _np().where(mask, a, b)


def _abs(x: BackendArray) -> BackendArray:
    return x.abs() if _is_torch(x) else _np().abs(x)


def _isnan(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        return _torch().isnan(x)
    return _np().isnan(x)


def _nanmean(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        if hasattr(torch, "nanmean"):
            return torch.nanmean(x, dim=axis, keepdim=keepdims)  # type: ignore[attr-defined]
        # Emulate nanmean
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, _as_tensor_like(x, 0.0)).sum(dim=axis, keepdim=keepdims)
        den = mask.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        all_nan = den == 0
        return torch.where(all_nan, _as_tensor_like(x, float("nan")), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)


def _torch_nanmedian(x, axis=None, keepdims=False):
    """
    Torch-compatible nanmedian: tries torch.nanmedian when available,
    else uses take_along_dim workaround, else NumPy bridge last-resort.
    """
    torch = _torch()
    # Newer torch
    if hasattr(torch, "nanmedian"):
        return torch.nanmedian(x, dim=axis, keepdim=keepdims).values  # type: ignore[attr-defined]

    # Fallback: mask nans by +inf, sort, then take middle index among valid count
    isnan = torch.isnan(x)
    fill = torch.full_like(x, float("inf"))
    xf = torch.where(isnan, fill, x)
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
        # if all valid==0 -> med == inf; replace those with nan
        all_nan = (valid == 0)
        if all_nan.any():
            repl = torch.where(all_nan.squeeze(dim=axis), torch.full_like(med, float("nan")), med)
            return repl
        return med

    # As a last resort (very old torch), fallback to NumPy bridge (CPU).
    return _to_numpy_and_back_nanmedian(x, axis=axis, keepdims=keepdims)


def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        return _torch_nanmedian(x, axis=axis, keepdims=keepdims)
    else:
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)


def _to_numpy_and_back_nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    """Helper: torch nanmedian fallback via CPU numpy (last resort)."""
    if not _is_torch(x):
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)
    torch = _torch()
    np = _np()
    x_np = x.detach().cpu().numpy()
    med_np = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
    med_t = torch.from_numpy(med_np).to(device=x.device, dtype=x.dtype)
    return med_t


def _nanstd(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        if hasattr(torch, "nanstd"):
            return torch.nanstd(x, dim=axis, keepdim=keepdims)  # type: ignore[attr-defined]
        m = _nanmean(x, axis=axis, keepdims=True)
        diff2 = (x - m) ** 2
        v = _nanmean(diff2, axis=axis, keepdims=keepdims)
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
# API dataclasses
# -----------------------------------------------------------------------------

AggMode = Literal["mean", "median"]
CDSMode = Literal["two_sample", "fowler", "rolling"]
WeightMode = Literal["counts", "ivar"]  # counts = #samples; ivar = inverse variance estimate


@dataclass
class CDSParams:
    """
    Parameters controlling the CDS stage.

    mode:            "two_sample" | "fowler" | "rolling"
    time_axis:       axis index for time dimension (default: -3 for [..., T, H, W])
    ref_count:       for Fowler / rolling, number of reference frames (N)
    sig_count:       for Fowler / rolling, number of signal frames (M)
    stride:          for rolling, step between windows along time (default 1)
    agg:             "mean" or "median" for temporal aggregation
    cosmic_reject:   if True, perform MAD-zscore CR rejection inside windows
    cr_zmax:         z-threshold for CR mask (e.g., 6.0)
    cr_iter:         number of robust iterations (recompute mean/median after masking)
    mask_saturated:  optional mask (same shape as frames) where True => ignore in agg
    mask_hot:        optional mask for hot/bad pixels (ignore)
    clip_out:        optional (low, high) to clip final CDS output
    return_intermediate: include debug info in meta
    dtype:           output float dtype (torch: "float32"/"float64"; numpy dtype or string)
    weight_mode:     "counts" -> effective samples; "ivar" -> inverse variance estimate
    """

    mode: CDSMode = "two_sample"
    time_axis: int = -3
    ref_count: int = 1
    sig_count: int = 1
    stride: int = 1
    agg: AggMode = "mean"
    cosmic_reject: bool = False
    cr_zmax: float = 6.0
    cr_iter: int = 1
    mask_saturated: Optional[BackendArray] = None
    mask_hot: Optional[BackendArray] = None
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    return_intermediate: bool = False
    dtype: Optional[Union[str, Any]] = None
    weight_mode: WeightMode = "counts"


@dataclass
class CDSResult:
    """
    CDS output bundle.

    cds: [..., H, W] for two_sample/fowler, or [..., Nout, H, W] for rolling
    weights: effective weights per output (counts or inverse variance)
    ivar: optional inverse variance (if weight_mode=='ivar') else None
    meta: additional diagnostic info
    """
    cds: BackendArray
    weights: Optional[BackendArray]
    ivar: Optional[BackendArray] = None
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Core CDS helpers
# -----------------------------------------------------------------------------

def _normalize_time_axis(x: BackendArray, time_axis: int) -> Tuple[BackendArray, int]:
    """
    Normalize the time axis to position -3 so that frames are shaped as [..., T, H, W].
    Returns:
        x_norm: possibly transposed view
        target_axis: canonical axis index (ndim-3)
    """
    nd = x.ndim
    if nd < 3:
        raise ValueError(f"Expected at least 3 dims [T,H,W], got shape {x.shape}")
    if time_axis < 0:
        time_axis = nd + time_axis
    target = nd - 3
    if time_axis == target:
        return x, target
    if _is_torch(x):
        perm = list(range(nd))
        perm[target], perm[time_axis] = perm[time_axis], perm[target]
        x2 = x.permute(*perm)
    else:
        x2 = x.swapaxes(target, time_axis)
    return x2, target


def _apply_masks(frames: BackendArray,
                 mask_sat: Optional[BackendArray],
                 mask_hot: Optional[BackendArray]) -> BackendArray:
    """
    Apply saturated / hot pixel masks by setting those samples to NaN
    to exclude them from subsequent aggregation.
    """
    if mask_sat is None and mask_hot is None:
        return frames
    if _is_torch(frames):
        torch = _torch()
        out = frames.clone()
        if mask_sat is not None:
            out = torch.where(mask_sat, _as_tensor_like(frames, float('nan')), out)
        if mask_hot is not None:
            out = torch.where(mask_hot, _as_tensor_like(frames, float('nan')), out)
        return out
    else:
        np = _np()
        out = frames.copy()
        if mask_sat is not None:
            out[mask_sat] = np.nan
        if mask_hot is not None:
            out[mask_hot] = np.nan
        return out


def _robust_aggregate(
    x: BackendArray,
    axis: int,
    agg: AggMode,
    cosmic_reject: bool,
    zmax: float,
    iters: int,
) -> Tuple[BackendArray, BackendArray, BackendArray]:
    """
    Robust temporal aggregation with optional cosmic-ray rejection.

    Args
    ----
    x : BackendArray
        Samples over 'axis'
    axis : int
        Aggregation axis (time)
    agg : "mean" | "median"
    cosmic_reject : bool
        Apply MAD-based z-score rejection if True
    zmax : float
        Robust z threshold
    iters : int
        Iterations of CR rejection

    Returns
    -------
    val : aggregated value (nanmean / nanmedian) over axis
    w   : effective sample count over axis
    var : estimated sample variance of the aggregate (per-pixel), i.e.
          var(val) ~ (std^2 / w) for mean; for median, we return an empirical
          robust proxy using asymptotic factor (pi/2)*sigma^2/n.
    """
    torch_mode = _is_torch(x)

    if not cosmic_reject:
        if agg == "mean":
            val = _nanmean(x, axis=axis, keepdims=False)
            std = _nanstd(x, axis=axis, keepdims=False)
        else:
            val = _nanmedian(x, axis=axis, keepdims=False)
            mad = _nanmedian(_abs(x - _nanmedian(x, axis=axis, keepdims=True)), axis=axis, keepdims=False)
            std = 1.4826 * mad
        if torch_mode:
            w = (~_torch().isnan(x)).sum(dim=axis)
            w_clamped = w.clamp_min(1)
        else:
            np = _np()
            w = np.sum(~np.isnan(x), axis=axis)
            w_clamped = w.copy()
            w_clamped[w_clamped < 1] = 1
        if agg == "mean":
            var = (std ** 2) / w_clamped
        else:
            var = (1.5708 * (std ** 2)) / w_clamped
        return val, w, var

    # CR rejection loop
    cur = x
    for _ in range(max(1, iters)):
        center = _nanmedian(cur, axis=axis, keepdims=True) if agg == "median" else _nanmean(cur, axis=axis, keepdims=True)
        dev = cur - center
        mad = _nanmedian(_abs(dev), axis=axis, keepdims=True)
        if torch_mode:
            mad = _where(mad <= _as_tensor_like(mad, 1e-12), _as_tensor_like(mad, 1e-12), mad)
        else:
            mad = _where(mad <= 1e-12, _np().array(1e-12, dtype=mad.dtype), mad)  # type: ignore[arg-type]
        z = 0.6745 * dev / mad  # 0.6745 approx => z ~ (x - median)/MAD
        rej = _abs(z) > zmax
        if torch_mode:
            cur = _where(rej, _as_tensor_like(cur, float('nan')), cur)
        else:
            cur = cur.copy()
            cur[rej] = _np().nan

    # final aggregation + variance estimate
    if agg == "mean":
        val = _nanmean(cur, axis=axis, keepdims=False)
        std = _nanstd(cur, axis=axis, keepdims=False)
    else:
        val = _nanmedian(cur, axis=axis, keepdims=False)
        mad = _nanmedian(_abs(cur - _nanmedian(cur, axis=axis, keepdims=True)), axis=axis, keepdims=False)
        std = 1.4826 * mad

    if torch_mode:
        w = (~_torch().isnan(cur)).sum(dim=axis)
        w_clamped = w.clamp_min(1)
    else:
        np = _np()
        w = np.sum(~np.isnan(cur), axis=axis)
        w_clamped = w.copy()
        w_clamped[w_clamped < 1] = 1

    if agg == "mean":
        var = (std ** 2) / w_clamped
    else:
        var = (1.5708 * (std ** 2)) / w_clamped
    return val, w, var


def _window_slices(T: int, ref: int, sig: int, stride: int) -> Tuple["list[Tuple[slice, slice]]", int]:
    """
    Build rolling window slices along time: for each output,
    take ref_count frames as reference and sig_count frames as signal.
    Returns (list_of_pairs, n_out).
    """
    pairs = []
    t = 0
    while t + ref + sig <= T:
        pairs.append((slice(t, t + ref), slice(t + ref, t + ref + sig)))
        t += stride
    return pairs, len(pairs)


def _combine_ref_sig(
    ref_val: BackendArray, ref_var: BackendArray,
    sig_val: BackendArray, sig_var: BackendArray,
) -> Tuple[BackendArray, BackendArray]:
    """
    Combine reference/signal aggregates into a CDS difference and its variance.

    cds = sig - ref
    var(cds) = var(sig) + var(ref)   (assuming independence)
    """
    cds_val = sig_val - ref_val
    cds_var = sig_var + ref_var
    return cds_val, cds_var


def _weights_from_variance(cds_var: BackendArray) -> BackendArray:
    """Return inverse-variance weights (avoid divide-by-zero)."""
    if _is_torch(cds_var):
        eps = _as_tensor_like(cds_var, 1e-24)
        return 1.0 / (cds_var + eps)
    else:
        np = _np()
        return 1.0 / (cds_var + np.array(1e-24, dtype=cds_var.dtype))


# -----------------------------------------------------------------------------
# Public CDS API
# -----------------------------------------------------------------------------

def cds(frames: BackendArray, params: CDSParams) -> CDSResult:
    """
    Run correlated double sampling on a time-series stack of frames.

    Parameters
    ----------
    frames : BackendArray
        Time-series frames [..., T, H, W].
    params : CDSParams
        Configuration for CDS.

    Returns
    -------
    CDSResult
        cds:
            CDS difference image(s) [..., H, W] for two_sample/fowler,
            or [..., Nout, H, W] for rolling
        weights:
            If weight_mode == "counts": effective sample counts used at each pixel.
            If weight_mode == "ivar": inverse-variance weights (per-pixel).
        ivar:
            Inverse-variance maps if computed (else None).
        meta:
            Diagnostics and intermediates (only when return_intermediate=True).
    """
    # Normalize dtype
    x = _to_float(frames, dtype=params.dtype)

    # Normalize time axis to [..., T, H, W]
    x, _ = _normalize_time_axis(x, params.time_axis)
    nd = x.ndim
    T = x.shape[-3]
    H, W = x.shape[-2], x.shape[-1]
    batch_shape = x.shape[:-3]

    # Apply pixel masks
    x = _apply_masks(x, params.mask_saturated, params.mask_hot)

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update(
            dict(
                input_shape=frames.shape,
                canonical_time_axis=nd - 3,
                ref_count=params.ref_count,
                sig_count=params.sig_count,
                stride=params.stride,
                agg=params.agg,
                cosmic_reject=params.cosmic_reject,
                cr_zmax=params.cr_zmax,
                cr_iter=params.cr_iter,
                weight_mode=params.weight_mode,
            )
        )

    # -----------------------------------------------
    # two-sample: first frame is ref, last frame is sig
    # -----------------------------------------------
    if params.mode == "two_sample":
        if T < 2:
            raise ValueError(f"two_sample CDS requires T>=2, got T={T}")
        ref = x[..., 0:1, :, :]
        sig = x[..., -1:, :, :]

        ref_val, ref_w, ref_var = _robust_aggregate(
            ref, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
            zmax=params.cr_zmax, iters=params.cr_iter,
        )
        sig_val, sig_w, sig_var = _robust_aggregate(
            sig, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
            zmax=params.cr_zmax, iters=params.cr_iter,
        )

        out, var = _combine_ref_sig(ref_val, ref_var, sig_val, sig_var)
        out = _clip(out, *(params.clip_out or (None, None)))

        if params.weight_mode == "ivar":
            weights = _weights_from_variance(var)
            ivar = weights
        else:
            weights = ref_w + sig_w
            ivar = None

        if params.return_intermediate:
            meta.update({"ref_w": ref_w, "sig_w": sig_w})
        return CDSResult(cds=out, weights=weights, ivar=ivar, meta=meta)

    # -----------------------------------------------
    # Fowler-N: average N ref and M sig frames
    # -----------------------------------------------
    elif params.mode == "fowler":
        N = int(params.ref_count)
        M = int(params.sig_count)
        if N <= 0 or M <= 0:
            raise ValueError("Fowler requires ref_count>0 and sig_count>0")
        if T < (N + M):
            raise ValueError(f"Fowler requires T >= N+M, got T={T}, N={N}, M={M}")

        ref = x[..., :N, :, :]
        sig = x[..., N:N + M, :, :]

        ref_val, ref_w, ref_var = _robust_aggregate(
            ref, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
            zmax=params.cr_zmax, iters=params.cr_iter,
        )
        sig_val, sig_w, sig_var = _robust_aggregate(
            sig, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
            zmax=params.cr_zmax, iters=params.cr_iter,
        )

        out, var = _combine_ref_sig(ref_val, ref_var, sig_val, sig_var)
        out = _clip(out, *(params.clip_out or (None, None)))

        if params.weight_mode == "ivar":
            weights = _weights_from_variance(var)
            ivar = weights
        else:
            weights = ref_w + sig_w
            ivar = None

        if params.return_intermediate:
            meta.update({"ref_w": ref_w, "sig_w": sig_w})
        return CDSResult(cds=out, weights=weights, ivar=ivar, meta=meta)

    # -----------------------------------------------
    # rolling: sliding Fowler windows -> multiple outputs along time
    # -----------------------------------------------
    elif params.mode == "rolling":
        N = int(params.ref_count)
        M = int(params.sig_count)
        stride = max(1, int(params.stride))
        if N <= 0 or M <= 0:
            raise ValueError("rolling CDS requires ref_count>0 and sig_count>0")
        pairs, n_out = _window_slices(T, N, M, stride)
        if n_out <= 0:
            raise ValueError(f"rolling CDS: no valid windows for T={T} with N={N}, M={M}, stride={stride}")

        cds_stack = _empty_like(x, (*batch_shape, n_out, H, W))
        w_stack = _empty_like(x, (*batch_shape, n_out, H, W))
        ivar_stack = _empty_like(x, (*batch_shape, n_out, H, W)) if params.weight_mode == "ivar" else None

        for i, (s_ref, s_sig) in enumerate(pairs):
            r = x[..., s_ref, :, :]
            s = x[..., s_sig, :, :]

            r_val, r_w, r_var = _robust_aggregate(r, axis=-3, agg=params.agg,
                                                  cosmic_reject=params.cosmic_reject,
                                                  zmax=params.cr_zmax, iters=params.cr_iter)
            s_val, s_w, s_var = _robust_aggregate(s, axis=-3, agg=params.agg,
                                                  cosmic_reject=params.cosmic_reject,
                                                  zmax=params.cr_zmax, iters=params.cr_iter)

            delta, var = _combine_ref_sig(r_val, r_var, s_val, s_var)
            delta = _clip(delta, *(params.clip_out or (None, None)))
            cds_stack[..., i, :, :] = delta

            if params.weight_mode == "ivar":
                iv = _weights_from_variance(var)
                w_stack[..., i, :, :] = iv
                if ivar_stack is not None:
                    ivar_stack[..., i, :, :] = iv
            else:
                w_stack[..., i, :, :] = r_w + s_w

        return CDSResult(cds=cds_stack, weights=w_stack, ivar=ivar_stack, meta=meta)

    else:
        raise ValueError(f"Unknown CDS mode: {params.mode}")


# -----------------------------------------------------------------------------
# Minimal self-checks
# -----------------------------------------------------------------------------

def _test_two_sample_mean():
    import numpy as np
    # frames [..., T, H, W] = [T, 2, 2]
    f0 = np.zeros((2, 2), dtype=np.float32) + 10
    f1 = np.zeros((2, 2), dtype=np.float32) + 25
    stack = np.stack([f0, f1], axis=0)  # [T,H,W]=[2,2,2]
    # Move time to canonical by adding a dummy batch axis => shape [1,T,H,W]
    stack = stack[None, ...]
    res = cds(stack, CDSParams(mode="two_sample", agg="mean", return_intermediate=True))
    expected = (f1 - f0)[None, ...]
    assert np.allclose(res.cds, expected)
    assert res.weights is not None
    return True


def _test_fowler_median_cr():
    import numpy as np
    rng = np.random.default_rng(0)
    T, H, W = 8, 4, 4
    stack = np.zeros((1, T, H, W), dtype=np.float32)
    stack[:, 0:3] = 10.0 + rng.normal(0, 0.3, size=(1, 3, H, W))
    stack[:, 3:6] = 20.0 + rng.normal(0, 0.3, size=(1, 3, H, W))
    # cosmic ray spike in one ref frame
    stack[:, 1, 1, 1] += 100.0
    params = CDSParams(
        mode="fowler",
        ref_count=3,
        sig_count=3,
        agg="median",
        cosmic_reject=True,
        cr_zmax=5.0,
        cr_iter=1,
        weight_mode="ivar",
    )
    res = cds(stack, params)
    # Expected near 10 difference (20-10)
    med_val = _np().median(res.cds)
    assert abs(med_val - 10.0) < 0.8
    # ivar should be positive and finite
    assert res.ivar is not None
    assert _np().isfinite(_np().mean(res.ivar))
    return True


if __name__ == "__main__":
    ok1 = _test_two_sample_mean()
    ok2 = _test_fowler_median_cr()
    print("cds.py self-tests:", ok1 and ok2)