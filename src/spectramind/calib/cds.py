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
#   - hot/saturated pixel masking and weight propagation
#
# Backend-agnostic API: works with NumPy ndarray or Torch tensor inputs.
# Shapes: time-series frames as [..., T, H, W] (configurable time_axis)
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821


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


def _to_float(x: BackendArray, dtype: Optional[str] = None) -> BackendArray:
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
        # No torch.nanmean on older versions; emulate
        torch = _torch()
        mask = ~torch.isnan(x)
        num = torch.where(mask, x, torch.tensor(0., dtype=x.dtype, device=x.device)).sum(dim=axis, keepdim=keepdims)
        den = mask.sum(dim=axis, keepdim=keepdims).clamp_min(1)
        out = num / den
        # Fill all-nan slices with nan
        all_nan = den == 0
        return torch.where(all_nan, torch.tensor(float('nan'), dtype=x.dtype, device=x.device), out)
    else:
        return _np().nanmean(x, axis=axis, keepdims=keepdims)


def _nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        # emulate nanmedian by masking
        torch = _torch()
        mask = ~torch.isnan(x)
        # Sort replacing nans by +inf to push them to the end
        fill = torch.tensor(float('inf'), dtype=x.dtype, device=x.device)
        xf = torch.where(mask, x, fill)
        # sort along axis
        sorted_xf, _ = torch.sort(xf, dim=axis)
        # count valid
        cnt = mask.sum(dim=axis, keepdim=keepdims).to(x.dtype)
        # median index(s)
        if keepdims:
            mid_low = (cnt - 1) // 2
            mid_high = cnt // 2
        else:
            # need to expand dims for gather
            mid_low = ((cnt - 1) // 2).to(torch.long)
            mid_high = (cnt // 2).to(torch.long)
        # We need to gather at positions mid_low and mid_high; make index grid
        # Simpler approach: compute both medians by selecting along sorted dimension.
        # For code brevity, fallback to converting to numpy for median in torch path:
        return _to_numpy_and_back_nanmedian(x, axis=axis, keepdims=keepdims)
    else:
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)


def _to_numpy_and_back_nanmedian(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    """Helper: torch nanmedian fallback via CPU numpy (small overhead OK for robustness pipeline)."""
    if not _is_torch(x):
        return _np().nanmedian(x, axis=axis, keepdims=keepdims)
    torch = _torch()
    np = _np()
    # move to CPU numpy
    x_np = x.detach().cpu().numpy()
    med_np = np.nanmedian(x_np, axis=axis, keepdims=keepdims)
    med_t = torch.from_numpy(med_np).to(device=x.device, dtype=x.dtype)
    return med_t


def _nanstd(x: BackendArray, axis=None, keepdims=False) -> BackendArray:
    if _is_torch(x):
        # emulate nanstd
        torch = _torch()
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


@dataclass
class CDSResult:
    """
    CDS output bundle.
    """
    cds: BackendArray               # difference image(s) after CDS
    weights: Optional[BackendArray] # weights per output (e.g., #effective frames used)
    meta: Dict[str, Any] = field(default_factory=dict)


# -----------------------------------------------------------------------------
# Core CDS helpers
# -----------------------------------------------------------------------------

def _move_time_axis(x: BackendArray, time_axis: int) -> Tuple[BackendArray, int]:
    """
    Normalize time axis to be a known position for implementation:
    We will move time axis to position -3 (i.e., [..., T, H, W]) if not already.
    Returns (x_moved, orig_axis).
    """
    nd = x.ndim
    # We expect [..., T, H, W], so time axis should be nd-3
    target = nd - 3
    if time_axis < 0:
        time_axis = nd + time_axis
    if time_axis == target:
        return x, time_axis
    # permute axes
    perm = list(range(nd))
    perm[target], perm[time_axis] = perm[time_axis], perm[target]
    if _is_torch(x):
        x2 = x.permute(*perm)
    else:
        x2 = x.transpose(target, time_axis)
    return x2, time_axis


def _apply_masks(frames: BackendArray,
                 mask_sat: Optional[BackendArray],
                 mask_hot: Optional[BackendArray]) -> BackendArray:
    """
    Apply saturated/hot masks by setting those samples to NaN
    to exclude from aggregation.
    """
    if mask_sat is None and mask_hot is None:
        return frames
    if _is_torch(frames):
        torch = _torch()
        out = frames.clone()
        if mask_sat is not None:
            out[mask_sat] = torch.tensor(float('nan'), dtype=out.dtype, device=out.device)
        if mask_hot is not None:
            out[mask_hot] = torch.tensor(float('nan'), dtype=out.dtype, device=out.device)
        return out
    else:
        import numpy as np
        out = frames.copy()
        if mask_sat is not None:
            out[mask_sat] = np.nan
        if mask_hot is not None:
            out[mask_hot] = np.nan
        return out


def _robust_aggregate(x: BackendArray, axis: int, agg: AggMode, cosmic_reject: bool,
                      zmax: float, iters: int) -> Tuple[BackendArray, BackendArray]:
    """
    Robust temporal aggregation with optional cosmic-ray rejection.

    Returns:
      (val, weights)
      val: nanmean/nanmedian after masking high-MAD z-scores
      weights: number of effective samples used (post-masking)
    """
    if not cosmic_reject:
        if agg == "mean":
            val = _nanmean(x, axis=axis, keepdims=False)
        else:
            val = _nanmedian(x, axis=axis, keepdims=False)
        # weights (#non-nan)
        if _is_torch(x):
            torch = _torch()
            w = (~torch.isnan(x)).sum(dim=axis)
        else:
            w = _np().sum(~_np().isnan(x), axis=axis)
        return val, w

    # CR rejection loop
    cur = x
    mask = None
    torch_mode = _is_torch(x)
    for _ in range(max(1, iters)):
        if agg == "mean":
            m = _nanmean(cur, axis=axis, keepdims=True)
        else:
            m = _nanmedian(cur, axis=axis, keepdims=True)
        dev = cur - m
        mad = _nanmedian(_abs(dev), axis=axis, keepdims=True)
        # Robust zscore ≈ 0.6745*(x - med) / MAD
        if torch_mode:
            torch = _torch()
            mad = _where(mad <= 1e-12, torch.tensor(1e-12, dtype=mad.dtype, device=mad.device), mad)
            z = 0.6745 * dev / mad
            # new mask: |z| > zmax -> set NaN
            rej = _abs(z) > zmax
            cur = _where(rej, torch.tensor(float('nan'), dtype=cur.dtype, device=cur.device), cur)
            mask = rej if mask is None else (mask | rej)
        else:
            np = _np()
            mad = _where(mad <= 1e-12, np.array(1e-12, dtype=mad.dtype), mad)
            z = 0.6745 * dev / mad
            rej = _abs(z) > zmax
            cur = cur.copy()
            cur[rej] = np.nan
            mask = rej if mask is None else (mask | rej)

    # final aggregation
    if agg == "mean":
        val = _nanmean(cur, axis=axis, keepdims=False)
    else:
        val = _nanmedian(cur, axis=axis, keepdims=False)
    if torch_mode:
        w = (~_torch().isnan(cur)).sum(dim=axis)
    else:
        w = _np().sum(~_np().isnan(cur), axis=axis)
    return val, w


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


# -----------------------------------------------------------------------------
# Public CDS APIs
# -----------------------------------------------------------------------------

def cds(
    frames: BackendArray,
    params: CDSParams,
) -> CDSResult:
    """
    Run correlated double sampling on a time-series stack of frames.

    Args
    ----
    frames : [..., T, H, W]
        Time-series frames. dtype will be converted to requested float dtype.
    params : CDSParams
        Configuration for CDS.

    Returns
    -------
    CDSResult
      - cds: CDS difference image(s) [..., H, W] for two_sample/fowler, or [..., Nout, H, W] for rolling
      - weights: effective sample counts for each output (post-masking / CR rejection)
      - meta: intermediates for debugging (optional)
    """
    # Normalize dtype
    x = _to_float(frames, dtype=params.dtype)

    # Move time axis to canonical position -3
    x, orig_taxis = _move_time_axis(x, params.time_axis)  # now [..., T, H, W]
    nd = x.ndim
    assert nd >= 3, "Expected at least [T, H, W]"

    # Apply masks (set to NaN)
    x = _apply_masks(x, params.mask_saturated, params.mask_hot)

    # Time dimension size
    T = x.shape[-3]
    H, W = x.shape[-2], x.shape[-1]
    batch_shape = x.shape[:-3]

    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta["input_shape"] = frames.shape
        meta["time_axis_canonical"] = nd - 3
        meta["ref_count"] = params.ref_count
        meta["sig_count"] = params.sig_count
        meta["stride"] = params.stride
        meta["agg"] = params.agg
        meta["cosmic_reject"] = params.cosmic_reject
        meta["cr_zmax"] = params.cr_zmax
        meta["cr_iter"] = params.cr_iter

    # Implement modes
    if params.mode == "two_sample":
        # Use first frame as reference and last as signal (typical CDS)
        if T < 2:
            raise ValueError(f"two_sample CDS requires T>=2, got T={T}")
        ref = x[..., 0:1, :, :]
        sig = x[..., -1:, :, :]
        # Aggregate (trivial here, but keep robust path for masks)
        ref_val, ref_w = _robust_aggregate(ref, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
                                           zmax=params.cr_zmax, iters=params.cr_iter)
        sig_val, sig_w = _robust_aggregate(sig, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
                                           zmax=params.cr_zmax, iters=params.cr_iter)
        out = sig_val - ref_val
        weights = ref_w + sig_w

        out = _clip(out, *(params.clip_out or (None, None)))
        if params.return_intermediate:
            meta.update({"ref_w": ref_w, "sig_w": sig_w})
        return CDSResult(cds=out, weights=weights, meta=meta)

    elif params.mode == "fowler":
        # Fowler-N: average N ref frames then N sig frames, take difference
        N = int(params.ref_count)
        M = int(params.sig_count)
        if N <= 0 or M <= 0:
            raise ValueError("Fowler requires ref_count>0 and sig_count>0")
        if T < (N + M):
            raise ValueError(f"Fowler requires T >= N+M, got T={T}, N={N}, M={M}")

        ref = x[..., :N, :, :]
        sig = x[..., N:N+M, :, :]

        ref_val, ref_w = _robust_aggregate(ref, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
                                           zmax=params.cr_zmax, iters=params.cr_iter)
        sig_val, sig_w = _robust_aggregate(sig, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
                                           zmax=params.cr_zmax, iters=params.cr_iter)
        out = sig_val - ref_val
        weights = ref_w + sig_w

        out = _clip(out, *(params.clip_out or (None, None)))
        if params.return_intermediate:
            meta.update({"ref_w": ref_w, "sig_w": sig_w})
        return CDSResult(cds=out, weights=weights, meta=meta)

    elif params.mode == "rolling":
        # Sliding windows producing multiple CDS outputs along time
        N = int(params.ref_count)
        M = int(params.sig_count)
        stride = max(1, int(params.stride))
        if N <= 0 or M <= 0:
            raise ValueError("rolling CDS requires ref_count>0 and sig_count>0")
        pairs, n_out = _window_slices(T, N, M, stride)
        if n_out <= 0:
            raise ValueError(f"rolling CDS: no valid windows for T={T} with N={N}, M={M}, stride={stride}")

        # Prepare outputs
        if _is_torch(x):
            torch = _torch()
            cds_stack = torch.empty((*batch_shape, n_out, H, W), dtype=x.dtype, device=x.device)
            w_stack = torch.empty((*batch_shape, n_out, H, W), dtype=x.dtype, device=x.device)
        else:
            np = _np()
            cds_stack = np.empty((*batch_shape, n_out, H, W), dtype=x.dtype)
            w_stack = np.empty((*batch_shape, n_out, H, W), dtype=x.dtype)

        for i, (s_ref, s_sig) in enumerate(pairs):
            r = x[..., s_ref, :, :]
            s = x[..., s_sig, :, :]
            r_val, r_w = _robust_aggregate(r, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
                                           zmax=params.cr_zmax, iters=params.cr_iter)
            s_val, s_w = _robust_aggregate(s, axis=-3, agg=params.agg, cosmic_reject=params.cosmic_reject,
                                           zmax=params.cr_zmax, iters=params.cr_iter)
            cds_stack[..., i, :, :] = _clip(s_val - r_val, *(params.clip_out or (None, None)))
            w_stack[..., i, :, :] = r_w + s_w

        return CDSResult(cds=cds_stack, weights=w_stack, meta=meta)

    else:
        raise ValueError(f"Unknown CDS mode: {params.mode}")


# -----------------------------------------------------------------------------
# Minimal self-checks
# -----------------------------------------------------------------------------

def _test_two_sample_mean():
    import numpy as np
    # frames [..., T, H, W] = [T, 2, 2]
    f0 = np.zeros((2,2), dtype=np.float32) + 10
    f1 = np.zeros((2,2), dtype=np.float32) + 25
    stack = np.stack([f0, f1], axis=0)  # [T,H,W]=[2,2,2]
    res = cds(stack, CDSParams(mode="two_sample", agg="mean", return_intermediate=True))
    expected = (f1 - f0)
    assert np.allclose(res.cds, expected)
    return True


def _test_fowler_median_cr():
    import numpy as np
    rng = np.random.default_rng(0)
    T, H, W = 8, 4, 4
    # baseline ramp: ref ~ 10, sig ~ 20
    stack = np.zeros((T, H, W), dtype=np.float32)
    stack[0:3] = 10.0 + rng.normal(0, 0.3, size=(3, H, W))
    stack[3:6] = 20.0 + rng.normal(0, 0.3, size=(3, H, W))
    # cosmic ray spike in one ref frame
    stack[1, 1, 1] += 100.0
    params = CDSParams(mode="fowler", ref_count=3, sig_count=3, agg="median", cosmic_reject=True, cr_zmax=5.0, cr_iter=1)
    res = cds(stack, params)
    # Expected near 10 difference (20-10)
    assert np.allclose(np.median(res.cds), 10.0, atol=0.7)
    return True


if __name__ == "__main__":
    ok1 = _test_two_sample_mean()
    ok2 = _test_fowler_median_cr()
    print("cds.py self-tests:", ok1 and ok2)
