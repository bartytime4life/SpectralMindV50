# src/spectramind/calib/phase.py
# =============================================================================
# SpectraMind V50 — Phase-locked systematics modeling & correction
# -----------------------------------------------------------------------------
# Goal:
#   - For time-series image stacks frames[... , T, H, W] with timestamps,
#     remove phase-locked patterns (e.g., pointing/orbit/scan-phase) that repeat
#     with a known (or scanned) period.
#
# Methods:
#   - Harmonic regression in phase: fit sum_{k=1..K} [a_k cos(k*phi) + b_k sin(k*phi)]
#     per-pixel (vectorized), with optional DC term and optional phase bins template.
#   - Optional smoothing (Savitzky–Golay if SciPy present; else moving average).
#   - Robust masking: saturated/hot/bad pixels -> NaN-safe operations.
#
# Backend-agnostic: NumPy or PyTorch
# Canonical time-axis is -3 (frames[..., T, H, W]); configurable via params.time_axis
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Literal

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

# -----------------------------------------------------------------------------
# Backend shims (aligned with adc/cds/dark/flat)
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
# Phase computation & smoothing
# -----------------------------------------------------------------------------

def _compute_phase(times: BackendArray, period: float, t0: float, *, wrap: bool = True) -> BackendArray:
    """
    Compute phase phi = 2*pi * ((t - t0)/period) mod 2pi (if wrap), else continuous.
    times may be NumPy or Torch 1D array of length T.
    """
    if _is_torch(times):
        torch = _torch()
        phi = 2.0 * torch.pi * ((times - t0) / period)
        if wrap:
            # wrap to [0, 2pi)
            two_pi = 2.0 * torch.pi
            phi = phi.remainder(two_pi)
        return phi
    else:
        np = _np()
        phi = 2.0 * np.pi * ((times - t0) / period)
        if wrap:
            phi = np.mod(phi, 2.0 * np.pi)
        return phi

def _savitzky_golay_1d(y: "np.ndarray", window_length: int, polyorder: int) -> "np.ndarray":
    """NumPy/Scipy: prefer scipy.signal.savgol_filter; fallback to simple moving average."""
    # This function will be used only if NumPy path; for torch we fallback to numpy path via CPU.
    try:
        from scipy.signal import savgol_filter
        return savgol_filter(y, window_length=window_length, polyorder=polyorder, mode="nearest")
    except Exception:
        # fallback: moving average
        import numpy as np
        if window_length <= 1:
            return y
        k = window_length
        pad = k // 2
        yp = np.pad(y, (pad, pad), mode="edge")
        c = np.ones(k, dtype=yp.dtype) / k
        return np.convolve(yp, c, mode="valid")

def _smooth_template(template: BackendArray, window: int, poly: int) -> BackendArray:
    """Apply Savitzky–Golay / moving average to 1D template along time."""
    if window <= 1:
        return template
    if _is_torch(template):
        # move to cpu numpy, filter, move back
        torch = _torch(); np = _np()
        t_np = template.detach().cpu().numpy()
        t_sm = _savitzky_golay_1d(t_np, window, poly)
        return torch.from_numpy(t_sm).to(device=template.device, dtype=template.dtype)
    else:
        np = _np()
        return _savitzky_golay_1d(template, window, poly)

# -----------------------------------------------------------------------------
# Dataclasses (params, model, results)
# -----------------------------------------------------------------------------

AggMode = Literal["mean", "median"]  # reserved for future residual pooling
PhaseTemplate = Literal["none", "bins"]

@dataclass
class PhaseParams:
    """
    Phase-locked systematics model configuration.

    period              : fundamental period (seconds)
    t0                  : phase reference time (seconds)
    time_axis           : time dimension index in frames (default -3 => [..., T, H, W])
    harmonics           : number of Fourier harmonics (K). K=0 -> no harmonic terms.
    include_dc          : include DC (constant) term in the regression
    template            : 'none' | 'bins' (bin-averaged phase template per pixel)
    n_bins              : if template='bins', number of phase bins
    smooth_window       : optional Savitzky–Golay (or MA fallback) window length (odd)
    smooth_poly         : polynomial order for Savitzky–Golay
    mask_saturated      : mask of saturated pixels (same shape as frames) -> NaN exclude
    mask_hot            : mask of hot/bad pixels -> NaN exclude
    dtype               : float dtype for internal ops
    return_intermediate : include debug meta
    """
    period: float
    t0: float
    time_axis: int = -3
    harmonics: int = 3
    include_dc: bool = True
    template: PhaseTemplate = "none"
    n_bins: int = 32
    smooth_window: int = 0
    smooth_poly: int = 2
    mask_saturated: Optional[BackendArray] = None
    mask_hot: Optional[BackendArray] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class PhaseModel:
    """
    Encapsulates per-pixel phase-locked model components:
      - coeffs: regression coefficients [C, H, W] with C = (dc?1:0) + 2*K
      - basis_info: dict holding basis description for reconstruction
      - template_bins: optional bin template [B, H, W]
      - meta: diagnostics
    """
    coeffs: Optional[BackendArray]
    basis_info: Dict[str, Any]
    template_bins: Optional[BackendArray]
    meta: Dict[str, Any] = field(default_factory=dict)

@dataclass
class PhaseBuildResult:
    model: PhaseModel

@dataclass
class PhaseApplyParams:
    """
    How to apply (subtract) the phase-locked model.

    subtract_model      : subtract the harmonic model
    subtract_template   : subtract phase-bin template (if available)
    clip_out            : optional (low, high)
    dtype               : output dtype
    return_intermediate : include debug meta
    """
    subtract_model: bool = True
    subtract_template: bool = False
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    dtype: Optional[Union[str, Any]] = None
    return_intermediate: bool = False

@dataclass
class PhaseApplyResult:
    corrected: BackendArray
    model_contrib: Optional[BackendArray]
    template_contrib: Optional[BackendArray]
    meta: Dict[str, Any] = field(default_factory=dict)

# -----------------------------------------------------------------------------
# Axis movement
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
# Harmonic regression builder
# -----------------------------------------------------------------------------

def _design_matrix(phi: BackendArray, K: int, include_dc: bool) -> BackendArray:
    """
    Build design matrix X[T, C] with columns:
      [1?] + cos(phi), sin(phi), cos(2phi), sin(2phi), ..., cos(Kphi), sin(Kphi)
    """
    if K <= 0 and not include_dc:
        # degenerate case: no features
        return None
    torch_mode = _is_torch(phi)
    T = int(phi.shape[0])
    C = (1 if include_dc else 0) + (2 * max(K, 0))
    if torch_mode:
        torch = _torch()
        X = torch.empty((T, C), dtype=phi.dtype, device=phi.device)
        idx = 0
        if include_dc:
            X[:, idx] = 1.0
            idx += 1
        for k in range(1, K + 1):
            X[:, idx] = torch.cos(k * phi)
            X[:, idx + 1] = torch.sin(k * phi)
            idx += 2
        return X
    else:
        import numpy as np
        X = np.empty((T, C), dtype=phi.dtype)
        idx = 0
        if include_dc:
            X[:, idx] = 1.0
            idx += 1
        for k in range(1, K + 1):
            X[:, idx] = np.cos(k * phi)
            X[:, idx + 1] = np.sin(k * phi)
            idx += 2
        return X

def _solve_least_squares(X: BackendArray, y: BackendArray) -> BackendArray:
    """
    Solve beta in X[T,C] @ beta[C, H, W] ≈ y[T, H, W] via normal equations / lstsq.
    Torch path falls back to CPU NumPy solve for stability/simplicity.
    """
    T, C = X.shape
    H, W = y.shape[-2], y.shape[-1]
    torch_mode = _is_torch(y) or _is_torch(X)
    if torch_mode:
        torch = _torch(); np = _np()
        X_np = X.detach().cpu().numpy() if _is_torch(X) else X
        y_np = y.detach().cpu().numpy() if _is_torch(y) else y
        # reshape y to [T, HW]
        Y2 = y_np.reshape(T, -1)
        beta, *_ = np.linalg.lstsq(X_np, Y2, rcond=None)  # [C, HW]
        beta = beta.reshape(C, H, W)
        beta_t = torch.from_numpy(beta).to(device=(y.device if _is_torch(y) else (X.device if _is_torch(X) else "cpu")),
                                          dtype=(y.dtype if _is_torch(y) else (X.dtype if _is_torch(X) else None)))
        return beta_t
    else:
        import numpy as np
        Y2 = y.reshape(T, -1)
        beta, *_ = np.linalg.lstsq(X, Y2, rcond=None)
        return beta.reshape(C, H, W)

def _predict_from_beta(X: BackendArray, beta: BackendArray) -> BackendArray:
    """
    Compute model m[T,H,W] = X[T,C] @ beta[C,H,W]
    """
    T, C = X.shape
    H, W = beta.shape[-2], beta.shape[-1]
    torch_mode = _is_torch(X) or _is_torch(beta)
    if torch_mode:
        torch = _torch()
        # reshape beta to [C, HW]; matmul -> [T, HW] -> reshape
        B2 = beta.reshape(C, -1)
        M2 = X @ B2  # [T, HW]
        return M2.reshape(T, H, W)
    else:
        import numpy as np
        B2 = beta.reshape(C, -1)
        M2 = X @ B2
        return M2.reshape(T, H, W)

def _apply_masks(frames: BackendArray,
                 mask_sat: Optional[BackendArray],
                 mask_hot: Optional[BackendArray]) -> BackendArray:
    """
    Apply saturated/hot masks by setting those samples to NaN to exclude from fits.
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

# -----------------------------------------------------------------------------
# Template building (optional bins)
# -----------------------------------------------------------------------------

def _build_phase_bins(phi: BackendArray, T: int, H: int, W: int, n_bins: int, y: BackendArray) -> BackendArray:
    """
    Compute phase-binned template: [B,H,W], average frames whose phase in bin b.
    NaN-safe averaging.
    """
    torch_mode = _is_torch(y) or _is_torch(phi)
    if torch_mode:
        torch = _torch(); np = _np()
        phi_np = phi.detach().cpu().numpy() if _is_torch(phi) else phi
        y_np = y.detach().cpu().numpy() if _is_torch(y) else y
        bins = np.linspace(0.0, 2.0*np.pi, n_bins+1)
        out = np.full((n_bins, H, W), np.nan, dtype=y_np.dtype)
        for b in range(n_bins):
            mask = (phi_np >= bins[b]) & (phi_np < bins[b+1]) if b < n_bins-1 else (phi_np >= bins[b])
            if not mask.any():
                continue
            sub = y_np[mask, ...]  # [Nb,H,W]
            out[b] = np.nanmean(sub, axis=0)
        return torch.from_numpy(out).to(device=(y.device if _is_torch(y) else "cpu"), dtype=(y.dtype if _is_torch(y) else None))
    else:
        import numpy as np
        bins = np.linspace(0.0, 2.0*np.pi, n_bins+1)
        out = np.full((n_bins, H, W), np.nan, dtype=y.dtype)
        for b in range(n_bins):
            mask = (phi >= bins[b]) & (phi < bins[b+1]) if b < n_bins-1 else (phi >= bins[b])
            if not mask.any():
                continue
            sub = y[mask, ...]
            out[b] = np.nanmean(sub, axis=0)
        return out

def _template_contrib_from_bins(phi: BackendArray, bins: BackendArray) -> BackendArray:
    """
    Interpolate nearest bin contribution for each time sample.
    Returns m[T,H,W] selecting nearest bin.
    """
    torch_mode = _is_torch(phi) or _is_torch(bins)
    B, H, W = bins.shape[-3], bins.shape[-2], bins.shape[-1]
    if torch_mode:
        torch = _torch(); np = _np()
        phi_np = phi.detach().cpu().numpy() if _is_torch(phi) else phi
        bins_np = bins.detach().cpu().numpy() if _is_torch(bins) else bins
        centers = np.linspace(0.0, 2.0*np.pi, B, endpoint=False) + (np.pi / B)
        # nearest index per sample
        idx = np.argmin(np.abs(phi_np[:, None] - centers[None, :]), axis=1)  # [T]
        out = bins_np[idx, ...]  # [T,H,W]
        return torch.from_numpy(out).to(device=(bins.device if _is_torch(bins) else "cpu"), dtype=(bins.dtype if _is_torch(bins) else None))
    else:
        import numpy as np
        centers = np.linspace(0.0, 2.0*np.pi, B, endpoint=False) + (np.pi / B)
        idx = np.argmin(np.abs(phi[:, None] - centers[None, :]), axis=1)  # [T]
        return bins[idx, ...]  # [T,H,W]

# -----------------------------------------------------------------------------
# Public API: build and apply
# -----------------------------------------------------------------------------

def build_phase_model(
    frames: BackendArray,
    times: BackendArray,
    params: PhaseParams,
) -> PhaseBuildResult:
    """
    Fit a per-pixel phase-locked harmonic model (and optional phase-binned template).

    frames : [..., T, H, W]
    times  : [T] timestamps (seconds)
    params : PhaseParams

    Returns
    -------
    PhaseBuildResult with PhaseModel
      - coeffs: [C,H,W] for harmonic model; None if K=0 and include_dc=False
      - template_bins: [B,H,W] if template='bins' else None
      - basis_info: dict describing basis (K, include_dc)
    """
    # dtype and axis handling
    F = _to_float(frames, dtype=params.dtype)
    T_raw = times
    if _is_torch(T_raw):
        T_raw = T_raw.to(getattr(_torch(), params.dtype or "float32"))
    else:
        T_raw = T_raw.astype((_np()).float32 if params.dtype is None else getattr(_np(), params.dtype) if isinstance(params.dtype, str) else params.dtype, copy=False)

    F, _ = _move_time_axis(F, params.time_axis)  # -> [..., T, H, W]
    T = F.shape[-3]; H, W = F.shape[-2], F.shape[-1]

    # Apply masks to exclude from fit
    Fm = _apply_masks(F, params.mask_saturated, params.mask_hot)

    # Compute wrapped phase in [0,2pi)
    phi = _compute_phase(T_raw, params.period, params.t0, wrap=True)  # [T]

    # Optional smoothing of a future template (only affects bins template)
    # Build design matrix for harmonics
    X = _design_matrix(phi, params.harmonics, params.include_dc)  # [T,C] or None
    coeffs = None
    if X is not None:
        # NaN-safe fit: we cannot directly handle NaNs in y[T,H,W] with single solve,
        # so we replace NaNs by column-wise means per pixel? Better: mask rows with NaNs.
        # Simple robust approach: fill NaNs with temporal nanmean (per pixel).
        if _is_torch(Fm):
            torch = _torch()
            y = Fm
            # compute per-pixel nanmean and fill
            mu = _nanmean(y, axis=-3, keepdims=True)
            y_filled = torch.where(torch.isnan(y), mu, y)
        else:
            np = _np()
            y = Fm
            mu = _nanmean(y, axis=-3, keepdims=True)
            y_filled = y.copy()
            y_filled[np.isnan(y_filled)] = mu[np.isnan(y_filled)]

        # Center per-pixel if no DC requested (optional but stabilizes)
        if not params.include_dc:
            if _is_torch(y_filled):
                y_filled = y_filled - _nanmean(y_filled, axis=-3, keepdims=True)
            else:
                y_filled = y_filled - _nanmean(y_filled, axis=-3, keepdims=True)

        coeffs = _solve_least_squares(X, y_filled)  # [C,H,W]

    # Optional phase-binned template
    bins = None
    if params.template == "bins":
        bins = _build_phase_bins(phi, T, H, W, params.n_bins, Fm)  # [B,H,W]
        # Optional smoothing of per-bin template along phase (across bins order)
        if params.smooth_window and params.smooth_window > 1:
            # Smooth by cycling through each (h,w) bin series
            if _is_torch(bins):
                torch = _torch(); np = _np()
                b_np = bins.detach().cpu().numpy()
                B = b_np.shape[0]
                for h in range(H):
                    for w in range(W):
                        b_np[:, h, w] = _smooth_template(
                            _torch().from_numpy(b_np[:, h, w]).to(dtype=torch.float32),  # temporary torch use
                            params.smooth_window, params.smooth_poly
                        ).detach().cpu().numpy()
                bins = torch.from_numpy(b_np).to(device=bins.device, dtype=bins.dtype)
            else:
                np = _np()
                B = bins.shape[0]
                for h in range(H):
                    for w in range(W):
                        bins[:, h, w] = _smooth_template(bins[:, h, w], params.smooth_window, params.smooth_poly)

    basis_info = {
        "harmonics": params.harmonics,
        "include_dc": params.include_dc,
    }
    meta: Dict[str, Any] = {}
    if params.return_intermediate:
        meta.update({
            "T": T, "H": H, "W": W,
            "period": params.period, "t0": params.t0,
            "template": params.template, "n_bins": params.n_bins,
        })

    model = PhaseModel(coeffs=coeffs, basis_info=basis_info, template_bins=bins, meta=meta)
    return PhaseBuildResult(model=model)

# -----------------------------------------------------------------------------
# Apply (subtract) phase-locked model
# -----------------------------------------------------------------------------

def apply_phase_correction(
    frames: BackendArray,
    times: BackendArray,
    model: PhaseModel,
    params_build: PhaseParams,
    params_apply: PhaseApplyParams,
) -> PhaseApplyResult:
    """
    Subtract the harmonic phase model and/or binned template from frames.

    frames      : [..., T, H, W]
    times       : [T]
    model       : PhaseModel returned by build_phase_model(...)
    params_build: PhaseParams used for build (needed for basis/phase)
    params_apply: PhaseApplyParams

    Returns
    -------
    PhaseApplyResult
      - corrected       : frames - contribs (per params_apply)
      - model_contrib   : harmonic model contribution [.., T, H, W] or None
      - template_contrib: binned template contribution [.., T, H, W] or None
    """
    F = _to_float(frames, dtype=params_apply.dtype)
    F, _ = _move_time_axis(F, params_build.time_axis)  # -> [..., T, H, W]
    T = F.shape[-3]; H, W = F.shape[-2], F.shape[-1]

    # Compute phase & basis
    phi = _compute_phase(times, params_build.period, params_build.t0, wrap=True)
    X = _design_matrix(phi, params_build.harmonics, params_build.include_dc) if (model.coeffs is not None) else None

    # Model contribution
    model_contrib = None
    if params_apply.subtract_model and (X is not None) and (model.coeffs is not None):
        m = _predict_from_beta(X, model.coeffs)  # [T,H,W]
        # broadcast to batch prefix
        # frames shape [..., T,H,W]; m is [T,H,W]; rely on broadcasting below
        model_contrib = m

    # Template contribution
    template_contrib = None
    if params_apply.subtract_template and (model.template_bins is not None):
        template_contrib = _template_contrib_from_bins(phi, model.template_bins)

    # Subtract contributions
    corrected = F
    if model_contrib is not None:
        # expand model_contrib across batch prefix if needed via broadcasting
        corrected = corrected - model_contrib
    if template_contrib is not None:
        corrected = corrected - template_contrib

    if params_apply.clip_out is not None:
        low, high = params_apply.clip_out
        corrected = _clip(corrected, low, high)

    meta: Dict[str, Any] = {}
    if params_apply.return_intermediate:
        meta.update({
            "subtract_model": params_apply.subtract_model,
            "subtract_template": params_apply.subtract_template,
            "clip_out": params_apply.clip_out
        })

    return PhaseApplyResult(
        corrected=corrected,
        model_contrib=model_contrib,
        template_contrib=template_contrib,
        meta=meta,
    )

# -----------------------------------------------------------------------------
# Self-tests (light)
# -----------------------------------------------------------------------------

def _test_harmonic_fit_and_subtract():
    import numpy as np
    rng = np.random.default_rng(0)
    T, H, W = 256, 8, 8
    t = np.linspace(0.0, 1000.0, T, dtype=np.float32)
    period = 100.0; t0 = 10.0
    phi = 2*np.pi*((t - t0)/period) % (2*np.pi)
    # True model: DC + 2 harmonics
    a0 = 5.0
    a1, b1 = 2.0, -1.0
    a2, b2 = 0.5, 0.7
    trend = a0 + a1*np.cos(phi) + b1*np.sin(phi) + a2*np.cos(2*phi) + b2*np.sin(2*phi)  # [T]
    # Construct frames: add per-pixel scale + noise
    base = rng.normal(0, 0.1, size=(H, W)).astype(np.float32)
    frames = trend[:, None, None].astype(np.float32) + base[None, ...] + rng.normal(0, 0.05, size=(T, H, W)).astype(np.float32)
    # Fit & apply
    build = PhaseParams(period=period, t0=t0, harmonics=2, include_dc=True, template="none", return_intermediate=True)
    res = build_phase_model(frames, t, build)
    apply_params = PhaseApplyParams(subtract_model=True, subtract_template=False)
    out = apply_phase_correction(frames, t, res.model, build, apply_params)
    # After subtraction, median over T should be ~ per-pixel base with small residuals
    med = np.median(out.corrected, axis=0)
    err = np.abs(med - base)
    assert np.percentile(err, 95) < 0.2, f"Phase subtraction too rough (95p={np.percentile(err,95)})"
    return True

def _test_bins_template_only():
    import numpy as np
    rng = np.random.default_rng(1)
    T, H, W = 128, 6, 6
    t = np.linspace(0, 640.0, T, dtype=np.float32)
    period = 80.0; t0 = 5.0
    phi = 2*np.pi*((t - t0)/period) % (2*np.pi)
    template_true = 1.5*np.cos(phi)[:, None, None]
    frames = template_true + rng.normal(0, 0.05, size=(T, H, W)).astype(np.float32)
    build = PhaseParams(period=period, t0=t0, harmonics=0, include_dc=False, template="bins", n_bins=16, return_intermediate=True)
    res = build_phase_model(frames, t, build)
    apply_params = PhaseApplyParams(subtract_model=False, subtract_template=True)
    out = apply_phase_correction(frames, t, res.model, build, apply_params)
    # residuals should be small
    rms = np.sqrt(np.nanmean(out.corrected**2))
    assert rms < 0.2, f"Residual too large after template subtraction: {rms}"
    return True

if __name__ == "__main__":
    ok1 = _test_harmonic_fit_and_subtract()
    ok2 = _test_bins_template_only()
    print("phase.py self-tests:", ok1 and ok2)
