# src/spectramind/calib/adc.py
# =============================================================================
# SpectraMind V50 — ADC calibration utilities
# -----------------------------------------------------------------------------
# Calibrates raw ADC digital numbers (DN) to linearized, gain/offset-corrected
# physical units (e.g., electrons or photo-electrons), with:
#   - offset (bias/dark) removal
#   - inverse nonlinearity correction (polynomial; Newton fallback)
#   - gain normalization
#   - quantization de-bias (optional)
#   - saturation/clip mask
#
# The API is backend-agnostic (NumPy or PyTorch) and fully vectorized.
# =============================================================================

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Tuple, Union

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821


def _is_torch(x: BackendArray) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "torch"


def _np() -> Any:
    import numpy as np

    return np


def _torch() -> Any:
    import torch

    return torch


def _to_float(x: BackendArray, dtype: Optional[str] = None) -> BackendArray:
    """
    Cast to a reasonable float dtype preserving backend.
    """
    if _is_torch(x):
        torch = _torch()
        if dtype is None:
            dtype = "float32"
        return x.to(getattr(torch, dtype))
    else:
        np = _np()
        if dtype is None:
            dtype = np.float32
        return x.astype(dtype, copy=False)


def _zeros_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.zeros_like(x)
    else:
        np = _np()
        return np.zeros_like(x)


def _ones_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.ones_like(x)
    else:
        np = _np()
        return np.ones_like(x)


def _where(cond: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(cond):
        torch = _torch()
        return torch.where(cond, a, b)
    else:
        np = _np()
        return np.where(cond, a, b)


def _abs(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        return x.abs()
    else:
        return _np().abs(x)


def _maximum(a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(a) or _is_torch(b):
        torch = _torch()
        return torch.maximum(a, b)
    else:
        np = _np()
        return np.maximum(a, b)


def _minimum(a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(a) or _is_torch(b):
        torch = _torch()
        return torch.minimum(a, b)
    else:
        np = _np()
        return np.minimum(a, b)


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


def _allclose(a: BackendArray, b: BackendArray, rtol=1e-5, atol=1e-8) -> bool:
    if _is_torch(a) or _is_torch(b):
        torch = _torch()
        return bool(torch.allclose(a, b, rtol=rtol, atol=atol))
    else:
        np = _np()
        return bool(np.allclose(a, b, rtol=rtol, atol=atol))


@dataclass
class NonLinearity:
    """
    Parametric nonlinearity model in 'forward' sense:

        y_meas = poly(x_true) = c0 + c1*x + c2*x^2 + ... + cN*x^N

    During calibration we want *inverse* mapping x_true = poly^{-1}(y_meas).
    We solve with Newton iterations using poly and its derivative.

    - coeffs: polynomial coefficients [c0, c1, c2, ...] (broadcastable to data).
              If None or length==2 with c0≈0,c1≈1, correction is skipped.
    - max_iter: Newton iterations
    - tol: termination tolerance in DN units
    - damping: optional damping factor in (0,1] for stability
    """

    coeffs: Optional[BackendArray] = None
    max_iter: int = 12
    tol: float = 1e-3
    damping: float = 1.0

    def is_identity(self) -> bool:
        if self.coeffs is None:
            return True
        # If coeffs ~ [0, 1] and others ~ 0, treat as identity
        coeffs = self.coeffs
        if _is_torch(coeffs):
            torch = _torch()
            if coeffs.numel() < 2:
                return True
            c0 = coeffs[..., 0]
            c1 = coeffs[..., 1]
            others = coeffs[..., 2:] if coeffs.shape[-1] > 2 else None
            ok = torch.allclose(c0, torch.zeros_like(c0), atol=1e-9) and torch.allclose(
                c1, torch.ones_like(c1), atol=1e-9
            )
            if ok and (others is None or torch.allclose(others, torch.zeros_like(others), atol=1e-12)):
                return True
        else:
            np = _np()
            if np.size(coeffs) < 2:
                return True
            c0 = coeffs[..., 0]
            c1 = coeffs[..., 1]
            others = coeffs[..., 2:] if coeffs.shape[-1] > 2 else None
            ok = np.allclose(c0, 0.0, atol=1e-9) and np.allclose(c1, 1.0, atol=1e-9)
            if ok and (others is None or np.allclose(others, 0.0, atol=1e-12)):
                return True
        return False

    def poly(self, x: BackendArray) -> BackendArray:
        """
        Horner evaluation of polynomial at x.
        coeffs are [..., degree+1] with last dim degree order ascending [c0..cN].
        """
        if self.coeffs is None:
            return x
        coeffs = self.coeffs
        # Horner scheme (ascending coeffs)
        # y = c0 + x*(c1 + x*(c2 + ...))
        # Implement with explicit fold for backends
        y = _zeros_like(x)
        # Evaluate from highest degree for numerical stability
        # rearrange coeffs into descending:
        if _is_torch(coeffs):
            torch = _torch()
            cs = torch.flip(coeffs, dims=(-1,))
            for i in range(cs.shape[-1]):
                y = y * x + cs[..., i]
        else:
            np = _np()
            cs = np.flip(coeffs, axis=-1)
            # broadcasting friendly loop
            for i in range(cs.shape[-1]):
                y = y * x + cs[..., i]
        return y

    def dpoly(self, x: BackendArray) -> BackendArray:
        """
        Derivative of polynomial at x.
        If coeffs = [c0, c1, c2, ... cN], then dpoly = c1 + 2*c2*x + ... + N*cN*x^{N-1}
        """
        if self.coeffs is None:
            return _ones_like(x)
        coeffs = self.coeffs
        if coeffs.shape[-1] <= 1:
            return _zeros_like(x)
        # Exclude c0
        if _is_torch(coeffs):
            torch = _torch()
            p = torch.arange(1, coeffs.shape[-1], device=x.device if _is_torch(x) else None, dtype=coeffs.dtype)
            # sum_{k=1..N} k*ck * x^{k-1}
            # build powers iteratively
            y = _zeros_like(x)
            # Evaluate derivative via Horner on derivative coefficients descending
            # For derivative, derivative coefficients ascending are [c1*1, c2*2, ...]
            dcs = coeffs[..., 1:] * p  # shape [..., N]
            dcs = torch.flip(dcs, dims=(-1,))
            for i in range(dcs.shape[-1]):
                y = y * x + dcs[..., i]
            return y
        else:
            np = _np()
            p = np.arange(1, coeffs.shape[-1], dtype=coeffs.dtype if hasattr(coeffs, "dtype") else None)
            dcs = coeffs[..., 1:] * p  # ascending derivative coefficients
            dcs = np.flip(dcs, axis=-1)
            y = _zeros_like(x)
            for i in range(dcs.shape[-1]):
                y = y * x + dcs[..., i]
            return y

    def invert(self, y_meas: BackendArray, x0: Optional[BackendArray] = None) -> BackendArray:
        """
        Solve x_true from y_meas ≈ poly(x_true) using Newton's method.

        Args
        ----
        y_meas: measured DN after offset removal
        x0    : optional initial guess; by default y_meas (identity init)

        Returns
        -------
        x_true: linearized DN
        """
        if self.is_identity():
            return y_meas

        x = y_meas if x0 is None else x0
        x = _to_float(x)
        tol = float(self.tol)
        damp = float(self.damping)

        # Newton iterations
        for _ in range(int(self.max_iter)):
            f = self.poly(x) - y_meas
            df = self.dpoly(x)
            # Avoid division by zero
            if _is_torch(df):
                torch = _torch()
                df = _where(df.abs() < 1e-12, _where(df >= 0, torch.tensor(1e-12, device=df.device, dtype=df.dtype), torch.tensor(-1e-12, device=df.device, dtype=df.dtype)), df)
            else:
                np = _np()
                df = _where(_abs(df) < 1e-12, _where(df >= 0, np.array(1e-12, dtype=df.dtype), np.array(-1e-12, dtype=df.dtype)), df)
            step = f / df
            x_new = x - damp * step
            # Convergence check
            if _is_torch(x):
                done = (step.abs() <= tol).all()
            else:
                done = bool((_abs(step) <= tol).all())
            x = x_new
            if done:
                break
        return x


@dataclass
class ADCParams:
    """
    ADC calibration parameters.

    The usual measurement model:
        DN_meas = poly( gain * signal + offset ) + quant_noise

    During calibration, we aim to recover 'signal' in physical units.

    Args
    ----
    gain            : electrons/DN (or physical_unit/DN); broadcastable to raw shape
    offset          : DN (bias/dark/zero-level); broadcastable
    bit_depth       : ADC bit depth (e.g., 12, 14, 16)
    full_scale_dn   : maximum code value (default: 2^bit_depth - 1)
    sat_dn          : saturation threshold DN; if None, defaults to full_scale_dn - sat_margin
    sat_margin      : margin below full-scale to consider as saturated (default: 0)
    nonlinearity    : NonLinearity model (forward coefficients to invert)
    quant_debias    : if True, subtract 0.5 LSB prior to dequant (assumes mid-tread)
    clip_out        : (low, high) clipping for output signal (None = no clip)
    dtype           : desired float dtype for outputs (torch: "float32"/"float64"; numpy: np.float32/64)
    """

    gain: BackendArray
    offset: BackendArray
    bit_depth: int
    full_scale_dn: Optional[int] = None
    sat_dn: Optional[float] = None
    sat_margin: float = 0.0
    nonlinearity: NonLinearity = field(default_factory=NonLinearity)
    quant_debias: bool = True
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None
    dtype: Optional[Union[str, Any]] = None  # allow torch dtype string or numpy dtype

    def __post_init__(self) -> None:
        if self.full_scale_dn is None:
            self.full_scale_dn = (1 << int(self.bit_depth)) - 1
        if self.sat_dn is None:
            # e.g., treat top-N codes as saturated
            self.sat_dn = float(self.full_scale_dn) - float(self.sat_margin)


@dataclass
class ADCResult:
    """
    Output structure for ADC calibration.
    """
    signal: BackendArray
    # saturated mask in input DN space (True where saturated or near-saturated)
    saturated: BackendArray
    # useful debug intermediates
    meta: Dict[str, Any] = field(default_factory=dict)


def calibrate_adc(
    raw_dn: BackendArray,
    params: ADCParams,
    *,
    return_intermediate: bool = False,
) -> ADCResult:
    """
    Calibrate raw ADC counts to linearized, gain/offset-corrected signal.

    Pipeline:
      1) cast to float
      2) detect saturation (>= sat_dn)
      3) optional quantization de-bias (DN - 0.5)
      4) remove offset (DN -> DN')
      5) inverse nonlinearity (DN' -> DN_lin)
      6) gain normalization (DN_lin / gain -> signal)
      7) optional clipping of signal
      8) return signal, saturation mask, and metadata

    Broadcast rules:
      - 'gain', 'offset', and polynomial coeffs must be broadcastable to raw_dn.

    Args
    ----
    raw_dn : array-like (NumPy or Torch)
        Raw ADC digital numbers.
    params : ADCParams
        Calibration parameters.
    return_intermediate : bool
        If True, returns additional intermediate arrays in meta.

    Returns
    -------
    ADCResult
      - signal: calibrated signal in physical units
      - saturated: boolean mask in input space
      - meta: dict of intermediate arrays and scalars
    """
    # Local handles
    gain = params.gain
    offset = params.offset
    nlin = params.nonlinearity

    # Cast to float
    x = _to_float(raw_dn, dtype=params.dtype)
    gain = _to_float(gain, dtype=params.dtype)
    offset = _to_float(offset, dtype=params.dtype)

    # Saturation detection
    sat_thr = float(params.sat_dn if params.sat_dn is not None else params.full_scale_dn)
    if _is_torch(x):
        torch = _torch()
        saturated = x >= torch.tensor(sat_thr, device=x.device, dtype=x.dtype)
    else:
        saturated = x >= sat_thr

    meta: Dict[str, Any] = {}
    if return_intermediate:
        meta["raw_dn"] = x
        meta["sat_threshold"] = sat_thr

    # Quantization de-bias (helps reduce 0.5 LSB bias for mid-tread ADCs)
    if params.quant_debias:
        if _is_torch(x):
            torch = _torch()
            x = x - torch.tensor(0.5, device=x.device, dtype=x.dtype)
        else:
            x = x - 0.5

    # Offset removal (bias/dark) -> DN'
    x_off = x - offset

    # Inverse non-linearity: DN' -> DN_lin
    # Initial guess: DN_lin ≈ DN' (identity)
    x_lin = nlin.invert(x_off, x0=x_off)

    # Gain normalization to physical units
    # Avoid division by zero in gain (rare, but robust)
    if _is_torch(gain):
        torch = _torch()
        z = torch.where(gain.abs() < 1e-20, torch.tensor(1e-20, device=gain.device, dtype=gain.dtype), gain)
    else:
        np = _np()
        z = _where(_abs(gain) < 1e-20, np.array(1e-20, dtype=gain.dtype), gain)
    signal = x_lin / z

    # Optional clipping
    low, high = (None, None) if params.clip_out is None else params.clip_out
    signal = _clip(signal, low, high)

    if return_intermediate:
        meta.update(
            {
                "quant_debias": params.quant_debias,
                "x_after_debias": x if params.quant_debias else None,
                "offset": offset,
                "dn_minus_offset": x_off,
                "dn_linearized": x_lin,
                "gain": gain,
                "bit_depth": params.bit_depth,
                "full_scale_dn": params.full_scale_dn,
                "clip_out": params.clip_out,
                "nonlinearity_coeffs": nlin.coeffs,
                "nonlinearity_max_iter": nlin.max_iter,
                "nonlinearity_tol": nlin.tol,
                "nonlinearity_damping": nlin.damping,
            }
        )

    return ADCResult(signal=signal, saturated=saturated, meta=meta)


# -----------------------------------------------------------------------------
# Convenience builders and runtime checks
# -----------------------------------------------------------------------------

def build_nonlinearity_from_coeffs(coeffs: Optional[BackendArray], *, max_iter: int = 12, tol: float = 1e-3, damping: float = 1.0) -> NonLinearity:
    """
    Helper to construct NonLinearity from a coefficients array or None.
    coeffs shape: [..., degree+1] ascending in degree: [c0, c1, ..., cN]
    """
    if coeffs is None:
        return NonLinearity(coeffs=None)
    return NonLinearity(coeffs=coeffs, max_iter=max_iter, tol=tol, damping=damping)


def sanity_check_params(raw_shape: Tuple[int, ...], params: ADCParams) -> None:
    """
    Perform basic broadcastability checks for gain/offset/coeffs.
    Raises ValueError if broadcast would fail (best-effort static check).
    """
    # We do a soft-check with numpy's broadcasting rules by attempting to create a dummy array
    # We won't allocate the full raw array; instead try shapes alignment with 1s.
    np = _np()
    dummy = np.empty(tuple(dim if dim > 1 else 1 for dim in raw_shape), dtype=np.float32)

    def _bshape(x: BackendArray, name: str) -> None:
        if x is None:
            return
        try:
            _ = dummy + np.zeros_like(np.empty(getattr(x, "shape", (1,)), dtype=np.float32))
        except Exception as e:
            raise ValueError(f"{name} with shape {getattr(x,'shape',None)} may not broadcast to raw shape {raw_shape}: {e}")

    _bshape(params.gain, "gain")
    _bshape(params.offset, "offset")
    if params.nonlinearity and params.nonlinearity.coeffs is not None:
        _bshape(params.nonlinearity.coeffs[..., 0], "nonlinearity.coeffs")


# -----------------------------------------------------------------------------
# Minimal internal tests (can be invoked from a notebook or CI smoke test)
# -----------------------------------------------------------------------------

def _test_identity_linear():
    import numpy as np
    raw = np.array([0, 100, 1024, 4095], dtype=np.uint16)
    p = ADCParams(
        gain=np.array(2.0, dtype=np.float32),   # 2 electrons per DN
        offset=np.array(10.0, dtype=np.float32),  # bias 10 DN
        bit_depth=12,
        nonlinearity=NonLinearity(coeffs=np.array([0.0, 1.0], dtype=np.float64)),  # identity
        quant_debias=False,
    )
    res = calibrate_adc(raw, p)
    # expected: (raw - offset)/gain
    expected = (raw.astype(np.float32) - 10.0) / 2.0
    assert np.allclose(res.signal, expected, atol=1e-5), f"Expected {expected}, got {res.signal}"
    assert res.saturated[-1] == True  # 4095 should be saturated for 12-bit full scale
    return True


def _test_poly_correction():
    """
    Check that a mild quadratic nonlinearity is inverted reasonably.
    """
    import numpy as np
    rng = np.random.default_rng(0)
    x_true = rng.uniform(0, 3000, size=(2048,), dtype=np.float64)
    # Forward model: y = c0 + c1*x + c2*x^2  (slight nonlinearity)
    c0, c1, c2 = 5.0, 1.0, 1e-6
    y_meas = c0 + c1 * x_true + c2 * x_true * x_true
    # Add offset & gain: DN = poly(gain*signal + offset)
    gain = 2.0
    offset = 50.0
    y_meas_dn = c0 + c1 * (gain * x_true + offset) + c2 * (gain * x_true + offset) ** 2

    p = ADCParams(
        gain=np.array(gain, dtype=np.float64),
        offset=np.array(offset, dtype=np.float64),
        bit_depth=16,
        nonlinearity=NonLinearity(coeffs=np.array([c0, c1, c2], dtype=np.float64), max_iter=16, tol=1e-6),
        quant_debias=False,
    )

    res = calibrate_adc(y_meas_dn.astype(np.float64), p)
    err = np.abs(res.signal - x_true)
    assert np.percentile(err, 95) < 1e-2, f"Nonlinearity inversion too rough (95p err={np.percentile(err,95)})"
    return True


if __name__ == "__main__":
    # Lightweight self-checks
    ok1 = _test_identity_linear()
    ok2 = _test_poly_correction()
    print("adc.py self-tests:", ok1 and ok2)
