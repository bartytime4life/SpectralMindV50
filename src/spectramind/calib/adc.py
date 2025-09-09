# =============================================================================
# SpectraMind V50 — ADC calibration utilities (NumPy/Torch backend)
# =============================================================================
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple, Union, Iterable

BackendArray = Union["np.ndarray", "torch.Tensor"]  # noqa: F821

__all__ = [
    "NonLinearity",
    "ADCParams",
    "ADCResult",
    "calibrate_adc",
    "build_nonlinearity_from_coeffs",
    "sanity_check_params",
]


# --------------------------- Backend helpers ---------------------------------
def _is_torch(x: BackendArray) -> bool:
    return x.__class__.__module__.split(".", 1)[0] == "torch"


def _np() -> Any:
    import numpy as np
    return np


def _torch() -> Any:
    import torch
    return torch


def _resolve_dtype(backend: str, dtype: Optional[Union[str, Any]]) -> Any:
    """
    Resolve dtype consistently for both backends.

    - backend: "torch" or "numpy"
    - dtype may be: None | str (e.g., "float32", "float64") | backend dtype object
    """
    if dtype is None:
        return None

    if backend == "torch":
        torch = _torch()
        if isinstance(dtype, str):
            # Map common strings to torch dtypes; fallback to attribute lookup
            map_ = {
                "float32": torch.float32,
                "float": torch.float32,
                "float64": torch.float64,
                "double": torch.float64,
                "float16": torch.float16,
                "half": torch.float16,
                "bfloat16": torch.bfloat16,
            }
            return map_.get(dtype.lower(), getattr(torch, dtype))
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
    if _is_torch(x):
        return _torch().zeros_like(x)
    return _np().zeros_like(x)


def _ones_like(x: BackendArray) -> BackendArray:
    if _is_torch(x):
        return _torch().ones_like(x)
    return _np().ones_like(x)


def _full_like(x: BackendArray, val: float) -> BackendArray:
    if _is_torch(x):
        torch = _torch()
        return torch.full_like(x, fill_value=val)
    np = _np()
    return np.full_like(x, fill_value=val)


def _where(cond: BackendArray, a: BackendArray, b: BackendArray) -> BackendArray:
    if _is_torch(cond):
        return _torch().where(cond, a, b)
    return _np().where(cond, a, b)


def _abs(x: BackendArray) -> BackendArray:
    return x.abs() if _is_torch(x) else _np().abs(x)


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
    return _np().clip(x, low, high)


# --------------------------- Nonlinearity model -------------------------------
@dataclass
class NonLinearity:
    """
    Forward polynomial model: y = c0 + c1*x + c2*x^2 + ... + cN*x^N
    During calibration we invert it with damped Newton iterations.

    coeffs  : [..., deg+1] ascending degree (c0..cN). None ≡ identity.
    max_iter: Newton iterations (element-wise).
    tol     : absolute step tolerance for early exit.
    damping : (0, 1] step damping for stability.
    clamp   : optional (low, high) bounds on x during Newton.
    """
    coeffs: Optional[BackendArray] = None
    max_iter: int = 12
    tol: float = 1e-3
    damping: float = 1.0
    clamp: Optional[Tuple[Optional[float], Optional[float]]] = None

    # --- fast checks / evals ---
    def is_identity(self) -> bool:
        c = self.coeffs
        if c is None:
            return True
        # Quickly accept identity if ~[0, 1] and rest ~0
        if _is_torch(c):
            torch = _torch()
            if c.numel() < 2:
                return True
            ok01 = torch.allclose(c[..., 0], torch.zeros_like(c[..., 0]), atol=1e-9) and \
                   torch.allclose(c[..., 1], torch.ones_like(c[..., 1]), atol=1e-9)
            rest_ok = True
            if c.shape[-1] > 2:
                rest_ok = torch.allclose(c[..., 2:], torch.zeros_like(c[..., 2:]), atol=1e-12)
            return bool(ok01 and rest_ok)
        else:
            np = _np()
            if c.shape[-1] < 2:
                return True
            ok01 = np.allclose(c[..., 0], 0.0, atol=1e-9) and np.allclose(c[..., 1], 1.0, atol=1e-9)
            rest_ok = True
            if c.shape[-1] > 2:
                rest_ok = np.allclose(c[..., 2:], 0.0, atol=1e-12)
            return bool(ok01 and rest_ok)

    def poly(self, x: BackendArray) -> BackendArray:
        if self.coeffs is None:
            return x
        c = self.coeffs
        y = _zeros_like(x)
        # Horner on descending coefficients
        if _is_torch(c):
            cs = _torch().flip(c, dims=(-1,))
            for i in range(cs.shape[-1]):
                y = y * x + cs[..., i]
        else:
            np = _np()
            cs = np.flip(c, axis=-1)
            for i in range(cs.shape[-1]):
                y = y * x + cs[..., i]
        return y

    def dpoly(self, x: BackendArray) -> BackendArray:
        if self.coeffs is None:
            return _ones_like(x)
        c = self.coeffs
        if c.shape[-1] <= 1:
            return _zeros_like(x)
        if _is_torch(c):
            torch = _torch()
            p = torch.arange(1, c.shape[-1], device=c.device, dtype=c.dtype)  # [1..N]
            dcs = c[..., 1:] * p
            dcs = torch.flip(dcs, dims=(-1,))
            y = _zeros_like(x)
            for i in range(dcs.shape[-1]):
                y = y * x + dcs[..., i]
            return y
        else:
            np = _np()
            p = np.arange(1, c.shape[-1], dtype=c.dtype if hasattr(c, "dtype") else None)
            dcs = c[..., 1:] * p
            dcs = np.flip(dcs, axis=-1)
            y = _zeros_like(x)
            for i in range(dcs.shape[-1]):
                y = y * x + dcs[..., i]
            return y

    def invert(self, y_meas: BackendArray, x0: Optional[BackendArray] = None) -> BackendArray:
        """
        Solve x from y_meas = poly(x) with damped Newton.
        Element-wise early exit when |Δx| ≤ tol.
        """
        if self.is_identity():
            return y_meas

        x = _to_float(y_meas if x0 is None else x0)
        tol = float(self.tol)
        damp = float(self.damping)
        lo, hi = (None, None) if self.clamp is None else self.clamp

        # Vectorized element-wise convergence
        for _ in range(int(self.max_iter)):
            f = self.poly(x) - y_meas
            df = self.dpoly(x)

            # Guard df≈0 without leaving device/dtype, preserve sign
            eps = _full_like(df, 1e-12)
            df = _where(_abs(df) < 1e-12, df + _where(df >= 0, eps, -eps), df)

            step = f / df
            x_new = x - damp * step
            if lo is not None or hi is not None:
                x_new = _clip(x_new, lo, hi)

            # Per-element done
            done = _abs(step) <= tol
            # Early exit only when all are converged
            if _is_torch(done):
                if bool(done.all()):
                    x = x_new
                    break
            else:
                if bool(done.all()):
                    x = x_new
                    break
            x = x_new

        return x


# --------------------------- Parameter structs --------------------------------
@dataclass
class ADCParams:
    """
    ADC measurement model (forward):
        DN = poly(gain * signal + offset) + quant_error

    We invert to estimate `signal`.

    gain          : electrons/DN (broadcastable)
    offset        : DN (broadcastable)
    bit_depth     : e.g., 12/14/16
    full_scale_dn : if None, uses (2^bit_depth - 1)
    sat_dn        : saturation threshold; default = full_scale_dn - sat_margin
    sat_margin    : margin below full-scale to count as saturated
    nonlinearity  : NonLinearity model to invert
    quant_debias  : subtract 0.5 LSB prior to offset removal (mid-tread ADC)
    clip_out      : optional (low, high) clamp on output signal
    dtype         : dtype (torch or numpy or string) for floats in computation
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
    dtype: Optional[Union[str, Any]] = None

    def __post_init__(self) -> None:
        if self.full_scale_dn is None:
            self.full_scale_dn = (1 << int(self.bit_depth)) - 1
        if self.sat_dn is None:
            self.sat_dn = float(self.full_scale_dn) - float(self.sat_margin)


@dataclass
class ADCResult:
    """Result of ADC calibration."""
    signal: BackendArray
    saturated: BackendArray  # boolean mask (same backend/device as input)
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------- Core calibration ---------------------------------
def calibrate_adc(
    raw_dn: BackendArray,
    params: ADCParams,
    *,
    return_intermediate: bool = False,
) -> ADCResult:
    """
    Calibrate raw ADC values → linearized, gain/offset-corrected signal.

    Steps:
      1) cast to float with requested dtype
      2) saturated := raw_dn >= sat_dn
      3) (optional) quant_debias: raw_dn -= 0.5
      4) offset removal: x_off = raw_dn - offset
      5) inverse nonlinearity: x_lin = nlin^{-1}(x_off)
      6) divide by gain: signal = x_lin / gain
      7) (optional) clip_out
    """
    # Cast inputs to float consistently
    x = _to_float(raw_dn, dtype=params.dtype)
    gain = _to_float(params.gain, dtype=params.dtype)
    offset = _to_float(params.offset, dtype=params.dtype)

    # Saturation detection on the input domain
    sat_thr = float(params.sat_dn if params.sat_dn is not None else params.full_scale_dn)
    saturated = x >= _full_like(x, sat_thr)

    meta: Dict[str, Any] = {}
    if return_intermediate:
        meta["raw_dn"] = x
        meta["sat_threshold"] = sat_thr
        # Count saturation for quick diagnostics
        if _is_torch(saturated):
            meta["saturated_count"] = int(saturated.sum().item())
            meta["saturated_ratio"] = float(saturated.float().mean().item())
        else:
            np = _np()
            meta["saturated_count"] = int(saturated.sum())
            meta["saturated_ratio"] = float(np.mean(saturated))

    # Optional 0.5 LSB debias (mid-tread ADC)
    if params.quant_debias:
        x = x - _full_like(x, 0.5)

    # Offset removal → DN'
    x_off = x - offset

    # Inverse nonlinearity → DN_lin
    x_lin = params.nonlinearity.invert(x_off, x0=x_off)

    # Gain normalization (robust to tiny/zero gain)
    tiny = _full_like(gain, 1e-20)
    z = _where(_abs(gain) < tiny, tiny, gain)
    signal = x_lin / z

    # Optional clipping
    low, high = (None, None) if params.clip_out is None else params.clip_out
    signal = _clip(signal, low, high)

    if return_intermediate:
        # Enrich metadata for reproducibility/debuggability
        meta.update(
            dict(
                quant_debias=params.quant_debias,
                offset=offset,
                dn_minus_offset=x_off,
                dn_linearized=x_lin,
                gain=gain,
                bit_depth=params.bit_depth,
                full_scale_dn=params.full_scale_dn,
                clip_out=params.clip_out,
                nonlinearity_coeffs=getattr(params.nonlinearity, "coeffs", None),
                nonlinearity_max_iter=params.nonlinearity.max_iter,
                nonlinearity_tol=params.nonlinearity.tol,
                nonlinearity_damping=params.nonlinearity.damping,
                nonlinearity_clamp=params.nonlinearity.clamp,
                compute_dtype=str(signal.dtype) if hasattr(signal, "dtype") else "float",
                backend="torch" if _is_torch(signal) else "numpy",
            )
        )

    return ADCResult(signal=signal, saturated=saturated, meta=meta)


# --------------------------- Builders / checks --------------------------------
def build_nonlinearity_from_coeffs(
    coeffs: Optional[BackendArray],
    *,
    max_iter: int = 12,
    tol: float = 1e-3,
    damping: float = 1.0,
    clamp: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> NonLinearity:
    return NonLinearity(coeffs=coeffs, max_iter=max_iter, tol=tol, damping=damping, clamp=clamp)


def _broadcastable(shape_a: Tuple[int, ...], shape_b: Tuple[int, ...]) -> bool:
    """
    True if NumPy-style broadcasting from B into A is possible.
    """
    # Align right
    a = list(shape_a)[::-1]
    b = list(shape_b)[::-1]
    for i in range(max(len(a), len(b))):
        da = a[i] if i < len(a) else 1
        db = b[i] if i < len(b) else 1
        if da != 1 and db != 1 and da != db:
            return False
    return True


def sanity_check_params(raw_shape: Tuple[int, ...], params: ADCParams) -> None:
    """
    Best-effort broadcastability checks for gain/offset/coeffs against raw shape.
    """
    g_shape = getattr(params.gain, "shape", ())
    o_shape = getattr(params.offset, "shape", ())

    if not _broadcastable(raw_shape, g_shape):
        raise ValueError(f"gain shape {g_shape} does not broadcast to raw shape {raw_shape}")
    if not _broadcastable(raw_shape, o_shape):
        raise ValueError(f"offset shape {o_shape} does not broadcast to raw shape {raw_shape}")

    coeffs = getattr(params.nonlinearity, "coeffs", None)
    if coeffs is not None:
        if coeffs.shape[-1] < 1:
            raise ValueError("nonlinearity.coeffs must have at least 1 coefficient (c0).")
        # For poly evaluation, coeffs[..., deg+1] broadcasts element-wise with x
        coeffs_shape = coeffs.shape[:-1]  # drop degree axis
        if not _broadcastable(raw_shape, coeffs_shape):
            raise ValueError(
                f"nonlinearity.coeffs shape {coeffs.shape} (sans degree {coeffs_shape}) "
                f"does not broadcast to raw shape {raw_shape}"
            )


# --------------------------- Lightweight self-tests ---------------------------
def _test_identity_linear() -> bool:
    import numpy as np
    raw = np.array([0, 100, 1024, 4095], dtype=np.uint16)
    p = ADCParams(
        gain=np.array(2.0, dtype=np.float32),
        offset=np.array(10.0, dtype=np.float32),
        bit_depth=12,
        nonlinearity=NonLinearity(coeffs=np.array([0.0, 1.0], dtype=np.float64)),
        quant_debias=False,
    )
    res = calibrate_adc(raw, p)
    expected = (raw.astype(np.float32) - 10.0) / 2.0
    assert np.allclose(res.signal, expected, atol=1e-5)
    assert bool(res.saturated[-1]) is True  # 4095 at 12-bit
    return True


def _test_poly_correction() -> bool:
    import numpy as np
    rng = np.random.default_rng(0)
    x_true = rng.uniform(0, 3000, size=(2048,)).astype(np.float64)
    c0, c1, c2 = 5.0, 1.0, 1e-6
    gain, offset = 2.0, 50.0
    y_meas_dn = c0 + c1 * (gain * x_true + offset) + c2 * (gain * x_true + offset) ** 2
    p = ADCParams(
        gain=np.array(gain, dtype=np.float64),
        offset=np.array(offset, dtype=np.float64),
        bit_depth=16,
        nonlinearity=NonLinearity(
            coeffs=np.array([c0, c1, c2], dtype=np.float64),
            max_iter=16,
            tol=1e-6,
            damping=0.9,
            clamp=(0.0, None),
        ),
        quant_debias=False,
    )
    res = calibrate_adc(y_meas_dn.astype(np.float64), p)
    err = np.abs(res.signal - x_true)
    assert np.percentile(err, 95) < 1e-2
    return True


if __name__ == "__main__":
    print("adc.py self-tests:", _test_identity_linear() and _test_poly_correction())