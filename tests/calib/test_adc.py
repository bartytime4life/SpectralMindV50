# src/spectramind/calib/adc.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


__all__ = [
    "NonLinearity",
    "ADCParams",
    "ADCResult",
    "calibrate_adc",
]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class NonLinearity:
    """
    Polynomial non-linearity model P(z) = sum_{k=0}^n c_k z^k.
    If coeffs is None, treat as identity: P(z) = z.

    Newton solve parameters (when degree >= 2):
      - max_iter: maximum Newton iterations
      - tol: absolute convergence tolerance on update
      - damping: step scaling (0 < damping <= 1), helps stability
    """
    coeffs: Optional[np.ndarray] = None
    max_iter: int = 12
    tol: float = 1e-8
    damping: float = 1.0


@dataclass
class ADCParams:
    """
    ADC calibration parameters.

    gain, offset may be scalars or array-like (broadcastable to the data).
    bit_depth determines the full-scale range [0, 2^bit_depth - 1].
    sat_margin = number of DN below full scale still considered 'safe'.
      e.g., sat_margin=1 -> only the max code is flagged saturated.
    quant_debias: if True, subtract 0.5 DN to approximately unbias quantization.
    """
    gain: np.ndarray
    offset: np.ndarray
    bit_depth: int
    nonlinearity: NonLinearity
    sat_margin: float = 0.0
    quant_debias: bool = False


@dataclass
class ADCResult:
    """Result of ADC calibration/inversion."""
    signal: np.ndarray      # recovered analog quantity x (float)
    saturated: np.ndarray   # boolean mask of saturated input codes


# ---------------------------------------------------------------------------
# Utilities: polynomial eval and derivative
# ---------------------------------------------------------------------------

def _poly_and_deriv(z: np.ndarray, coeffs: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Evaluate polynomial P(z) and its derivative P'(z) for vector z.
    Horner scheme for stability.
    """
    # Handle deg 0 and deg 1 quickly
    if coeffs.size == 0:
        P = np.zeros_like(z, dtype=z.dtype)
        dP = np.zeros_like(z, dtype=z.dtype)
        return P, dP
    if coeffs.size == 1:
        P = np.full_like(z, coeffs[0], dtype=z.dtype)
        dP = np.zeros_like(z, dtype=z.dtype)
        return P, dP
    # Horner for P; also compute derivative via Horner on derived coeffs
    P = np.zeros_like(z, dtype=z.dtype)
    dP = np.zeros_like(z, dtype=z.dtype)
    # P(z) = c0 + z*(c1 + z*(c2 + ...))
    # dP(z) = c1 + 2*c2*z + 3*c3*z^2 + ...
    # We do both in one pass from highest degree down
    # Build arrays so indices are simple
    c = coeffs
    # Start from highest degree term
    P = np.zeros_like(z, dtype=z.dtype) + c[-1]
    dP = np.zeros_like(z, dtype=z.dtype) + (len(c) - 1) * c[-1]
    for k in range(len(c) - 2, -1, -1):
        dP = dP * z + (k) * c[k]  # derivative Horner
        P = P * z + c[k]
    return P, dP


def _invert_poly(y: np.ndarray, coeffs: np.ndarray, init: Optional[np.ndarray],
                 max_iter: int, tol: float, damping: float) -> np.ndarray:
    """
    Solve P(z) = y for z, vectorized, using damped Newton iterations.
    - coeffs: polynomial coefficients (c0, c1, c2, ...)
    - init: initial guess (if None, use (y - c0)/c1 when c1 != 0 else y)
    """
    # Promote to float64 for robust inversion, then cast back later
    y64 = y.astype(np.float64, copy=False)
    c = coeffs.astype(np.float64, copy=False)

    # Initial guess: linearized inversion if possible
    if init is None:
        if c.size >= 2 and c[1] != 0.0:
            z = (y64 - c[0]) / c[1]
        else:
            z = y64.copy()
    else:
        z = init.astype(np.float64, copy=False)

    # Newton iterations
    for _ in range(max_iter):
        Pz, dPz = _poly_and_deriv(z, c)
        f = Pz - y64
        # Avoid zero derivative: where |dPz| tiny, take a safe step (skip update)
        safe = np.abs(dPz) > 1e-16
        step = np.zeros_like(z)
        step[safe] = f[safe] / dPz[safe]
        # Damped update
        z_new = z - damping * step
        # Convergence check (absolute)
        if np.max(np.abs(z_new - z)) <= tol:
            z = z_new
            break
        z = z_new

    return z.astype(y.dtype, copy=False)


# ---------------------------------------------------------------------------
# Main calibration entrypoint
# ---------------------------------------------------------------------------

def calibrate_adc(raw_dn: np.ndarray, params: ADCParams) -> ADCResult:
    """
    Invert the ADC forward model:
        raw_dn ≈ P(gain * x + offset)
    returning x and a saturation mask.

    Args
    ----
    raw_dn : np.ndarray
        Raw digital numbers (integers or floats).
    params : ADCParams
        Calibration parameters (gain, offset, bit depth, non-linearity, etc.).

    Returns
    -------
    ADCResult
        signal: float array of same shape as raw_dn
        saturated: boolean mask where raw_dn hits/approaches full scale
    """
    # Prepare arrays / dtypes
    y = np.asarray(raw_dn)
    # Work in float for math; keep a copy for saturation test on original codes
    y_float = y.astype(np.float64, copy=False)

    # Optional quantization de-biasing: mid-tread estimator ~ y - 0.5 DN
    if params.quant_debias:
        y_float = y_float - 0.5

    # Saturation mask: values at (or within margin of) the top code
    full_scale = float((1 << int(params.bit_depth)) - 1)
    margin = float(params.sat_margin or 0.0)
    saturated = (y.astype(np.float64, copy=False) >= (full_scale - margin))

    # Invert polynomial non-linearity to recover z = gain*x + offset
    nl = params.nonlinearity
    if nl.coeffs is None:
        # Identity
        z = y_float
    else:
        c = np.asarray(nl.coeffs)
        if c.ndim != 1:
            raise ValueError("NonLinearity.coeffs must be a 1-D array of polynomial coefficients.")
        deg = c.size - 1
        if deg <= 0:
            # Constant or deg-0 is nonsensical as a forward model; treat as identity offset
            z = y_float - float(c[0])
        elif deg == 1:
            # Linear: y = c0 + c1*z  => z = (y - c0)/c1
            if c[1] == 0:
                raise ValueError("NonLinearity linear term (c1) must be non-zero.")
            z = (y_float - float(c[0])) / float(c[1])
        else:
            # Degree >= 2: Newton solve P(z)=y
            init = (y_float - float(c[0])) / float(c[1]) if c[1] != 0 else None
            z = _invert_poly(y_float, c, init, nl.max_iter, nl.tol, nl.damping)

    # Linear calibration to signal domain: z = gain*x + offset  ->  x = (z - offset)/gain
    gain = np.asarray(params.gain).astype(np.float64, copy=False)
    offset = np.asarray(params.offset).astype(np.float64, copy=False)
    # Broadcast gain/offset to data shape if needed
    x = (z - offset) / gain

    # Return float array (float32 is fine for most pipelines; tests use float checks)
    signal = x.astype(np.float64, copy=False)
    return ADCResult(signal=signal, saturated=saturated.astype(np.bool_))