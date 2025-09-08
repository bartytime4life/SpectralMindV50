# tests/calib/test_trace.py
from __future__ import annotations

import importlib
from typing import Callable, Optional

import numpy as np
import pytest

# ---------------------------------------------------------------------
# Module under test: support multiple public names for forward/backward compat
# ---------------------------------------------------------------------
mod = importlib.import_module("spectramind.calib.trace")


def _fit_poly(points: np.ndarray, order: int = 2):
    """
    Calls spectramind.calib.trace.{fit_polynomial|fit}(points, order=?)
    """
    fn: Optional[Callable] = getattr(mod, "fit_polynomial", None) or getattr(mod, "fit", None)
    assert fn is not None, "spectramind.calib.trace must expose `fit_polynomial` or `fit`"
    # not all impls accept 'order'; detect by signature
    return fn(points, order=order) if "order" in fn.__code__.co_varnames else fn(points)


def _eval_poly(coeffs, x: np.ndarray):
    """
    Calls spectramind.calib.trace.{evaluate|predict}(coeffs, x)
    """
    fn: Optional[Callable] = getattr(mod, "evaluate", None) or getattr(mod, "predict", None)
    assert fn is not None, "trace module should expose `evaluate` or `predict`"
    return fn(coeffs, x)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2)))


def _mk_quadratic(n: int = 64, a: float = 1e-2, b: float = 1e-1, c: float = 10.0, dtype=np.float32):
    x = np.arange(0, n, dtype=dtype)
    y = a * x**2 + b * x + c
    return x, y


# ---------------------------------------------------------------------
# Core happy-path tests
# ---------------------------------------------------------------------
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_trace_fit_and_eval_quadratic_low_error(dtype):
    x, y = _mk_quadratic(dtype=dtype)
    points = np.stack([x, y], axis=1)
    coeffs = _fit_poly(points, order=2)
    y_hat = _eval_poly(coeffs, x)
    rmse = _rmse(y_hat, y)
    assert rmse < 1e-1, f"Quadratic fit RMSE too high: {rmse:.6f}"


@pytest.mark.parametrize("sigma", [0.0, 0.05, 0.1])
def test_trace_fit_with_noise_is_reasonable(sigma):
    x, y_true = _mk_quadratic(dtype=np.float32)
    rng = np.random.default_rng(0)
    noise = rng.normal(0.0, sigma, size=x.shape).astype(np.float32)
    y_noisy = y_true + noise
    points = np.stack([x, y_noisy], axis=1)
    coeffs = _fit_poly(points, order=2)
    y_hat = _eval_poly(coeffs, x)
    # should denoise to within ~noise level
    rmse = _rmse(y_hat, y_true)
    assert rmse < max(0.15, 2.5 * sigma), f"RMSE {rmse:.4f} unexpectedly high for noise {sigma}"


def test_monotonicity_for_monotonic_mapping():
    """
    Wavelength solution must be strictly increasing in pixel index for realistic trace data.
    Simulate monotonic mapping with slight curvature.
    """
    x = np.arange(0, 256, dtype=np.float32)
    # gentle curvature + positive slope baseline
    y = 0.0005 * x**2 + 0.5 * x + 500.0
    points = np.stack([x, y], axis=1)
    coeffs = _fit_poly(points, order=2)
    x_dense = np.linspace(x.min(), x.max(), 1024, dtype=np.float32)
    y_hat = _eval_poly(coeffs, x_dense)
    diffs = np.diff(y_hat)
    assert np.all(diffs > 0), "Fitted wavelength grid must be strictly increasing"


# ---------------------------------------------------------------------
# Order sensitivity / robustness
# ---------------------------------------------------------------------
@pytest.mark.parametrize("order", [1, 2, 3, 4])
def test_higher_order_does_not_break_quadratic(order):
    x, y = _mk_quadratic(dtype=np.float32)
    points = np.stack([x, y], axis=1)
    coeffs = _fit_poly(points, order=order)
    y_hat = _eval_poly(coeffs, x)
    rmse = _rmse(y_hat, y)
    # Linear fit will be worse; quadratic or higher should be good
    if order == 1:
        assert rmse < 3.0, f"Linear fit should still be bounded; RMSE={rmse:.3f}"
    else:
        assert rmse < 0.2, f"Order {order} fit should recover quadratic well; RMSE={rmse:.3f}"


# ---------------------------------------------------------------------
# Shape / dtype / extrapolation behavior
# ---------------------------------------------------------------------
def test_supports_float32_and_float64_consistently():
    x32, y32 = _mk_quadratic(dtype=np.float32)
    x64, y64 = _mk_quadratic(dtype=np.float64)
    c32 = _fit_poly(np.stack([x32, y32], 1), order=2)
    c64 = _fit_poly(np.stack([x64, y64], 1), order=2)
    # evaluate both on float64 grid for a fair numeric compare
    y32_hat = _eval_poly(c32, x64)
    y64_hat = _eval_poly(c64, x64)
    np.testing.assert_allclose(y32_hat, y64_hat, rtol=1e-5, atol=1e-3)


def test_accepts_column_stacked_points():
    x, y = _mk_quadratic()
    pts = np.c_[x, y]  # same as stack([x,y],1)
    coeffs = _fit_poly(pts, order=2)
    y_hat = _eval_poly(coeffs, x)
    assert _rmse(y_hat, y) < 0.2


def test_extrapolation_is_finite_and_monotonic():
    x = np.arange(0, 128, dtype=np.float32)
    y = 0.002 * x**2 + 0.4 * x + 100
    coeffs = _fit_poly(np.stack([x, y], 1), order=2)
    x_ext = np.linspace(-20, 200, 1024, dtype=np.float32)
    y_ext = _eval_poly(coeffs, x_ext)
    assert np.all(np.isfinite(y_ext)), "Extrapolated values must be finite"
    diffs = np.diff(y_ext)
    # Outside the fit range some curvature is expected, but no wild oscillations
    assert np.all(diffs > -5.0), "Extrapolation produced large non-physical reversals"


# ---------------------------------------------------------------------
# Error handling: NaNs, duplicates, unsorted
# ---------------------------------------------------------------------
def test_raises_or_handles_nans_gracefully():
    x, y = _mk_quadratic()
    y[10] = np.nan
    pts = np.stack([x, y], 1)
    try:
        _ = _fit_poly(pts, order=2)
    except Exception as e:  # noqa: BLE001
        # Accept a clear failure mode
        assert isinstance(e, (ValueError, AssertionError)), f"Unexpected exception type: {type(e)}"
    else:
        # If the impl internally masks NaNs, ensure outputs are finite on valid domain
        coeffs = _fit_poly(pts[~np.isnan(pts).any(1)], order=2)
        y_hat = _eval_poly(coeffs, x)
        assert np.isfinite(y_hat).all()


def test_unsorted_and_duplicate_x_are_handled_or_error_useful():
    x, y = _mk_quadratic()
    # shuffle and add duplicates
    rng = np.random.default_rng(42)
    idx = rng.permutation(len(x))
    x_shuf = x[idx]
    y_shuf = y[idx]
    # append duplicates
    x_dups = np.concatenate([x_shuf, x_shuf[:5]])
    y_dups = np.concatenate([y_shuf, y_shuf[:5]])
    pts = np.stack([x_dups, y_dups], 1)
    try:
        coeffs = _fit_poly(pts, order=2)
        y_hat = _eval_poly(coeffs, np.sort(x))
        assert np.isfinite(y_hat).all()
    except Exception as e:  # noqa: BLE001
        assert isinstance(e, (ValueError, AssertionError)), f"Unexpected exception: {e}"


# ---------------------------------------------------------------------
# Affine scaling invariance (numerical sanity)
# ---------------------------------------------------------------------
def test_affine_scaling_of_axes_is_consistent():
    """
    If we scale pixel and wavelength by known factors then unscale predictions, we should
    recover the same numeric mapping (within tolerance). This guards basic conditioning.
    """
    x, y = _mk_quadratic(dtype=np.float64)
    pts = np.stack([x, y], 1)
    c_ref = _fit_poly(pts, order=2)
    y_ref = _eval_poly(c_ref, x)

    sx, tx = 2.5, -7.0
    sy, ty = 0.75, 123.0
    x_s = sx * x + tx
    y_s = sy * y + ty
    c_s = _fit_poly(np.stack([x_s, y_s], 1), order=2)

    # predict in scaled space, then invert scaling
    y_s_hat = _eval_poly(c_s, sx * x + tx)
    y_hat = (y_s_hat - ty) / sy

    np.testing.assert_allclose(y_hat, y_ref, rtol=5e-5, atol=5e-3)