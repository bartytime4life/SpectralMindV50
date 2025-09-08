from __future__ import annotations
import importlib
import numpy as np

mod = importlib.import_module("spectramind.calib.trace")

def _fit_poly(points, order=2):
    fn = getattr(mod, "fit_polynomial", None) or getattr(mod, "fit", None)
    assert fn is not None, "spectramind.calib.trace must expose `fit_polynomial` or `fit`"
    return fn(points, order=order) if "order" in fn.__code__.co_varnames else fn(points)

def _eval_poly(coeffs, x):
    fn = getattr(mod, "evaluate", None) or getattr(mod, "predict", None)
    assert fn is not None, "trace module should expose `evaluate` or `predict`"
    return fn(coeffs, x)

def test_trace_fit_and_eval():
    x = np.arange(0, 64, dtype=np.float32)
    y = 0.01 * x**2 + 0.1 * x + 10.0
    points = np.stack([x, y], axis=1)
    coeffs = _fit_poly(points, order=2)
    y_hat = _eval_poly(coeffs, x)
    rmse = float(np.sqrt(np.mean((y_hat - y) ** 2)))
    assert rmse < 0.1

