from __future__ import annotations
import importlib
import numpy as np

mod = importlib.import_module("spectramind.calib.detrend")

def _remove_linear_trend(y: np.ndarray, X: np.ndarray):
    fn = getattr(mod, "remove_linear_trend", None) or getattr(mod, "linear_detrend", None)
    assert fn is not None, "spectramind.calib.detrend must expose `remove_linear_trend` or `linear_detrend`"
    return fn(y, X)

def test_linear_detrend_reduces_correlation():
    n = 512
    t = np.linspace(0, 1, n)
    trend = 3.0 * t + 1.0
    noise = np.random.default_rng(0).normal(0, 0.1, size=n)
    y = trend + noise
    X = np.column_stack([np.ones_like(t), t])
    resid, beta = _remove_linear_trend(y.astype(np.float32), X.astype(np.float32))
    assert abs(resid.mean()) < 0.02
    corr = float(np.corrcoef(resid, t)[0, 1])
    assert abs(corr) < 0.1
    assert beta.shape == (2,)
    assert abs(beta[0] - 1.0) < 0.2 and abs(beta[1] - 3.0) < 0.2

