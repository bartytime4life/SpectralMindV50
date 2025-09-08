from __future__ import annotations
import numpy as np
import pytest

from spectramind.calib.adc import ADCParams, NonLinearity, calibrate_adc

def test_adc_identity_no_quant_debias():
    raw = np.array([0, 100, 1024, 4095], dtype=np.uint16)
    p = ADCParams(
        gain=np.array(2.0, dtype=np.float32),
        offset=np.array(10.0, dtype=np.float32),
        bit_depth=12,
        nonlinearity=NonLinearity(coeffs=np.array([0.0, 1.0], dtype=np.float32)),
        quant_debias=False,
    )
    res = calibrate_adc(raw, p)
    expected = (raw.astype(np.float32) - 10.0) / 2.0
    assert np.allclose(res.signal, expected, atol=1e-6)
    assert res.saturated.dtype == np.bool_

def test_adc_quadratic_nlin_inversion():
    rng = np.random.default_rng(0)
    x_true = rng.uniform(0, 3000, size=(4096,)).astype(np.float64)
    c0, c1, c2 = 5.0, 1.0, 1e-6
    gain, offset = 2.0, 50.0
    y_dn = c0 + c1 * (gain * x_true + offset) + c2 * (gain * x_true + offset) ** 2
    p = ADCParams(
        gain=np.array(gain, dtype=np.float64),
        offset=np.array(offset, dtype=np.float64),
        bit_depth=16,
        nonlinearity=NonLinearity(coeffs=np.array([c0, c1, c2], dtype=np.float64), max_iter=16, tol=1e-6, damping=0.9),
        quant_debias=False,
    )
    res = calibrate_adc(y_dn, p)
    err = np.abs(res.signal - x_true)
    assert np.percentile(err, 95) < 1e-2

def test_adc_saturation_mask_trigger():
    raw = np.array([0, 4094, 4095], dtype=np.uint16)
    p = ADCParams(
        gain=np.array(1.0, dtype=np.float32),
        offset=np.array(0.0, dtype=np.float32),
        bit_depth=12,
        sat_margin=0.0,
        nonlinearity=NonLinearity(coeffs=None),
        quant_debias=False,
    )
    res = calibrate_adc(raw, p)
    assert res.saturated.tolist() == [False, False, True]
