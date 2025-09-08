from __future__ import annotations
import importlib
import numpy as np

mod = importlib.import_module("spectramind.calib.dark")

def _apply(img: np.ndarray, dark: np.ndarray) -> np.ndarray:
    fn = getattr(mod, "apply", None) or getattr(mod, "correct", None)
    assert fn is not None, "spectramind.calib.dark must expose `apply` or `correct`"
    return fn(img, dark)

def test_dark_subtraction_reduces_offset(synthetic_image, master_dark):
    img = synthetic_image + master_dark
    out = _apply(img, master_dark)
    assert out.shape == img.shape
    assert np.isfinite(out).all()
    delta = float(abs(out.mean() - synthetic_image.mean()))
    assert delta < 1.0
