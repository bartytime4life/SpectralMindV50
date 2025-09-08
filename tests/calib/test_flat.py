from __future__ import annotations
import importlib

mod = importlib.import_module("spectramind.calib.flat")

def _apply(img, flat, eps: float = 1e-6):
    fn = getattr(mod, "apply", None) or getattr(mod, "correct", None)
    assert fn is not None, "spectramind.calib.flat must expose `apply` or `correct`"
    return fn(img, flat, eps=eps) if "eps" in fn.__code__.co_varnames else fn(img, flat)

def test_flat_field_normalization(synthetic_image, master_flat):
    img = synthetic_image * master_flat
    out = _apply(img, master_flat)
    rel_err = abs(out.mean() - synthetic_image.mean()) / (synthetic_image.mean() + 1e-6)
    assert out.shape == img.shape
    assert rel_err < 0.02

