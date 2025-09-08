from __future__ import annotations
import importlib
import numpy as np

mod = importlib.import_module("spectramind.calib.cds")

def _apply(frames: np.ndarray) -> np.ndarray:
    fn = getattr(mod, "apply", None) or getattr(mod, "cds", None)
    assert fn is not None, "spectramind.calib.cds must expose `apply` or `cds`"
    return fn(frames)

def test_cds_increases_signal_contrast(frame_stack: np.ndarray):
    out = _apply(frame_stack)
    assert out.ndim == 2
    pooled = frame_stack.std(axis=0).mean()
    assert float(out.mean()) > 5.0
    assert out.std() < 2.5 * pooled

