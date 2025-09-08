from __future__ import annotations
import importlib
import numpy as np

mod = importlib.import_module("spectramind.calib.photometry")

def _aperture_sum(img, x, y, r):
    fn = getattr(mod, "aperture_sum", None) or getattr(mod, "aperture_flux", None)
    assert fn is not None, "spectramind.calib.photometry must expose `aperture_sum` or `aperture_flux`"
    return fn(img, x, y, r)

def _annulus_median(img, x, y, r_in, r_out):
    fn = getattr(mod, "annulus_median", None) or getattr(mod, "annulus_background", None)
    assert fn is not None, "needs `annulus_median` or `annulus_background`"
    return fn(img, x, y, r_in, r_out)

def test_aperture_flux_monotonic_with_radius(synthetic_image):
    x = y = 32.0
    f1 = _aperture_sum(synthetic_image, x, y, r=2.0)
    f2 = _aperture_sum(synthetic_image, x, y, r=4.0)
    assert f2 > f1

def test_background_subtraction_reasonable(synthetic_image):
    x = y = 32.0
    src = _aperture_sum(synthetic_image, x, y, r=4.0)
    bkg = _annulus_median(synthetic_image, x, y, r_in=8.0, r_out=12.0)
    area = np.pi * 4.0**2
    net = src - bkg * area
    assert net > 100.0

