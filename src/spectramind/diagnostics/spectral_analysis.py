from __future__ import annotations
import numpy as np
from typing import Dict, Any

def run_fft_analysis(lightcurve: np.ndarray, dt: float = 1.0) -> Dict[str, Any]:
    """
    Compute FFT power spectrum of a lightcurve or spectral time series.

    Args:
        lightcurve: Array [T] or [N, T] with flux vs time.
        dt: Time step (default 1.0 unit).

    Returns:
        dict with frequency array and power spectrum.
    """
    n = lightcurve.shape[-1]
    freqs = np.fft.rfftfreq(n, d=dt)
    fft_vals = np.fft.rfft(lightcurve, axis=-1)
    power = np.abs(fft_vals) ** 2

    return {"freqs": freqs, "power": power}
