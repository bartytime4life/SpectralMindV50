from __future__ import annotations
import numpy as np
from pathlib import Path
from spectramind.diagnostics.adc import run_adc_diagnostics

def test_run_adc_diag(tmp_path: Path):
    raw = np.linspace(0, 4095, 1000, dtype=np.float32)
    out = run_adc_diagnostics(
        raw_dn=raw,
        gain=np.array(2.0, dtype=np.float32),
        offset=np.array(10.0, dtype=np.float32),
        bit_depth=12,
        out_dir=tmp_path/"adc",
        bins=32,
        quant_debias=False,
    )
    assert out.report_html.exists()
    assert (out.out_dir/"histogram_raw.png").exists()
    assert (out.out_dir/"histogram_signal.png").exists()
    assert 0.0 <= out.stats["saturation_frac"] <= 1.0
