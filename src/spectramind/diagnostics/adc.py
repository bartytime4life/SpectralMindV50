from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np

try:
    import torch  # optional; only used if caller passes tensors
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

from ..calib.adc import ADCParams, NonLinearity, calibrate_adc

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821

@dataclass
class ADCDiagResult:
    out_dir: Path
    figs: Dict[str, Path]
    report_html: Path
    stats: Dict[str, Any]

def _to_numpy(x: ArrayLike) -> np.ndarray:
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)

def _save_hist_png(arr: ArrayLike, title: str, out_path: Path, bins: int = 128) -> Path:
    import matplotlib.pyplot as plt
    fig = plt.figure()
    plt.hist(_to_numpy(arr).ravel(), bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    fig.savefig(out_path, bbox_inches="tight", dpi=120)
    plt.close(fig)
    return out_path

def _png_to_base64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")

def _render_html(before_png: Path, after_png: Path, stats: Dict[str, Any]) -> str:
    b64_before = _png_to_base64(before_png)
    b64_after = _png_to_base64(after_png)
    s_items = "".join(
        f"<li><b>{k}</b>: {v}</li>" for k, v in stats.items()
    )
    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ADC Diagnostics</title></head>
<body>
<h1>ADC Diagnostics</h1>
<ul>{s_items}</ul>
<h2>Histograms</h2>
<div style="display:flex;gap:16px;flex-wrap:wrap;">
  <figure><img src="data:image/png;base64,{b64_before}"/><figcaption>Raw DN</figcaption></figure>
  <figure><img src="data:image/png;base64,{b64_after}"/><figcaption>Calibrated Signal</figcaption></figure>
</div>
</body></html>"""

def run_adc_diagnostics(
    raw_dn: ArrayLike,
    *,
    gain: ArrayLike,
    offset: ArrayLike,
    bit_depth: int,
    coeffs: Optional[ArrayLike] = None,
    coeffs_tol: float = 1e-3,
    coeffs_max_iter: int = 12,
    coeffs_damping: float = 1.0,
    nlin_clamp: Optional[Tuple[Optional[float], Optional[float]]] = None,
    quant_debias: bool = True,
    clip_out: Optional[Tuple[Optional[float], Optional[float]]] = None,
    out_dir: Union[str, Path] = "artifacts/diagnostics/adc",
    bins: int = 128,
) -> ADCDiagResult:
    """
    Generate quick-look diagnostics for ADC calibration.

    Saves:
      - histogram_raw.png
      - histogram_signal.png
      - report.html
    Returns ADCDiagResult with basic stats.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    nlin = NonLinearity(
        coeffs=None if coeffs is None else coeffs,
        max_iter=coeffs_max_iter,
        tol=coeffs_tol,
        damping=coeffs_damping,
        clamp=nlin_clamp,
    )
    params = ADCParams(
        gain=gain,
        offset=offset,
        bit_depth=bit_depth,
        nonlinearity=nlin,
        quant_debias=quant_debias,
        clip_out=clip_out,
    )

    res = calibrate_adc(raw_dn, params, return_intermediate=True)

    raw_np = _to_numpy(res.meta.get("raw_dn"))
    sig_np = _to_numpy(res.signal)
    sat_np = _to_numpy(res.saturated)

    # Stats
    stats: Dict[str, Any] = {
        "bit_depth": bit_depth,
        "sat_threshold": res.meta.get("sat_threshold"),
        "saturation_frac": float(sat_np.mean()),
        "raw_dn_min": float(np.nanmin(raw_np)),
        "raw_dn_max": float(np.nanmax(raw_np)),
        "signal_min": float(np.nanmin(sig_np)),
        "signal_max": float(np.nanmax(sig_np)),
        "signal_mean": float(np.nanmean(sig_np)),
        "signal_std": float(np.nanstd(sig_np)),
        "quant_debias": bool(quant_debias),
        "nlin_coeffs_len": int(0 if coeffs is None else (coeffs.shape[-1] if hasattr(coeffs, "shape") else 0)),
    }

    figs: Dict[str, Path] = {}
    figs["histogram_raw"] = _save_hist_png(raw_np, "Raw DN", out_dir / "histogram_raw.png", bins=bins)
    figs["histogram_signal"] = _save_hist_png(sig_np, "Calibrated Signal", out_dir / "histogram_signal.png", bins=bins)

    html = _render_html(figs["histogram_raw"], figs["histogram_signal"], stats)
    report_html = out_dir / "report.html"
    report_html.write_text(html, encoding="utf-8")

    return ADCDiagResult(out_dir=out_dir, figs=figs, report_html=report_html, stats=stats)
