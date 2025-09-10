# src/spectramind/diagnostics/adc.py
# =============================================================================
# SpectraMind V50 — ADC Calibration Diagnostics (Upgraded)
# -----------------------------------------------------------------------------
# Quick-look diagnostics around ADC calibration:
#   • Robust ArrayLike (NumPy / Torch) handling with float64 math
#   • Deterministic JSON (and HTML) reports; optional SHA-256 sidecar
#   • Smarter saturation metrics; explicit clip fractions
#   • Optional ROI mask; optional downsample for CI/Kaggle
#   • Matplotlib optional — plots skipped gracefully if unavailable
# =============================================================================

from __future__ import annotations

import json
import math
import base64
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union, Sequence

import numpy as np

try:  # optional torch support
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore[assignment]

# Prefer the richer report generator if present
try:
    from .report import generate_diagnostics_report, generate_json_and_html  # type: ignore
except Exception:  # pragma: no cover
    generate_diagnostics_report = None  # type: ignore
    generate_json_and_html = None  # type: ignore

from ..calib.adc import ADCParams, NonLinearity, calibrate_adc

ArrayLike = Union[np.ndarray, "torch.Tensor"]  # noqa: F821


# -----------------------------------------------------------------------------
# Result type
# -----------------------------------------------------------------------------
@dataclass
class ADCDiagResult:
    out_dir: Path
    figs: Dict[str, Path]
    report_html: Path
    stats: Dict[str, Any]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------
def _to_numpy(x: ArrayLike) -> np.ndarray:
    """ArrayLike → np.ndarray (float64 where numeric), preserving NaNs."""
    if torch is not None and isinstance(x, torch.Tensor):  # type: ignore[arg-type]
        return x.detach().cpu().numpy()
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float64, copy=False)
    return arr


def _finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def _basic_stats(arr: np.ndarray, finite_only: bool = True) -> Dict[str, float]:
    a = arr
    if finite_only:
        m = _finite_mask(a)
        if not m.any():
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        a = a[m]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }


def _percentiles(arr: np.ndarray, qs: Sequence[float] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    m = _finite_mask(arr)
    if not m.any():
        return {f"p{int(q)}": np.nan for q in qs}
    vals = np.percentile(arr[m], qs)
    return {f"p{int(q)}": float(v) for q, v in zip(qs, vals)}


def _auto_hist_range(a: np.ndarray) -> Optional[Tuple[float, float]]:
    """Return a percentile-capped (1, 99) range for cleaner plots; None if empty."""
    m = _finite_mask(a)
    if not m.any():
        return None
    lo, hi = np.percentile(a[m], (1, 99))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return None
    return float(lo), float(hi)


def _save_hist_png(arr: ArrayLike, title: str, out_path: Path, bins: Optional[int] = None) -> Optional[Path]:
    """
    Save histogram PNG if matplotlib is available; return path or None on skip.
    Uses finite values only and percentile-capped range to avoid long tails.
    """
    try:  # lazy import
        import matplotlib.pyplot as plt  # type: ignore
    except Exception:
        return None

    a = _to_numpy(arr).ravel()
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None

    # Reasonable defaults for CI
    if bins is None:
        bins = min(256, max(32, int(np.sqrt(a.size))))
    hr = _auto_hist_range(a)

    fig = plt.figure(figsize=(5.0, 3.2), dpi=120)
    plt.hist(a, bins=bins, range=hr)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _png_to_base64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _fallback_html(before_png: Optional[Path], after_png: Optional[Path], stats: Dict[str, Any]) -> str:
    """Lightweight HTML when rich reporter is unavailable."""
    s_items = "".join(f"<li><b>{k}</b>: {stats[k]}</li>" for k in sorted(stats.keys()))

    def _img(p: Optional[Path], cap: str) -> str:
        if p and p.exists():
            return f'<figure><img src="data:image/png;base64,{_png_to_base64(p)}" style="max-width:100%;height:auto;border-radius:6px"/><figcaption>{cap}</figcaption></figure>'
        return f'<figure><div style="padding:12px;background:#f6f8fa;border:1px solid #e2e2e2;">(Plot unavailable)</div><figcaption>{cap}</figcaption></figure>'

    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>ADC Diagnostics</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:980px;margin:40px auto;padding:0 16px}}
pre,code{{background:#f6f8fa;padding:4px 6px;border-radius:4px}}
figure{{margin:0}} h2{{margin-top:1rem}}
</style></head>
<body>
<h1>ADC Diagnostics</h1>
<h2>Stats</h2>
<ul>{s_items}</ul>
<h2>Histograms</h2>
<div style="display:flex;gap:16px;flex-wrap:wrap;">
  {_img(before_png, "Raw DN")}
  {_img(after_png, "Calibrated Signal")}
</div>
</body></html>"""


def _downsample_1d(x: np.ndarray, max_len: int = 250_000) -> np.ndarray:
    """Optionally downsample huge arrays to keep CI/matplotlib snappy."""
    x = np.asarray(x)
    if x.ndim != 1 or x.size <= max_len:
        return x
    step = math.ceil(x.size / max_len)
    return x[::step]


# -----------------------------------------------------------------------------
# Main entrypoint
# -----------------------------------------------------------------------------
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
    roi_mask: Optional[ArrayLike] = None,            # new: optional ROI (bool mask over samples)
    downsample_max: Optional[int] = 250_000,         # new: keep CI plotting fast
) -> ADCDiagResult:
    """
    Generate quick-look diagnostics for ADC calibration.

    Saves (when matplotlib is available):
      - histogram_raw.png
      - histogram_signal.png
      - report.html  (always produced; embeds plots if available)
      - summary.json (+ .sha256 if rich reporter is available)

    Returns
    -------
    ADCDiagResult
    """
    # --- Validate inputs
    if bit_depth <= 0:
        raise ValueError(f"bit_depth must be positive; got {bit_depth}")

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build non-linearity and ADC parameter objects (caller-provided coeffs optional)
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

    # Run calibration with intermediate outputs for diagnostics
    res = calibrate_adc(raw_dn, params, return_intermediate=True)

    # Extract numpy buffers (robustly)
    raw_np = _to_numpy(res.meta.get("raw_dn"))
    sig_np = _to_numpy(res.signal)
    sat_np = _to_numpy(res.meta.get("saturated")) if "saturated" in res.meta else np.zeros_like(sig_np, dtype=bool)

    # Optional ROI mask
    if roi_mask is not None:
        m = _to_numpy(roi_mask).astype(bool, copy=False).ravel()
        if m.size != raw_np.size:
            raise ValueError(f"roi_mask length {m.size} != raw length {raw_np.size}")
        raw_np = raw_np[m]
        sig_np = sig_np[m]
        if sat_np.size == m.size:
            sat_np = sat_np[m]

    # Optional downsample for speed (display/percentiles tolerant)
    if downsample_max is not None:
        raw_np = _downsample_1d(raw_np.ravel(), max_len=int(downsample_max))
        sig_np = _downsample_1d(sig_np.ravel(), max_len=int(downsample_max))
        if sat_np.ndim == 1:
            sat_np = _downsample_1d(sat_np.ravel(), max_len=int(downsample_max))

    # Masks
    finite_raw = _finite_mask(raw_np)
    finite_sig = _finite_mask(sig_np)

    # Derived constants
    max_code = float((1 << bit_depth) - 1)
    # Saturation by raw code proximity (== max) when available
    sat_from_raw = float(np.mean((raw_np >= max_code) & finite_raw)) if raw_np.size else 0.0
    sat_from_flag = float(np.mean(sat_np)) if sat_np.size else 0.0

    # Clip stats for calibrated signal
    clip_low, clip_high = (None, None)
    clip_frac_low, clip_frac_high = (0.0, 0.0)
    if clip_out is not None:
        clip_low, clip_high = clip_out
        if clip_low is not None:
            clip_frac_low = float(np.mean((sig_np <= clip_low) & finite_sig))
        if clip_high is not None:
            clip_frac_high = float(np.mean((sig_np >= clip_high) & finite_sig))

    # Stats blocks
    stats_raw = _basic_stats(raw_np)
    stats_sig = _basic_stats(sig_np)
    pct_raw = _percentiles(raw_np)
    pct_sig = _percentiles(sig_np)

    # Compose stats summary (provenance-friendly)
    stats: Dict[str, Any] = {
        "bit_depth": int(bit_depth),
        "max_code": max_code,
        "quant_debias": bool(quant_debias),
        "nlin": {
            "coeffs_len": int(0 if coeffs is None else (coeffs.shape[-1] if hasattr(coeffs, "shape") else 0)),
            "tol": float(coeffs_tol),
            "max_iter": int(coeffs_max_iter),
            "damping": float(coeffs_damping),
            "clamp": nlin_clamp,
        },
        "saturation": {
            "by_raw_maxcode_frac": sat_from_raw,
            "by_flag_frac": sat_from_flag,
            "threshold_meta": float(res.meta.get("sat_threshold")) if "sat_threshold" in res.meta and res.meta.get("sat_threshold") is not None else None,
        },
        "clip_out": {
            "low": clip_low,
            "high": clip_high,
            "frac_low": float(clip_frac_low),
            "frac_high": float(clip_frac_high),
        },
        "raw_dn": {
            "finite_frac": float(np.mean(finite_raw)) if raw_np.size else 0.0,
            **stats_raw,
            **pct_raw,
        },
        "signal": {
            "finite_frac": float(np.mean(finite_sig)) if sig_np.size else 0.0,
            **stats_sig,
            **pct_sig,
        },
        "_counts": {
            "n_samples_considered": int(raw_np.size),
        },
    }

    # Plots (optional)
    figs: Dict[str, Path] = {}
    raw_png = _save_hist_png(raw_np, "Raw DN", out_dir / "histogram_raw.png", bins=bins)
    sig_png = _save_hist_png(sig_np, "Calibrated Signal", out_dir / "histogram_signal.png", bins=bins)
    if raw_png:
        figs["histogram_raw"] = raw_png
    if sig_png:
        figs["histogram_signal"] = sig_png

    # Reports (prefer rich reporter if available)
    report_html = out_dir / "report.html"
    if generate_json_and_html is not None:
        # Rich: both JSON (with .sha256) and HTML with figure gallery
        generate_json_and_html(
            {"adc_stats": stats, "figures": [str(p) for p in figs.values()]},
            out_base=out_dir / "summary",
            title="SpectraMind V50 — ADC Diagnostics",
        )
        # Use the same HTML name as before to keep tests/back-compat happy
        if generate_diagnostics_report is not None:
            generate_diagnostics_report(
                {"adc_stats": stats, "figures": [str(p) for p in figs.values()]},
                out_path=report_html,
                title="SpectraMind V50 — ADC Diagnostics",
            )
        else:
            # Minimal fallback if only generate_json_and_html exists (unlikely)
            report_html.write_text(_fallback_html(raw_png, sig_png, stats), encoding="utf-8")
    else:
        # Minimal JSON + HTML without the rich generator
        (out_dir / "summary.json").write_text(json.dumps({"adc_stats": stats, "figures": [str(p) for p in figs.values()]}, indent=2), encoding="utf-8")
        report_html.write_text(_fallback_html(raw_png, sig_png, stats), encoding="utf-8")

    return ADCDiagResult(out_dir=out_dir, figs=figs, report_html=report_html, stats=stats)