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


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _to_numpy(x: ArrayLike) -> np.ndarray:
    """Convert ArrayLike → np.ndarray (float64 where sensible), preserving NaNs."""
    if torch is not None and isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    # keep dtype if numeric; convert to float64 for safety on basic ops
    arr = np.asarray(x)
    if np.issubdtype(arr.dtype, np.number):
        return arr.astype(np.float64, copy=False)
    return arr


def _finite_mask(arr: np.ndarray) -> np.ndarray:
    return np.isfinite(arr)


def _basic_stats(arr: np.ndarray, finite_only: bool = True) -> Dict[str, float]:
    """Compute robust stats with finite-only mask if requested."""
    a = arr
    if finite_only:
        mask = _finite_mask(a)
        if not mask.any():
            return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
        a = a[mask]
    if a.size == 0:
        return {"min": np.nan, "max": np.nan, "mean": np.nan, "std": np.nan}
    return {
        "min": float(np.nanmin(a)),
        "max": float(np.nanmax(a)),
        "mean": float(np.nanmean(a)),
        "std": float(np.nanstd(a)),
    }


def _percentiles(arr: np.ndarray, qs: Tuple[float, ...] = (1, 5, 25, 50, 75, 95, 99)) -> Dict[str, float]:
    mask = _finite_mask(arr)
    if not mask.any():
        return {f"p{int(q)}": np.nan for q in qs}
    vals = np.percentile(arr[mask], qs)
    return {f"p{int(q)}": float(v) for q, v in zip(qs, vals)}


def _save_hist_png(arr: ArrayLike, title: str, out_path: Path, bins: int = 128) -> Optional[Path]:
    """
    Save histogram PNG if matplotlib is available; return path or None if skipped.
    Uses finite values only to avoid skew from NaNs/Infs.
    """
    try:
        import matplotlib.pyplot as plt  # lazy import
    except Exception:
        return None

    a = _to_numpy(arr).ravel()
    a = a[np.isfinite(a)]
    fig = plt.figure(figsize=(5, 3.2), dpi=120)
    plt.hist(a, bins=bins)
    plt.title(title)
    plt.xlabel("value")
    plt.ylabel("count")
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _png_to_base64(p: Path) -> str:
    return base64.b64encode(p.read_bytes()).decode("ascii")


def _render_html(before_png: Optional[Path], after_png: Optional[Path], stats: Dict[str, Any]) -> str:
    s_items = "".join(f"<li><b>{k}</b>: {stats[k]}</li>" for k in sorted(stats.keys()))
    def _img_fig(p: Optional[Path], caption: str) -> str:
        if p and p.exists():
            return f'<figure><img src="data:image/png;base64,{_png_to_base64(p)}"/><figcaption>{caption}</figcaption></figure>'
        return f'<figure><div style="padding:12px;background:#f6f8fa;border:1px solid #e2e2e2;">(Plot unavailable — matplotlib not installed)</div><figcaption>{caption}</figcaption></figure>'

    return f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>ADC Diagnostics</title>
<style>
body{{font-family:system-ui,-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:1000px;margin:40px auto;padding:0 16px}}
code,pre{{background:#f6f8fa;padding:4px 6px;border-radius:4px}}
figure{{margin:0}}
</style>
</head>
<body>
<h1>ADC Diagnostics</h1>
<h2>Stats</h2>
<ul>{s_items}</ul>
<h2>Histograms</h2>
<div style="display:flex;gap:16px;flex-wrap:wrap;">
  {_img_fig(before_png, "Raw DN")}
  {_img_fig(after_png, "Calibrated Signal")}
</div>
</body></html>"""


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
) -> ADCDiagResult:
    """
    Generate quick-look diagnostics for ADC calibration.

    Saves (when matplotlib is available):
      - histogram_raw.png
      - histogram_signal.png
      - report.html  (always produced; embeds plots if available)

    Returns:
      ADCDiagResult with paths and a comprehensive stats dictionary.
    """
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
    sat_np = _to_numpy(res.saturated) if "saturated" in res.meta or hasattr(res, "saturated") else np.zeros_like(sig_np, dtype=bool)

    # Basic sanity masks
    finite_raw = _finite_mask(raw_np)
    finite_sig = _finite_mask(sig_np)

    # Fractions and thresholds
    sat_threshold = res.meta.get("sat_threshold", None)
    saturation_frac = float(np.mean(sat_np)) if sat_np.size else 0.0

    # If clipping was applied, estimate fraction clipped on output bounds
    clip_low, clip_high = (None, None)
    clip_frac_low, clip_frac_high = (0.0, 0.0)
    if clip_out is not None:
        clip_low, clip_high = clip_out
        if clip_low is not None:
            clip_frac_low = float(np.mean(sig_np <= clip_low))
        if clip_high is not None:
            clip_frac_high = float(np.mean(sig_np >= clip_high))

    # Stats blocks
    stats_raw = _basic_stats(raw_np)
    stats_sig = _basic_stats(sig_np)
    pct_raw = _percentiles(raw_np)
    pct_sig = _percentiles(sig_np)

    # Compose stats summary (provenance-friendly)
    stats: Dict[str, Any] = {
        "bit_depth": int(bit_depth),
        "quant_debias": bool(quant_debias),
        "nlin": {
            "coeffs_len": int(0 if coeffs is None else (coeffs.shape[-1] if hasattr(coeffs, "shape") else 0)),
            "tol": float(coeffs_tol),
            "max_iter": int(coeffs_max_iter),
            "damping": float(coeffs_damping),
            "clamp": nlin_clamp,
        },
        "sat_threshold": float(sat_threshold) if sat_threshold is not None else None,
        "saturation_frac": float(saturation_frac),
        "clip_out": clip_out,
        "clip_frac_low": float(clip_frac_low),
        "clip_frac_high": float(clip_frac_high),
        "raw_dn": {
            "finite_frac": float(np.mean(finite_raw)),
            **stats_raw,
            **pct_raw,
        },
        "signal": {
            "finite_frac": float(np.mean(finite_sig)),
            **stats_sig,
            **pct_sig,
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

    # HTML report (always produced; embeds plots if available)
    html = _render_html(raw_png, sig_png, stats)
    report_html = out_dir / "report.html"
    report_html.write_text(html, encoding="utf-8")

    # Also emit machine-readable JSON summary
    (out_dir / "summary.json").write_text(
        json.dumps(stats, indent=2),
        encoding="utf-8",
    )

    return ADCDiagResult(out_dir=out_dir, figs=figs, report_html=report_html, stats=stats)
