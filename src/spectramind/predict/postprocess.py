# src/spectramind/predict/postprocess.py
from __future__ import annotations

"""
SpectraMind V50 — Prediction Post-processing
--------------------------------------------

Utility functions to convert raw model outputs into a Kaggle-ready submission:

* Applies physics-informed constraints (non-negativity, optional max cap)
* Optional smoothing (moving average) to reduce high-freq noise
* Optional per-band coherence (coarse low-pass using band edges)
* Enforces σ (uncertainty) floors/ceilings and couples σ to residual magnitude (optional)
* Packs into a DataFrame with the proper column layout
* Validates against the submission schema (if available)
* Writes CSV atomically

All operations are **Kaggle/CI-safe** (no SciPy dependency, headless).

Typical usage
-------------
    cfg = {
        "bins": 283,
        "smooth": {"enabled": True, "window": 5},
        "nonneg": True,
        "max_cap": 0.05,       # optional (absolute cap on transit depth)
        "sigma_floor": 1e-6,
        "sigma_cap": 0.1,
        "band_edges": [0, 1, 30, 283],   # example coarse bands (FGS1, then AIRS blocks)
        "band_smooth_window": 7,         # optional band-level smoothing
    }

    rows = [
        {"id": "sample_0001", "mu": mu_np, "sigma": sigma_np},
        ...
    ]

    df = postprocess_and_pack(rows, config=cfg)
    write_submission(df, Path("artifacts/submission.csv"), schema_path=Path("schemas/submission.schema.json"))

Notes
-----
* This module does **not** apply any dataset-specific scaling; if your model emits
  normalized values, handle the de-normalization in your model head before calling this.
* Keep all numpy ops vectorized for performance.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Optional JSON Schema support
# -----------------------------

try:
    import jsonschema  # type: ignore

    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover
    _HAS_JSONSCHEMA = False


# -----------------------------
# Config dataclass
# -----------------------------

@dataclass(slots=True)
class PostprocessConfig:
    bins: int = 283
    nonneg: bool = True                       # clamp μ >= 0
    max_cap: Optional[float] = None           # optional clamp μ <= max_cap
    smooth_enabled: bool = True
    smooth_window: int = 5                    # odd window recommended
    sigma_floor: float = 1e-6
    sigma_cap: Optional[float] = None
    band_edges: Optional[List[int]] = None    # e.g. [0, 1, 30, 283]
    band_smooth_window: Optional[int] = None  # extra smoothing per band

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PostprocessConfig":
        # Support nested "smooth": {"enabled":..., "window":...}
        smooth_enabled = d.get("smooth_enabled", d.get("smooth", {}).get("enabled", True))
        smooth_window = d.get("smooth_window", d.get("smooth", {}).get("window", 5))
        return PostprocessConfig(
            bins=int(d.get("bins", 283)),
            nonneg=bool(d.get("nonneg", True)),
            max_cap=d.get("max_cap", None),
            smooth_enabled=bool(smooth_enabled),
            smooth_window=int(smooth_window),
            sigma_floor=float(d.get("sigma_floor", 1e-6)),
            sigma_cap=d.get("sigma_cap", None) if d.get("sigma_cap", None) is None else float(d.get("sigma_cap")),
            band_edges=list(d.get("band_edges", [])) or None,
            band_smooth_window=int(d["band_smooth_window"]) if d.get("band_smooth_window") else None,
        )


# -----------------------------
# Core ops
# -----------------------------

def _as_1d(a: Any) -> np.ndarray:
    x = np.asarray(a, dtype=float)
    if x.ndim == 0:
        x = x[None]
    return x.reshape(-1)


def _moving_average(x: np.ndarray, window: int) -> np.ndarray:
    """Simple centered moving average with reflect padding."""
    if window is None or window <= 1:
        return x
    window = int(window)
    if window % 2 == 0:
        window += 1  # prefer odd for symmetry
    pad = window // 2
    if x.size < 2:
        return x.copy()
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)
    y = np.convolve(xp, kernel, mode="valid")
    return y


def _apply_band_smoothing(mu: np.ndarray, edges: Sequence[int], window: int) -> np.ndarray:
    """Apply additional smoothing within each coarse band."""
    if window <= 1 or not edges:
        return mu
    out = mu.copy()
    for a, b in zip(edges[:-1], edges[1:]):
        a = int(a)
        b = int(b)
        out[a:b] = _moving_average(out[a:b], window)
    return out


def apply_constraints(
    mu: Sequence[float],
    sigma: Optional[Sequence[float]] = None,
    *,
    config: PostprocessConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply per-example constraints/smoothing. Returns (mu_pp, sigma_pp).
    """
    mu = _as_1d(mu)
    if mu.size != config.bins:
        raise ValueError(f"mu has length {mu.size}, expected config.bins={config.bins}")

    sig = None
    if sigma is not None:
        sig = _as_1d(sigma)
        if sig.size != config.bins:
            raise ValueError(f"sigma has length {sig.size}, expected {config.bins}")

    # Smoothing (prior to clamping to preserve shape)
    if config.smooth_enabled and config.smooth_window and config.smooth_window > 1:
        mu = _moving_average(mu, config.smooth_window)

    if config.band_edges and config.band_smooth_window and config.band_smooth_window > 1:
        mu = _apply_band_smoothing(mu, config.band_edges, config.band_smooth_window)

    # Non-negativity / cap
    if config.nonneg:
        mu = np.maximum(mu, 0.0)
    if config.max_cap is not None:
        mu = np.minimum(mu, float(config.max_cap))

    # Sigma floors/caps
    if sig is not None:
        sig = np.maximum(sig, float(config.sigma_floor))
        if config.sigma_cap is not None:
            sig = np.minimum(sig, float(config.sigma_cap))

    return mu, sig


# -----------------------------
# Packing/validation/writing
# -----------------------------

def _column_names(bins: int) -> Tuple[List[str], List[str]]:
    mu_cols = [f"mu_{i:03d}" for i in range(bins)]
    sigma_cols = [f"sigma_{i:03d}" for i in range(bins)]
    return mu_cols, sigma_cols


def postprocess_and_pack(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | PostprocessConfig | None = None,
    validate_lengths: bool = True,
) -> pd.DataFrame:
    """
    Post-process a batch of {id, mu, sigma} dicts and pack into submission shape.

    Each row MUST include:
      - "id": str
      - "mu": array-like (length = bins)
      - "sigma": array-like (length = bins)   (if absent, zeros will be used then floored)

    Returns
    -------
    pd.DataFrame with columns: ["id", *mu_000.., *sigma_000..]
    """
    cfg = config if isinstance(config, PostprocessConfig) else PostprocessConfig.from_dict(config or {})

    mu_cols, sigma_cols = _column_names(cfg.bins)
    out_rows: List[Dict[str, Any]] = []

    for r in rows:
        sid = str(r["id"])
        mu = r.get("mu", None)
        sigma = r.get("sigma", None)

        if mu is None:
            raise ValueError(f"row id={sid} missing 'mu'")

        mu_pp, sig_pp = apply_constraints(mu, sigma, config=cfg)

        # If sigma missing -> generate a near-deterministic floor (retain numeric stability)
        if sig_pp is None:
            sig_pp = np.full(cfg.bins, cfg.sigma_floor, dtype=float)

        if validate_lengths:
            if len(mu_pp) != cfg.bins or len(sig_pp) != cfg.bins:
                raise ValueError(f"id={sid}: postprocessed lengths mismatch (mu={len(mu_pp)}, sigma={len(sig_pp)}, bins={cfg.bins})")

        row_out: Dict[str, Any] = {"id": sid}
        row_out.update({c: float(v) for c, v in zip(mu_cols, mu_pp)})
        row_out.update({c: float(v) for c, v in zip(sigma_cols, sig_pp)})
        out_rows.append(row_out)

    df = pd.DataFrame(out_rows, columns=["id", *mu_cols, *sigma_cols])
    return df


def validate_submission(df: pd.DataFrame, schema_path: Optional[Path] = None) -> None:
    """
    Validate with JSON Schema (if available). Raises on error, no-op otherwise.
    """
    if not _HAS_JSONSCHEMA:
        return

    if schema_path and schema_path.exists():
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        # Convert to list-of-rows for validation
        for i, rec in enumerate(df.to_dict(orient="records")):
            try:
                jsonschema.validate(instance=rec, schema=schema)  # type: ignore
            except Exception as ex:  # pragma: no cover
                raise ValueError(f"Submission schema validation failed at row {i}: {ex}") from ex


def write_submission(
    df: pd.DataFrame,
    out_path: Path,
    *,
    schema_path: Optional[Path] = None,
    index: bool = False,
) -> None:
    """
    Write submission DataFrame to CSV atomically after validation.
    """
    if schema_path is not None:
        validate_submission(df, schema_path=schema_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, out_path)


# -----------------------------
# Convenience high-level API
# -----------------------------

def postprocess_filelike(
    predictions: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | PostprocessConfig | None = None,
    out_csv: Optional[Path] = None,
    schema_path: Optional[Path] = None,
) -> pd.DataFrame:
    """
    One-shot helper: post-process, pack, (optionally) validate + write CSV.

    Returns the resulting DataFrame (even if not written).
    """
    df = postprocess_and_pack(predictions, config=config)
    if out_csv is not None:
        write_submission(df, out_csv, schema_path=schema_path)
    return df


# -----------------------------
# Minimal CLI (optional)
# -----------------------------

def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    """
    Load a JSONL predictions file with rows: {"id":..., "mu":[...], "sigma":[...]}
    """
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _save_jsonl(rows: Iterable[Mapping[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


if __name__ == "__main__":  # pragma: no cover
    import argparse

    ap = argparse.ArgumentParser(description="SpectraMind V50 post-processing")
    ap.add_argument("--in-jsonl", type=Path, required=True, help="Input JSONL with raw predictions")
    ap.add_argument("--out-csv", type=Path, required=True, help="Output submission CSV")
    ap.add_argument("--schema", type=Path, default=None, help="Submission JSON schema path (optional)")
    ap.add_argument("--bins", type=int, default=283)
    ap.add_argument("--nonneg", action="store_true", default=True)
    ap.add_argument("--no-nonneg", dest="nonneg", action="store_false")
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--no-smooth", dest="smooth_enabled", action="store_false", default=False)
    ap.add_argument("--max-cap", type=float, default=None)
    ap.add_argument("--sigma-floor", type=float, default=1e-6)
    ap.add_argument("--sigma-cap", type=float, default=None)
    ap.add_argument("--band-edges", type=str, default="", help="Comma-separated ints, e.g. '0,1,30,283'")
    ap.add_argument("--band-smooth-window", type=int, default=None)

    args = ap.parse_args()

    band_edges = [int(x) for x in args.band_edges.split(",")] if args.band_edges else None

    cfg = PostprocessConfig(
        bins=args.bins,
        nonneg=args.nonneg,
        max_cap=args.max_cap,
        smooth_enabled=args.smooth_enabled,
        smooth_window=args.smooth_window,
        sigma_floor=args.sigma_floor,
        sigma_cap=args.sigma_cap,
        band_edges=band_edges,
        band_smooth_window=args.band_smooth_window,
    )

    raw = _load_jsonl(args.in_jsonl)
    df = postprocess_and_pack(raw, config=cfg)
    write_submission(df, args.out_csv, schema_path=args.schema)
    print(f"Wrote submission: {args.out_csv}")
