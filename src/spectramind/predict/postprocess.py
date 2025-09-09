from __future__ import annotations

"""
SpectraMind V50 — Prediction Post-processing
--------------------------------------------

Turn raw model outputs into a Kaggle-ready submission:

• Vectorized smoothing (moving average) and optional band-wise smoothing
• Physics-informed constraints: clamp μ ≥ 0 (optional), cap μ (optional), σ floors/caps
• Non-finite guards
• Packs via submit.format.format_predictions to ensure canonical column order
• Validates via submit.validate.validate_dataframe (schema/order/physics guards)
• Atomic write

Kaggle/CI-safe: numpy + pandas + stdlib only.
"""

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from spectramind.submit.format import format_predictions, build_expected_columns
from spectramind.submit.validate import N_BINS_DEFAULT, validate_dataframe

# =============================================================================
# Config
# =============================================================================


@dataclass(slots=True)
class PostprocessConfig:
    bins: int = N_BINS_DEFAULT        # number of spectral bins
    nonneg: bool = True               # clamp μ >= 0
    max_cap: Optional[float] = None   # optional clamp μ <= max_cap
    smooth_enabled: bool = True
    smooth_window: int = 5            # center window (odd recommended)
    sigma_floor: float = 1e-6
    sigma_cap: Optional[float] = None
    # Optional coarse bands (e.g., [0, 1, 30, 283]) and smoothing within each band
    band_edges: Optional[List[int]] = None
    band_smooth_window: Optional[int] = None

    @staticmethod
    def from_dict(d: Mapping[str, Any]) -> "PostprocessConfig":
        """Build config from nested dicts (compatible with Hydra)."""
        smooth_cfg = d.get("smooth", {})
        smooth_enabled = d.get("smooth_enabled", smooth_cfg.get("enabled", True))
        smooth_window = d.get("smooth_window", smooth_cfg.get("window", 5))
        return PostprocessConfig(
            bins=int(d.get("bins", N_BINS_DEFAULT)),
            nonneg=bool(d.get("nonneg", True)),
            max_cap=(None if d.get("max_cap") is None else float(d["max_cap"])),
            smooth_enabled=bool(smooth_enabled),
            smooth_window=int(smooth_window),
            sigma_floor=float(d.get("sigma_floor", 1e-6)),
            sigma_cap=(None if d.get("sigma_cap") is None else float(d["sigma_cap"])),
            band_edges=list(d.get("band_edges", [])) or None,
            band_smooth_window=(None if d.get("band_smooth_window") is None else int(d["band_smooth_window"])),
        )


# =============================================================================
# Core ops (vectorized)
# =============================================================================


def _moving_average_1d(x: np.ndarray, window: int) -> np.ndarray:
    """Centered moving average with reflect padding; shape preserved."""
    if window <= 1 or x.size == 0:
        return x
    if window % 2 == 0:
        window += 1
    pad = window // 2
    xp = np.pad(x, (pad, pad), mode="reflect")
    kernel = np.ones(window, dtype=float) / float(window)
    return np.convolve(xp, kernel, mode="valid")


def _moving_average_batch(x: np.ndarray, window: int) -> np.ndarray:
    """Apply centered moving average over last axis for batch [N, B]."""
    if window <= 1:
        return x
    if x.ndim != 2:
        raise ValueError(f"moving_average_batch expects 2D [N, B], got {x.shape}")
    return np.stack([_moving_average_1d(row, window) for row in x], axis=0)


def _apply_band_smoothing_batch(x: np.ndarray, edges: Sequence[int], window: int) -> np.ndarray:
    """Apply additional smoothing inside each coarse band for batch [N, B]."""
    if window <= 1 or not edges:
        return x
    out = x.copy()
    for a, b in zip(edges[:-1], edges[1:]):
        a, b = int(a), int(b)
        if a < 0 or b > x.shape[1] or a >= b:
            continue
        out[:, a:b] = _moving_average_batch(out[:, a:b], window)
    return out


def _apply_constraints_batch(
    mu: np.ndarray,
    sigma: Optional[np.ndarray],
    cfg: PostprocessConfig,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Apply smoothing and physics-informed constraints to batch predictions.

    Inputs
    ------
    mu:    [N, B] floats
    sigma: [N, B] floats or None

    Returns
    -------
    (mu_pp, sigma_pp) with same shapes (sigma_pp may be None if sigma was None).
    """
    if mu.ndim != 2 or mu.shape[1] != cfg.bins:
        raise ValueError(f"mu must be [N, {cfg.bins}], got {mu.shape}")
    if sigma is not None and (sigma.ndim != 2 or sigma.shape != mu.shape):
        raise ValueError(f"sigma must match mu shape, got mu={mu.shape}, sigma={None if sigma is None else sigma.shape}")

    # Smoothing first (preserves overall shape before clamps)
    if cfg.smooth_enabled and cfg.smooth_window and cfg.smooth_window > 1:
        mu = _moving_average_batch(mu, cfg.smooth_window)

    if cfg.band_edges and cfg.band_smooth_window and cfg.band_smooth_window > 1:
        mu = _apply_band_smoothing_batch(mu, cfg.band_edges, cfg.band_smooth_window)

    # μ constraints
    if cfg.nonneg:
        mu = np.maximum(mu, 0.0)
    if cfg.max_cap is not None:
        mu = np.minimum(mu, float(cfg.max_cap))

    # σ constraints
    sig_out = None
    if sigma is not None:
        sig = np.maximum(sigma, float(cfg.sigma_floor))
        if cfg.sigma_cap is not None:
            sig = np.minimum(sig, float(cfg.sigma_cap))
        sig_out = sig

    return mu, sig_out


# =============================================================================
# Packing / Validation / Writing
# =============================================================================


def postprocess_and_pack(
    rows: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | PostprocessConfig | None = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    Vectorized post-process and pack into submission DataFrame.

    Input rows (iterable) must each include:
      - "sample_id" (preferred) or "id": string identifier
      - "mu":    array-like of length bins
      - "sigma": array-like of length bins (optional -> will be set to floor)

    Returns a DataFrame with columns: ["sample_id", mu_000.., sigma_000..]
    """
    cfg = config if isinstance(config, PostprocessConfig) else PostprocessConfig.from_dict(config or {})

    # Collect batch
    sample_ids: List[str] = []
    mu_list: List[np.ndarray] = []
    sigma_list: List[Optional[np.ndarray]] = []

    for r in rows:
        sid = r.get("sample_id", r.get("id"))
        if sid is None:
            raise ValueError("Each row must include 'sample_id' or 'id'")
        sample_ids.append(str(sid))

        mu = np.asarray(r.get("mu"), dtype=float).reshape(-1)
        if mu.size != cfg.bins:
            raise ValueError(f"id={sid}: mu length {mu.size}, expected {cfg.bins}")
        mu_list.append(mu)

        s = r.get("sigma", None)
        if s is None:
            sigma_list.append(None)
        else:
            s_arr = np.asarray(s, dtype=float).reshape(-1)
            if s_arr.size != cfg.bins:
                raise ValueError(f"id={sid}: sigma length {s_arr.size}, expected {cfg.bins}")
            sigma_list.append(s_arr)

    N = len(sample_ids)
    mu_np = np.vstack(mu_list) if N > 0 else np.zeros((0, cfg.bins), dtype=float)
    # If any sigma missing, create array with floors; else stack
    if any(s is None for s in sigma_list):
        sigma_np = np.full((N, cfg.bins), cfg.sigma_floor, dtype=float)
        for i, s in enumerate(sigma_list):
            if s is not None:
                sigma_np[i] = s
    else:
        sigma_np = np.vstack([s for s in sigma_list if s is not None]) if N > 0 else np.zeros((0, cfg.bins), dtype=float)

    # Post-process in batch
    mu_pp, sigma_pp = _apply_constraints_batch(mu_np, sigma_np, cfg)
    # Non-finite guards
    if not np.all(np.isfinite(mu_pp)) or not np.all(np.isfinite(sigma_pp)):
        raise ValueError("Non-finite values detected after post-processing.")

    # Pack via canonical formatter (ensures exact column order)
    df = format_predictions(sample_ids, mu_pp, sigma_pp, n_bins=cfg.bins)

    if validate:
        report = validate_dataframe(df, n_bins=cfg.bins, strict_order=True, check_unique_ids=True)
        report.raise_if_failed()

    return df


def write_submission(df: pd.DataFrame, out_path: Union[str, Path], *, index: bool = False) -> None:
    """
    Atomic CSV write.
    """
    out_p = Path(out_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_p.with_suffix(".tmp")
    df.to_csv(tmp, index=index)
    os.replace(tmp, out_p)


# =============================================================================
# Convenience one-shot
# =============================================================================


def postprocess_filelike(
    predictions: Iterable[Mapping[str, Any]],
    *,
    config: Mapping[str, Any] | PostprocessConfig | None = None,
    out_csv: Optional[Union[str, Path]] = None,
    validate: bool = True,
) -> pd.DataFrame:
    """
    One-shot: post-process rows and optionally write CSV.
    """
    df = postprocess_and_pack(predictions, config=config, validate=validate)
    if out_csv is not None:
        write_submission(df, out_csv)
    return df


# =============================================================================
# Minimal CLI (Kaggle-safe)
# =============================================================================


def _load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            rows.append(json.loads(ln))
    return rows


def _save_jsonl(rows: Iterable[Mapping[str, Any]], path: Union[str, Path]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser(description="SpectraMind V50 post-processing")
    ap.add_argument("--in-jsonl", type=str, required=True, help="Input JSONL with rows: {sample_id|id, mu, sigma}")
    ap.add_argument("--out-csv", type=str, required=True, help="Output submission CSV")
    ap.add_argument("--bins", type=int, default=N_BINS_DEFAULT)
    # smoothing / constraints
    ap.add_argument("--smooth", dest="smooth_enabled", action="store_true", default=True)
    ap.add_argument("--no-smooth", dest="smooth_enabled", action="store_false")
    ap.add_argument("--smooth-window", type=int, default=5)
    ap.add_argument("--nonneg", action="store_true", default=True)
    ap.add_argument("--no-nonneg", dest="nonneg", action="store_false")
    ap.add_argument("--max-cap", type=float, default=None)
    ap.add_argument("--sigma-floor", type=float, default=1e-6)
    ap.add_argument("--sigma-cap", type=float, default=None)
    ap.add_argument("--band-edges", type=str, default="", help="Comma-separated ints, e.g. '0,1,30,283'")
    ap.add_argument("--band-smooth-window", type=int, default=None)
    ap.add_argument("--no-validate", action="store_true", help="Skip schema/order validation")
    args = ap.parse_args()

    band_edges = [int(x) for x in args.band_edges.split(",")] if args.band_edges else None

    cfg = PostprocessConfig(
        bins=args.bins,
        nonneg=args.nonneg,
        max_cap=(None if args.max_cap is None else float(args.max_cap)),
        smooth_enabled=args.smooth_enabled,
        smooth_window=args.smooth_window,
        sigma_floor=args.sigma_floor,
        sigma_cap=(None if args.sigma_cap is None else float(args.sigma_cap)),
        band_edges=band_edges,
        band_smooth_window=args.band_smooth_window,
    )

    raw = _load_jsonl(args.in_jsonl)
    df = postprocess_and_pack(raw, config=cfg, validate=(not args.no_validate))
    write_submission(df, args.out_csv)
    print(f"Wrote submission: {args.out_csv}")


if __name__ == "__main__":  # pragma: no cover
    main()
