# tools/make_golden_submission.py
"""
SpectraMind V50 — Golden Submission Generator (Upgraded)
========================================================

Creates a deterministic "golden" submission CSV for unit tests.

Column contract (canonical order):
  - sample_id
  - mu_000 .. mu_(N-1)
  - sigma_000 .. sigma_(N-1)

Bin count (N) resolution precedence:
  1) --bins CLI flag
  2) Infer from schemas/submission.schema.json (scan mu_### in properties)
  3) SM_SUBMISSION_BINS env var
  4) Default 283

Examples:
    python tools/make_golden_submission.py
    python tools/make_golden_submission.py --rows 3 --seed 42
    python tools/make_golden_submission.py --out tests/golden/submission_valid.csv.gz --gzip
    python tools/make_golden_submission.py --parquet --out tests/golden/submission_valid.parquet
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------------------
# Schema & column helpers
# --------------------------------------------------------------------------------------
def _find_repo_root(start: Optional[Path] = None) -> Path:
    cur = (start or Path(__file__)).resolve()
    for parent in [cur] + list(cur.parents):
        if (parent.parent / "schemas").is_dir():  # tools/ → project root has /schemas
            return parent.parent
        if (parent / "schemas").is_dir():
            return parent
    return Path.cwd()


def _load_json(p: Path) -> dict:
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _collect_properties_bags(d: dict) -> Iterable[dict]:
    # Scan all nested dicts/lists for "properties" bags
    stack: List[object] = [d]
    while stack:
        cur = stack.pop()
        if isinstance(cur, dict):
            if "properties" in cur and isinstance(cur["properties"], dict):
                yield cur["properties"]
            stack.extend(v for v in cur.values() if isinstance(v, (dict, list)))
        elif isinstance(cur, list):
            stack.extend(cur)


def _infer_bins_from_schema(schema_path: Path) -> Optional[int]:
    try:
        schema = _load_json(schema_path)
    except Exception:
        return None
    max_mu = -1
    pat = re.compile(r"^mu_(\d{3})$")
    for bag in _collect_properties_bags(schema):
        for k in bag.keys():
            m = pat.match(k)
            if m:
                idx = int(m.group(1))
                if idx > max_mu:
                    max_mu = idx
    return (max_mu + 1) if max_mu >= 0 else None


def resolve_n_bins(cli_bins: Optional[int], schema_hint: Optional[Path]) -> int:
    if cli_bins and cli_bins > 0:
        return cli_bins
    if schema_hint and schema_hint.exists():
        inferred = _infer_bins_from_schema(schema_hint)
        if inferred:
            return inferred
    try:
        env_bins = int(os.environ.get("SM_SUBMISSION_BINS", "283"))
        if env_bins > 0:
            return env_bins
    except Exception:
        pass
    return 283


def build_columns(n_bins: int) -> List[str]:
    mu_cols = [f"mu_{i:03d}" for i in range(n_bins)]
    sigma_cols = [f"sigma_{i:03d}" for i in range(n_bins)]
    return ["sample_id"] + mu_cols + sigma_cols


# --------------------------------------------------------------------------------------
# Data generation
# --------------------------------------------------------------------------------------
def generate_data(
    n_rows: int,
    n_bins: int,
    seed: int,
    sample_ids: Optional[Sequence[str]] = None,
    mu_loc: float = 0.0,
    mu_scale: float = 0.1,
    sigma_low: float = 1e-3,
    sigma_high: float = 0.2,
) -> pd.DataFrame:
    """
    Generate reproducible Gaussian μ and strictly-positive uniform σ.
    """
    rng = np.random.default_rng(seed)
    mus = rng.normal(mu_loc, mu_scale, (n_rows, n_bins)).astype(np.float64)
    sigmas = rng.uniform(sigma_low, sigma_high, (n_rows, n_bins)).astype(np.float64)

    if sample_ids is None:
        sample_ids = [f"row_{i}" for i in range(n_rows)]
    else:
        if len(sample_ids) < n_rows:
            raise ValueError(f"Provided {len(sample_ids)} sample_ids, need {n_rows}")

    data: dict[str, object] = {"sample_id": list(sample_ids)[:n_rows]}
    for i in range(n_bins):
        data[f"mu_{i:03d}"] = mus[:, i]
    for i in range(n_bins):
        data[f"sigma_{i:03d}"] = sigmas[:, i]

    cols = build_columns(n_bins)
    return pd.DataFrame(data, columns=cols)


# --------------------------------------------------------------------------------------
# Lightweight structural validation
# --------------------------------------------------------------------------------------
def validate_structure(df: pd.DataFrame, n_bins: int) -> Tuple[bool, list[str]]:
    errors: list[str] = []
    expected = build_columns(n_bins)

    # Columns exact set:
    got = list(df.columns)
    missing = [c for c in expected if c not in got]
    extra = [c for c in got if c not in expected]
    if missing:
        errors.append(f"Missing columns: {missing[:10]}{' …' if len(missing) > 10 else ''}")
    if extra:
        errors.append(f"Extra columns: {extra[:10]}{' …' if len(extra) > 10 else ''}")

    # Types: μ/σ numeric; σ > 0; finite
    mu_cols = [c for c in expected if c.startswith("mu_")]
    sigma_cols = [c for c in expected if c.startswith("sigma_")]
    num_cols = mu_cols + sigma_cols
    if not set(num_cols).issubset(df.columns):
        # If columns missing we already flagged; skip type checks
        return (len(errors) == 0), errors

    # Check numeric dtypes and finiteness
    num_df = df[num_cols]
    if not all(np.issubdtype(num_df[c].dtype, np.number) for c in num_cols):
        errors.append("Non-numeric values detected in μ/σ columns")

    # Convert to numpy for robust finite checks (ignores non-numeric already caught)
    vals = num_df.to_numpy(copy=False)
    if not np.isfinite(vals).all():
        errors.append("NaN/Inf detected in μ/σ columns")

    # Sigma strictly positive
    sig = df[sigma_cols].to_numpy(copy=False)
    if not (sig > 0.0).all():
        errors.append("σ columns must be strictly positive (no zeros/negatives)")

    # sample_id duplicates
    if df["sample_id"].duplicated().any():
        errors.append("Duplicate sample_id values found")

    return (len(errors) == 0), errors


# --------------------------------------------------------------------------------------
# IO
# --------------------------------------------------------------------------------------
def write_output(df: pd.DataFrame, out: Path, gzip: bool, parquet: bool, float_format: Optional[str]) -> Path:
    out.parent.mkdir(parents=True, exist_ok=True)
    if parquet:
        # Write Parquet; use pandas default engine (pyarrow preferred if available)
        df.to_parquet(out, index=False)
        return out
    if gzip:
        # Ensure .gz suffix
        if out.suffix != ".gz":
            out = out.with_suffix(out.suffix + ".gz")
        df.to_csv(out, index=False, compression="gzip", float_format=float_format)
        return out
    df.to_csv(out, index=False, float_format=float_format)
    return out


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------
def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate a deterministic golden submission CSV for tests.")
    p.add_argument("--rows", type=int, default=2, help="Number of rows to generate (default: 2)")
    p.add_argument("--bins", type=int, default=None, help="Override bin count; else infer/env/default")
    p.add_argument("--seed", type=int, default=7, help="RNG seed (default: 7)")
    p.add_argument(
        "--schema",
        type=Path,
        default=_find_repo_root() / "schemas" / "submission.schema.json",
        help="Path to submission JSON Schema for bin inference",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=_find_repo_root() / "tests" / "golden" / "submission_valid.csv",
        help="Output file path (.csv, .csv.gz, or .parquet when --parquet)",
    )
    p.add_argument("--ids", type=str, nargs="*", default=None, help="Explicit sample_id values (space separated)")
    p.add_argument("--gzip", action="store_true", help="Write gzip-compressed CSV (.csv.gz)")
    p.add_argument("--parquet", action="store_true", help="Write Parquet instead of CSV")
    p.add_argument("--float-format", type=str, default=None, help="Optional float format for CSV (e.g., '%.6g')")
    p.add_argument("--no-validate", action="store_true", help="Skip structural validation before writing")
    return p.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)

    n_bins = resolve_n_bins(args.bins, args.schema)
    cols = build_columns(n_bins)

    df = generate_data(
        n_rows=args.rows,
        n_bins=n_bins,
        seed=args.seed,
        sample_ids=args.ids,
    )

    if not args.no_validate:
        ok, errs = validate_structure(df, n_bins)
        if not ok:
            msg = " | ".join(errs)
            raise SystemExit(f"[make_golden_submission] Structural validation failed: {msg}")

    out_path = write_output(df, args.out, gzip=args.gzip, parquet=args.parquet, float_format=args.float_format)

    # Pretty preview: show first row and column count
    print(f"✅ Wrote golden submission to: {out_path.resolve()}")
    print(f"   rows={len(df):d}  cols={len(df.columns):d}  bins={n_bins:d}")
    with pd.option_context("display.width", 160, "display.max_columns", 12):
        print(df.head(1).to_string(index=False))

    # Sanity: exact expected column order (info only)
    if df.columns.tolist() != cols:
        print("⚠️  Note: column order differs from canonical; tests may reorder before compare.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
