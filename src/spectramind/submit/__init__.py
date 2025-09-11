# src/spectramind/submit/__init__.py
"""
Submission toolkit for SpectraMind V50.

Clean boundary between raw model outputs and Kaggle-ready artifacts:
  • formatting (arrays → CSV with required columns/order),
  • validation (header & numeric sanity; optional schema for manifest),
  • packaging (submission.csv + manifest.json + optional ZIP).

Typical usage:

    from spectramind.submit import (
        format_predictions, validate_dataframe, package_submission, N_BINS_DEFAULT
    )

    df = format_predictions(sample_ids, mu, sigma, n_bins=N_BINS_DEFAULT)
    validate_dataframe(df).raise_if_failed()
    package_submission(df, "artifacts/submit")

All functions are safe to call from CI and Kaggle (offline) environments.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Iterator, Mapping, Optional, Sequence, Tuple, Union

# --- public APIs and building blocks from submodules ---
from .format import (  # re-export
    FGS1_INDEX,
    N_BINS_DEFAULT,
    SubmissionRow,
    iter_rows_from_predictions,
    mu_column_names,
    sigma_column_names,
    submission_columns,
    write_csv,
    format_row,
)

from .bundle import (  # re-export main builders + validators
    build_submission_from_predictions,
    build_submission_from_rows,
    validate_columns_present,
    write_manifest,
)

# Optional pandas support
try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


# --------------------------------------------------------------------------- #
# Compatibility shims / high-level conveniences (as advertised in docstring)
# --------------------------------------------------------------------------- #

def build_expected_columns(n_bins: int = N_BINS_DEFAULT) -> list[str]:
    """Alias: full ordered header for the CSV submission."""
    return submission_columns(n_bins)


def format_predictions(
    sample_ids: Sequence[str],
    mu: Sequence[Sequence[float]] | "pd.DataFrame" | None = None,
    sigma: Sequence[Sequence[float]] | "pd.DataFrame" | None = None,
    *,
    n_bins: int = N_BINS_DEFAULT,
    clamp_nonneg_sigma: bool = True,
    round_ndigits: Optional[int] = None,
) -> "pd.DataFrame":
    """
    High-level formatter producing a pandas DataFrame with the exact submission columns.

    Parameters
    ----------
    sample_ids : list[str]
        Sequence of sample IDs.
    mu, sigma : 2D sequences (len == len(sample_ids) and each row has n_bins floats)
        If pandas DataFrames are provided, they must be aligned row-wise with sample_ids
        and contain n_bins columns (order ignored; will be reordered).

    Returns
    -------
    pd.DataFrame
        Columns: ['sample_id', mu_000.., sigma_000..] in the exact required order.
    """
    if pd is None:  # pragma: no cover
        raise RuntimeError("pandas is required for format_predictions()")

    # Normalize mu/sigma into row-wise iterables
    if isinstance(mu, pd.DataFrame):
        mu_rows = [list(map(float, row)) for row in mu.to_numpy()]
    else:
        mu_rows = [list(map(float, r)) for r in (mu or [])]

    if isinstance(sigma, pd.DataFrame):
        sigma_rows = [list(map(float, row)) for row in sigma.to_numpy()]
    else:
        sigma_rows = [list(map(float, r)) for r in (sigma or [])]

    if len(sample_ids) != len(mu_rows) or len(sample_ids) != len(sigma_rows):
        raise ValueError(
            f"Length mismatch: sample_ids={len(sample_ids)} mu={len(mu_rows)} sigma={len(sigma_rows)}"
        )

    # Build SubmissionRow objects
    rows: list[SubmissionRow] = []
    for sid, mu_vec, sg_vec in zip(sample_ids, mu_rows, sigma_rows):
        rows.append(
            format_row(
                sid,
                mu_vec,
                sg_vec,
                n_bins=n_bins,
                clamp_nonneg_sigma=clamp_nonneg_sigma,
                round_ndigits=round_ndigits,
            )
        )

    # Assemble DataFrame in the exact required column order
    data = [r.to_dict(n_bins=n_bins) for r in rows]
    cols = submission_columns(n_bins)
    return pd.DataFrame(data, columns=cols)


# --- validation report and helpers (lightweight) ---

@dataclass
class ValidationErrorReport:
    """
    Collects validation issues; call .raise_if_failed() to convert into exception.
    """
    ok: bool
    messages: list[str]

    def add(self, msg: str) -> None:
        self.ok = False
        self.messages.append(msg)

    def raise_if_failed(self) -> None:
        if not self.ok:
            raise ValueError("Submission validation failed:\n  - " + "\n  - ".join(self.messages))


def validate_dataframe(df: "pd.DataFrame", *, n_bins: int = N_BINS_DEFAULT) -> ValidationErrorReport:
    """
    Validate a pandas DataFrame has the correct header & numeric finiteness.
    Returns a ValidationErrorReport; call .raise_if_failed() to enforce.
    """
    rep = ValidationErrorReport(ok=True, messages=[])
    if pd is None:  # pragma: no cover
        rep.add("pandas validation requested but pandas is not available")
        return rep

    expected = submission_columns(n_bins)
    cols = list(df.columns)
    if cols != expected:
        rep.add("Column order mismatch with required submission header.")
        missing = [c for c in expected if c not in cols]
        extra = [c for c in cols if c not in expected]
        if missing:
            rep.add(f"Missing columns: {missing[:5]}{'...' if len(missing)>5 else ''}")
        if extra:
            rep.add(f"Unexpected columns: {extra[:5]}{'...' if len(extra)>5 else ''}")

    # numeric sanity: all mu_* and sigma_* must be finite numbers
    mu_cols = [c for c in expected if c.startswith("mu_")]
    sg_cols = [c for c in expected if c.startswith("sigma_")]
    try:
        if not df[mu_cols].applymap(lambda v: isinstance(v, (float, int))).all().all():
            rep.add("mu columns contain non-numeric values")
        if not df[sg_cols].applymap(lambda v: isinstance(v, (float, int))).all().all():
            rep.add("sigma columns contain non-numeric values")
        # pandas is finite-friendly via numpy
        import numpy as _np  # type: ignore
        if ~_np.isfinite(df[mu_cols].to_numpy(dtype=float)).all():
            rep.add("mu columns contain non-finite values")
        if ~_np.isfinite(df[sg_cols].to_numpy(dtype=float)).all():
            rep.add("sigma columns contain non-finite values")
    except Exception as e:  # pragma: no cover
        rep.add(f"Numeric validation error: {type(e).__name__}: {e}")

    return rep


def validate_csv(csv_path: str, *, n_bins: int = N_BINS_DEFAULT) -> ValidationErrorReport:
    """
    Validate a CSV file's header matches the submission spec.
    (Row-wise numeric checks are intentionally light to stay I/O-cheap.)
    """
    rep = ValidationErrorReport(ok=True, messages=[])
    try:
        validate_columns_present(csv_path, n_bins=n_bins)
    except Exception as e:
        rep.add(f"Header validation failed: {type(e).__name__}: {e}")
    return rep


def validate_row_dict(row: Mapping[str, object], *, n_bins: int = N_BINS_DEFAULT) -> ValidationErrorReport:
    """
    Validate a single row dict has necessary keys and numeric finiteness.
    Useful for spot checks.
    """
    rep = ValidationErrorReport(ok=True, messages=[])
    expected = submission_columns(n_bins)
    missing = [k for k in expected if k not in row]
    if missing:
        rep.add(f"Row missing keys: {missing[:5]}{'...' if len(missing)>5 else ''}")
        return rep  # no further checks

    # cheap numeric checks (avoid heavy deps)
    import math as _m
    for i in range(n_bins):
        mu = row.get(f"mu_{i:03d}")
        sg = row.get(f"sigma_{i:03d}")
        if not isinstance(mu, (float, int)) or not isinstance(sg, (float, int)):
            rep.add(f"Non-numeric value in bin {i}")
            continue
        if not (_m.isfinite(float(mu)) and _m.isfinite(float(sg))):
            rep.add(f"Non-finite value in bin {i}")
    return rep


# --- packaging convenience (pandas-friendly wrapper) ---

def package_submission(
    df_or_rows: "pd.DataFrame | Iterable[SubmissionRow]",
    out_dir: str,
    *,
    filename: str = "submission.csv",
    n_bins: int = N_BINS_DEFAULT,
    zip_bundle: bool = False,
    zip_name: str = "submission.zip",
    run_id: Optional[str] = None,
    notes: Optional[str] = None,
    manifest_schema: Optional[str] = None,
    strict_manifest_schema: bool = True,
) -> tuple[str, Optional[str], str]:
    """
    Unified packaging helper:
      - If given a DataFrame, write CSV directly and produce manifest (and optional ZIP).
      - If given SubmissionRow iterable, build via bundle helpers.

    Returns (csv_path, manifest_path, bundle_dir).
    """
    os.makedirs(out_dir or ".", exist_ok=True)
    if pd is not None and isinstance(df_or_rows, pd.DataFrame):
        # Write DataFrame to CSV deterministically in required column order
        cols = submission_columns(n_bins)
        csv_path = os.path.join(out_dir, filename)
        # Always write header + rows freshly
        df_or_rows[cols].to_csv(csv_path, index=False)
        # quick header sanity
        validate_columns_present(csv_path, n_bins=n_bins)
        # manifest (+ zip)
        from .bundle import write_manifest  # local import to avoid cycles
        manifest_path = write_manifest(
            out_dir,
            csv_path,
            n_bins=n_bins,
            run_id=run_id,
            notes=notes,
            schema_path=manifest_schema,
            strict_schema=strict_manifest_schema,
        )
        if zip_bundle:
            import zipfile
            zpath = os.path.join(out_dir, zip_name)
            with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as zf:
                zf.write(csv_path, arcname=os.path.basename(csv_path))
                zf.write(manifest_path, arcname=os.path.basename(manifest_path))
        return csv_path, manifest_path, out_dir

    # Iterable[SubmissionRow] path
    return build_submission_from_rows(
        df_or_rows,  # type: ignore[arg-type]
        out_dir,
        filename=filename,
        n_bins=n_bins,
        manifest_schema=manifest_schema,
        strict_manifest_schema=strict_manifest_schema,
        run_id=run_id,
        notes=notes,
        zip_bundle=zip_bundle,
        zip_name=zip_name,
    )


# Optional CLI (do not hard-require Typer in library contexts)
try:  # pragma: no cover - thin convenience re-export
    from .cli import app as cli_app  # type: ignore
except Exception:  # pragma: no cover
    cli_app = None


__all__ = [
    # constants & datatypes
    "N_BINS_DEFAULT",
    "FGS1_INDEX",
    "SubmissionRow",
    # column helpers
    "mu_column_names",
    "sigma_column_names",
    "submission_columns",
    "build_expected_columns",
    # formatting
    "format_row",
    "iter_rows_from_predictions",
    "format_predictions",
    "write_csv",
    # validation
    "ValidationErrorReport",
    "validate_dataframe",
    "validate_csv",
    "validate_row_dict",
    "validate_columns_present",
    # packaging
    "package_submission",
    "build_submission_from_predictions",
    "build_submission_from_rows",
    "write_manifest",
    # optional CLI app
    "cli_app",
]
