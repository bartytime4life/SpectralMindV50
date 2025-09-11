from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    np = None  # numpy is optional; we can work with lists/tuples

# Public constants
N_BINS_DEFAULT = 283  # per challenge spec
FGS1_INDEX = 0        # first bin corresponds to white-light FGS1


FloatArray = Union[Sequence[float], "np.ndarray"]  # noqa: F821


def _len(x: FloatArray) -> int:
    # np arrays (including 0-d) are handled: fall back to len for sequences
    if np is not None and hasattr(x, "shape"):
        # 1-D vectors expected; try to coerce if 0-D
        shape = getattr(x, "shape")
        if shape is not None and len(shape) == 0:
            return 1
        return int(shape[0])  # type: ignore[index]
    return len(x)  # type: ignore[no-any-return]


def _as_list(x: FloatArray) -> List[float]:
    """
    Coerce an incoming vector to a python list[float].
    Supports: list/tuple, numpy ndarray, numpy masked arrays.
    """
    if np is not None and isinstance(x, np.ndarray):
        # Handle masked arrays by filling masked values with NaN (we will reject later)
        if np.ma.isMaskedArray(x):  # type: ignore[attr-defined]
            filled = x.filled(np.nan)
            return [float(v) for v in np.asarray(filled, dtype=float).tolist()]
        return np.asarray(x, dtype=float).astype(float).tolist()
    # Convert any sequence; ensure numeric cast
    return [float(v) for v in x]


def mu_column_names(n_bins: int = N_BINS_DEFAULT) -> List[str]:
    """Return the ordered list of μ column names: mu_000..mu_(n_bins-1)."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return [f"mu_{i:03d}" for i in range(n_bins)]


def sigma_column_names(n_bins: int = N_BINS_DEFAULT) -> List[str]:
    """Return the ordered list of σ column names: sigma_000..sigma_(n_bins-1)."""
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    return [f"sigma_{i:03d}" for i in range(n_bins)]


def submission_columns(n_bins: int = N_BINS_DEFAULT) -> List[str]:
    """
    Return the complete ordered set of submission columns:
      sample_id, mu_000..mu_(n_bins-1), sigma_000..sigma_(n_bins-1)
    """
    return ["sample_id"] + mu_column_names(n_bins) + sigma_column_names(n_bins)


def _round_if(v: float, ndigits: Optional[int]) -> float:
    if ndigits is None:
        return float(v)
    r = round(float(v), ndigits)
    # Avoid "-0.0" which can creep in with negatives near zero
    if r == 0.0:
        return 0.0
    return r


def _is_finite_vec(vec: Sequence[float]) -> bool:
    return all(math.isfinite(float(v)) for v in vec)


@dataclass(frozen=True)
class SubmissionRow:
    """
    In-memory representation of a single Kaggle submission row.
    Use .to_dict() to write via csv.DictWriter.
    """
    sample_id: str
    mu: List[float]
    sigma: List[float]

    def to_dict(self, n_bins: int = N_BINS_DEFAULT) -> Mapping[str, Union[str, float]]:
        if len(self.mu) != n_bins:
            raise ValueError(f"mu has length {len(self.mu)} but expected {n_bins}")
        if len(self.sigma) != n_bins:
            raise ValueError(f"sigma has length {len(self.sigma)} but expected {n_bins}")
        row: dict[str, Union[str, float]] = {"sample_id": self.sample_id}
        for i, v in enumerate(self.mu):
            row[f"mu_{i:03d}"] = float(v)
        for i, v in enumerate(self.sigma):
            row[f"sigma_{i:03d}"] = float(v)
        return row


def validate_values(name: str, values: Sequence[float]) -> None:
    """
    Raise ValueError if any element is non-finite.
    """
    if not _is_finite_vec(values):
        bad = [i for i, v in enumerate(values) if not math.isfinite(float(v))]
        preview = bad[:5]
        raise ValueError(f"{name} contains non-finite values at indices {preview}")


def format_row(
    sample_id: str,
    mu: FloatArray,
    sigma: FloatArray,
    *,
    n_bins: int = N_BINS_DEFAULT,
    clamp_nonneg_sigma: bool = True,
    round_ndigits: Optional[int] = None,
) -> SubmissionRow:
    """
    Format one prediction into a SubmissionRow.
    - Ensures correct bin counts.
    - Optionally clamps sigma to be non-negative.
    - Optional rounding to control CSV size & consistency.
    """
    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")

    mu_list = _as_list(mu)
    sigma_list = _as_list(sigma)
    if _len(mu_list) != n_bins or _len(sigma_list) != n_bins:
        raise ValueError(
            f"Expected mu/sigma length {n_bins}, got {len(mu_list)}/{len(sigma_list)}"
        )

    # sanitize sigma
    out_sigma: List[float] = []
    for s in sigma_list:
        val = float(s)
        if clamp_nonneg_sigma and val < 0.0:
            val = 0.0
        out_sigma.append(_round_if(val, round_ndigits))

    out_mu = [_round_if(float(v), round_ndigits) for v in mu_list]

    # sanity checks: finite values only
    validate_values("mu", out_mu)
    validate_values("sigma", out_sigma)

    return SubmissionRow(sample_id=sample_id, mu=out_mu, sigma=out_sigma)


def iter_rows_from_predictions(
    preds: Iterable[Tuple[str, FloatArray, FloatArray]],
    *,
    n_bins: int = N_BINS_DEFAULT,
    clamp_nonneg_sigma: bool = True,
    round_ndigits: Optional[int] = None,
) -> Iterator[SubmissionRow]:
    """
    Convenience: iterate (sample_id, mu, sigma) tuples and yield SubmissionRow instances.
    """
    for sample_id, mu, sigma in preds:
        yield format_row(
            sample_id,
            mu,
            sigma,
            n_bins=n_bins,
            clamp_nonneg_sigma=clamp_nonneg_sigma,
            round_ndigits=round_ndigits,
        )


def write_csv(
    rows: Iterable[SubmissionRow],
    out_path: str,
    *,
    n_bins: int = N_BINS_DEFAULT,
    newline: str = "",
    atomic: bool = True,
) -> None:
    """
    Write submission.csv without requiring pandas.
    Column order:
        sample_id, mu_000..mu_(n_bins-1), sigma_000..sigma_(n_bins-1)

    If atomic=True (default), write to a temp sibling, then os.replace() to avoid partial files.
    """
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    cols = submission_columns(n_bins)

    target = out_path
    tmp = f"{out_path}.tmp" if atomic else out_path

    with open(tmp, "w", encoding="utf-8", newline=newline) as fh:
        writer = csv.DictWriter(fh, fieldnames=cols, extrasaction="raise")
        writer.writeheader()
        wrote_any = False
        for row in rows:
            writer.writerow(row.to_dict(n_bins=n_bins))
            wrote_any = True

    if atomic:
        # Replace target atomically, only if the write succeeded
        os.replace(tmp, target)
    elif not wrote_any:
        # Non-atomic path but empty: still ensure file exists with header written above
        pass


# Optional pandas helpers (only if pandas is installed)
try:
    import pandas as pd  # type: ignore

    def to_dataframe(
        rows: Iterable[SubmissionRow], n_bins: int = N_BINS_DEFAULT
    ) -> "pd.DataFrame":  # noqa: F821
        data = [r.to_dict(n_bins=n_bins) for r in rows]
        cols = submission_columns(n_bins)
        return pd.DataFrame(data, columns=cols)

except Exception:  # pragma: no cover
    pd = None
    # No pandas in env; write_csv is sufficient.
