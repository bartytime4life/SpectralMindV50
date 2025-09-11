from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

try:
    import pandas as pd  # type: ignore
except Exception:  # pragma: no cover
    pd = None  # type: ignore


def _require_pandas() -> None:
    if pd is None:
        raise RuntimeError("pandas is required for diagnostics I/O")


def read_predictions_any(path: Path) -> "pd.DataFrame":
    """
    Read predictions table from CSV/Parquet/JSON.
    Supports:
      - wide:  columns [id|sample_id] + mu_000.. + sigma_000..
      - narrow: columns [id|sample_id, bin, mu, sigma] -> pivoted to wide
    """
    _require_pandas()
    low = path.suffix.lower()
    if low == ".csv":
        df = pd.read_csv(path)  # type: ignore
    elif low == ".parquet":
        df = pd.read_parquet(path)  # type: ignore
    elif low == ".json":
        df = pd.read_json(path)  # type: ignore
    else:
        raise ValueError(f"Unsupported predictions format: {path}")

    return to_wide_predictions(df)


def to_wide_predictions(df: "pd.DataFrame") -> "pd.DataFrame":
    """
    Coerce predictions to 'wide':
      id/sample_id + mu_000.. + sigma_000..
    """
    _require_pandas()

    cols = set(df.columns)
    id_col = "id" if "id" in cols else ("sample_id" if "sample_id" in cols else None)

    # wide?
    if any(c.startswith("mu_") for c in cols) and any(c.startswith("sigma_") for c in cols):
        if id_col is None:
            df = df.copy()
            df.insert(0, "id", range(len(df)))
            id_col = "id"
        # enforce id column name
        if id_col != "id":
            df = df.rename(columns={id_col: "id"})
        return df

    # narrow?
    expected = {"mu", "sigma", "bin"}
    if id_col is not None and expected.issubset(cols):
        tmp = df.rename(columns={id_col: "id"}).copy()
        tmp["bin"] = tmp["bin"].astype(int)
        # pivot μ
        mu_w = tmp.pivot(index="id", columns="bin", values="mu")
        sg_w = tmp.pivot(index="id", columns="bin", values="sigma")
        mu_w.columns = [f"mu_{i:03d}" for i in mu_w.columns]
        sg_w.columns = [f"sigma_{i:03d}" for i in sg_w.columns]
        out = mu_w.join(sg_w)
        out = out.reset_index()
        return out

    raise ValueError("Predictions table must be wide (mu_***, sigma_***) or narrow (id,bin,mu,sigma).")


def read_truth_any(path: Optional[Path]) -> Optional["pd.DataFrame"]:
    """
    Read optional ground truth with columns:
      - narrow: [id|sample_id, bin, target]
      - wide: [id|sample_id] + y_000.. (coerced to narrow internally)
    Returns a narrow DataFrame [id, bin, target].
    """
    if path is None:
        return None
    _require_pandas()
    low = path.suffix.lower()
    if low == ".csv":
        df = pd.read_csv(path)  # type: ignore
    elif low == ".parquet":
        df = pd.read_parquet(path)  # type: ignore
    elif low == ".json":
        df = pd.read_json(path)  # type: ignore
    else:
        raise ValueError(f"Unsupported truth format: {path}")

    cols = set(df.columns)
    id_col = "id" if "id" in cols else ("sample_id" if "sample_id" in cols else None)

    # narrow?
    if id_col is not None and {"bin", "target"}.issubset(cols):
        return df.rename(columns={id_col: "id"})[["id", "bin", "target"]].copy()

    # wide → narrow (y_000..)
    y_cols = [c for c in cols if c.startswith("y_")]
    if id_col is not None and y_cols:
        tmp = df.rename(columns={id_col: "id"}).copy()
        records = []
        for _, row in tmp.iterrows():
            rid = row["id"]
            for c in y_cols:
                i = int(c.split("_", 1)[1])
                records.append({"id": rid, "bin": i, "target": row[c]})
        out = pd.DataFrame.from_records(records)  # type: ignore
        return out

    raise ValueError("Truth table must be narrow (id,bin,target) or wide (id + y_***).")
