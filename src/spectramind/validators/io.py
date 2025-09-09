
from __future__ import annotations
from pathlib import Path
from typing import Optional
import pandas as pd

def read_table_any(path: Path, max_rows: Optional[int] = None, max_cols: Optional[int] = None) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf == ".csv":
        df = pd.read_csv(path)
    elif suf == ".parquet":
        df = pd.read_parquet(path)
    elif suf == ".json":
        df = pd.read_json(path)
    else:
        raise ValueError(f"unsupported table format: {path}")
    if max_cols and df.shape[1] > max_cols:
        df = df.iloc[:, :max_cols]
    if max_rows and df.shape[0] > max_rows:
        df = df.head(max_rows)
    return df
