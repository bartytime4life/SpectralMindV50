
from __future__ import annotations
from typing import Iterable
import pandas as pd
from .base import ValidationResult, ValidationError, ok

def validate_dataset_basic(df: pd.DataFrame, expect_cols: Iterable[str]) -> ValidationResult:
    missing = [c for c in expect_cols if c not in df.columns]
    if missing:
        return ValidationResult(False, [ValidationError("missing columns", {"missing": missing})])
    if df.isna().any().any():
        return ValidationResult(False, [ValidationError("NaN present in dataset")])
    return ok()

def validate_split_disjointness(train_ids: set, eval_ids: set) -> ValidationResult:
    inter = train_ids.intersection(eval_ids)
    if inter:
        return ValidationResult(False, [ValidationError("split leakage", {"overlap": list(inter)[:10]})])
    return ok()
