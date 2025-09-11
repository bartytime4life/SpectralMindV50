from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Mapping, Optional, Sequence, Tuple, Union

from .format import (
    N_BINS_DEFAULT,
    SubmissionRow,
    iter_rows_from_predictions,
    submission_columns,
    write_csv,
)

try:
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


# ---------------------------
# Validation utilities
# ---------------------------

def _sha256_file(path: Union[str, os.PathLike]) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 64), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_columns_present(csv_path: str, *, n_bins: int = N_BINS_DEFAULT) -> None:
    """
    Quick CSV header validation: ensure submission has the exact required columns.
    """
    expected = submission_columns(n_bins)
    with open(csv_path, "r", encoding="utf-8", newline="") as fh:
        reader = csv.reader(fh)
        try:
            header = next(reader)
        except StopIteration:
            raise ValueError("CSV appears empty (no header)")
    if header != expected:
        # Construct a nice diff summary (shortened)
        missing = [c for c in expected if c not in header]
        extra = [c for c in header if c not in expected]
        raise ValueError(
            "CSV header mismatch.\n"
            f"  Expected {len(expected)} columns.\n"
            f"  Got {len(header)}.\n"
            f"  Missing: {missing[:5]}{'...' if len(missing) > 5 else ''}\n"
            f"  Extra: {extra[:5]}{'...' if len(extra) > 5 else ''}"
        )


def validate_json_schema(
    obj: Mapping[str, object],
    *,
    schema_path: Optional[Union[str, os.PathLike]],
    strict: bool = True,
) -> None:
    """
    Validate 'obj' against a JSON schema (file path). No-op if schema_path is None or jsonschema missing.
    """
    if schema_path is None or jsonschema is None:
        return
    with open(schema_path, "r", encoding="utf-8") as fh:
        schema = json.load(fh)
    try:
        jsonschema.validate(obj, schema)  # type: ignore
    except Exception as e:
        if strict:
            raise
        else:
            # degrade to warning; do not interrupt the bundle
            print(f"[submit] Warning: manifest schema validation failed: {e}", file=sys.stderr)


# ---------------------------
# Manifest
# ---------------------------

@dataclass
class Manifest:
    created_at: str
    submission_csv: str
    n_bins: int
    sha256: str
    run_id: Optional[str] = None
    config_hash: Optional[str] = None
    git_commit: Optional[str] = None
    notes: Optional[str] = None

    def to_dict(self) -> Mapping[str, object]:
        return asdict(self)


def _env_or_none(key: str) -> Optional[str]:
    val = os.getenv(key)
    return
