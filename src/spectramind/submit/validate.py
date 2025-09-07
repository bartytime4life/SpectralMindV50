from __future__ import annotations
import json, pathlib
from jsonschema import validate
import pandas as pd

SUB_SCHEMA = json.loads(pathlib.Path("schemas/submission.schema.json").read_text())

def validate_csv(csv_path: str) -> None:
    df = pd.read_csv(csv_path)
    for _, row in df.iterrows():
        rec = {"id": row["id"], "mu": json.loads(row["mu"]), "sigma": json.loads(row["sigma"])}
        validate(rec, SUB_SCHEMA)
        assert all(s >= 0 for s in rec["sigma"]), "negative sigma"
