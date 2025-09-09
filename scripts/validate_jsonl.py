#!/usr/bin/env python3
"""
Validate a JSONL file line-by-line against a JSON Schema.

Usage:
  python scripts/validate_jsonl.py --schema schemas/diagnostics.schema.json --file artifacts/diagnostics/report.jsonl
  python scripts/validate_jsonl.py --schema schemas/metrics.schema.json --file artifacts/metrics/metrics.jsonl
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

try:
    import jsonschema  # type: ignore
except Exception:
    print("jsonschema not installed; please add to requirements-dev.txt", file=sys.stderr)
    sys.exit(2)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--schema", required=True, help="Path to JSON schema file")
    ap.add_argument("--file", required=True, help="Path to JSONL file")
    args = ap.parse_args()

    schema_path = Path(args.schema)
    jsonl_path = Path(args.file)
    if not schema_path.exists():
        print(f"schema not found: {schema_path}", file=sys.stderr)
        return 2
    if not jsonl_path.exists():
        print(f"jsonl not found: {jsonl_path}", file=sys.stderr)
        return 2

    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    validator = jsonschema.Draft202012Validator(schema)

    n = 0
    errs = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            n += 1
            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"{jsonl_path}:{n}: invalid JSON: {e}", file=sys.stderr)
                errs += 1
                continue
            errors = sorted(validator.iter_errors(obj), key=lambda e: e.path)
            if errors:
                errs += 1
                for e in errors[:5]:  # show first few per line
                    path = ".".join(map(str, e.path)) or "(root)"
                    print(f"{jsonl_path}:{n}: schema error at {path}: {e.message}", file=sys.stderr)

    if errs:
        print(f"{jsonl_path}: {errs} invalid line(s) out of {n}", file=sys.stderr)
        return 1
    print(f"{jsonl_path}: OK ({n} line(s))")
    return 0


if __name__ == "__main__":
    sys.exit(main())
