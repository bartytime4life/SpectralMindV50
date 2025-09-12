#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
SpectraMind V50 — Self-Test JSONL Validator
-------------------------------------------
Validates self-test event lines against the JSON Schema.

Default inputs:
  - JSONL:  artifacts/selftest/last_selftest.jsonl
  - Schema: schemas/selftest.events.schema.json

Usage:
  python tools/validate_selftest_jsonl.py
  python tools/validate_selftest_jsonl.py path/to/events.jsonl
  python tools/validate_selftest_jsonl.py --schema schemas/events.schema.json
  python tools/validate_selftest_jsonl.py --quiet --max-errors 20

Exit codes:
  0 = all lines valid
  1 = any invalid line(s) or I/O / schema errors
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Optional

# Optional dependency message shown if jsonschema isn't installed.
_JSONSCHEMA_HELP = "Install with: python -m pip install jsonschema>=4"

try:
    import jsonschema
    from jsonschema.validators import Draft202012Validator
except Exception:
    print(f"[validate] ERROR: 'jsonschema' is not installed.\n{_JSONSCHEMA_HELP}", file=sys.stderr)
    sys.exit(1)


def colorize(text: str, color: Optional[str]) -> str:
    """Add ANSI color if stdout is a TTY; otherwise return plain text."""
    if not sys.stderr.isatty() or not color:
        return text
    colors = {
        "red": "\033[1;31m",
        "yel": "\033[1;33m",
        "grn": "\033[1;32m",
        "blu": "\033[1;34m",
        "dim": "\033[2m",
        "off": "\033[0m",
    }
    return f"{colors.get(color,'')}{text}{colors['off']}"


def load_schema(schema_path: Path) -> dict[str, Any]:
    try:
        raw = schema_path.read_text(encoding="utf-8")
    except Exception as e:
        raise RuntimeError(f"cannot read schema: {schema_path} ({e})")
    try:
        return json.loads(raw)
    except Exception as e:
        raise RuntimeError(f"invalid JSON in schema: {schema_path} ({e})")


def iter_jsonl_lines(jsonl_path: Path):
    try:
        with jsonl_path.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, 1):
                yield i, line.rstrip("\n")
    except Exception as e:
        raise RuntimeError(f"cannot read JSONL: {jsonl_path} ({e})")


def validate_jsonl(jsonl_path: Path, schema_path: Path, quiet: bool, max_errors: int) -> int:
    schema = load_schema(schema_path)
    validator = Draft202012Validator(schema)

    errors = 0
    total = 0

    # Optional lightweight semantic checks (beyond schema)
    last_ts_by_run: dict[str, float] = {}

    for lineno, raw in iter_jsonl_lines(jsonl_path):
        if not raw.strip():
            # Allow blank lines silently
            continue
        total += 1

        try:
            obj = json.loads(raw)
        except json.JSONDecodeError as e:
            errors += 1
            if not quiet:
                msg = colorize(f"[validate] line {lineno}: invalid JSON: {e}", "red")
                print(msg, file=sys.stderr)
            if errors >= max_errors > 0:
                break
            continue

        # Schema validation
        line_errs = list(validator.iter_errors(obj))
        if line_errs:
            for err in line_errs:
                errors += 1
                if not quiet:
                    # Build a compact path display (e.g., payload.stage)
                    path = ".".join(str(p) for p in err.path) or "(root)"
                    where = f"{path}"
                    msg = colorize(f"[validate] line {lineno}: {where}: {err.message}", "red")
                    print(msg, file=sys.stderr)
                if errors >= max_errors > 0:
                    break
            if errors >= max_errors > 0:
                break

        # Lightweight monotonic ts check per run_id (if both present)
        try:
            run_id = obj.get("run_id")
            ts = obj.get("ts")
            if run_id is not None and isinstance(ts, (int, float)):
                last = last_ts_by_run.get(run_id)
                if last is not None and ts < last:
                    errors += 1
                    if not quiet:
                        msg = colorize(
                            f"[validate] line {lineno}: ts not monotonic for run_id={run_id}: {ts} < {last}",
                            "yel",
                        )
                        print(msg, file=sys.stderr)
                last_ts_by_run[run_id] = ts  # update regardless
        except Exception:
            # Never fail the entire run on advisory check
            pass

    # Summary
    if errors == 0:
        if not quiet:
            print(colorize(f"[validate] OK — {total} line(s) valid", "grn"), file=sys.stderr)
        return 0
    else:
        print(colorize(f"[validate] FAILED — {errors} error(s) across {total} line(s)", "red"), file=sys.stderr)
        return 1


def main(argv: list[str]) -> int:
    root = Path(__file__).resolve().parents[1]
    default_jsonl = root / "artifacts" / "selftest" / "last_selftest.jsonl"
    # Keep schema name aligned with the file we added earlier
    default_schema = root / "schemas" / "selftest.events.schema.json"

    p = argparse.ArgumentParser(description="Validate self-test events JSONL against schema.")
    p.add_argument(
        "jsonl",
        nargs="?",
        default=str(default_jsonl),
        help=f"Path to events JSONL (default: {default_jsonl})",
    )
    p.add_argument(
        "--schema",
        default=str(default_schema),
        help=f"Path to JSON Schema (default: {default_schema})",
    )
    p.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress per-line messages; only print final summary.",
    )
    p.add_argument(
        "--max-errors",
        type=int,
        default=200,
        help="Stop after this many errors (>0). 0 disables early stop. (default: 200)",
    )
    args = p.parse_args(argv)

    jsonl_path = Path(args.jsonl)
    schema_path = Path(args.schema)

    if not jsonl_path.exists():
        print(colorize(f"[validate] ERROR: JSONL not found: {jsonl_path}", "red"), file=sys.stderr)
        return 1
    if not schema_path.exists():
        print(colorize(f"[validate] ERROR: Schema not found: {schema_path}", "red"), file=sys.stderr)
        return 1

    return validate_jsonl(jsonl_path, schema_path, args.quiet, args.max_errors)


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))