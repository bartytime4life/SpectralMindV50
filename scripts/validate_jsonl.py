#!/usr/bin/env python3
"""
Validate a JSONL file line-by-line against a JSON Schema.

Usage:
  scripts/validate_jsonl.py --schema schemas/diagnostics.schema.json --file artifacts/diagnostics/report.jsonl
  scripts/validate_jsonl.py --schema schemas/metrics.schema.json --file -  # read from stdin
  scripts/validate_jsonl.py --schema schemas/metrics.schema.json --file artifacts/metrics/metrics.jsonl.gz

Exit codes:
  0 = all valid, 1 = validation error(s), 2 = usage/deps/read error
"""
from __future__ import annotations

import argparse
import gzip
import io
import json
import sys
from pathlib import Path
from typing import Iterable, Tuple, Optional

try:
    import jsonschema  # type: ignore
    from jsonschema import Draft202012Validator, validators, FormatChecker
except Exception:
    print("jsonschema not installed; please add to requirements-dev.txt", file=sys.stderr)
    sys.exit(2)


def _argparse() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="validate_jsonl.py",
        description="Line-by-line JSONL validation against a JSON Schema.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--schema", required=True, help="Path to JSON schema file")
    p.add_argument(
        "--file",
        required=True,
        help="Path to JSONL file (use '-' for stdin). Supports .gz",
    )
    p.add_argument(
        "--max-errors",
        type=int,
        default=50,
        help="Stop after this many total schema errors (per-file), 0 = unlimited",
    )
    p.add_argument(
        "--fail-fast",
        action="store_true",
        help="Exit on the first schema error",
    )
    p.add_argument(
        "--progress",
        action="store_true",
        help="Print a progress counter every 10k lines",
    )
    p.add_argument(
        "--show-pass",
        action="store_true",
        help="Show an OK line summary at the end",
    )
    p.add_argument(
        "--utf8-strict",
        action="store_true",
        help="Strict UTF-8 decode (default is 'ignore' decode errors)",
    )
    return p.parse_args()


def _open_jsonl(path: str, strict: bool) -> Iterable[Tuple[int, str]]:
    """
    Open a JSONL path or '-' for stdin. Yield (lineno, raw_line).
    Supports .gz files by extension.
    """
    text_mode = {"encoding": "utf-8", "errors": ("strict" if strict else "ignore")}
    if path == "-":
        # stdin may be bytes; normalize to text
        if isinstance(sys.stdin, io.TextIOBase):
            src = sys.stdin
        else:
            src = io.TextIOWrapper(sys.stdin.buffer, **text_mode)  # type: ignore
        for i, line in enumerate(src, 1):
            yield i, line
        return

    fpath = Path(path)
    if not fpath.exists():
        print(f"{path}: not found", file=sys.stderr)
        sys.exit(2)

    if fpath.suffix == ".gz":
        with gzip.open(fpath, "rb") as fh:
            with io.TextIOWrapper(fh, **text_mode) as tfh:
                for i, line in enumerate(tfh, 1):
                    yield i, line
    else:
        with fpath.open("r", **text_mode) as fh:
            for i, line in enumerate(fh, 1):
                yield i, line


def _load_schema(schema_path: Path) -> Tuple[jsonschema.Validator, dict]:
    """
    Load schema, infer correct validator class (Draft), and construct with:
      • format checker
      • base_uri set to the schema file location (for relative $ref)
    """
    if not schema_path.exists():
        print(f"schema not found: {schema_path}", file=sys.stderr)
        sys.exit(2)
    try:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"failed to read schema: {e}", file=sys.stderr)
        sys.exit(2)

    # Pick the right validator for the given schema
    ValidatorCls = validators.validator_for(schema)
    ValidatorCls.check_schema(schema)  # upfront schema validation

    # Base URI for relative $ref resolution
    base_uri = schema_path.resolve().as_uri()
    # jsonschema >=4.18 recommends referencing via resource loaders; base_uri is sufficient for file refs
    validator = ValidatorCls(
        schema,
        format_checker=FormatChecker(),
        resolver=jsonschema.RefResolver(base_uri=base_uri, referrer=schema),  # type: ignore[arg-type]
    )
    return validator, schema


def _json_pointer_from_path(path_iter) -> str:
    """
    Convert jsonschema error.path (deque / list) to a JSON Pointer string
    e.g., ["items", 3, "foo"] → /items/3/foo
    """
    parts = []
    for p in path_iter:
        s = str(p)
        s = s.replace("~", "~0").replace("/", "~1")
        parts.append(s)
    return "/" + "/".join(parts) if parts else "/"


def _snippet(value, width: int = 160) -> str:
    try:
        s = json.dumps(value, ensure_ascii=False)
    except Exception:
        return "<unserializable>"
    return (s if len(s) <= width else s[: width - 3] + "...").replace("\n", "\\n")


def main() -> int:
    args = _argparse()
    schema_path = Path(args.schema)

    validator, _ = _load_schema(schema_path)

    fname = args.file
    n = 0
    errs = 0
    parse_errs = 0

    try:
        for lineno, raw in _open_jsonl(fname, strict=args.utf8-strict):  # type: ignore
            line = raw.strip()
            if not line:
                continue
            n += 1

            try:
                obj = json.loads(line)
            except Exception as e:
                print(f"{fname}:{lineno}: invalid JSON: {e}", file=sys.stderr)
                parse_errs += 1
                if args.fail_fast:
                    break
                continue

            found_issue = False
            # iterate for detailed multi-error reporting per line
            for err in sorted(validator.iter_errors(obj), key=lambda e: (list(e.path), e.validator)):
                found_issue = True
                errs += 1
                ptr = _json_pointer_from_path(err.path)
                # include offending instance snippet if available
                inst = getattr(err, "instance", None)
                inst_snip = _snippet(inst) if inst is not None else ""
                print(
                    f"{fname}:{lineno}: schema error at {ptr}: {err.message}"
                    + (f" | instance={inst_snip}" if inst_snip else ""),
                    file=sys.stderr,
                )
                if args.fail_fast:
                    break

            if args.fail_fast and (found_issue or parse_errs):
                break

            if args.progress and n % 10000 == 0:
                print(f"[progress] {n} lines...", file=sys.stderr)

            # cap total errors if requested
            if args.max_errors and errs >= args.max_errors:
                print(f"[limit] reached --max-errors={args.max_errors}; stopping early", file=sys.stderr)
                break

    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        return 2

    total_errs = errs + parse_errs
    if total_errs:
        print(f"{fname}: {total_errs} invalid line(s) out of {n}", file=sys.stderr)
        return 1
    if args.show_pass:
        print(f"{fname}: OK ({n} line(s))")
    else:
        # keep stdout quiet for CI; still return 0
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())