# src/spectramind/utils/schema.py
"""
SpectraMind V50 — JSON/YAML Schema Utilities
---------------------------------------------
Utilities to load configuration and artifact files (JSON/YAML) and validate
them against JSON Schemas, with fast and friendly error reporting.

Features
--------
- Loads JSON or YAML (if PyYAML is available; otherwise JSON-only).
- Validates via `fastjsonschema` when present (fast), else falls back to
  `jsonschema` (robust).
- Path-aware error messages (e.g. "data.items[3].mu: must be number").
- Lightweight, Kaggle/CI friendly; no hard deps beyond stdlib + optional libs.

Typical Usage
-------------
from spectramind.utils.schema import load_doc, load_schema, validate_doc

cfg = load_doc("configs/train.yaml")
schema = load_schema("schemas/config_snapshot.schema.json")
validate_doc(cfg, schema, doc_name="train.yaml")

# or validate a file against a schema file directly:
from spectramind.utils.schema import validate_file
validate_file("artifacts/submission.json", "schemas/submission.schema.json")
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

# Optional YAML support
try:  # pragma: no cover - external dep
    import yaml  # type: ignore
    _HAS_YAML = True
except Exception:  # pragma: no cover
    _HAS_YAML = False

# Prefer fastjsonschema if available, else fallback to jsonschema
_ValidatorFactory: Optional[Callable[[dict], Callable[[Any], None]]] = None
try:  # pragma: no cover - external dep
    import fastjsonschema  # type: ignore

    def _compile_fast(schema: dict) -> Callable[[Any], None]:
        return fastjsonschema.compile(schema)

    _ValidatorFactory = _compile_fast
    _ENGINE = "fastjsonschema"
except Exception:  # pragma: no cover
    try:
        import jsonschema  # type: ignore

        def _compile_jsonschema(schema: dict) -> Callable[[Any], None]:
            def _validate(instance: Any) -> None:
                jsonschema.validate(instance=instance, schema=schema)
            return _validate

        _ValidatorFactory = _compile_jsonschema
        _ENGINE = "jsonschema"
    except Exception:
        _ENGINE = None  # no validator available


@dataclass
class SchemaError:
    path: str
    message: str

    def __str__(self) -> str:
        return f"{self.path}: {self.message}"


def _is_json(path: Path) -> bool:
    return path.suffix.lower() in {".json", ".jsonl"}


def _is_yaml(path: Path) -> bool:
    return path.suffix.lower() in {".yml", ".yaml"}


def load_doc(path: str | Path) -> Any:
    """
    Load a JSON or YAML document from disk.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Document not found: {p}")
    if _is_json(p):
        with p.open("r", encoding="utf-8") as f:
            return json.load(f)
    if _is_yaml(p):
        if not _HAS_YAML:
            raise RuntimeError(
                f"YAML requested but PyYAML not installed. Cannot load {p}"
            )
        with p.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)  # type: ignore
    raise ValueError(f"Unsupported file extension for {p.name} (expect .json/.yaml)")


def load_schema(path: str | Path) -> dict:
    """
    Load a JSON schema from disk.

    Notes
    -----
    JSON Schema should be valid JSON; YAML schema is accepted if PyYAML is present.
    """
    p = Path(path)
    schema = load_doc(p)
    if not isinstance(schema, dict):
        raise TypeError(f"Schema did not parse to a dict: {p}")
    return schema


def _coerce_path_from_exc(exc: Exception) -> str:
    """
    Attempt to build a JSON path string from validation exceptions across engines.
    """
    # fastjsonschema exceptions have .path (list) or .field (str)
    path_parts: Tuple[str, ...] = ()
    if hasattr(exc, "path") and getattr(exc, "path"):
        try:
            parts = getattr(exc, "path")
            if isinstance(parts, (list, tuple)):
                path_parts = tuple(str(x) for x in parts)
        except Exception:
            pass
    elif hasattr(exc, "relative_path") and getattr(exc, "relative_path"):
        try:
            parts = getattr(exc, "relative_path")
            path_parts = tuple(str(x) for x in parts)
        except Exception:
            pass
    elif hasattr(exc, "field"):
        try:
            fld = getattr(exc, "field")
            if isinstance(fld, str):
                return fld
        except Exception:
            pass

    if not path_parts:
        return "data"

    # Build a human-friendly JSONPath-like string, e.g. data.items[3].mu
    out = "data"
    for seg in path_parts:
        if seg.isdigit():
            out += f"[{seg}]"
        else:
            # escape dots if needed (kept simple here)
            out += f".{seg}"
    return out


def _compile_validator(schema: dict) -> Callable[[Any], None]:
    if _ValidatorFactory is None or _ENGINE is None:
        raise RuntimeError(
            "No JSON Schema validator available. Install 'fastjsonschema' or 'jsonschema'."
        )
    try:
        validator = _ValidatorFactory(schema)
        logger.debug("Compiled JSON Schema validator using %s", _ENGINE)
        return validator
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"Failed to compile schema with {_ENGINE}: {e}") from e


def validate_doc(doc: Any, schema: dict, *, doc_name: str = "document") -> None:
    """
    Validate a parsed document (dict/list/etc.) against the provided schema.

    Raises
    ------
    ValueError
        If validation fails, with a path-aware, readable message.
    RuntimeError
        If no validator engine is available or compilation fails.
    """
    validator = _compile_validator(schema)
    try:
        validator(doc)
    except Exception as e:
        # Construct a friendly error
        path = _coerce_path_from_exc(e)
        msg = getattr(e, "message", str(e))
        err = SchemaError(path=path, message=msg)
        raise ValueError(f"[Schema] {doc_name}: {err}") from e


def validate_file(doc_path: str | Path, schema_path: str | Path) -> None:
    """
    Validate a document file against a schema file.
    """
    doc_path = Path(doc_path)
    schema_path = Path(schema_path)
    doc = load_doc(doc_path)
    schema = load_schema(schema_path)
    validate_doc(doc, schema, doc_name=doc_path.name)
    logger.info("[Schema] %s validated against %s", doc_path.name, schema_path.name)