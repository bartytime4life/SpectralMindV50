
from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, Tuple, Union
import json

from .base import ValidationResult, ValidationError, ok, fail

PathLike = Union[str, Path]

def _read(path: PathLike) -> Dict[str, Any]:
    p = Path(path)
    txt = p.read_text(encoding="utf-8")
    if p.suffix.lower() in {".yml", ".yaml"}:
        try:
            import yaml  # type: ignore
        except Exception:
            return {"__raw__": txt}  # fallback: no yaml in env
        return yaml.safe_load(txt) or {}
    return json.loads(txt)

def validate_config(config_path: PathLike, schema_path: PathLike | None = None) -> ValidationResult:
    try:
        cfg = _read(config_path)
    except Exception as e:
        return fail("config read error", error=str(e))
    if schema_path is None:
        return ok()
    try:
        import jsonschema  # type: ignore
        schema = json.loads(Path(schema_path).read_text(encoding="utf-8"))
        jsonschema.validate(cfg, schema)  # may raise
        return ok()
    except Exception as e:
        return ValidationResult(False, [ValidationError("config schema validation failed", {"error": str(e)})])
