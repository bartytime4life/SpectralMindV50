from __future__ import annotations
from dataclasses import dataclass
from typing import List, Any, Optional

@dataclass(slots=True)
class ValidationError:
    msg: str
    context: Optional[dict[str, Any]] = None

@dataclass(slots=True)
class ValidationResult:
    ok: bool
    errors: List[ValidationError]

def ok() -> ValidationResult:
    return ValidationResult(True, [])

def fail(msg: str, **ctx: Any) -> ValidationResult:
    return ValidationResult(False, [ValidationError(msg, ctx or None)])

def cap_list(xs: list[str], n: int = 10) -> list[str]:
    return xs[:n] + (["…"] if len(xs) > n else [])
