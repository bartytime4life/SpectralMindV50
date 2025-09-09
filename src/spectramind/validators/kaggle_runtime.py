
from __future__ import annotations
from pathlib import Path
from .base import ValidationResult, ValidationError, ok

def validate_kaggle_runtime() -> ValidationResult:
    root = Path("/kaggle")
    if not root.exists():
        return ok()  # not in Kaggle
    working = root / "working"
    inputd  = root / "input"
    errs = []
    if not working.exists() or not working.is_dir():
        errs.append("missing /kaggle/working")
    if not inputd.exists() or not inputd.is_dir():
        errs.append("missing /kaggle/input")
    try:
        test_file = working / ".rw_test"
        test_file.write_text("ok", encoding="utf-8")
        test_file.unlink(missing_ok=True)  # py3.8+: set to False if needed
    except Exception as e:
        errs.append(f"/kaggle/working not writeable: {e}")
    return ok() if not errs else ValidationResult(False, [ValidationError("kaggle runtime invalid", {"errors": errs})])
