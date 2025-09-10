from __future__ import annotations
import io, json, os
from typing import Any, Optional
from .utils import iso_now, safe_mkdir, to_serializable

class JSONLLogger:
    """
    Append-only JSON Lines logger. Each event is a one-line JSON object.
    Safe in Kaggle/offline CI; no external services.
    """
    def __init__(self, path: str, autoflush: bool = True) -> None:
        self.path = os.fspath(path)
        safe_mkdir(os.path.dirname(self.path) or ".")
        # use append+ buffering to prevent truncation
        self._fh: Optional[io.TextIOWrapper] = open(self.path, "a", buffering=1, encoding="utf-8")
        self.autoflush = autoflush

    def log(self, **fields: Any) -> None:
        if not self._fh:
            raise RuntimeError("JSONLLogger is closed")
        event = {"ts": iso_now()}
        for k, v in fields.items():
            event[k] = to_serializable(v)
        line = json.dumps(event, sort_keys=False, ensure_ascii=False)
        self._fh.write(line + "\n")
        if self.autoflush:
            self._fh.flush()

    def close(self) -> None:
        if self._fh:
            self._fh.flush()
            self._fh.close()
            self._fh = None

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
