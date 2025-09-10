from __future__ import annotations
from typing import Any, Dict, Optional
from .jsonl import JSONLLogger
from .utils import iso_now

class EventLogger:
    """
    Convenience wrapper over JSONLLogger with semantic helpers.
    """
    def __init__(self, jsonl: JSONLLogger) -> None:
        self._j = jsonl

    def info(self, message: str, **kws) -> None:
        self._j.log(level="INFO", msg=message, **kws)

    def warn(self, message: str, **kws) -> None:
        self._j.log(level="WARN", msg=message, **kws)

    def error(self, message: str, **kws) -> None:
        self._j.log(level="ERROR", msg=message, **kws)

    def metric(self, **metrics: float) -> None:
        self._j.log(kind="metric", **metrics)

    def artifact(self, path: str, kind: str = "file", **kws) -> None:
        self._j.log(kind="artifact", path=path, artifact_kind=kind, **kws)

    def step(self, phase: str, step: int, **kws) -> None:
        self._j.log(kind="step", phase=phase, step=int(step), **kws)
