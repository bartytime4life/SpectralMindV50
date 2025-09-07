from __future__ import annotations
import json, time, pathlib
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class Event:
    ts: float
    kind: str
    payload: dict[str, Any]

def jsonl_logger(path: str | pathlib.Path):
    p = pathlib.Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    def _log(kind: str, **payload: Any) -> None:
        e = Event(ts=time.time(), kind=kind, payload=payload)
        with p.open("a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(e), separators=(",", ":")) + "\n")
    return _log
