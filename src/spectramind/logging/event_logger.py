# src/spectramind/logging/event_logger.py
from __future__ import annotations

import json
import os
import socket
import sys
import threading
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, TextIO

try:
    import jsonschema  # type: ignore
    _HAS_JSONSCHEMA = True
except Exception:  # pragma: no cover - optional dependency
    _HAS_JSONSCHEMA = False


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_json_dumps(obj: Any) -> str:
    """Reliable JSON encoding for arbitrary payloads (falls back to str())."""
    def _default(o: Any) -> Any:
        try:
            return asdict(o)  # dataclass?
        except Exception:
            return str(o)
    return json.dumps(obj, default=_default, ensure_ascii=False)


def _env_flag(name: str, default: bool = False) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.lower() in {"1", "true", "yes", "on"}


# -----------------------------------------------------------------------------
# Event model
# -----------------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Event:
    """
    Structured JSONL event.

    Fields:
      ts      : ISO-8601 UTC timestamp
      level   : "DEBUG" | "INFO" | "WARN" | "ERROR"
      event   : short machine-readable event key, e.g., "train/epoch_end"
      message : optional human-readable message
      run_id  : run correlation id (from manifest)
      stage   : pipeline stage, e.g., "train"
      step    : optional numeric step/iteration/epoch counter
      data    : free-form dict payload
      tags    : set of string tags (e.g., ["metrics", "val"])
      pid     : process id
      thread  : thread name
      host    : hostname
    """
    ts: str
    level: str
    event: str
    message: Optional[str] = None
    run_id: Optional[str] = None
    stage: Optional[str] = None
    step: Optional[float] = None
    data: Dict[str, Any] = field(default_factory=dict)
    tags: Iterable[str] = field(default_factory=list)
    pid: int = field(default_factory=lambda: os.getpid())
    thread: str = field(default_factory=lambda: threading.current_thread().name)
    host: str = field(default_factory=lambda: socket.gethostname())

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return _safe_json_dumps(self.to_dict())


# -----------------------------------------------------------------------------
# Event Logger
# -----------------------------------------------------------------------------

class EventLogger:
    """
    JSONL event logger with optional schema validation and stdout mirroring.

    Typical usage:
        logger = EventLogger.for_run(stage="train", run_id="abc123")
        logger.info("train/start", message="begin training", data={"epochs": 50})
        ...
        logger.metric("train/epoch_end", step=1, data={"loss": 0.123, "acc": 0.91})
        ...
        logger.close()

    Thread-safe append-only writer; lines are flushed immediately to minimize loss
    in notebook/CI environments (e.g., Kaggle).
    """

    def __init__(
        self,
        path: Path,
        *,
        stage: Optional[str] = None,
        run_id: Optional[str] = None,
        validate_schema: bool = False,
        schema_path: Optional[Path] = None,
        mirror_stdout: Optional[bool] = None,
    ) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh: TextIO = open(self.path, "a", encoding="utf-8")
        self._lock = threading.Lock()
        self.stage = stage
        self.run_id = run_id

        # JSON Schema validation setup (optional)
        self._validate = validate_schema and _HAS_JSONSCHEMA
        self._schema = None
        if self._validate:
            if schema_path and schema_path.exists():
                self._schema = json.loads(schema_path.read_text(encoding="utf-8"))
            else:
                # Soft-disable if schema is not provided
                self._validate = False

        # stdout mirror: default on in CI/Kaggle or when env flag is set
        if mirror_stdout is None:
            mirror_stdout = _env_flag("SPECTRAMIND_LOG_STDOUT", default=True)
        self._mirror_stdout = mirror_stdout

    # ---- factories -----------------------------------------------------------

    @classmethod
    def for_run(
        cls,
        *,
        stage: str,
        run_id: str,
        root: Optional[Path] = None,
        validate_schema: bool = False,
        schema_path: Optional[Path] = None,
        mirror_stdout: Optional[bool] = None,
    ) -> "EventLogger":
        """
        Construct an EventLogger using a conventional path:
          logs/events/{stage}/{run_id}.jsonl
        """
        root = root or Path("logs") / "events"
        path = root / stage / f"{run_id}.jsonl"
        return cls(
            path=path,
            stage=stage,
            run_id=run_id,
            validate_schema=validate_schema,
            schema_path=schema_path,
            mirror_stdout=mirror_stdout,
        )

    # ---- core write ----------------------------------------------------------

    def log(
        self,
        level: str,
        event: str,
        *,
        message: Optional[str] = None,
        step: Optional[float] = None,
        data: Optional[Dict[str, Any]] = None,
        tags: Optional[Iterable[str]] = None,
        ts: Optional[str] = None,
    ) -> Event:
        """Create and persist a structured event (JSONL)."""
        e = Event(
            ts=ts or _utc_now_iso(),
            level=level.upper(),
            event=event,
            message=message,
            run_id=self.run_id,
            stage=self.stage,
            step=step,
            data=data or {},
            tags=tags or [],
        )
        payload = e.to_dict()

        # Optional validation
        if self._validate and self._schema is not None:
            try:
                jsonschema.validate(instance=payload, schema=self._schema)  # type: ignore
            except Exception as ex:  # pragma: no cover - schema is optional
                # Non-fatal: include validation error in payload and continue
                payload["_validation_error"] = str(ex)

        line = _safe_json_dumps(payload)
        with self._lock:
            self._fh.write(line + "\n")
            self._fh.flush()
            os.fsync(self._fh.fileno())
            if self._mirror_stdout:
                # Keep stdout line-buffered for notebooks/CI
                sys.stdout.write(line + "\n")
                sys.stdout.flush()
        return e

    # ---- convenience levels --------------------------------------------------

    def debug(self, event: str, **kwargs: Any) -> Event:
        return self.log("DEBUG", event, **kwargs)

    def info(self, event: str, **kwargs: Any) -> Event:
        return self.log("INFO", event, **kwargs)

    def warn(self, event: str, **kwargs: Any) -> Event:
        return self.log("WARN", event, **kwargs)

    def error(self, event: str, **kwargs: Any) -> Event:
        return self.log("ERROR", event, **kwargs)

    # ---- high-level helpers --------------------------------------------------

    def metric(
        self,
        event: str,
        *,
        step: Optional[float] = None,
        metrics: Optional[Dict[str, float]] = None,
        **kwargs: Any,
    ) -> Event:
        """
        Log a metrics event. Adds a 'metrics' sub-dict into data and tags with 'metrics'.
        """
        data = dict(kwargs.pop("data", {}) or {})
        if metrics:
            data["metrics"] = metrics
        tags = set(kwargs.pop("tags", []) or [])
        tags.add("metrics")
        return self.info(event, step=step, data=data, tags=sorted(tags), **kwargs)

    def artifact(
        self,
        event: str,
        *,
        path: Path,
        kind: str = "file",
        description: Optional[str] = None,
        **kwargs: Any,
    ) -> Event:
        """
        Log an artifact reference (file/dir URL or path) for lineage.
        """
        data = dict(kwargs.pop("data", {}) or {})
        data["artifact"] = {
            "path": str(path),
            "kind": kind,
            "exists": path.exists(),
            "size": path.stat().st_size if path.exists() and path.is_file() else None,
            "mtime": path.stat().st_mtime if path.exists() else None,
            "description": description,
        }
        tags = set(kwargs.pop("tags", []) or [])
        tags.add("artifact")
        return self.info(event, data=data, tags=sorted(tags), **kwargs)

    def heartbeat(self, *, every_sec: float = 60.0) -> None:
        """
        Emit periodic heartbeat events; non-blocking sleep between emits.
        Intended for long-running stages to signal liveness.
        """
        self.info("heartbeat", message=f"alive (interval={every_sec}s)")
        time.sleep(max(0.0, float(every_sec)))

    # ---- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        with self._lock:
            try:
                self._fh.flush()
                os.fsync(self._fh.fileno())
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass

    def __enter__(self) -> "EventLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


# -----------------------------------------------------------------------------
# Module-level helpers
# -----------------------------------------------------------------------------

def default_events_path(stage: str, run_id: str, root: Optional[Path] = None) -> Path:
    """
    Conventional JSONL path for events of a specific run.
    """
    root = root or Path("logs") / "events"
    return root / stage / f"{run_id}.jsonl"
