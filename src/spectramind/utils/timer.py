# src/spectramind/utils/timer.py
"""
SpectraMind V50 — Timer Utility
--------------------------------
Mission-grade, zero-dependency timing helpers for benchmarking, diagnostics,
and lightweight telemetry (Kaggle/CI-safe).

Highlights
----------
- Monotonic high-resolution timers (perf_counter_ns) + optional CPU time
- Context manager **and** decorator forms (sync + async)
- Named laps/checkpoints with cumulative stats
- Pretty humanized durations (μs..days) and machine-readable dicts
- Pluggable log sink and optional JSONL event callback
- Stable logging via spectramind.get_logger if available

Examples
--------
from spectramind.utils.timer import Timer, timeit, timeit_async

# As context manager
with Timer("Calibration") as t:
    run_calibration()
    t.lap("dark_current")
    run_dark_current()
    t.lap("flat_field")
# -> logs: [Timer] Calibration: 2 laps, total 12.34 s (wall), 11.98 s (cpu)

# As decorator (sync)
@timeit("Model training")
def train_model(cfg): ...

# As decorator (async)
@timeit_async("Download blobs")
async def fetch_all(): ...

# With JSONL event stream
def write_event(rec: dict) -> None:
    Path("artifacts/logs/events.jsonl").parent.mkdir(parents=True, exist_ok=True)
    with open("artifacts/logs/events.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(rec) + "\n")

with Timer("Predict", on_event=write_event):
    predict()
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
import json
import logging
import time
from functools import wraps
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, TypeVar, Union, Awaitable, Coroutine

__all__ = [
    "Timer",
    "timeit",
    "timeit_async",
    "format_duration",
]

# --------------------------------------------------------------------------------------
# Logging: prefer spectramind.get_logger if present, else stdlib
# --------------------------------------------------------------------------------------
def _resolve_logger() -> logging.Logger:
    try:
        from spectramind import get_logger  # type: ignore
        return get_logger(__name__)
    except Exception:
        return logging.getLogger(__name__)

logger = _resolve_logger()

# --------------------------------------------------------------------------------------
# Utilities
# --------------------------------------------------------------------------------------
def format_duration(seconds: float) -> str:
    """Humanize a duration (seconds) using adaptive units with 3 sig figs."""
    if seconds < 1e-6:
        return f"{seconds*1e9:.3g} ns"
    if seconds < 1e-3:
        return f"{seconds*1e6:.3g} μs"
    if seconds < 1.0:
        return f"{seconds*1e3:.3g} ms"
    # ≥ 1s
    mins, sec = divmod(seconds, 60.0)
    if mins < 1:
        return f"{sec:.3g} s"
    hrs, mins = divmod(int(mins), 60)
    if hrs < 24:
        return f"{hrs:d}h {mins:d}m {sec:0.3f}s"
    days, hrs = divmod(hrs, 24)
    return f"{days:d}d {hrs:d}h {mins:d}m {sec:0.3f}s"

def _now_wall_ns() -> int:
    return time.perf_counter_ns()  # monotonic, high-res

def _now_cpu_ns() -> int:
    # process_time_ns is Python 3.7+; safe to fall back
    get = getattr(time, "process_time_ns", None)
    if callable(get):
        return get()
    return int(time.process_time() * 1e9)

# --------------------------------------------------------------------------------------
# Data structures
# --------------------------------------------------------------------------------------
@dataclass
class Lap:
    name: str
    t_ns: int
    cpu_ns: int

@dataclass
class TimerRecord:
    label: str
    start_ns: int
    end_ns: int
    cpu_start_ns: int
    cpu_end_ns: int
    laps: List[Lap] = field(default_factory=list)
    ok: bool = True
    error: Optional[str] = None

    @property
    def wall_seconds(self) -> float:
        return max(0.0, (self.end_ns - self.start_ns) / 1e9)

    @property
    def cpu_seconds(self) -> float:
        return max(0.0, (self.cpu_end_ns - self.cpu_start_ns) / 1e9)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "ok": self.ok,
            "error": self.error,
            "wall_s": self.wall_seconds,
            "cpu_s": self.cpu_seconds,
            "laps": [
                {"name": l.name, "t_s": (l.t_ns - self.start_ns) / 1e9, "cpu_s": (l.cpu_ns - self.cpu_start_ns) / 1e9}
                for l in self.laps
            ],
        }

# --------------------------------------------------------------------------------------
# Timer
# --------------------------------------------------------------------------------------
OnEvent = Optional[Callable[[Dict[str, Any]], None]]
LogFn = Optional[Callable[[str], None]]

class Timer:
    """Context manager + utility for timing code blocks and recording laps.

    Parameters
    ----------
    label : str
        A friendly label for logs and telemetry.
    log_fn : Callable[[str], None] | None
        Custom logger. Defaults to `logger.info`.
    on_event : Callable[[dict], None] | None
        Optional sink for JSON-serializable events (e.g., append to events.jsonl).
        Called on 'start', 'lap', and 'end' with a compact dict.
    include_cpu : bool
        Also measure CPU time (cheap and useful on shared runners).
    """

    def __init__(
        self,
        label: str = "",
        *,
        log_fn: LogFn = None,
        on_event: OnEvent = None,
        include_cpu: bool = True,
    ) -> None:
        self.label = label or "timer"
        self._log: Callable[[str], None] = log_fn or (lambda msg: logger.info(msg))
        self._on_event = on_event
        self._include_cpu = include_cpu

        self._start_ns: Optional[int] = None
        self._end_ns: Optional[int] = None
        self._cpu_start_ns: Optional[int] = None
        self._cpu_end_ns: Optional[int] = None
        self._laps: List[Lap] = []
        self._exc: Optional[BaseException] = None

    # ---- lifecycle ---------------------------------------------------------
    def __enter__(self) -> "Timer":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self._exc = exc  # may be None
        self.stop()
        # do not suppress exceptions
        return None  # type: ignore[return-value]

    # ---- core API ----------------------------------------------------------
    def start(self) -> None:
        self._start_ns = _now_wall_ns()
        self._cpu_start_ns = _now_cpu_ns() if self._include_cpu else 0
        if self._on_event:
            self._on_event({
                "t": time.time(),
                "kind": "timer:start",
                "label": self.label,
            })
        self._log(f"[Timer] {self.label} started")

    def lap(self, name: str) -> None:
        if self._start_ns is None:
            raise RuntimeError("Timer.lap() called before start")
        l = Lap(name=name, t_ns=_now_wall_ns(), cpu_ns=_now_cpu_ns() if self._include_cpu else 0)
        self._laps.append(l)
        if self._on_event:
            self._on_event({
                "t": time.time(),
                "kind": "timer:lap",
                "label": self.label,
                "lap": name,
                "since_start_s": (l.t_ns - self._start_ns) / 1e9,
            })
        self._log(f"[Timer] {self.label} lap '{name}' at {format_duration((l.t_ns - self._start_ns) / 1e9)}")

    def stop(self) -> TimerRecord:
        if self._start_ns is None:
            raise RuntimeError("Timer.stop() called before start")
        self._end_ns = _now_wall_ns()
        self._cpu_end_ns = _now_cpu_ns() if self._include_cpu else 0

        rec = TimerRecord(
            label=self.label,
            start_ns=self._start_ns,
            end_ns=self._end_ns,
            cpu_start_ns=self._cpu_start_ns or 0,
            cpu_end_ns=self._cpu_end_ns or 0,
            laps=self._laps[:],
            ok=self._exc is None,
            error=None if self._exc is None else f"{type(self._exc).__name__}: {self._exc}",
        )

        wall, cpu = rec.wall_seconds, rec.cpu_seconds
        laps_count = len(self._laps)
        cpu_str = f", {format_duration(cpu)} (cpu)" if self._include_cpu else ""
        status = "finished" if rec.ok else "failed"
        self._log(f"[Timer] {self.label}: {laps_count} laps, {status} in {format_duration(wall)} (wall){cpu_str}")

        if self._on_event:
            ev: Dict[str, Any] = {
                "t": time.time(),
                "kind": "timer:end",
                "label": self.label,
                "ok": rec.ok,
                "wall_s": wall,
                "cpu_s": cpu if self._include_cpu else None,
                "laps": [{"name": l.name, "t_s": (l.t_ns - rec.start_ns) / 1e9} for l in self._laps],
            }
            self._on_event(ev)

        return rec

    # ---- properties --------------------------------------------------------
    @property
    def started(self) -> bool:
        return self._start_ns is not None

    @property
    def elapsed(self) -> Optional[float]:
        """Elapsed wall-clock seconds so far (None if not started)."""
        if self._start_ns is None:
            return None
        end = _now_wall_ns() if self._end_ns is None else self._end_ns
        return (end - self._start_ns) / 1e9

# --------------------------------------------------------------------------------------
# Decorators
# --------------------------------------------------------------------------------------
F = TypeVar("F", bound=Callable[..., Any])
AF = TypeVar("AF", bound=Callable[..., Awaitable[Any]])

def timeit(label: Optional[str] = None, *, on_event: OnEvent = None, include_cpu: bool = True) -> Callable[[F], F]:
    """Decorator for synchronous functions."""
    def deco(func: F) -> F:
        msg = label or func.__name__
        @wraps(func)
        def wrapper(*args, **kwargs):  # type: ignore[override]
            with Timer(msg, on_event=on_event, include_cpu=include_cpu):
                return func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return deco

def timeit_async(label: Optional[str] = None, *, on_event: OnEvent = None, include_cpu: bool = True) -> Callable[[AF], AF]:
    """Decorator for coroutine functions."""
    def deco(func: AF) -> AF:
        msg = label or func.__name__
        @wraps(func)
        async def wrapper(*args, **kwargs):  # type: ignore[override]
            # We keep Timer synchronous; it measures around the awaited call.
            with Timer(msg, on_event=on_event, include_cpu=include_cpu):
                return await func(*args, **kwargs)
        return wrapper  # type: ignore[return-value]
    return deco
