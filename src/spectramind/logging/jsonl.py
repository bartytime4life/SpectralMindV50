from __future__ import annotations

import gzip
import io
import json
import os
import sys
import time
import threading
from pathlib import Path
from typing import Any, Optional, Union

from .utils import iso_now, safe_mkdir, to_serializable

try:  # optional; only used if schema validation is enabled
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


class JSONLLogger:
    """
    Append-only JSON Lines logger. Each event is one JSON object per line.
    Safe for Kaggle/offline CI. Supports size-based rotation, optional schema validation,
    fallback to stdout, periodic flush, gzip-on-close, and injectable audit fields.

    (Docstring of your current class kept intentionally concise; see parameter doc in __init__.)
    """

    def __init__(
        self,
        path: str,
        *,
        autoflush: bool = True,
        flush_secs: float = 5.0,
        pretty: bool = False,
        indent: int = 2,
        include_timestamp: bool = True,
        include_config_hash: Optional[str] = None,
        include_git_commit: Optional[str] = None,
        schema_path: Optional[Union[str, os.PathLike]] = None,
        strict_validation: bool = True,
        safe_mode: bool = True,
        fallback_stdout: bool = True,
        max_file_size_mb: Optional[int] = None,
        compress_on_close: bool = False,
        thread_safe: bool = False,
    ) -> None:
        self.path = os.fspath(path)
        self.autoflush = autoflush
        self.flush_secs = float(max(0.0, flush_secs))
        self.pretty = bool(pretty)
        self.indent = int(max(0, indent))
        self.include_timestamp = include_timestamp

        self._fixed_fields: dict[str, Any] = {}
        if include_config_hash:
            self._fixed_fields["config_hash"] = include_config_hash
        if include_git_commit:
            self._fixed_fields["git_commit"] = include_git_commit

        self.strict_validation = strict_validation
        self.safe_mode = safe_mode
        self.fallback_stdout = fallback_stdout
        self.compress_on_close = compress_on_close

        self._lock: Optional[threading.Lock] = threading.Lock() if thread_safe else None

        # rotation
        self.max_bytes = None
        if max_file_size_mb is not None:
            self.max_bytes = int(max(1, max_file_size_mb)) * 1024 * 1024
        self._rotate_keep = 3  # keep .1, .2, .3

        # schema (optional)
        self._schema: Optional[dict[str, Any]] = None
        if schema_path is not None:
            try:
                with open(schema_path, "r", encoding="utf-8") as fh:
                    self._schema = json.load(fh)
            except Exception as e:  # pragma: no cover
                if strict_validation:
                    raise
                self._schema = None
                self._fixed_fields["_schema_load_error"] = f"{type(e).__name__}: {e}"

        # open file
        safe_mkdir(os.path.dirname(self.path) or ".")
        self._fh: Optional[io.TextIOWrapper] = None
        self._last_flush = time.monotonic()
        self._use_stdout = False
        self._open_file()

    # ------------------------------------------------------------------ #
    # Public API
    # ------------------------------------------------------------------ #

    def log(self, **fields: Any) -> None:
        """Log one event using keyword arguments."""
        self._log_event(fields)

    def log_dict(self, fields: dict[str, Any]) -> None:
        """Log one event from a dict (supports keys that aren't valid identifiers)."""
        self._log_event(dict(fields))  # shallow copy

    def log_exception(self, **fields: Any) -> None:
        """
        Log an exception with context. Use within an except block.
        Adds: _exc_type, _exc_msg, and (non-strict) _schema_error if applicable.
        """
        etype, evalue, _ = sys.exc_info()
        if etype is not None:
            fields.setdefault("_exc_type", etype.__name__)
            fields.setdefault("_exc_msg", str(evalue))
        self._log_event(fields)

    def reopen(self, path: Optional[str] = None) -> None:
        """
        Close and reopen the logger, optionally switching to a new path.
        Useful when rotating log directories per run or after tmpfs moves.
        """
        if path:
            self.path = os.fspath(path)
            safe_mkdir(os.path.dirname(self.path) or ".")
        self._close_file()
        self._open_file()

    def close(self) -> None:
        """Flush, optionally compress, and close."""
        with self._maybe_locked():
            self._close_file()
            if self.compress_on_close and not self._use_stdout:
                self._gzip_inplace()

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    def _log_event(self, fields: dict[str, Any]) -> None:
        if self._lock:
            with self._lock:
                self._log_event_unlocked(fields)
        else:
            self._log_event_unlocked(fields)

    def _log_event_unlocked(self, fields: dict[str, Any]) -> None:
        if not self._fh and not self._use_stdout:
            raise RuntimeError("JSONLLogger is closed")

        event: dict[str, Any] = {}
        if self.include_timestamp:
            event["ts"] = iso_now()
        if self._fixed_fields:
            event.update(self._fixed_fields)

        # serialize
        for k, v in fields.items():
            event[k] = to_serializable(v)

        # validate (optional)
        if self._schema is not None and jsonschema is not None:
            try:
                jsonschema.validate(event, self._schema)  # type: ignore
            except Exception as e:
                if self.strict_validation:
                    raise
                event["_schema_error"] = f"{type(e).__name__}: {e}"

        # encode
        if self.pretty:
            line = json.dumps(event, ensure_ascii=False, indent=self.indent)
        else:
            line = json.dumps(event, ensure_ascii=False, separators=(",", ":"))

        # write (with fallback)
        try:
            self._write_line(line)
        except Exception as e:
            if not self.fallback_stdout:
                raise
            self._use_stdout = True
            self._fh = None
            # attach fallback reason once
            self._fixed_fields.setdefault(
                "_fallback", f"stdout: {type(e).__name__}: {e}"
            )
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            return

        # periodic flush
        if self.autoflush:
            self._flush_file()
        elif self.flush_secs > 0:
            now = time.monotonic()
            if now - self._last_flush >= self.flush_secs:
                self._flush_file()

        # rotation
        if self.max_bytes is not None and not self._use_stdout:
            try:
                self._rotate_if_needed()
            except Exception:
                # Non-fatal; keep going.
                pass

    def _open_file(self) -> None:
        try:
            # Append + line buffered; never truncates.
            self._fh = open(
                self.path, "a", buffering=1, encoding="utf-8", errors="backslashreplace"
            )
            self._use_stdout = False
        except Exception as e:
            if not self.fallback_stdout:
                raise
            self._fh = None
            self._use_stdout = True
            self._fixed_fields.setdefault("_fallback", f"stdout: {type(e).__name__}: {e}")

    def _close_file(self) -> None:
        if self._fh:
            try:
                self._fh.flush()
            finally:
                try:
                    self._fh.close()
                finally:
                    self._fh = None

    def _write_line(self, line: str) -> None:
        if self._use_stdout:
            sys.stdout.write(line + "\n")
            sys.stdout.flush()
            return
        if not self._fh:
            raise RuntimeError("JSONLLogger file handle is closed")
        self._fh.write(line + "\n")

    def _flush_file(self) -> None:
        if self._fh:
            self._fh.flush()
        self._last_flush = time.monotonic()
        if self._use_stdout:
            sys.stdout.flush()

    def _file_size(self) -> int:
        try:
            return os.stat(self.path).st_size
        except Exception:
            return 0

    # --- rotation helpers ------------------------------------------------

    @staticmethod
    def _rotated_name(base: Union[str, Path], n: int) -> str:
        """
        Return a rotated sibling filename: 'name.ext.n' or 'name.n' (if no ext).
        We avoid Path.with_suffix() because it fails when there is no suffix.
        """
        p = Path(base)
        s = str(p)
        return f"{s}.{n}"

    def _rotate_if_needed(self) -> None:
        if self.max_bytes is None or self._use_stdout or not self._fh:
            return
        if self._file_size() < self.max_bytes:
            return

        # Close current file before rotation
        try:
            self._fh.flush()
        except Exception:
            pass
        try:
            self._fh.close()
        except Exception:
            pass
        self._fh = None

        base = self.path

        # Shift .(k) -> .(k+1) (reverse order to prevent clobber)
        for idx in range(self._rotate_keep, 0, -1):
            src = self._rotated_name(base, idx)
            dst = self._rotated_name(base, idx + 1)
            if os.path.exists(src):
                try:
                    if not self.safe_mode or (self.safe_mode and not os.path.exists(dst)):
                        os.replace(src, dst)  # atomic on POSIX/Windows
                except Exception:
                    # ignore per-file rotation failure
                    pass

        # Move current -> .1 (only if it exists)
        if os.path.exists(base):
            dst = self._rotated_name(base, 1)
            try:
                if not self.safe_mode or (self.safe_mode and not os.path.exists(dst)):
                    os.replace(base, dst)
            except Exception:
                pass

        # Reopen fresh log file
        self._open_file()

    def _gzip_inplace(self) -> None:
        """
        Gzip the current file (path -> path + '.gz') and remove the original.
        No-op if file missing or already gzipped.
        """
        p = Path(self.path)
        if not p.exists() or p.suffix == ".gz":
            return
        gz_path = str(p) + ".gz"
        try:
            with open(p, "rb") as src, gzip.open(gz_path, "wb") as dst:
                for chunk in iter(lambda: src.read(1024 * 64), b""):
                    dst.write(chunk)
            try:
                p.unlink()  # Py3.8+: no missing_ok for safety
            except FileNotFoundError:
                pass
        except Exception:
            # Non-fatal; leave original in place.
            pass

    # context manager for optional locking
    def _maybe_locked(self):
        class _NullCtx:
            def __enter__(self_s):  # noqa: N805
                return None
            def __exit__(self_s, exc_type, exc, tb):  # noqa: N805
                return False
        return self._lock or _NullCtx()
