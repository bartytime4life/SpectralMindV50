from __future__ import annotations

import gzip
import io
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Optional, Union

from .utils import iso_now, safe_mkdir, to_serializable

try:  # optional; only used if schema validation is enabled
    import jsonschema  # type: ignore
except Exception:  # pragma: no cover
    jsonschema = None


class JSONLLogger:
    """
    Append-only JSON Lines logger. Each event is a single JSON object per line.
    - Safe for Kaggle/offline CI (append-only; no truncation; small buffers).
    - Supports size-based rotation, optional schema validation, fallback to stdout,
      periodic flush, and gzip-on-close.

    Parameters
    ----------
    path : str
        Destination file path (created if missing). Opened in append mode.
    autoflush : bool
        If True, flush on each write; otherwise use flush_secs cadence.
    flush_secs : float
        If > 0, force a flush when this many seconds have elapsed since last flush.
    pretty : bool
        If True, pretty-print JSON (larger files; not recommended for competition).
    indent : int
        Indentation to use if pretty=True.
    include_timestamp : bool
        If True, include ISO timestamp field "ts" on each event.
    include_config_hash : Optional[str]
        If set, include a "config_hash" attribute on each event.
    include_git_commit : Optional[str]
        If set, include a "git_commit" attribute on each event.
        (Prefer passing env-provided short SHA to avoid `git` calls in offline envs.)
    schema_path : Optional[Union[str, os.PathLike]]
        Path to a JSON schema for event validation. If provided and jsonschema is
        available, each record is validated before write.
    strict_validation : bool
        If True, raise on schema errors; if False, attach "_schema_error" and continue.
    safe_mode : bool
        If True, disables risky operations (e.g., overwrite/truncate). We always append,
        but safe_mode also guards rotation rename races on exotic FS.
    fallback_stdout : bool
        If True, on write/open errors the logger falls back to stdout.
    max_file_size_mb : Optional[int]
        If set, rotate when file exceeds this size. Rotation scheme: `.1`, `.2`, `.3`.
    compress_on_close : bool
        If True, gzip the final file on close (keeps `.gz` and removes original).
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

        # rotation
        self.max_bytes = None
        if max_file_size_mb is not None:
            self.max_bytes = int(max(1, max_file_size_mb)) * 1024 * 1024
        self._rotate_keep = 3

        # schema (optional)
        self._schema: Optional[dict[str, Any]] = None
        if schema_path is not None:
            try:
                with open(schema_path, "r", encoding="utf-8") as fh:
                    self._schema = json.load(fh)
            except Exception as e:  # pragma: no cover
                if strict_validation:
                    raise
                else:
                    # Non-strict: proceed without schema but mark the reason
                    self._schema = None
                    self._fixed_fields["_schema_load_error"] = str(e)

        # open file
        safe_mkdir(os.path.dirname(self.path) or ".")
        self._fh: Optional[io.TextIOWrapper] = None
        self._last_flush = time.monotonic()
        self._use_stdout = False
        self._open_file()

    # ------------------------------------------------------------------ #
    # public API
    # ------------------------------------------------------------------ #

    def log(self, **fields: Any) -> None:
        """
        Write one JSONL event. Serializes all values safely; adds audit fields and timestamp.
        Performs optional schema validation and rotation checks.
        """
        if not self._fh and not self._use_stdout:
            raise RuntimeError("JSONLLogger is closed")

        event: dict[str, Any] = {}
        if self.include_timestamp:
            event["ts"] = iso_now()

        # audit fields (constant per-run)
        if self._fixed_fields:
            event.update(self._fixed_fields)

        for k, v in fields.items():
            event[k] = to_serializable(v)

        # validate (optional)
        if self._schema is not None and jsonschema is not None:
            try:
                jsonschema.validate(event, self._schema)  # type: ignore
            except Exception as e:
                if self.strict_validation:
                    raise
                else:
                    # record the error inline; do not drop the event
                    event["_schema_error"] = str(e)

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
            # Fallback to stdout and mark the failure reason
            self._use_stdout = True
            self._fh = None
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

        # rotate if needed
        if self.max_bytes is not None and not self._use_stdout:
            try:
                self._rotate_if_needed()
            except Exception:
                # Rotation failures should not kill training; best-effort.
                pass

    def close(self) -> None:
        """Flush, optionally compress, and close."""
        if self._fh:
            try:
                self._fh.flush()
            finally:
                self._fh.close()
            self._fh = None

        if self.compress_on_close and not self._use_stdout:
            self._gzip_inplace()

    def __enter__(self) -> "JSONLLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # ------------------------------------------------------------------ #
    # internal helpers
    # ------------------------------------------------------------------ #

    def _open_file(self) -> None:
        try:
            # Line-buffered text mode; append only.
            self._fh = open(self.path, "a", buffering=1, encoding="utf-8")
            self._use_stdout = False
        except Exception as e:
            if not self.fallback_stdout:
                raise
            # Cannot open file (e.g., read-only FS). Fall back to stdout.
            self._fh = None
            self._use_stdout = True
            # Attach reason once (as a fixed field)
            self._fixed_fields.setdefault("_fallback", f"stdout: {type(e).__name__}: {e}")

    def _write_line(self, line: str) -> None:
        if self._use_stdout:
            sys.stdout.write(line + "\n")
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
            return Path(self.path).stat().st_size
        except Exception:
            return 0

    def _rotate_if_needed(self) -> None:
        if self.max_bytes is None or self._use_stdout or not self._fh:
            return
        size = self._file_size()
        if size < self.max_bytes:
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

        base = Path(self.path)
        # Guard rotations in "safe_mode" (best effort to avoid destructive ops)
        try:
            # Shift .2 -> .3, .1 -> .2, current -> .1
            for idx in range(self._rotate_keep, 0, -1):
                src = base.with_suffix(base.suffix + f".{idx}")
                dst = base.with_suffix(base.suffix + f".{idx+1}")
                if src.exists():
                    try:
                        if not self.safe_mode or (self.safe_mode and not dst.exists()):
                            src.replace(dst)
                    except Exception:
                        # ignore per-file rotation failure
                        pass
            # Move current to .1
            if base.exists():
                dst = base.with_suffix(base.suffix + ".1")
                if not self.safe_mode or (self.safe_mode and not dst.exists()):
                    base.replace(dst)
        finally:
            # Reopen a fresh file for continued logging
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
            p.unlink(missing_ok=True)
        except Exception:
            # Non-fatal; leave original in place.
            pass
