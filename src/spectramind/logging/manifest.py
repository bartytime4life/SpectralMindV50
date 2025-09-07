# src/spectramind/logging/manifest.py
from __future__ import annotations

import hashlib
import json
import os
import platform
import socket
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

try:
    import jsonschema
    _HAS_JSONSCHEMA = True
except Exception:
    _HAS_JSONSCHEMA = False


# -------------------------------------------------------------------
# Data model
# -------------------------------------------------------------------

@dataclass(slots=True, frozen=True)
class Manifest:
    """Structured run manifest for SpectraMind V50 runs."""

    run_id: str
    stage: str
    timestamp: str
    user: str
    host: str
    python: str
    argv: list[str]
    cwd: str
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    git_commit: Optional[str] = None
    dvc_rev: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self, indent: Optional[int] = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, sort_keys=True)


# -------------------------------------------------------------------
# Factory
# -------------------------------------------------------------------

def generate_manifest(
    *,
    stage: str,
    config_snapshot: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Manifest:
    """Generate a run manifest from runtime context + config snapshot."""
    now = datetime.now(timezone.utc).isoformat()

    uid = run_id or hashlib.sha256(f"{stage}-{now}".encode()).hexdigest()[:12]

    # System context
    user = os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    host = socket.gethostname()
    python = sys.version.replace("\n", " ")
    argv = sys.argv
    cwd = str(Path.cwd())

    # Git & DVC info (best-effort)
    git_commit = _get_git_commit()
    dvc_rev = _get_dvc_rev()

    return Manifest(
        run_id=uid,
        stage=stage,
        timestamp=now,
        user=user,
        host=host,
        python=python,
        argv=argv,
        cwd=cwd,
        config_snapshot=config_snapshot or {},
        git_commit=git_commit,
        dvc_rev=dvc_rev,
        extra=extra or {},
    )


def _get_git_commit() -> Optional[str]:
    try:
        import subprocess
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True)
            .strip()
        )
    except Exception:
        return None


def _get_dvc_rev() -> Optional[str]:
    try:
        import subprocess
        return (
            subprocess.check_output(["dvc", "rev-parse"], text=True)
            .strip()
        )
    except Exception:
        return None


# -------------------------------------------------------------------
# Persistence & validation
# -------------------------------------------------------------------

def save_manifest(manifest: Manifest, path: Path) -> None:
    """Write manifest JSON file to disk (atomic)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(manifest.to_json())
    tmp.replace(path)


def validate_manifest(manifest: Manifest, schema_path: Optional[Path] = None) -> None:
    """Validate manifest against JSON schema if available."""
    if not _HAS_JSONSCHEMA:
        return
    schema = None
    if schema_path and schema_path.exists():
        schema = json.loads(schema_path.read_text())
    if schema:
        jsonschema.validate(instance=manifest.to_dict(), schema=schema)
