from __future__ import annotations

"""
SpectraMind V50 — Cryptographic & Canonical Hash Utilities

Implements:
  • sha256_bytes(data: bytes|bytearray|memoryview) -> bytes
  • sha256_hex(data: bytes|str) -> str
  • hash_bytes == sha256_bytes
  • hash_hex / hash_str == sha256_hex
  • hash_stream(fileobj, *, chunk_size=1024*1024) -> bytes
  • hash_file(pathlike, *, as_bytes: bool = False, chunk_size=1024*1024) -> str|bytes
  • sha256_file == hash_file
  • hash_dir(pathlike, *, ignore: tuple[str,...]|list[str]=(), excludes=None, patterns=None,
             globs=None, follow_symlinks: bool = False, as_bytes: bool = False) -> str|bytes
     (aliases: sha256_dir, hash_tree)
  • hash_dict(obj) / hash_json(obj) / sha256_json(obj) / hash_mapping(obj) -> str
    (canonicalizes: key order, tuple->list, -0.0→0.0, bytes→base64 strings)

Design goals:
  • Deterministic across OSes and filesystems
  • Ignores mtime/permissions by default for directory hashing
  • Path order independence via explicit sorting of relative paths
  • Flexible ignore globs (fnmatch), including directory globs like "logs/**"
  • No external deps; pure stdlib
"""

from dataclasses import is_dataclass, asdict
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping, Sequence, Union, Optional

import base64
import fnmatch
import hashlib
import io
import json
import os
import stat

# ---------------------------------------------------------------------------
# Public API surface
# ---------------------------------------------------------------------------

__all__ = [
    "sha256_bytes",
    "sha256_hex",
    "hash_bytes",
    "hash_hex",
    "hash_str",
    "hash_stream",
    "hash_file",
    "sha256_file",
    "hash_dir",
    "sha256_dir",
    "hash_tree",
    "hash_dict",
    "hash_json",
    "sha256_json",
    "hash_mapping",
]

# ---------------------------------------------------------------------------
# Core SHA-256 primitives
# ---------------------------------------------------------------------------

def _to_bytes(data: Union[bytes, bytearray, memoryview, str]) -> bytes:
    if isinstance(data, (bytes, bytearray, memoryview)):
        return bytes(data)
    if isinstance(data, str):
        return data.encode("utf-8")
    raise TypeError(f"Unsupported data type for hashing: {type(data).__name__}")

def sha256_bytes(data: Union[bytes, bytearray, memoryview, str]) -> bytes:
    """Return raw 32-byte SHA-256 digest for data (str encoded as UTF-8)."""
    return hashlib.sha256(_to_bytes(data)).digest()

def sha256_hex(data: Union[bytes, bytearray, memoryview, str]) -> str:
    """Return lowercase hex SHA-256 digest for data (str encoded as UTF-8)."""
    return hashlib.sha256(_to_bytes(data)).hexdigest()

# ergonomic aliases
hash_bytes = sha256_bytes
hash_hex = sha256_hex
hash_str = sha256_hex

# ---------------------------------------------------------------------------
# Stream / file hashing
# ---------------------------------------------------------------------------

def hash_stream(fileobj: io.BufferedIOBase, *, chunk_size: int = 1_048_576) -> bytes:
    """
    Hash a readable binary file-like object from current position to EOF.
    Returns raw 32-byte digest.
    """
    if not (hasattr(fileobj, "read") and callable(fileobj.read)):
        raise TypeError("hash_stream expects a readable binary file-like object")
    h = hashlib.sha256()
    while True:
        chunk = fileobj.read(chunk_size)
        if not chunk:
            break
        if not isinstance(chunk, (bytes, bytearray, memoryview)):
            raise TypeError("hash_stream received non-bytes chunk from file object")
        h.update(chunk)
    return h.digest()

def _coerce_path(p: Union[str, os.PathLike]) -> Path:
    return p if isinstance(p, Path) else Path(p)

def hash_file(path: Union[str, os.PathLike], *, as_bytes: bool = False, chunk_size: int = 1_048_576) -> Union[str, bytes]:
    """
    Hash a file by content only. Returns hex str by default, or raw bytes if as_bytes=True.
    Raises on missing paths or directories.
    """
    p = _coerce_path(path)
    if not p.exists():
        raise FileNotFoundError(f"No such file: {p}")
    if p.is_dir():
        raise IsADirectoryError(f"Path is a directory, not a file: {p}")
    with p.open("rb") as f:
        dig = hash_stream(f, chunk_size=chunk_size)
    return dig if as_bytes else dig.hex()

# alias
sha256_file = hash_file

# ---------------------------------------------------------------------------
# Directory hashing (content + relative paths, independent of root name)
# ---------------------------------------------------------------------------

def _iter_files_sorted(root: Path, *, follow_symlinks: bool) -> Iterator[Path]:
    """
    Yield all *files* under root as sorted relative paths (Posix style),
    honoring follow_symlinks flag.
    """
    # Collect candidates in a list first for deterministic sort
    rels: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=follow_symlinks):
        # Normalize order for determinism regardless of FS or creation order
        dirnames.sort()
        filenames.sort()
        d = Path(dirpath)
        for fn in filenames:
            rels.append((d / fn).relative_to(root))
    # Sort by POSIX path string
    for rel in sorted(rels, key=lambda p: p.as_posix()):
        yield rel

def _match_any_glob(path: Path, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    s = path.as_posix()
    return any(fnmatch.fnmatch(s, pat) for pat in patterns)

def _normalize_ignore_kwargs(
    ignore: Optional[Sequence[str]] = None,
    *,
    excludes: Optional[Sequence[str]] = None,
    patterns: Optional[Sequence[str]] = None,
    globs: Optional[Sequence[str]] = None,
) -> tuple[str, ...]:
    # Accept multiple keyword names; merge unique patterns
    pats: list[str] = []
    for seq in (ignore, excludes, patterns, globs):
        if not seq:
            continue
        pats.extend([str(x) for x in seq])
    # Deduplicate while preserving order
    seen: set[str] = set()
    out: list[str] = []
    for p in pats:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return tuple(out)

def hash_dir(
    path: Union[str, os.PathLike],
    *,
    ignore: Optional[Sequence[str]] = None,
    excludes: Optional[Sequence[str]] = None,
    patterns: Optional[Sequence[str]] = None,
    globs: Optional[Sequence[str]] = None,
    follow_symlinks: bool = False,
    as_bytes: bool = False,
) -> Union[str, bytes]:
    """
    Compute a deterministic hash of a directory tree.

    • Depends on file contents and each file's *relative path* under the root.
    • Ignores directory mtime/permissions; file mode bits do not affect digest.
    • `ignore`/`excludes`/`patterns`/`globs`: fnmatch patterns on POSIX-style
      relative paths, e.g., "*.tmp", ".DS_Store", "logs/**".
    • `follow_symlinks`: if True, includes the target file content where os.walk follows links.

    Returns hex string by default, or raw bytes if as_bytes=True.
    """
    root = _coerce_path(path)
    if not root.exists():
        raise FileNotFoundError(f"No such directory: {root}")
    if not root.is_dir():
        raise NotADirectoryError(f"Path is not a directory: {root}")

    patt = _normalize_ignore_kwargs(ignore, excludes=excludes, patterns=patterns, globs=globs)

    h = hashlib.sha256()

    # Incorporate a small header to lock hashing scheme version
    h.update(b"SM-V50-DIRHASH-V1\0")

    for rel in _iter_files_sorted(root, follow_symlinks=follow_symlinks):
        if _match_any_glob(rel, patt):
            continue
        file_path = root / rel
        # incorporate relative path (POSIX) + content length for extra safety
        rel_posix = rel.as_posix().encode("utf-8")
        h.update(b"P\0"); h.update(rel_posix); h.update(b"\0")
        try:
            size = file_path.stat().st_size
        except FileNotFoundError:
            # File vanished between walk and stat; skip deterministically
            continue
        h.update(str(size).encode("ascii"))
        h.update(b"\0C\0")
        # content hash stream
        with file_path.open("rb") as f:
            while True:
                chunk = f.read(1_048_576)
                if not chunk:
                    break
                h.update(chunk)

    dig = h.digest()
    return dig if as_bytes else dig.hex()

# aliases
sha256_dir = hash_dir
hash_tree = hash_dir

# ---------------------------------------------------------------------------
# Canonical dict/JSON hashing
# ---------------------------------------------------------------------------

def _canonicalize(obj: Any) -> Any:
    """
    Canonicalize arbitrary Python objects to a JSON-serializable form:

      • dataclasses -> dict
      • Path -> str (POSIX)
      • bytes/bytearray/memoryview -> base64 string with type tag
      • tuples -> lists
      • sets/frozensets -> sorted lists
      • floats: normalize -0.0 → 0.0
      • mappings: sort keys (recursively canonicalize values)
    """
    # dataclass
    if is_dataclass(obj):
        obj = asdict(obj)

    # basic types straight through
    if obj is None or isinstance(obj, (bool, int, str)):
        return obj

    # float normalization (-0.0 -> 0.0)
    if isinstance(obj, float):
        return 0.0 if obj == 0.0 else obj

    # path -> str
    if isinstance(obj, (os.PathLike, Path)):
        return Path(obj).as_posix()

    # bytes-like -> base64 string with tag to avoid collisions vs text
    if isinstance(obj, (bytes, bytearray, memoryview)):
        return {"__bytes__": base64.b64encode(bytes(obj)).decode("ascii")}

    # mapping
    if isinstance(obj, Mapping):
        # sort keys for determinism; canonicalize recursively
        items = []
        for k in sorted(obj.keys(), key=lambda x: str(x)):
            items.append((str(k), _canonicalize(obj[k])))
        return {"__dict__": items}

    # sequence types -> list
    if isinstance(obj, (list, tuple)):
        return [_canonicalize(x) for x in obj]

    # sets -> sorted list
    if isinstance(obj, (set, frozenset)):
        return sorted((_canonicalize(x) for x in obj), key=lambda v: json.dumps(v, separators=(",", ":"), sort_keys=True))

    # fallback to string repr (stable for common types)
    return str(obj)

def _json_dumps_canonical(obj: Any) -> str:
    canon = _canonicalize(obj)
    # separators minimize whitespace; ensure ascii for stability
    return json.dumps(canon, ensure_ascii=True, sort_keys=True, separators=(",", ":"))

def hash_dict(obj: Any) -> str:
    """Hash a mapping/JSON-like object canonically; returns lowercase hex digest."""
    return sha256_hex(_json_dumps_canonical(obj))

# ergonomic aliases
hash_json = hash_dict
sha256_json = hash_dict
hash_mapping = hash_dict
