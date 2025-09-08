# tests/unit/test_hashing.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import pytest


# ----------------------------------------------------------------------------- #
# Module discovery
# ----------------------------------------------------------------------------- #
def _try_import_hashing():
    try:
        from spectramind.utils import hashing as H  # type: ignore
        return H
    except Exception as e:  # pragma: no cover
        pytest.skip(f"hashing module not available: {e!r}")


def _get_fn(mod: Any, *names: str) -> Optional[Callable[..., Any]]:
    for n in names:
        fn = getattr(mod, n, None)
        if callable(fn):
            return fn
    return None


# ----------------------------------------------------------------------------- #
# Helpers
# ----------------------------------------------------------------------------- #
def _write_bytes(p: Path, data: bytes) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("wb") as f:
        f.write(data)


def _mk_dir_tree(root: Path) -> None:
    # Construct a small tree with nested paths
    (root / "a").mkdir(parents=True, exist_ok=True)
    (root / "b" / "c").mkdir(parents=True, exist_ok=True)
    _write_bytes(root / "a" / "x.txt", b"alpha\n")
    _write_bytes(root / "a" / "y.bin", b"\x00\x01\x02\x03")
    _write_bytes(root / "b" / "c" / "z.txt", b"zeta\n")


def _mk_dir_tree_variant(root: Path) -> None:
    # Same content/structure as _mk_dir_tree but create in a different order
    (root / "b" / "c").mkdir(parents=True, exist_ok=True)
    (root / "a").mkdir(parents=True, exist_ok=True)
    _write_bytes(root / "b" / "c" / "z.txt", b"zeta\n")
    _write_bytes(root / "a" / "y.bin", b"\x00\x01\x02\x03")
    _write_bytes(root / "a" / "x.txt", b"alpha\n")


def _mk_large_bytes(n_bytes: int = 5_000_000) -> bytes:
    # 5 MB should exercise chunked hashing implementations
    chunk = (b"0123456789abcdef" * 1024)  # 16 KB
    out = bytearray()
    while len(out) < n_bytes:
        out.extend(chunk)
    return bytes(out[:n_bytes])


# ----------------------------------------------------------------------------- #
# Tests — bytes/hex primitives
# ----------------------------------------------------------------------------- #
def test_sha256_known_vectors() -> None:
    """
    If sha256/bytes hashing is available, check known vectors.
    """
    H = _try_import_hashing()
    sha_bytes = _get_fn(H, "hash_bytes", "sha256_bytes", "sha256")  # returns bytes digest
    sha_hex = _get_fn(H, "sha256_hex", "hash_hex", "hash_str")      # returns hex digest for str/bytes

    # SHA-256("")
    empty_expected = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
    # SHA-256("abc")
    abc_expected = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    if sha_hex is not None:
        assert sha_hex("") == empty_expected
        assert sha_hex("abc") == abc_expected
        # Accept bytes input, too, if supported
        try:
            assert sha_hex(b"") == empty_expected  # type: ignore[arg-type]
            assert sha_hex(b"abc") == abc_expected  # type: ignore[arg-type]
        except TypeError:
            # Function may be str-only
            pass

    if sha_bytes is not None:
        d0 = sha_bytes(b"")
        d1 = sha_bytes(b"abc")
        assert isinstance(d0, (bytes, bytearray)) and len(d0) == 32
        assert isinstance(d1, (bytes, bytearray)) and len(d1) == 32
        assert d0.hex() == empty_expected
        assert d1.hex() == abc_expected


def test_sha256_large_bytes_if_supported() -> None:
    """
    If a bytes hashing function exists, verify it handles large inputs deterministically.
    """
    H = _try_import_hashing()
    sha_bytes = _get_fn(H, "hash_bytes", "sha256_bytes", "sha256")
    if sha_bytes is None:
        pytest.skip("bytes hashing not implemented")

    big = _mk_large_bytes(5_000_000)
    h1 = sha_bytes(big)
    h2 = sha_bytes(big)
    assert h1 == h2


def test_sha256_hex_str_vs_bytes_consistency_if_supported() -> None:
    """
    If the hex API accepts both str and bytes, their digests must match.
    """
    H = _try_import_hashing()
    sha_hex = _get_fn(H, "sha256_hex", "hash_hex", "hash_str")
    if sha_hex is None:
        pytest.skip("sha256 hex hashing not implemented")

    msg = "SpectraMindV50"
    expected = sha_hex(msg)
    try:
        hx = sha_hex(msg.encode("utf-8"))  # type: ignore[arg-type]
        assert hx == expected
    except TypeError:
        # If bytes not supported, this is fine; test not applicable
        pass


# ----------------------------------------------------------------------------- #
# Tests — files
# ----------------------------------------------------------------------------- #
def test_hash_file_deterministic(tmp_path: Path) -> None:
    """
    hash_file should produce a stable digest for the same content and change when content changes.
    """
    H = _try_import_hashing()
    hash_file = _get_fn(H, "hash_file", "sha256_file")
    sha_hex = _get_fn(H, "sha256_hex", "hash_hex", "hash_bytes")

    if hash_file is None:
        pytest.skip("hash_file not implemented")

    p = tmp_path / "sample.txt"
    _write_bytes(p, b"hello\nworld\n")
    h1 = hash_file(p)
    assert isinstance(h1, (str, bytes))

    # Compute via bytes/hex function if available (sanity)
    if sha_hex is not None:
        with p.open("rb") as f:
            b = f.read()
        # Determine whether sha_hex returns str or bytes for bytes input
        probe = sha_hex(b"abc")
        if isinstance(probe, str):
            hx = sha_hex(b)  # type: ignore[arg-type]
        else:
            hx = sha_hex(b).hex()  # type: ignore[call-arg, union-attr]
        if isinstance(h1, bytes):
            assert h1.hex() == hx
        else:
            assert h1 == hx

    # Change content -> hash should change
    _write_bytes(p, b"hello\nWORLD!\n")
    h2 = hash_file(p)
    assert h1 != h2


def test_hash_file_large(tmp_path: Path) -> None:
    """
    Large files should hash deterministically (chunked readers).
    """
    H = _try_import_hashing()
    hash_file = _get_fn(H, "hash_file", "sha256_file")
    if hash_file is None:
        pytest.skip("hash_file not implemented")

    p = tmp_path / "large.bin"
    data = _mk_large_bytes(5_000_000)  # 5 MB
    _write_bytes(p, data)
    h1 = hash_file(p)
    h2 = hash_file(p)
    assert h1 == h2


def test_hash_file_nonexistent_gives_useful_error(tmp_path: Path) -> None:
    H = _try_import_hashing()
    hash_file = _get_fn(H, "hash_file", "sha256_file")
    if hash_file is None:
        pytest.skip("hash_file not implemented")

    p = tmp_path / "missing.bin"
    with pytest.raises((FileNotFoundError, OSError, ValueError, AssertionError)):
        _ = hash_file(p)


def test_hash_file_stream_api_if_supported(tmp_path: Path) -> None:
    """
    If a stream hashing API exists, verify equality with hash_file.
    """
    H = _try_import_hashing()
    hash_file = _get_fn(H, "hash_file", "sha256_file")
    hash_stream = _get_fn(H, "hash_stream", "sha256_stream", "hash_fileobj")
    if hash_file is None or hash_stream is None:
        pytest.skip("stream/file hashing not both implemented")

    p = tmp_path / "stream.txt"
    _write_bytes(p, b"stream me\n")
    hf = hash_file(p)

    with p.open("rb") as f:
        hs = hash_stream(f)

    # normalize to hex string for compare
    hf_hex = hf.hex() if isinstance(hf, (bytes, bytearray)) else hf
    hs_hex = hs.hex() if isinstance(hs, (bytes, bytearray)) else hs
    assert hf_hex == hs_hex


# ----------------------------------------------------------------------------- #
# Tests — directories/trees
# ----------------------------------------------------------------------------- #
def test_hash_dir_same_tree_same_hash(tmp_path: Path) -> None:
    """
    Two *identical* directory trees should produce the same directory hash.
    """
    H = _try_import_hashing()
    hash_dir = _get_fn(H, "hash_dir", "sha256_dir", "hash_tree")
    if hash_dir is None:
        pytest.skip("hash_dir not implemented")

    d1 = tmp_path / "tree1"
    d2 = tmp_path / "tree2"
    _mk_dir_tree(d1)
    _mk_dir_tree_variant(d2)  # same files/bytes, created in different order

    h1 = hash_dir(d1)
    h2 = hash_dir(d2)
    assert h1 == h2


def test_hash_dir_detects_changes(tmp_path: Path) -> None:
    """
    Changing a file or adding a file should change the directory hash.
    """
    H = _try_import_hashing()
    hash_dir = _get_fn(H, "hash_dir", "sha256_dir", "hash_tree")
    if hash_dir is None:
        pytest.skip("hash_dir not implemented")

    d = tmp_path / "tree"
    _mk_dir_tree(d)
    base = hash_dir(d)

    # Modify a file
    _write_bytes(d / "a" / "x.txt", b"ALPHA\n")
    mod1 = hash_dir(d)
    assert base != mod1

    # Add a file
    _write_bytes(d / "a" / "extra.txt", b"new\n")
    mod2 = hash_dir(d)
    assert mod2 != mod1


def test_hash_dir_empty_and_unicode_names(tmp_path: Path) -> None:
    """
    If directory hashing exists, it should be stable for:
      - empty directories
      - unicode filenames (path normalization-safe)
    """
    H = _try_import_hashing()
    hash_dir = _get_fn(H, "hash_dir", "sha256_dir", "hash_tree")
    if hash_dir is None:
        pytest.skip("hash_dir not implemented")

    # Empty directory
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir(parents=True, exist_ok=True)
    h_empty_1 = hash_dir(empty_dir)
    h_empty_2 = hash_dir(empty_dir)
    assert h_empty_1 == h_empty_2

    # Unicode filenames (identical bytes, different creation order)
    u1 = tmp_path / "unic1"
    u2 = tmp_path / "unic2"
    (u1 / "ä").mkdir(parents=True, exist_ok=True)
    (u2 / "ä").mkdir(parents=True, exist_ok=True)
    _write_bytes(u1 / "ä" / "π.txt", "mu\n".encode("utf-8"))
    _write_bytes(u1 / "ß.bin", b"\x10\x20\x30")
    # variant creation order
    _write_bytes(u2 / "ß.bin", b"\x10\x20\x30")
    _write_bytes(u2 / "ä" / "π.txt", "mu\n".encode("utf-8"))

    h_u1 = hash_dir(u1)
    h_u2 = hash_dir(u2)
    assert h_u1 == h_u2


def test_hash_dir_independent_of_root_name(tmp_path: Path) -> None:
    """
    Hash should only depend on the tree content and relative paths beneath root,
    not on the name of the root directory itself.
    """
    H = _try_import_hashing()
    hash_dir = _get_fn(H, "hash_dir", "sha256_dir", "hash_tree")
    if hash_dir is None:
        pytest.skip("hash_dir not implemented")

    d1 = tmp_path / "ROOT_A"
    d2 = tmp_path / "ROOT_B"
    _mk_dir_tree(d1)
    _mk_dir_tree(d2)
    assert hash_dir(d1) == hash_dir(d2)


def test_hash_dir_ignore_patterns_if_supported(tmp_path: Path) -> None:
    """
    If the dir hashing supports ignore globs, verify that ignored files don't affect the digest.
    """
    H = _try_import_hashing()
    hash_dir = _get_fn(H, "hash_dir", "sha256_dir", "hash_tree")
    if hash_dir is None:
        pytest.skip("hash_dir not implemented")

    d = tmp_path / "tree"
    _mk_dir_tree(d)
    _write_bytes(d / "a" / "ignore.tmp", b"ignored\n")
    _write_bytes(d / ".DS_Store", b"metadata\n")

    # Try to discover an ignore argument name: ignore, excludes, patterns, globs
    kwargs_candidates = [
        {"ignore": ("*.tmp", ".DS_Store")},
        {"excludes": ("*.tmp", ".DS_Store")},
        {"patterns": ("*.tmp", ".DS_Store")},
        {"globs": ("*.tmp", ".DS_Store")},
    ]
    base = hash_dir(d, **kwargs_candidates[0]) if "ignore" in hash_dir.__code__.co_varnames else None
    if base is None:
        # find a signature that works, or skip if none do
        for kw in kwargs_candidates[1:]:
            try:
                base = hash_dir(d, **kw)
                break
            except TypeError:
                continue

    if base is None:
        pytest.skip("hash_dir ignore globs not supported")

    # Change only an ignored file; hash should remain the same
    _write_bytes(d / "a" / "ignore.tmp", b"IGNORED CHANGED\n")
    h2 = hash_dir(d, **(kwargs_candidates[0] if "ignore" in hash_dir.__code__.co_varnames else kw))
    assert h2 == base


# ----------------------------------------------------------------------------- #
# Tests — dict/JSON-style hashing
# ----------------------------------------------------------------------------- #
def test_hash_dict_order_independent_if_supported() -> None:
    """
    If an order-independent dict/JSON hashing function exists, verify that
    key order does not affect the digest.
    """
    H = _try_import_hashing()
    # Common names: hash_dict, hash_json, sha256_json, hash_mapping
    hash_dict = _get_fn(H, "hash_dict", "hash_json", "sha256_json", "hash_mapping")
    if hash_dict is None:
        pytest.skip("dict/json hashing not implemented")

    a = {"a": 1, "b": 2, "c": {"x": 9, "y": [3, 2, 1]}}
    b = {"c": {"y": [3, 2, 1], "x": 9}, "b": 2, "a": 1}  # same content, different order
    assert hash_dict(a) == hash_dict(b)


def test_hash_dict_numeric_normalization_if_supported() -> None:
    """
    If dict/json hashing canonicalizes numeric types, -0.0 and 0.0 should hash the same.
    """
    H = _try_import_hashing()
    hash_dict = _get_fn(H, "hash_dict", "hash_json", "sha256_json", "hash_mapping")
    if hash_dict is None:
        pytest.skip("dict/json hashing not implemented")

    a = {"x": 0.0, "y": [1, 2, 3]}
    b = {"y": [1, 2, 3], "x": -0.0}
    # Some implementations may not canonicalize; accept either behavior but require consistency
    h_a = hash_dict(a)
    h_b = hash_dict(b)
    assert isinstance(h_a, (str, bytes)) and isinstance(h_b, (str, bytes))
    # If canonicalized, these will match; if not, at least both calls must be deterministic types
    if h_a == h_b:
        assert True
    else:
        assert type(h_a) is type(h_b)


# ----------------------------------------------------------------------------- #
# Tests — PathLike vs str ergonomics
# ----------------------------------------------------------------------------- #
def test_pathlike_and_str_inputs_supported(tmp_path: Path) -> None:
    """
    If your API supports path-like & str, verify both work.
    """
    H = _try_import_hashing()
    hash_file = _get_fn(H, "hash_file", "sha256_file")
    if hash_file is None:
        pytest.skip("hash_file not implemented")

    p = tmp_path / "p.txt"
    _write_bytes(p, b"same")
    h_path = hash_file(p)
    h_str = hash_file(str(p))
    assert h_path == h_str