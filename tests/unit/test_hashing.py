# tests/unit/test_hashing.py
from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Optional

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
# Tests
# ----------------------------------------------------------------------------- #
def test_sha256_known_vectors() -> None:
    """
    If sha256/bytes hashing is available, check known vectors.
    """
    H = _try_import_hashing()
    sha_bytes = _get_fn(H, "hash_bytes", "sha256_bytes", "sha256")  # returns bytes digest
    sha_hex = _get_fn(H, "sha256_hex", "hash_hex", "hash_str")      # returns hex digest for str/bytes

    # SHA-256("abc")
    expected_hex = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"

    if sha_hex is not None:
        assert sha_hex("abc") == expected_hex
        # Accept bytes input, too, if supported
        try:
            assert sha_hex(b"abc") == expected_hex  # type: ignore[arg-type]
        except TypeError:
            pass

    if sha_bytes is not None:
        digest = sha_bytes(b"abc")
        assert isinstance(digest, (bytes, bytearray)) and len(digest) == 32
        assert digest.hex() == expected_hex


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

    # Compute via bytes function if available (sanity)
    if sha_hex is not None:
        with p.open("rb") as f:
            b = f.read()
        hx = sha_hex(b) if isinstance(sha_hex(b"abc"), str) else sha_hex(b).hex()  # type: ignore[call-arg]
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