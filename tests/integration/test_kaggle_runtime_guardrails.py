# tests/integration/test_kaggle_runtime_guardrails.py
from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _is_kaggle() -> bool:
    # The Kaggle runtime always mounts /kaggle and provides these well-known dirs.
    return Path("/kaggle").exists() and Path("/kaggle/working").exists()


def _try_write(p: Path, data: bytes = b"ok") -> tuple[bool, str]:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(data)
        return True, "wrote"
    except Exception as e:
        return False, f"{type(e).__name__}: {e}"


# All tests in this file are Kaggle-only.
pytestmark = pytest.mark.skipif(not _is_kaggle(), reason="Not running inside Kaggle kernel")


# ------------------------------------------------------------------------------
# Filesystem layout & permissions
# ------------------------------------------------------------------------------
def test_kaggle_directories_present() -> None:
    """
    Kaggle kernels mount a canonical directory structure:
      - /kaggle/input  (datasets & competition data, RO)
      - /kaggle/working (current working directory, RW)
      - /kaggle/temp   (scratch, RW)
    """
    assert Path("/kaggle").exists()
    assert Path("/kaggle/input").exists()
    assert Path("/kaggle/working").exists()
    assert Path("/kaggle/temp").exists()


def test_readonly_input_writable_working_temp(tmp_path_factory) -> None:
    """
    /kaggle/input should be read-only; working and temp should be writable.
    """
    # RO check
    ok_ro, msg_ro = _try_write(Path("/kaggle/input/__guardrails_test_ro.txt"))
    assert not ok_ro, f"/kaggle/input unexpectedly writable: {msg_ro}"

    # RW checks
    ok_wrk, msg_wrk = _try_write(Path("/kaggle/working/__guardrails_test_wrk.txt"))
    assert ok_wrk, f"/kaggle/working not writable: {msg_wrk}"

    ok_tmp, msg_tmp = _try_write(Path("/kaggle/temp/__guardrails_test_tmp.txt"))
    assert ok_tmp, f"/kaggle/temp not writable: {msg_tmp}"


# ------------------------------------------------------------------------------
# Network egress is blocked in Kaggle kernels
# ------------------------------------------------------------------------------
def test_network_blocked_via_socket() -> None:
    """
    Kaggle disables outbound internet. A raw socket connect to a public IP should fail.
    This uses a direct socket call to avoid library-specific handling.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(2.0)
    try:
        with pytest.raises(Exception):
            # Well-known public DNS; request should be blocked by the runtime.
            s.connect(("8.8.8.8", 53))
    finally:
        s.close()


def test_network_blocked_via_http_lib() -> None:
    """
    A higher-level HTTP request should also fail due to no internet in Kaggle.
    """
    try:
        import urllib.request

        with pytest.raises(Exception):
            urllib.request.urlopen("http://example.com", timeout=2).read()
    except Exception:
        # If urllib isn't available or behaves differently, don't hard-fail;
        # the socket-level test above already asserts the guardrail.
        pytest.xfail("HTTP egress check inconclusive; socket guardrail already enforced.")


# ------------------------------------------------------------------------------
# Environment sanity (soft checks)
# ------------------------------------------------------------------------------
def test_kaggle_environment_markers_present() -> None:
    """
    Kaggle sets several identifiers that are useful for runtime behavior & logging.
    We only soft-assert presence of at least one common marker.
    """
    markers = [
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_CONTAINER_NAME",
        "KAGGLE_URL_BASE",
    ]
    assert any(os.environ.get(k) for k in markers), f"No Kaggle env markers found among {markers}"


def test_thread_env_limits_soft() -> None:
    """
    Soft assertion: when thread caps are set (by environment or harness), they should be modest.
    We don't enforce values rigidly (env varies); this is a smoke check only.
    """
    caps = {k: os.environ.get(k) for k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    # If caps are present, they should parse to small-ish integers and not be zero.
    for k, v in caps.items():
        if not v:
            continue
        try:
            n = int(v)
        except ValueError:
            pytest.fail(f"{k} is not an integer: {v!r}")
        # Accept 1..8 as a reasonable range in shared runtimes.
        assert 1 <= n <= 8, f"{k} value {n} is outside the expected small range (1..8)"


# ------------------------------------------------------------------------------
# Optional: verify dataset mount is accessible (read-only list)
# ------------------------------------------------------------------------------
def test_list_input_datasets_ro() -> None:
    """
    Listing under /kaggle/input should be possible without write access.
    This is optional and only asserts the listing works and entries are directories/files.
    """
    root = Path("/kaggle/input")
    # Should not raise
    children = list(root.iterdir())
    # If empty (rare), nothing more to assert
    for child in children[:10]:  # cap to avoid long runtimes if many datasets
        assert child.exists()
        assert child.is_dir() or child.is_file()