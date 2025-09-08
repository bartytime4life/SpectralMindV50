from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Tuple

import pytest


# ------------------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------------------
def _kaggle_env_markers() -> list[str]:
    return [
        "KAGGLE_KERNEL_RUN_TYPE",
        "KAGGLE_CONTAINER_NAME",
        "KAGGLE_URL_BASE",
        "KAGGLE_DATA_PROXY_TOKEN",  # sometimes present
    ]


def _is_kaggle() -> bool:
    """
    Heuristic: Kaggle mounts /kaggle tree and sets one or more env markers.
    Both checks are useful to avoid false positives in local CI containers that mirror the FS.
    """
    dirs_ok = Path("/kaggle").exists() and Path("/kaggle/working").exists()
    env_ok = any(os.environ.get(k) for k in _kaggle_env_markers())
    return dirs_ok and env_ok


def _try_write(p: Path, data: bytes = b"ok") -> tuple[bool, str]:
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("wb") as f:
            f.write(data)
        return True, "wrote"
    except Exception as e:  # noqa: BLE001 - we want the exact exception string
        return False, f"{type(e).__name__}: {e}"


def _tcp_connect_blocked(host: str, port: int, timeout: float = 2.0) -> bool:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(timeout)
    try:
        try:
            s.connect((host, port))
            return False  # connected: not blocked
        except Exception:
            return True
    finally:
        try:
            s.close()
        except Exception:
            pass


def _udp_send_blocked(host: str, port: int, timeout: float = 2.0) -> bool:
    """
    UDP "connect" doesn't handshake; we emulate an egress check by send+recv with timeout.
    In a no-internet environment, we expect timeouts or network-level errors.

    Return True if we see clear evidence of egress being blocked (timeout or socket error).
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    s.settimeout(timeout)
    try:
        try:
            s.sendto(b"\x00", (host, port))
            # Most environments won't route/rsp; we expect timeout here
            s.recvfrom(1)
            return False  # got a response; egress not blocked
        except Exception:
            return True
    finally:
        try:
            s.close()
        except Exception:
            pass


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
    assert Path("/kaggle").exists(), "/kaggle root missing"
    assert Path("/kaggle/input").exists(), "/kaggle/input missing"
    assert Path("/kaggle/working").exists(), "/kaggle/working missing"
    assert Path("/kaggle/temp").exists(), "/kaggle/temp missing"


def test_cwd_is_working_dir() -> None:
    """
    By convention, Kaggle kernels cd into /kaggle/working.
    """
    assert Path.cwd().resolve() == Path("/kaggle/working"), f"cwd is not /kaggle/working: {Path.cwd()}"


def test_readonly_input_writable_working_temp() -> None:
    """
    /kaggle/input should be read-only; working and temp should be writable.
    We check both os.access + a real write attempt for /kaggle/input.
    """
    # RO check via permission bit + write attempt
    ro_root = Path("/kaggle/input")
    assert not os.access(ro_root, os.W_OK), "/kaggle/input appears writable via os.access"
    ok_ro, msg_ro = _try_write(ro_root / "__guardrails_test_ro.txt")
    assert not ok_ro, f"/kaggle/input unexpectedly writable: {msg_ro}"

    # RW checks
    ok_wrk, msg_wrk = _try_write(Path("/kaggle/working/__guardrails_test_wrk.txt"))
    assert ok_wrk, f"/kaggle/working not writable: {msg_wrk}"

    ok_tmp, msg_tmp = _try_write(Path("/kaggle/temp/__guardrails_test_tmp.txt"))
    assert ok_tmp, f"/kaggle/temp not writable: {msg_tmp}"


# ------------------------------------------------------------------------------
# Network egress is blocked in Kaggle kernels
# ------------------------------------------------------------------------------
def test_network_blocked_via_tcp_and_udp() -> None:
    """
    Kaggle disables outbound internet. Both raw TCP and UDP attempts should fail or timeout quickly.
    We probe a couple of well-known endpoints.
    """
    # Public DNS over TCP/UDP
    blocked_tcp = any(
        _tcp_connect_blocked(host, 53)
        for host in ("8.8.8.8", "1.1.1.1")
    )
    blocked_udp = any(
        _udp_send_blocked(host, 53)
        for host in ("8.8.8.8", "1.1.1.1")
    )
    assert blocked_tcp, "TCP egress unexpectedly allowed (DNS TCP connect succeeded)"
    assert blocked_udp, "UDP egress unexpectedly allowed (DNS UDP exchange succeeded)"


def test_network_blocked_via_http_lib() -> None:
    """
    A higher-level HTTP request should also fail due to no internet in Kaggle.
    This is a "nice to have": we xfail rather than fail if urllib is missing or behaves differently.
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
    markers = _kaggle_env_markers()
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
