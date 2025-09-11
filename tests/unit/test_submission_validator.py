from __future__ import annotations
import os
import shlex
import subprocess
from pathlib import Path
import pytest

class _Result:
    def __init__(self, exit_code: int, stdout: str, stderr: str):
        self.exit_code = exit_code
        self.stdout = stdout
        self.stderr = stderr

class _CliRunner:
    """
    Lightweight subprocess-based CLI runner.
    Tries multiple launchers for SpectraMind's CLI:
      • spectramind <args>
      • python -m spectramind <args>
      • python -m spectramind.submit <args>
      • spectramind-submit <args>
    Returns _Result(exit_code, stdout, stderr).
    """
    def __init__(self, cwd: Path | None = None, env: dict | None = None):
        self.cwd = cwd
        self.env = {**os.environ, **(env or {})}

    def invoke(self, args: list[str]) -> _Result:
        candidates = [
            ["spectramind", *args],
            ["python", "-m", "spectramind", *args],
            ["python", "-m", "spectramind.submit", *args],
            ["spectramind-submit", *args],
        ]
        last_rc, last_out, last_err = 127, "", "No such command"
        for cmd in candidates:
            try:
                proc = subprocess.run(
                    cmd,
                    cwd=self.cwd,
                    env=self.env,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                return _Result(proc.returncode, proc.stdout, proc.stderr)
            except FileNotFoundError as e:
                last_err = str(e)
                continue
        return _Result(last_rc, last_out, last_err)

@pytest.fixture(scope="session")
def cli_runner(tmp_path_factory: pytest.TempPathFactory) -> _CliRunner:
    return _CliRunner(cwd=tmp_path_factory.mktemp("cli"))
