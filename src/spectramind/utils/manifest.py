from __future__ import annotations
import json, os, subprocess, sys, time
from pathlib import Path

def _sh(cmd: str) -> str:
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except Exception:
        return ""

def write_run_manifest(path: str | Path, extra: dict | None = None) -> None:
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "run_id": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "env": {
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "nvidia_smi": _sh("nvidia-smi | sed -n '1,10p'"),
            "cuda_version_txt": _sh("cat /usr/local/cuda/version.txt 2>/dev/null || true"),
        },
    }
    if extra:
        payload.update(extra)
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
