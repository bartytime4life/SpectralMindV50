from __future__ import annotations
import json, os, subprocess, sys, time, hashlib
from pathlib import Path
from typing import Any, Dict, Optional


def _sh(cmd: str) -> str:
    """Run a shell command and return stdout (strip), or empty string on failure."""
    try:
        return subprocess.check_output(
            cmd, shell=True, stderr=subprocess.STDOUT, text=True
        ).strip()
    except Exception:
        return ""


def _git_info() -> Dict[str, Any]:
    return {
        "commit": _sh("git rev-parse HEAD"),
        "branch": _sh("git rev-parse --abbrev-ref HEAD"),
        "remote": _sh("git remote get-url origin"),
        "dirty": bool(_sh("git status --porcelain")),
        "tag": _sh("git describe --tags --always"),
    }


def _dvc_info() -> Dict[str, Any]:
    try:
        import dvc  # type: ignore
        version = getattr(dvc, "__version__", "")
    except Exception:
        version = ""
    return {
        "version": version,
        "status": _sh("dvc status -q || true"),
        "remote_list": _sh("dvc remote list || true"),
    }


def _torch_info() -> Dict[str, Any]:
    try:
        import torch  # type: ignore

        # Device inventory (resilient to CPU-only)
        cuda_ok = bool(getattr(torch, "cuda", None) and torch.cuda.is_available())
        ndev = torch.cuda.device_count() if cuda_ok else 0
        devices = []
        for i in range(ndev):
            try:
                props = torch.cuda.get_device_properties(i)
                devices.append(
                    {
                        "index": i,
                        "name": props.name,
                        "capability": f"{props.major}.{props.minor}",
                        "total_memory_bytes": getattr(props, "total_memory", None),
                    }
                )
            except Exception:
                devices.append({"index": i})

        return {
            "version": getattr(torch, "__version__", ""),
            "cuda": getattr(getattr(torch, "version", None), "cuda", None) or ("cpu" if not cuda_ok else "cuda"),
            "cudnn": (
                torch.backends.cudnn.version() if getattr(torch.backends, "cudnn", None) and torch.backends.cudnn.is_available() else None
            ),
            "cuda_available": cuda_ok,
            "device_count": ndev,
            "devices": devices,
            "determinism": {
                "torch.use_deterministic_algorithms": True,  # our policy; can't introspect directly
                "cudnn.deterministic": bool(getattr(getattr(torch.backends, "cudnn", None), "deterministic", False)),
                "cudnn.benchmark": bool(getattr(getattr(torch.backends, "cudnn", None), "benchmark", False)),
                "float32_matmul_precision": getattr(torch, "get_float32_matmul_precision", lambda: None)(),
                "env": {
                    "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED"),
                    "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                    "NVIDIA_TF32_OVERRIDE": os.environ.get("NVIDIA_TF32_OVERRIDE"),
                },
            },
        }
    except Exception:
        return {}


def _nvidia_runtime_info() -> Dict[str, Any]:
    # Keep calls short to avoid slowdowns in restricted envs
    return {
        "smi_header": _sh("nvidia-smi | sed -n '1,10p'"),
        "smi_devices": _sh("nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader || true"),
        "cuda_version_txt": _sh("cat /usr/local/cuda/version.txt 2>/dev/null || true"),
        "cuda_version_json": _sh("cat /usr/local/cuda/version.json 2>/dev/null || true"),
        "driver": _sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || true"),
    }


def _container_env_info() -> Dict[str, Any]:
    # Useful hints for reproducibility
    return {
        "hostname": _sh("hostname"),
        "whoami": _sh("whoami || true"),
        "workdir": os.getcwd(),
        "container_image": os.environ.get("CONTAINER_IMAGE") or _sh("grep -E 'PRETTY_NAME=' /etc/os-release | cut -d= -f2- | tr -d '\"'"),
        "kaggle": {
            "is_kaggle": bool(os.path.isdir("/kaggle") and os.path.isdir("/kaggle/working")),
            "kernel_run_type": os.environ.get("KAGGLE_KERNEL_RUN_TYPE"),
            "dataset_paths": [p for p in ("/kaggle/input", "/kaggle/working") if os.path.isdir(p)],
        },
        "env_overrides": {
            "PYTHONPATH": os.environ.get("PYTHONPATH"),
            "PATH": os.environ.get("PATH"),
        },
    }


def _sha256_file(p: Path) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def write_run_manifest(path: str | Path, extra: dict | None = None) -> None:
    """
    Write a rich JSON manifest for the current run.

    Args:
        path: Destination file path ('.json' or '.jsonl' both supported).
        extra: Optional dict to merge into top-level payload (e.g., seeds, hydra snapshot path/hash, cfg).

    Behavior:
        - Creates parent directory.
        - If extension == '.jsonl', appends a single JSON line; else writes a pretty JSON file.
        - Robust to missing components (torch/dvc/git/nvidia).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    payload: Dict[str, Any] = {
        "run_id": time.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "pid": os.getpid(),
        "env": {
            "python": sys.version.split()[0],
            "platform": sys.platform,
        },
        "git": _git_info(),
        "dvc": _dvc_info(),
        "torch": _torch_info(),
        "nvidia": _nvidia_runtime_info(),
        "container": _container_env_info(),
    }

    # If caller provided a hydra snapshot path, hash it (helps equivalence checks)
    try:
        hydra_snapshot = None
        if extra and "hydra_snapshot" in extra and extra["hydra_snapshot"]:
            hydra_snapshot = Path(str(extra["hydra_snapshot"]))
            payload.setdefault("hydra", {})
            payload["hydra"]["snapshot"] = str(hydra_snapshot)
            payload["hydra"]["sha256"] = _sha256_file(hydra_snapshot) if hydra_snapshot.exists() else None
    except Exception:
        # never fail manifest on convenience extras
        pass

    if extra:
        # Shallow merge — reserved keys can be overridden intentionally by caller
        payload.update(extra)

    # Write JSONL (append) vs JSON (overwrite pretty)
    if path.suffix.lower() == ".jsonl":
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    else:
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
