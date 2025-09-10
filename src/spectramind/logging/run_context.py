from __future__ import annotations
import json, os, socket, subprocess, sys, uuid
from datetime import datetime
from typing import Any, Dict, Optional
from .jsonl import JSONLLogger
from .utils import iso_now, safe_mkdir, hash_json, to_serializable

def _git_info(cwd: Optional[str] = None) -> Dict[str, str]:
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=cwd).decode().strip()
        dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=cwd).decode()
        return {"git_sha": sha, "git_dirty": "1" if dirty.strip() else "0"}
    except Exception:
        return {}

def _env_info() -> Dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "executable": sys.executable,
        "hostname": socket.gethostname(),
        "platform": sys.platform,
    }

def _cfg_to_dict(cfg: Any) -> Dict[str, Any]:
    # Hydra/OmegaConf: supports .to_container(resolve=True) and ._get_full_key support
    try:
        import omegaconf
        if isinstance(cfg, omegaconf.DictConfig) or isinstance(cfg, omegaconf.ListConfig):
            return omegaconf.OmegaConf.to_container(cfg, resolve=True)  # type: ignore
    except Exception:
        pass
    # Plain dict or dataclass
    if hasattr(cfg, "to_dict"):
        return cfg.to_dict()
    if isinstance(cfg, dict):
        return cfg
    if hasattr(cfg, "__dict__"):
        return dict(cfg.__dict__)
    return {"config": str(cfg)}

def init_run_dir(base_dir: str, run_id: Optional[str] = None) -> str:
    run_id = run_id or datetime.utcnow().strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:6]
    run_dir = os.path.join(base_dir, run_id)
    safe_mkdir(run_dir)
    return run_dir

class RunContext:
    """
    Creates a run directory with:
      - manifest.json (config snapshot + hashes + env + git)
      - events.jsonl  (append-only event stream)
    """
    def __init__(self, run_dir: str, cfg: Optional[Any] = None) -> None:
        self.run_dir = os.fspath(run_dir)
        safe_mkdir(self.run_dir)
        self.manifest_path = os.path.join(self.run_dir, "manifest.json")
        self.events_path = os.path.join(self.run_dir, "events.jsonl")
        self.logger = JSONLLogger(self.events_path)
        self.created_at = iso_now()
        self.config = _cfg_to_dict(cfg) if cfg is not None else {}
        self._write_manifest()

    def _write_manifest(self) -> None:
        env = _env_info()
        git = _git_info()
        cfg_hash = hash_json(self.config) if self.config else ""
        manifest = {
            "run_dir": self.run_dir,
            "created_at": self.created_at,
            "config": self.config,
            "config_hash": cfg_hash,
            "env": env,
            **git,
        }
        with open(self.manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

    def log_event(self, **fields) -> None:
        self.logger.log(**fields)

    def close(self) -> None:
        self.logger.close()

    def __enter__(self) -> "RunContext":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
