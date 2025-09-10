from __future__ import annotations
import json, hashlib, os
from datetime import datetime
from typing import Any, Dict, Iterable, Tuple

def iso_now() -> str:
    return datetime.utcnow().replace(microsecond=0).isoformat() + "Z"

def to_serializable(x: Any) -> Any:
    try:
        json.dumps(x)
        return x
    except Exception:
        if hasattr(x, "to_dict"):
            return x.to_dict()
        if hasattr(x, "__dict__"):
            return x.__dict__
        return str(x)

def flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k, v in d.items():
        nk = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            out.update(flatten_dict(v, nk, sep))
        else:
            out[nk] = v
    return out

def hash_text(s: str, algo: str = "sha256") -> str:
    h = hashlib.new(algo)
    h.update(s.encode("utf-8"))
    return h.hexdigest()

def hash_json(obj: Any, algo: str = "sha256") -> str:
    s = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hash_text(s, algo)

def safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)
