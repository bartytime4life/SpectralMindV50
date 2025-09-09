# src/spectramind/pipeline/runner.py
from __future__ import annotations

"""
Stage runner & plan executor with deterministic seeding, simple logging,
and JSON-serializable StageResult outputs. Supports DictConfig but does not
require Hydra (callers pass plain dicts in CI/Kaggle).
"""

import json
import logging
import os
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

try:  # optional Hydra support
    from omegaconf import DictConfig, OmegaConf  # type: ignore
except Exception:  # pragma: no cover
    DictConfig = None  # type: ignore
    OmegaConf = None  # type: ignore

from .stages import make_stage_callable, STAGE_NAMES

LOGGER = logging.getLogger("spectramind.pipeline")


@dataclass
class StageResult:
    stage: str
    ok: bool
    started_at: float
    ended_at: float
    duration_s: float
    data: Dict[str, Any]
    error: Optional[str] = None
    # Optional: path to artifacts (if the stage provides)
    artifacts: Optional[Dict[str, str]] = None


class PipelineError(RuntimeError):
    pass


def _now() -> float:
    return time.time()


def _coerce_mapping(cfg: Any) -> Dict[str, Any]:
    if OmegaConf is not None and isinstance(cfg, DictConfig):  # type: ignore[arg-type]
        return OmegaConf.to_container(cfg, resolve=True)  # type: ignore[return-value]
    if isinstance(cfg, Mapping):
        return dict(cfg)
    raise TypeError(f"stage config must be mapping-like; got {type(cfg)}")


def _seed_everything(seed: Optional[int]) -> None:
    if seed is None:
        return
    try:
        import numpy as _np  # local import to avoid import cost if unused

        _np.random.seed(int(seed))
    except Exception:  # pragma: no cover
        pass
    try:
        import torch as _torch  # pragma: no cover

        _torch.manual_seed(int(seed))
        _torch.cuda.manual_seed_all(int(seed))
        _torch.backends.cudnn.deterministic = True
        _torch.backends.cudnn.benchmark = False
    except Exception:
        pass
    os.environ["PYTHONHASHSEED"] = str(seed)


def _dump_json(p: Path, obj: Mapping[str, Any]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(obj, indent=2), encoding="utf-8")


def run_stage(stage: str, cfg: Mapping[str, Any], *, seed: Optional[int] = 42, snapshot_dir: Optional[Path] = None) -> StageResult:
    """
    Run a single stage. Returns structured StageResult. Does not raise unless stage
    import fails (for clear feedback).
    """
    cfg_dict = _coerce_mapping(cfg)
    _seed_everything(seed)

    fn = make_stage_callable(stage)
    started = _now()
    error_msg: Optional[str] = None
    data: Dict[str, Any] = {}
    artifacts: Optional[Dict[str, str]] = None

    LOGGER.info("==> stage=%s started", stage)
    try:
        result = fn(cfg_dict)  # stage callable should accept dict-like and return dict-like
        if isinstance(result, Mapping):
            data = dict(result)
            artifacts = result.get("artifacts") if isinstance(result.get("artifacts"), Mapping) else None  # type: ignore
    except Exception as e:
        error_msg = f"{type(e).__name__}: {e}"
        LOGGER.exception("Stage '%s' failed: %s", stage, e)
    ended = _now()

    sr = StageResult(
        stage=stage,
        ok=(error_msg is None),
        started_at=started,
        ended_at=ended,
        duration_s=(ended - started),
        data=data,
        error=error_msg,
        artifacts=(dict(artifacts) if artifacts else None),
    )

    # Optional snapshot
    if snapshot_dir is not None:
        snapshot = {
            "stage_result": asdict(sr),
            "config": cfg_dict,
        }
        _dump_json(Path(snapshot_dir) / f
