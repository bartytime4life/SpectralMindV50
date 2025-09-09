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
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

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
    # Optional: arbitrary artifact metadata (paths, counts, hashes, etc.)
    artifacts: Optional[Dict[str, Any]] = None


class PipelineError(RuntimeError):
    """Raised when a stage fails (unless suppressed by stop_on_error)."""
    pass


# -----------------------------------------------------------------------------#
# Internals
# -----------------------------------------------------------------------------#

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


# -----------------------------------------------------------------------------#
# Public API
# -----------------------------------------------------------------------------#

def run_stage(
    stage: str,
    cfg: Mapping[str, Any],
    *,
    seed: Optional[int] = 42,
    snapshot_dir: Optional[Path] = None,
) -> StageResult:
    """
    Run a single stage. Returns structured StageResult.

    Parameters
    ----------
    stage : str
        Stage name (must be in STAGE_NAMES).
    cfg : Mapping[str, Any]
        Mapping-like config passed to stage callable (`run(cfg)` signature).
    seed : Optional[int]
        Deterministic seeding for numpy/torch and PYTHONHASHSEED.
    snapshot_dir : Optional[Path]
        If provided, writes a {stage}_result.json containing the result & config.

    Raises
    ------
    PipelineError
        If the stage callable raises or returns an error.
    """
    if stage not in STAGE_NAMES:
        raise PipelineError(f"Unknown stage '{stage}'. Known: {', '.join(STAGE_NAMES)}")

    cfg_dict = _coerce_mapping(cfg)
    _seed_everything(seed)

    fn = make_stage_callable(stage)

    started = _now()
    error_msg: Optional[str] = None
    data: Dict[str, Any] = {}
    artifacts: Optional[Dict[str, Any]] = None

    LOGGER.info("==> stage=%s started", stage)
    try:
        result = fn(cfg_dict)  # stage callable should accept dict-like and return dict-like
        if isinstance(result, Mapping):
            data = dict(result)
            # Allow flexible artifact representation; ensure dict shape
            _arts = result.get("artifacts")
            if isinstance(_arts, Mapping):
                artifacts = dict(_arts)
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
        artifacts=artifacts,
    )

    # Optional snapshot (per-stage result + config)
    if snapshot_dir is not None:
        try:
            snapshot = {
                "stage_result": asdict(sr),
                "config": cfg_dict,
            }
            _dump_json(Path(snapshot_dir) / f"{stage}_result.json", snapshot)
        except Exception as e:  # pragma: no cover
            LOGGER.warning("Failed to write snapshot for stage '%s': %s", stage, e)

    if not sr.ok:
        raise PipelineError(f"Stage '{stage}' failed: {sr.error}")

    LOGGER.info("<== stage=%s ok duration=%.2fs", stage, sr.duration_s)
    return sr


def run_plan(
    plan: Iterable[Tuple[str, Mapping[str, Any]]],
    *,
    seed: Optional[int] = 42,
    snapshot_dir: Optional[Path] = None,
    stop_on_error: bool = True,
) -> List[StageResult]:
    """
    Run a series of (stage_name, config) pairs. Returns the list of StageResult.

    Parameters
    ----------
    plan : Iterable[Tuple[str, Mapping[str, Any]]]
        Sequence of (stage, cfg) pairs to execute in order.
    seed : Optional[int]
        Global seed applied before each stage (repeatable across plan runs).
    snapshot_dir : Optional[Path]
        If provided, writes one {stage}_result.json per stage and a plan summary.
    stop_on_error : bool
        If True (default), aborts on first stage failure; else continues and returns
        partial results (failures raise PipelineError only at stage boundary).

    Returns
    -------
    List[StageResult]
        Per-stage results in order.

    Raises
    ------
    PipelineError
        If a stage fails and stop_on_error=True.
    """
    results: List[StageResult] = []
    for stage, cfg in plan:
        if stage not in STAGE_NAMES:
            raise PipelineError(f"Unknown stage '{stage}'. Known: {', '.join(STAGE_NAMES)}")
        try:
            sr = run_stage(stage, cfg, seed=seed, snapshot_dir=snapshot_dir)
            results.append(sr)
        except PipelineError as e:
            if stop_on_error:
                # Write partial summary when snapshot_dir provided
                if snapshot_dir is not None:
                    _write_plan_summary(snapshot_dir, results, failed=str(e))
                raise
            LOGGER.warning("Continuing after failure: %s", e)
            results.append(
                StageResult(
                    stage=stage,
                    ok=False,
                    started_at=0.0,
                    ended_at=0.0,
                    duration_s=0.0,
                    data={},
                    error=str(e),
                    artifacts=None,
                )
            )

    if snapshot_dir is not None:
        _write_plan_summary(snapshot_dir, results)
    return results


# -----------------------------------------------------------------------------#
# Snapshots
# -----------------------------------------------------------------------------#

def _write_plan_summary(snapshot_dir: Path, results: Sequence[StageResult], *, failed: Optional[str] = None) -> None:
    """Write an aggregate plan summary JSON next to per-stage snapshots."""
    try:
        snapshot_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "ok": all(r.ok for r in results) and (failed is None),
            "failed": failed,
            "results": [
                {
                    "stage": r.stage,
                    "ok": r.ok,
                    "duration_s": r.duration_s,
                    "error": r.error,
                    "artifacts": (r.artifacts or {}),
                }
                for r in results
            ],
            "total_duration_s": round(sum(r.duration_s for r in results), 6),
            "started_at": (min((r.started_at for r in results if r.started_at), default=0.0)),
            "ended_at": (max((r.ended_at for r in results if r.ended_at), default=0.0)),
        }
        _dump_json(Path(snapshot_dir) / "plan_summary.json", summary)
    except Exception as e:  # pragma: no cover
        LOGGER.warning("Failed to write plan summary: %s", e)
