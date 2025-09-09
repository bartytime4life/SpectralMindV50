# src/spectramind/pipeline/cli.py
from __future__ import annotations

"""
Unified Pipeline CLI (minimal, Kaggle-safe argparse).

Examples:
  python -m spectramind.pipeline.cli run --stage calibrate --cfg configs/calib/local.json
  python -m spectramind.pipeline.cli plan --plan-file pipeline.jsonl --snapshots outputs/snapshots
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .runner import run_stage, run_plan, PipelineError
from .stages import STAGE_NAMES

_LOG_FMT = "[%(asctime)s] %(levelname)s: %(message)s"


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_LOG_FMT)


def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_plan_jsonl(path: Path) -> List[Tuple[str, Mapping[str, Any]]]:
    """Each line: {"stage": "...", "config": {...}}"""
    pairs: List[Tuple[str, Mapping[str, Any]]] = []
    with path.open("r", encoding="utf-8") as fh:
        for ln in fh:
            ln = ln.strip()
            if not ln:
                continue
            obj = json.loads(ln)
            st = obj.get("stage")
            cfg = obj.get("config", {})
            if not st:
                raise ValueError(f"Missing 'stage' in plan line: {obj}")
            pairs.append((st, cfg))
    return pairs


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser("SpectraMind V50 — Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run single stage
    p_run = sub.add_parser("run", help="Run a single stage with a JSON config")
    p_run.add_argument("--stage", required=True, choices=STAGE_NAMES)
    p_run.add_argument("--cfg", required=True, type=Path, help="Path to JSON config")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--snapshots", type=Path, default=None)
    p_run.add_argument("--log-level", type=str, default="INFO")

    # run plan from JSONL file
    p_plan = sub.add_parser("plan", help="Run multiple stages from a JSONL plan")
    p_plan.add_argument("--plan-file", required=True, type=Path, help="JSONL lines: {stage, config}")
    p_plan.add_argument("--seed", type=int, default=42)
    p_plan.add_argument("--snapshots", type=Path, default=None)
    p_plan.add_argument("--stop-on-error", action="store_true", default=True)
    p_plan.add_argument("--log-level", type=str, default="INFO")

    args = ap.parse_args()
    _setup_logging(args.log_level)

    if args.cmd == "run":
        cfg = _load_json(args.cfg)
        try:
            sr = run_stage(args.stage, cfg, seed=args.seed, snapshot_dir=args.snapshots)
            print(json.dumps({"ok": True, "data": sr.data, "duration_s": sr.duration_s}, indent=2))
        except PipelineError as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            raise SystemExit(1)

    elif args.cmd == "plan":
        plan = _load_plan_jsonl(args.plan_file)
        try:
            results = run_plan(plan, seed=args.seed, snapshot_dir=args.snapshots, stop_on_error=args.stop_on_error)
            summary = [{"stage": r.stage, "ok": r.ok, "duration_s": r.duration_s} for r in results]
            print(json.dumps({"ok": True, "summary": summary}, indent=2))
        except PipelineError as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            raise SystemExit(1)


if __name__ == "__main__":  # pragma: no cover
    main()
