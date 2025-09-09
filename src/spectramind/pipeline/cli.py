from __future__ import annotations

"""
Unified Pipeline CLI (minimal, Kaggle-safe argparse).

Examples:
  # Single stage from file (JSON or YAML)
  python -m spectramind.pipeline.cli run --stage calibrate --cfg configs/calib/local.json

  # Single stage with inline JSON and overrides
  python -m spectramind.pipeline.cli run --stage predict \
    --cfg-inline '{"checkpoint":"artifacts/ckpt.pth","output_csv":"outputs/submission.csv"}' \
    --set runtime.seed=1337 --set predict.device=cpu

  # Run a plan from JSONL (each line {"stage": "...", "config": {...}})
  python -m spectramind.pipeline.cli plan \
    --plan-file pipeline.jsonl --snapshots outputs/snapshots --stop-on-error

  # List stages
  python -m spectramind.pipeline.cli stages
"""

import argparse
import json
import logging
import signal
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Tuple

from .runner import run_stage, run_plan, PipelineError
from .stages import STAGE_NAMES
from . import get_stage_help

_LOG_FMT = "[%(asctime)s] %(levelname)s: %(message)s"


def _setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO), format=_LOG_FMT)


# --- config helpers ------------------------------------------------------------

def _load_json(path: Path) -> Mapping[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_yaml(path: Path) -> Mapping[str, Any]:
    # Very small YAML shim: try yaml if available, else error out
    try:
        import yaml  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(f"YAML config requested but PyYAML is not installed: {e}") from e
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _load_cfg_auto(path: Path) -> Mapping[str, Any]:
    """Auto-detect JSON or YAML by extension."""
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        return _load_yaml(path)
    if suffix == ".json":
        return _load_json(path)
    # Fallback try JSON first then YAML
    try:
        return _load_json(path)
    except Exception:
        return _load_yaml(path)


def _apply_overrides(cfg: Dict[str, Any], kv_list: Iterable[str]) -> Dict[str, Any]:
    """Apply dotpath overrides like key.subkey=value into a nested dict."""
    for kv in kv_list or []:
        kv = kv.strip()
        if not kv or "=" not in kv:
            continue
        key, val = kv.split("=", 1)
        key = key.strip()
        val = val.strip()
        # try to json-decode value to keep types, else keep as str
        try:
            val_parsed = json.loads(val)
        except Exception:
            val_parsed = val
        # set into dict by dotpath
        parts = [p for p in key.split(".") if p]
        cur = cfg
        for p in parts[:-1]:
            if p not in cur or not isinstance(cur[p], dict):
                cur[p] = {}
            cur = cur[p]  # type: ignore
        cur[parts[-1]] = val_parsed
    return cfg


# --- plan loader ---------------------------------------------------------------

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


def _load_plan_array(path: Path) -> List[Tuple[str, Mapping[str, Any]]]:
    """Load a JSON array: [{"stage": "...", "config": {...}}, ...]"""
    obj = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(obj, list):
        raise ValueError("Plan JSON must be a list of {stage, config} objects")
    pairs: List[Tuple[str, Mapping[str, Any]]] = []
    for item in obj:
        if not isinstance(item, dict):
            raise ValueError("Plan array entries must be objects")
        st = item.get("stage")
        cfg = item.get("config", {})
        if not st:
            raise ValueError(f"Missing 'stage' in plan item: {item}")
        pairs.append((st, cfg))
    return pairs


def _load_plan_auto(path: Path) -> List[Tuple[str, Mapping[str, Any]]]:
    """Auto-detect JSONL or JSON-array by peeking the first non-empty char."""
    text = path.read_text(encoding="utf-8")
    first = next((ch for ch in text if not ch.isspace()), "")
    if first == "[":
        return _load_plan_array(path)
    return _load_plan_jsonl(path)


# --- SIGINT graceful -----------------------------------------------------------

def _install_sigint_handler() -> None:
    def _handler(signum, frame):
        print(json.dumps({"ok": False, "error": "Interrupted by SIGINT"}))
        raise SystemExit(130)

    try:
        signal.signal(signal.SIGINT, _handler)
    except Exception:
        pass


# --- CLI ----------------------------------------------------------------------

def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser("SpectraMind V50 — Pipeline")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # run single stage
    p_run = sub.add_parser("run", help="Run a single stage with a JSON/YAML config")
    p_run.add_argument("--stage", required=True, choices=STAGE_NAMES)
    p_run.add_argument("--cfg", type=Path, help="Path to JSON/YAML config (mutually exclusive with --cfg-inline)")
    p_run.add_argument("--cfg-inline", type=str, default=None, help="Inline JSON config (mutually exclusive with --cfg)")
    p_run.add_argument("--set", dest="sets", action="append", default=[], help="Dotpath override(s): key.subkey=value")
    p_run.add_argument("--seed", type=int, default=42)
    p_run.add_argument("--snapshots", type=Path, default=None)
    p_run.add_argument("--log-level", type=str, default="INFO")

    # run plan from file (JSONL or JSON array)
    p_plan = sub.add_parser("plan", help="Run multiple stages from a plan file (JSONL or JSON array)")
    p_plan.add_argument("--plan-file", required=True, type=Path, help="JSONL lines: {stage, config} OR JSON array")
    p_plan.add_argument("--seed", type=int, default=42)
    p_plan.add_argument("--snapshots", type=Path, default=None)
    p_plan.add_argument("--stop-on-error", action="store_true", default=True)
    p_plan.add_argument("--log-level", type=str, default="INFO")

    # list stages
    p_stages = sub.add_parser("stages", help="List available stages")
    p_stages.add_argument("--verbose", action="store_true", help="Show stage module HELP/doc if available")
    p_stages.add_argument("--log-level", type=str, default="WARNING")

    args = ap.parse_args()
    _setup_logging(args.log_level)
    _install_sigint_handler()

    if args.cmd == "run":
        if (args.cfg is None) == (args.cfg_inline is None):
            # must provide exactly one of these
            print(json.dumps({"ok": False, "error": "Specify exactly one of --cfg or --cfg-inline"}))
            raise SystemExit(2)

        # Load config (file or inline), then apply overrides
        if args.cfg is not None:
            cfg = _load_cfg_auto(args.cfg)
        else:
            try:
                cfg = json.loads(args.cfg_inline)
            except Exception as e:
                print(json.dumps({"ok": False, "error": f"Invalid --cfg-inline JSON: {e}"}))
                raise SystemExit(2)

        if not isinstance(cfg, dict):
            print(json.dumps({"ok": False, "error": "Config must be an object/dict"}))
            raise SystemExit(2)

        cfg = _apply_overrides(dict(cfg), args.sets)

        try:
            sr = run_stage(args.stage, cfg, seed=args.seed, snapshot_dir=args.snapshots)
            # pretty summary
            resp: Dict[str, Any] = {
                "ok": True,
                "data": sr.data,
                "duration_s": sr.duration_s,
            }
            # show artifact summary if available
            arts = sr.data.get("artifacts") if isinstance(sr.data, dict) else None
            if isinstance(arts, dict):
                resp["artifacts"] = {k: len(v) for k, v in arts.items() if isinstance(v, list)}
            print(json.dumps(resp, indent=2))
        except PipelineError as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            raise SystemExit(1)

    elif args.cmd == "plan":
        try:
            plan = _load_plan_auto(args.plan_file)
        except Exception as e:
            print(json.dumps({"ok": False, "error": f"Failed to load plan: {e}"}))
            raise SystemExit(2)

        # Validate stage names early
        bad = [st for st, _ in plan if st not in STAGE_NAMES]
        if bad:
            print(json.dumps({"ok": False, "error": f"Unknown stage(s): {bad}. Known={list(STAGE_NAMES)}"}))
            raise SystemExit(2)

        try:
            results = run_plan(plan, seed=args.seed, snapshot_dir=args.snapshots, stop_on_error=args.stop_on_error)
            summary = [{"stage": r.stage, "ok": r.ok, "duration_s": round(r.duration_s, 3)} for r in results]
            print(json.dumps({"ok": True, "summary": summary}, indent=2))
        except PipelineError as e:
            print(json.dumps({"ok": False, "error": str(e)}))
            raise SystemExit(1)

    elif args.cmd == "stages":
        # simple list
        if not args.verbose:
            print(json.dumps({"ok": True, "stages": list(STAGE_NAMES)}))
            raise SystemExit(0)

        # verbose with HELP/doc
        enriched: List[Dict[str, Any]] = []
        for st in STAGE_NAMES:
            enriched.append(
                {
                    "stage": st,
                    "help": get_stage_help(st),
                }
            )
        print(json.dumps({"ok": True, "stages": enriched}, indent=2))
        raise SystemExit(0)


if __name__ == "__main__":  # pragma: no cover
    main()
