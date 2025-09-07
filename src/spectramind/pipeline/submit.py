# src/spectramind/pipeline/submit.py
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

__all__ = ["run"]


# ----------------------------- Data model -------------------------------------


@dataclass(frozen=True, slots=True)
class SubmitPayload:
    """Strongly-typed submission invocation payload."""
    config_name: str
    overrides: list[str]
    predictions: Optional[str]
    out_zip: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


# --------------------------------- API ----------------------------------------


def run(
    *,
    config_name: str = "submit",
    overrides: Iterable[str] | None = None,
    predictions: str | os.PathLike[str] | None = None,
    out_zip: str | os.PathLike[str] | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Submission packager: validate schema + package artifacts (zip/csv/manifest).

    Responsibilities
    ---------------
    • Compose Hydra config (submit + io paths)
    • Validate predictions file (existence + optional JSON schema)
    • Package predictions and manifest into a deterministic zip

    Parameters
    ----------
    config_name
        Name of the top-level Hydra config under `configs/` (default: "submit").
    overrides
        Hydra override strings, e.g. `["submit.out_zip=outputs/submissions/submission.zip"]`.
    predictions
        Optional explicit predictions file path; overrides `cfg.submit.predictions`.
    out_zip
        Optional explicit submission zip path; overrides `cfg.submit.out_zip`.
    strict
        If True, propagate composition/validation errors; else warn and continue with defaults.
    quiet
        Suppress resolved config echoing to stderr.
    env
        Optional execution context, e.g. {"is_kaggle": True, "is_ci": False}.

    Raises
    ------
    FileNotFoundError
        When `configs/` directory is missing or predictions file cannot be found.
    RuntimeError
        When schema validation fails (if schema present).
    """
    payload = SubmitPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        predictions=str(predictions) if predictions is not None else None,
        out_zip=str(out_zip) if out_zip is not None else None,
        strict=strict,
        quiet=quiet,
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    # honor explicit CLI overrides
    if payload.predictions is not None:
        cfg.submit.predictions = payload.predictions
    if payload.out_zip is not None:
        cfg.submit.out_zip = payload.out_zip

    # validate & normalize io
    preds_file = _resolve_predictions(repo_root, cfg)
    out_zip_file = _prepare_out_zip(repo_root, cfg)

    # optional schema and basic shape checks
    _validate_predictions(cfg, repo_root, preds_file, quiet=payload.quiet, strict=payload.strict)

    # emit run snapshot for traceability
    _emit_config_snapshot(out_zip_file, cfg, preds_file)

    # create zip (deterministic-ish: fixed arcname order & times)
    _package_submission(cfg, preds_file, out_zip_file, quiet=payload.quiet)

    if not payload.quiet:
        sys.stderr.write(f"[submit] package ready → {out_zip_file}\n")


# ---------------------------- Validation & Packaging -------------------------


def _validate_predictions(
    cfg: DictConfig,
    repo_root: Path,
    preds_path: Path,
    *,
    quiet: bool,
    strict: bool,
) -> None:
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    # Basic sanity: non-empty, readable, CSV-ish
    try:
        with preds_path.open("r", newline="") as f:
            sniff = csv.Sniffer().sniff(f.read(4096))
            f.seek(0)
            reader = csv.reader(f, dialect=sniff)
            rows_peek = []
            for i, row in enumerate(reader):
                rows_peek.append(row)
                if i >= 2:
                    break
            if not rows_peek:
                raise ValueError("empty CSV")
    except Exception as e:
        msg = f"[submit] basic CSV check failed for {preds_path}: {e}"
        if strict:
            raise RuntimeError(msg) from e
        if not quiet:
            sys.stderr.write(msg + " (continuing)\n")

    # Optional JSON Schema validation if present
    schema = repo_root / "schemas" / "submission.schema.json"
    if schema.exists():
        try:
            import jsonschema  # type: ignore
            import pandas as pd  # type: ignore

            df = pd.read_csv(preds_path)
            # Convert DataFrame to a simple dict-of-lists for schema validation
            with schema.open("r") as f:
                submission_schema = json.load(f)
            jsonschema.validate(df.to_dict(orient="list"), submission_schema)
            if not quiet:
                sys.stderr.write("[submit] predictions validated against JSON schema.\n")
        except Exception as e:
            raise RuntimeError(f"Submission schema validation failed: {e}") from e
    else:
        if not quiet:
            sys.stderr.write("[submit] no JSON schema found; skipping schema validation.\n")


def _package_submission(
    cfg: DictConfig,
    preds_file: Path,
    out_zip_file: Path,
    *,
    quiet: bool,
) -> None:
    # Prepare manifest with content hash & environment meta
    preds_sha256 = _sha256_of_file(preds_file)
    manifest = {
        "schema": "spectramind/submission@v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "predictions": preds_file.name,  # arcname in zip
        "predictions_sha256": preds_sha256,
        "tool": "spectramind-v50",
        "config": {
            "out_zip": str(out_zip_file),
        },
    }

    # Write zip with a deterministic ordering
    with zipfile.ZipFile(out_zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # store predictions under their basename (per Kaggle norms)
        zf.write(preds_file, arcname=preds_file.name)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    if not quiet:
        sys.stderr.write(f"[submit] wrote package: {out_zip_file}\n")


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: SubmitPayload) -> DictConfig:
    """Compose Hydra config from repo `configs/` with robust defaults + strict mode."""
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per repo blueprint)"
        )

    with initialize(config_path=str(config_dir), version_base=None):
        try:
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)
        except Exception:
            if payload.strict:
                raise
            # Best-effort: attempt to compose base config with no overrides
            cfg = compose(config_name=payload.config_name, overrides=[])

    _ensure_defaults(cfg)
    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved submission config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")
    _validate_minimal_schema(cfg, strict=payload.strict)
    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    cfg.setdefault("submit", {})
    # DVC-/Kaggle-friendly defaults under repo
    cfg.submit.setdefault("predictions", "outputs/predictions/predictions.csv")
    cfg.submit.setdefault("out_zip", "outputs/submissions/submission.zip")


def _validate_minimal_schema(cfg: DictConfig, *, strict: bool) -> None:
    missing: list[str] = []
    if "submit" not in cfg:
        missing.append("submit")
    else:
        if "predictions" not in cfg.submit:
            missing.append("submit.predictions")
        if "out_zip" not in cfg.submit:
            missing.append("submit.out_zip")
    if missing:
        msg = f"[submit] missing required config keys: {', '.join(missing)}"
        if strict:
            raise KeyError(msg)
        sys.stderr.write(msg + " (continuing)\n")


def _resolve_predictions(repo_root: Path, cfg: DictConfig) -> Path:
    """Return absolute predictions path (ensure parent exists; file may be absent until validated)."""
    raw = str(cfg.submit.predictions)
    preds = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    if not preds.exists():
        # Let caller raise a clearer error; here we only return normalized path.
        return preds
    # Optional: sanity on file size zero
    try:
        if preds.stat().st_size == 0:
            sys.stderr.write(f"[submit] warning: predictions file is empty: {preds}\n")
    except Exception:
        pass
    return preds


def _prepare_out_zip(repo_root: Path, cfg: DictConfig) -> Path:
    """Ensure parent dirs exist for zip under repo root; returns absolute zip path."""
    raw = str(cfg.submit.out_zip)
    out_zip = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    out_zip.parent.mkdir(parents=True, exist_ok=True)
    # Also prepare sibling 'snapshots/' for config snapshots
    (out_zip.parent / "snapshots").mkdir(exist_ok=True)
    # Normalize extension
    if out_zip.suffix.lower() != ".zip":
        out_zip = out_zip.with_suffix(".zip")
        cfg.submit.out_zip = str(out_zip)
    return out_zip


def _emit_config_snapshot(out_zip_file: Path, cfg: DictConfig, preds_file: Path) -> None:
    """Write resolved config & minimal run metadata beside the out_zip (snapshots/)."""
    try:
        snap_dir = out_zip_file.parent / "snapshots"
        snap_dir.mkdir(exist_ok=True)

        # YAML snapshot (human-friendly)
        snap_yaml = snap_dir / "config_snapshot.yaml"
        OmegaConf.save(cfg, snap_yaml, resolve=True)

        # JSON run metadata (machine-friendly)
        meta = {
            "schema": "spectramind/submit_config_snapshot@v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "git_root": str(_nearest_git_root(out_zip_file.parent) or ""),
            "predictions": str(preds_file),
            "predictions_sha256": _sha256_of_file(preds_file) if preds_file.exists() else "",
            "out_zip": str(out_zip_file),
            "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
        }
        meta_json = snap_dir / "config_snapshot.json"
        meta_json.write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[submit] warning: failed to write config snapshot: {e}\n")


# ---------------------------- Common helpers ---------------------------------


def _sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


def _nearest_git_root(path: Path) -> Path | None:
    """Return nearest parent that contains `.git`, else None."""
    for p in [path, *path.parents]:
        if (p / ".git").exists():
            return p
    return None


def _find_repo_root() -> Path:
    """
    Heuristic: start from this file, walk up until we hit a project marker.

    Accepts:
      • pyproject.toml (preferred)
      • setup.cfg
      • .git
    """
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            # If we’re sitting in .../src, prefer its parent as root (layout: root/src/...)
            if p.name == "src":
                return p.parent
            return p
    # Fallback: best-effort three levels up (keeps Kaggle/colab from exploding)
    return here.parents[3]


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))
