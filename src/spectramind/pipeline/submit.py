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


# ======================================================================================
# Data model
# ======================================================================================


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


# ======================================================================================
# Public API
# ======================================================================================


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
    • Validate predictions file (shape, columns, basic numeric checks; optional JSON schema)
    • Package predictions and manifest into a deterministic zip & emit a config snapshot
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

    # CLI overrides take precedence
    if payload.predictions is not None:
        cfg.submit.predictions = payload.predictions
    if payload.out_zip is not None:
        cfg.submit.out_zip = payload.out_zip

    # Resolve IO
    preds_file = _resolve_predictions(repo_root, cfg)
    out_zip_file = _prepare_out_zip(repo_root, cfg)

    # Validate predictions (fast CSV & strict column contract)
    _validate_predictions(cfg, repo_root, preds_file, quiet=payload.quiet, strict=payload.strict)

    # Emit config snapshot (resolved Hydra YAML + JSON meta) next to zip
    _emit_config_snapshot(out_zip_file, cfg, preds_file)

    # Create zip (stable filenames; compressed)
    _package_submission(cfg, preds_file, out_zip_file, quiet=payload.quiet)

    if not payload.quiet:
        sys.stderr.write(f"[submit] package ready → {out_zip_file}\n")


# ======================================================================================
# Validation & Packaging
# ======================================================================================


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

    # 1) Basic CSV sniff (non-empty, parseable)
    try:
        with preds_path.open("r", newline="") as f:
            blob = f.read(4096)
            if not blob.strip():
                raise ValueError("empty file")
            sniff = csv.Sniffer().sniff(blob)
            f.seek(0)
            reader = csv.reader(f, dialect=sniff)
            peek = [row for _, row in zip(range(3), reader)]
            if not peek:
                raise ValueError("no rows parsed")
    except Exception as e:
        msg = f"[submit] basic CSV check failed for {preds_path}: {e}"
        if strict:
            raise RuntimeError(msg) from e
        if not quiet:
            sys.stderr.write(msg + " (continuing)\n")

    # 2) Column contract & numeric sanity (use pandas for convenience)
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for submission validation") from e

    df = pd.read_csv(preds_path)

    # Required columns
    required_id = "id"
    if required_id not in df.columns:
        raise RuntimeError(f"[submit] missing required column: '{required_id}'")

    # Ensure exactly 283 mu_* and 283 sigma_* columns
    mu_cols = sorted([c for c in df.columns if c.startswith("mu_")])
    sg_cols = sorted([c for c in df.columns if c.startswith("sigma_")])

    def _expected(prefix: str) -> list[str]:
        return [f"{prefix}{i:03d}" for i in range(283)]

    exp_mu = _expected("mu_")
    exp_sg = _expected("sigma_")

    if mu_cols != exp_mu:
        missing = [c for c in exp_mu if c not in mu_cols]
        extra = [c for c in mu_cols if c not in exp_mu]
        raise RuntimeError(
            "[submit] mu columns mismatch: "
            f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}"
        )
    if sg_cols != exp_sg:
        missing = [c for c in exp_sg if c not in sg_cols]
        extra = [c for c in sg_cols if c not in exp_sg]
        raise RuntimeError(
            "[submit] sigma columns mismatch: "
            f"missing={missing[:5]}{'...' if len(missing) > 5 else ''}, "
            f"extra={extra[:5]}{'...' if len(extra) > 5 else ''}"
        )

    # id column sanity (non-null, mostly string-like; unique)
    if df[required_id].isna().any():
        raise RuntimeError("[submit] 'id' column contains nulls")
    if df[required_id].duplicated().any():
        dup = df[required_id][df[required_id].duplicated()].iloc[:5].tolist()
        raise RuntimeError(f"[submit] 'id' column contains duplicates (e.g., {dup})")
    # Cast to str to ensure safe CSV identity
    df[required_id] = df[required_id].astype(str)

    # mu/sigma numeric sanity (no NaNs; finite)
    numeric = df[mu_cols + sg_cols].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any():
        bad = numeric.columns[numeric.isna().any()].tolist()[:5]
        raise RuntimeError(f"[submit] numeric columns contain NaN/non-numeric values (e.g., {bad})")
    # Replace original block with numeric-cast (keeps exact column order)
    df[mu_cols + sg_cols] = numeric

    # Optional JSON Schema (column-level validation can be brittle; treat as informational)
    schema_path = repo_root / "schemas" / "submission.schema.json"
    if schema_path.exists():
        try:
            import jsonschema  # type: ignore
            with schema_path.open("r") as f:
                submission_schema = json.load(f)
            # Validate a minimal object capturing columns -> types, not the entire table
            obj = {
                "id": str(df["id"].iloc[0]) if len(df) else "",
                # we only validate presence via pattern; numeric content already checked
                **{c: float(df[c].iloc[0]) if len(df) else 0.0 for c in mu_cols[:1] + sg_cols[:1]},
            }
            jsonschema.validate(obj, submission_schema)
            if not quiet:
                sys.stderr.write("[submit] schema present; basic sample validation passed.\n")
        except Exception as e:
            # Keep the strong, explicit checks above as the source of truth
            if not quiet:
                sys.stderr.write(f"[submit] schema validation warning: {e}\n")

    # Write back normalized CSV (string ids, numeric mu/sigma) to ensure downstream stability
    df.to_csv(preds_path, index=False)


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

    # Write zip (stable arcnames)
    with zipfile.ZipFile(out_zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        # store predictions under basename (common Kaggle norm)
        zf.write(preds_file, arcname=preds_file.name)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    if not quiet:
        sys.stderr.write(f"[submit] wrote package: {out_zip_file}\n")


# ======================================================================================
# Hydra / Snapshot / Utils
# ======================================================================================


def _compose_hydra_config(repo_root: Path, payload: SubmitPayload) -> DictConfig:
    """Compose Hydra config from repo `configs/` with robust defaults + strict mode."""
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per repository blueprint)"
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
        # Caller raises precise error; here we only normalize
        return preds
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
            "predictions": str(preds_file),
            "predictions_sha256": _sha256_of_file(preds_file) if preds_file.exists() else "",
            "out_zip": str(out_zip_file),
            "hydra_overrides": list(getattr(cfg, "hydra", {}).get("overrides", [])),
        }
        meta_json = snap_dir / "config_snapshot.json"
        meta_json.write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[submit] warning: failed to write config snapshot: {e}\n")


# ======================================================================================
# Common helpers
# ======================================================================================


def _sha256_of_file(path: Path, chunk: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            data = f.read(chunk)
            if not data:
                break
            h.update(data)
    return h.hexdigest()


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
            if p.name == "src":  # /root/src/… → project root is parent
                return p.parent
            return p
    # Fallback: best-effort three levels up (safe for notebook contexts)
    return here.parents[3]


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))
