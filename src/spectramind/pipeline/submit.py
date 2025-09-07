# src/spectramind/pipeline/submit.py
from __future__ import annotations

import csv
import gzip
import hashlib
import io
import json
import os
import platform
import sys
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf

__all__ = ["run"]

# --------------------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------------------

BIN_COUNT = 283
ID_COL = "id"
MU_PREFIX = "mu_"
SIGMA_PREFIX = "sigma_"
EXPECTED_MU = [f"{MU_PREFIX}{i:03d}" for i in range(BIN_COUNT)]
EXPECTED_SIGMA = [f"{SIGMA_PREFIX}{i:03d}" for i in range(BIN_COUNT)]
CSV_MAX_PREVIEW_BYTES = 4096


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
    • Validate predictions file (shape, columns, numeric checks; optional JSON schema)
    • Normalize CSV (string ids; numeric mu/sigma; column order)
    • Package predictions + manifest (+ optional extras) into a *deterministic* ZIP
    • Emit resolved config snapshot & meta beside the ZIP
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
    preds_path = _resolve_predictions(repo_root, cfg)
    out_zip_path = _prepare_out_zip(repo_root, cfg)

    # Validate predictions (CSV sniff + contract + numeric)
    _validate_and_normalize_predictions(cfg, repo_root, preds_path, quiet=payload.quiet, strict=payload.strict)

    # Emit config snapshot (resolved Hydra YAML + JSON meta) next to zip
    _emit_config_snapshot(out_zip_path, cfg, preds_path)

    # Create deterministic zip (stable timestamps, permissions, and order)
    _package_submission(cfg, preds_path, out_zip_path, quiet=payload.quiet)

    if not payload.quiet:
        sys.stderr.write(f"[submit] package ready → {out_zip_path}\n")


# ======================================================================================
# Validation & Packaging
# ======================================================================================


def _read_csv_head(path: Path, max_bytes: int = CSV_MAX_PREVIEW_BYTES) -> str:
    if path.suffix.lower() == ".gz":
        with gzip.open(path, "rt", newline="") as f:
            return f.read(max_bytes)
    with path.open("r", newline="") as f:
        return f.read(max_bytes)


def _read_csv_df(path: Path):
    try:
        import pandas as pd  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError("pandas is required for submission validation") from e

    if path.suffix.lower() == ".gz":
        return pd.read_csv(path, compression="gzip")
    return pd.read_csv(path)


def _write_csv_df(path: Path, df) -> None:
    if path.suffix.lower() == ".gz":
        df.to_csv(path, index=False, compression="gzip")
    else:
        df.to_csv(path, index=False)


def _validate_and_normalize_predictions(
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
        blob = _read_csv_head(preds_path)
        if not blob.strip():
            raise ValueError("empty file")
        sniff = csv.Sniffer().sniff(blob)
        # sanity: try reading a couple of rows using the detected dialect
        sample_reader = csv.reader(io.StringIO(blob), dialect=sniff)
        _ = [row for _, row in zip(range(3), sample_reader)]
    except Exception as e:
        msg = f"[submit] basic CSV check failed for {preds_path}: {e}"
        if strict:
            raise RuntimeError(msg) from e
        if not quiet:
            sys.stderr.write(msg + " (continuing)\n")

    # 2) Column contract & numeric sanity
    df = _read_csv_df(preds_path)

    # Required id column
    if ID_COL not in df.columns:
        raise RuntimeError(f"[submit] missing required column: '{ID_COL}'")

    # Collect mu/sigma columns and enforce exact match + order
    mu_cols = sorted([c for c in df.columns if c.startswith(MU_PREFIX)])
    sg_cols = sorted([c for c in df.columns if c.startswith(SIGMA_PREFIX)])

    missing_mu = [c for c in EXPECTED_MU if c not in mu_cols]
    extra_mu = [c for c in mu_cols if c not in EXPECTED_MU]
    missing_sg = [c for c in EXPECTED_SIGMA if c not in sg_cols]
    extra_sg = [c for c in sg_cols if c not in EXPECTED_SIGMA]

    if missing_mu or extra_mu:
        raise RuntimeError(
            "[submit] mu columns mismatch: "
            f"missing={missing_mu[:5]}{'...' if len(missing_mu) > 5 else ''}, "
            f"extra={extra_mu[:5]}{'...' if len(extra_mu) > 5 else ''}"
        )
    if missing_sg or extra_sg:
        raise RuntimeError(
            "[submit] sigma columns mismatch: "
            f"missing={missing_sg[:5]}{'...' if len(missing_sg) > 5 else ''}, "
            f"extra={extra_sg[:5]}{'...' if len(extra_sg) > 5 else ''}"
        )

    # Reorder columns to the canonical submission layout
    canonical_cols = [ID_COL] + EXPECTED_MU + EXPECTED_SIGMA
    df = df[canonical_cols]

    # 'id' sanity (non-null, unique, string-cast)
    if df[ID_COL].isna().any():
        raise RuntimeError("[submit] 'id' column contains nulls")
    if df[ID_COL].duplicated().any():
        dup = df[ID_COL][df[ID_COL].duplicated()].iloc[:5].tolist()
        raise RuntimeError(f"[submit] 'id' column contains duplicates (e.g., {dup})")
    df[ID_COL] = df[ID_COL].astype(str)

    # Numeric sanity (no NaNs; finite; sigma >= 0)
    import numpy as np  # local import is fine; required at runtime anyway
    numeric = df[EXPECTED_MU + EXPECTED_SIGMA].apply(
        _to_numeric_strict if strict else _to_numeric_lenient, errors="ignore"
    )

    # Coerce object columns to numeric (both helpers above return floats/NaN)
    numeric = numeric.apply(lambda s: _coerce_series_to_float(s))

    if not np.isfinite(numeric.to_numpy(dtype="float64")).all():
        # Identify a few offenders to help users debug
        bad_cols = []
        arr = numeric.to_numpy(dtype="float64")
        for i, col in enumerate(numeric.columns):
            col_vals = arr[:, i]
            if not np.isfinite(col_vals).all():
                bad_cols.append(col)
            if len(bad_cols) >= 8:
                break
        raise RuntimeError(f"[submit] numeric columns contain non-finite values (e.g., {bad_cols[:8]})")

    # Sigma non-negativity
    if (numeric[EXPECTED_SIGMA] < 0).any().any():
        if strict:
            neg_cols = [c for c in EXPECTED_SIGMA if (numeric[c] < 0).any()]
            raise RuntimeError(f"[submit] sigma must be ≥ 0; negatives in {neg_cols[:5]}")
        else:
            num_neg = int((numeric[EXPECTED_SIGMA] < 0).to_numpy().sum())
            if not quiet:
                sys.stderr.write(f"[submit] warning: clamping {num_neg} negative sigma values to 0 (non-strict mode)\n")
            numeric[EXPECTED_SIGMA] = numeric[EXPECTED_SIGMA].clip(lower=0)

    # Replace original numeric block (ensures exact order and dtypes)
    df[EXPECTED_MU + EXPECTED_SIGMA] = numeric.astype("float64")

    # Optional JSON Schema (informational; our checks are the source of truth)
    schema_path = repo_root / "schemas" / "submission.schema.json"
    if schema_path.exists():
        try:
            import jsonschema  # type: ignore

            with schema_path.open("r", encoding="utf-8") as f:
                submission_schema = json.load(f)

            # Validate a minimal representative instance; full tabular schema is brittle.
            sample_obj = {
                "id": str(df[ID_COL].iloc[0]) if len(df) else "",
                # We just verify types via a couple of representative fields.
                EXPECTED_MU[0]: float(df[EXPECTED_MU[0]].iloc[0]) if len(df) else 0.0,
                EXPECTED_SIGMA[0]: float(df[EXPECTED_SIGMA[0]].iloc[0]) if len(df) else 0.0,
            }
            jsonschema.validate(sample_obj, submission_schema)
            if not quiet:
                sys.stderr.write("[submit] schema present; sample validation passed.\n")
        except Exception as e:
            if not quiet:
                sys.stderr.write(f"[submit] schema validation warning: {e}\n")

    # Persist normalized CSV back to the same path (stable types & order)
    _write_csv_df(preds_path, df)


def _to_numeric_strict(s):
    import pandas as pd
    out = pd.to_numeric(s, errors="coerce")
    if out.isna().any():
        raise RuntimeError(f"[submit] non-numeric values detected in column '{s.name}'")
    return out


def _to_numeric_lenient(s):
    import pandas as pd
    return pd.to_numeric(s, errors="coerce")


def _coerce_series_to_float(s):
    import numpy as np
    # cast to float64, preserving NaNs (handled upstream)
    return np.asarray(s, dtype="float64")


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
        "env": {
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": {
            "out_zip": str(out_zip_file),
        },
    }

    # Optional extras (configured via Hydra): list of file paths to include alongside predictions
    extras: list[str] = []
    try:
        extras = list(getattr(cfg.submit, "extra_files", []))
    except Exception:
        extras = []
    extra_paths = _resolve_existing_files(extras, base_dir=_find_repo_root())

    # Write deterministic ZIP: fixed timestamps/permissions; sorted entries
    files_to_add = [(preds_file, preds_file.name)] + [(p, p.name) for p in extra_paths]
    files_to_add.sort(key=lambda x: x[1])  # sort by arcname for deterministic order

    with zipfile.ZipFile(out_zip_file, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for src, arc in files_to_add:
            _zip_write_deterministic(zf, src, arc)
        # manifest last (but deterministic name)
        _zip_writestr_deterministic(zf, "manifest.json", json.dumps(manifest, indent=2))

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
    cfg.submit.setdefault("extra_files", [])  # optional list of extra artifacts to include


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
    """Return absolute predictions path; supports .csv and .csv.gz. Warn if empty."""
    raw = str(cfg.submit.predictions)
    preds = (_as_path(raw) if os.path.isabs(raw) else repo_root / raw).resolve()
    if not preds.exists():
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
            "hydra_overrides": _extract_hydra_overrides(cfg),
        }
        meta_json = snap_dir / "config_snapshot.json"
        meta_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[submit] warning: failed to write config snapshot: {e}\n")


def _extract_hydra_overrides(cfg: DictConfig) -> list[str]:
    try:
        # Hydra stores overrides under cfg.hydra.overrides.task (list[str]) at runtime contexts.
        # We tolerate absence in non-Hydra entrypoints.
        overrides = []
        hydra_node = getattr(cfg, "hydra", None)
        if hydra_node and hasattr(hydra_node, "overrides"):
            ov = getattr(hydra_node, "overrides")
            # common patterns: ov.task (list[str]) or just list-like
            if hasattr(ov, "task"):
                overrides = list(ov.task)
            else:
                try:
                    overrides = list(ov)
                except Exception:
                    overrides = []
        return overrides
    except Exception:
        return []


# ======================================================================================
# Deterministic ZIP helpers
# ======================================================================================

_FIXED_ZIP_DATE = (1980, 1, 1, 0, 0, 0)  # DOS epoch; ensures byte-for-byte reproducibility
_FIXED_PERMS = 0o644  # rw-r--r--


def _zip_write_deterministic(zf: zipfile.ZipFile, src: Path, arcname: str) -> None:
    """
    Write a file into the zip with stable timestamp/permissions.
    """
    data = src.read_bytes()
    zi = zipfile.ZipInfo(filename=arcname, date_time=_FIXED_ZIP_DATE)
    zi.compress_type = zipfile.ZIP_DEFLATED
    # Set Unix perms in external_attr (upper 16 bits)
    zi.external_attr = (_FIXED_PERMS & 0xFFFF) << 16
    zf.writestr(zi, data)


def _zip_writestr_deterministic(zf: zipfile.ZipFile, arcname: str, text: str) -> None:
    data = text.encode("utf-8")
    zi = zipfile.ZipInfo(filename=arcname, date_time=_FIXED_ZIP_DATE)
    zi.compress_type = zipfile.ZIP_DEFLATED
    zi.external_attr = (_FIXED_PERMS & 0xFFFF) << 16
    zf.writestr(zi, data)


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


def _resolve_existing_files(paths: Iterable[str], base_dir: Path) -> list[Path]:
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if not p.is_absolute():
            p = (base_dir / raw).resolve()
        if p.exists() and p.is_file():
            out.append(p)
    return out


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
    # Fallback: best-effort three levels up (safe for notebook/Kaggle contexts)
    return here.parents[3]


def _as_path(p: str | os.PathLike[str]) -> Path:
    return Path(p) if isinstance(p, Path) else Path(os.fspath(p))
