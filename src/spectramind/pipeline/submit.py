from __future__ import annotations

"""
SpectraMind V50 — Submission Runner (Hydra + Packaging Orchestrator)
====================================================================

Key capabilities
----------------
• Hydra config composition with robust defaults
• Kaggle/CI guardrails (determinism, low-mem)
• Standardized validation via spectramind.submit.validate
• Canonical column normalization (sample_id + mu_000.. / sigma_000..)
• Packaging via spectramind.submit.package.package_submission
• Config snapshot + packaging manifest (via package_submission)

Notes
-----
• Keep the business logic here; CLI wrapper (if any) should remain thin.
• This runner returns a dict payload that the pipeline can consume.

Config contract (Hydra)
-----------------------
submit:
  predictions: outputs/predictions/predictions.csv  # input predictions CSV
  out_dir:     outputs/submissions                   # directory to write package
  csv_name:    submission.csv                        # output CSV filename
  zip_name:    submission.zip                        # output ZIP name
  n_bins:      283                                   # number of spectral bins
  strict:      true                                  # raise on invalid submission
"""

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import pandas as pd
from hydra import compose, initialize_config_dir
from omegaconf import DictConfig, OmegaConf

from spectramind.submit.validate import (
    N_BINS_DEFAULT,
    build_expected_columns,
    validate_dataframe,
)
from spectramind.submit.package import package_submission

__all__ = ["run"]


# =============================================================================
# Data model
# =============================================================================

@dataclass(frozen=True, slots=True)
class SubmitPayload:
    config_name: str
    overrides: list[str]
    predictions: Optional[str]
    out_dir: Optional[str]
    csv_name: Optional[str]
    zip_name: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


# =============================================================================
# Public API
# =============================================================================

def run(
    *,
    config_name: str = "submit",
    overrides: Iterable[str] | None = None,
    predictions: str | os.PathLike[str] | None = None,
    out_dir: str | os.PathLike[str] | None = None,
    csv_name: str | None = None,
    zip_name: str | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Validate & package a predictions CSV into a Kaggle-ready bundle.

    Returns
    -------
    dict
      {
        "ok": True,
        "out_dir": "<abs path>",
        "csv": "submission.csv",
        "zip": "submission.zip",
        "n_rows": <int>,
        "sha256_csv": <hex>,
        "sha256_zip": <hex>,
      }
    """
    payload = SubmitPayload(
        config_name=config_name,
        overrides=[str(x).strip() for x in (overrides or []) if str(x).strip()],
        predictions=(str(predictions) if predictions is not None else None),
        out_dir=(str(out_dir) if out_dir is not None else None),
        csv_name=csv_name,
        zip_name=zip_name,
        strict=bool(strict),
        quiet=bool(quiet),
        env=dict(env or {}),
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    # CLI overrides
    if payload.predictions is not None:
        cfg.submit.predictions = payload.predictions
    if payload.out_dir is not None:
        cfg.submit.out_dir = payload.out_dir
    if payload.csv_name is not None:
        cfg.submit.csv_name = payload.csv_name
    if payload.zip_name is not None:
        cfg.submit.zip_name = payload.zip_name
    if payload.strict is not None:
        cfg.submit.strict = bool(payload.strict)

    # Resolve I/O
    preds_file = _resolve_predictions(repo_root, cfg, strict=cfg.submit.strict)
    out_dir_abs = _resolve_out_dir(repo_root, cfg)
    n_bins = int(getattr(cfg.submit, "n_bins", N_BINS_DEFAULT))
    csv_name = str(getattr(cfg.submit, "csv_name", "submission.csv"))
    zip_name = str(getattr(cfg.submit, "zip_name", "submission.zip"))

    # Load → normalize → validate via repo-native validator
    df = pd.read_csv(preds_file)

    # Accept legacy 'id' and normalize to 'sample_id'
    if "sample_id" not in df.columns and "id" in df.columns:
        df = df.rename(columns={"id": "sample_id"})

    # Ensure canonical ordering (sample_id, mu_000.., sigma_000..)
    expected_cols = build_expected_columns(n_bins)
    missing = [c for c in expected_cols if c not in df.columns]
    if missing:
        _raise_or_warn(f"[submit] missing required columns: {missing[:8]}", strict=cfg.submit.strict, quiet=quiet)

    # If extra columns exist, keep only the canonical set
    df = df[[c for c in expected_cols if c in df.columns]]

    # Validate (schema + physics guards) — raises if invalid
    validate_dataframe(
        df,
        n_bins=n_bins,
        strict_order=True,
        check_unique_ids=True,
    ).raise_if_failed()

    # Package using repo-native packager (writes CSV + manifest + ZIP)
    out_zip_or_csv = package_submission(
        df_or_csv=df,
        out_dir=out_dir_abs,
        filename=csv_name,
        make_zip=True,
        zip_name=zip_name,
        n_bins=n_bins,
        extra_meta={"submit_config": OmegaConf.to_container(cfg, resolve=True)},  # type: ignore
    )

    # Compute digests for both csv / zip
    csv_path = out_dir_abs / csv_name
    zip_path = out_dir_abs / zip_name
    sha_csv = _sha256(csv_path) if csv_path.exists() else ""
    sha_zip = _sha256(zip_path) if zip_path.exists() else ""

    if not quiet:
        sys.stderr.write(f"[submit] CSV → {csv_path} (rows={len(df)})\n")
        sys.stderr.write(f"[submit] ZIP → {zip_path}\n")

    # Snapshot resolved config near the ZIP for provenance
    _emit_config_snapshot(out_dir_abs, cfg, preds_file)

    return {
        "ok": True,
        "out_dir": str(out_dir_abs),
        "csv": csv_name,
        "zip": zip_name,
        "n_rows": int(len(df)),
        "sha256_csv": sha_csv,
        "sha256_zip": sha_zip,
    }


# =============================================================================
# Hydra / Defaults / Guardrails
# =============================================================================

def _compose_hydra_config(repo_root: Path, payload: SubmitPayload) -> DictConfig:
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(
            f"Missing Hydra config directory: {config_dir} "
            "(expected at repo root per blueprint)"
        )
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        try:
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)
        except Exception:
            if payload.strict:
                raise
            cfg = compose(config_name=payload.config_name, overrides=[])

    _ensure_defaults(cfg)
    _validate_minimal_schema(cfg, strict=payload.strict)
    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    cfg.setdefault("submit", {})
    cfg.submit.setdefault("predictions", "outputs/predictions/predictions.csv")
    cfg.submit.setdefault("out_dir", "outputs/submissions")
    cfg.submit.setdefault("csv_name", "submission.csv")
    cfg.submit.setdefault("zip_name", "submission.zip")
    cfg.submit.setdefault("n_bins", N_BINS_DEFAULT)
    cfg.submit.setdefault("strict", True)


def _validate_minimal_schema(cfg: DictConfig, *, strict: bool) -> None:
    missing: list[str] = []
    if "submit" not in cfg:
        missing.append("submit")
    else:
        for k in ("predictions", "out_dir", "csv_name", "zip_name"):
            if k not in cfg.submit:
                missing.append(f"submit.{k}")
    if missing:
        _raise_or_warn(f"[submit] missing required config keys: {', '.join(missing)}", strict=strict, quiet=False)


# =============================================================================
# IO & Snapshots
# =============================================================================

def _resolve_predictions(repo_root: Path, cfg: DictConfig, *, strict: bool) -> Path:
    raw = str(cfg.submit.predictions)
    preds = (Path(raw) if os.path.isabs(raw) else (repo_root / raw)).resolve()
    if not preds.exists():
        _raise_or_warn(f"[submit] predictions file not found: {preds}", strict=strict, quiet=False)
    return preds


def _resolve_out_dir(repo_root: Path, cfg: DictConfig) -> Path:
    raw = str(cfg.submit.out_dir)
    out = (Path(raw) if os.path.isabs(raw) else (repo_root / raw)).resolve()
    out.mkdir(parents=True, exist_ok=True)
    return out


def _emit_config_snapshot(out_dir: Path, cfg: DictConfig, preds_file: Path) -> None:
    try:
        snap_dir = out_dir / "snapshots"
        snap_dir.mkdir(exist_ok=True)

        # YAML snapshot (human-friendly)
        snap_yaml = snap_dir / "config_snapshot.yaml"
        OmegaConf.save(cfg, snap_yaml, resolve=True)

        # JSON run metadata (machine-friendly)
        meta = {
            "schema": "spectramind/submit_config_snapshot@v1",
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "predictions": str(preds_file),
            "out_dir": str(out_dir),
            "csv_name": str(getattr(cfg.submit, "csv_name", "submission.csv")),
            "zip_name": str(getattr(cfg.submit, "zip_name", "submission.zip")),
            "hydra_overrides": list(getattr(getattr(cfg, "hydra", {}), "overrides", [])) if hasattr(cfg, "hydra") else [],
        }
        (snap_dir / "config_snapshot.json").write_text(json.dumps(meta, indent=2))
    except Exception as e:  # pragma: no cover
        sys.stderr.write(f"[submit] warning: failed to write config snapshot: {e}\n")


# =============================================================================
# Helpers
# =============================================================================

def _sha256(p: Path, chunk: int = 1024 * 1024) -> str:
    import hashlib

    h = hashlib.sha256()
    try:
        with p.open("rb") as f:
            while True:
                b = f.read(chunk)
                if not b:
                    break
                h.update(b)
        return h.hexdigest()
    except Exception:
        return ""


def _raise_or_warn(msg: str, *, strict: bool, quiet: bool) -> None:
    if strict:
        raise RuntimeError(msg)
    if not quiet:
        sys.stderr.write(msg + " (continuing)\n")


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            if p.name == "src":
                return p.parent
            return p
    return here.parents[2]
