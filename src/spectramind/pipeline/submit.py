# src/spectramind/pipeline/submit.py
from __future__ import annotations

import json
import sys
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

from hydra import compose, initialize
from omegaconf import DictConfig, OmegaConf


@dataclass
class SubmitPayload:
    config_name: str
    overrides: list[str]
    predictions: Optional[str]
    out_zip: Optional[str]
    strict: bool
    quiet: bool
    env: Dict[str, Any]


def run(
    *,
    config_name: str = "submit",
    overrides: Iterable[str] | None = None,
    predictions: str | None = None,
    out_zip: str | None = None,
    strict: bool = True,
    quiet: bool = False,
    env: Dict[str, Any] | None = None,
) -> None:
    """
    Submission packager: validate schema + package artifacts (zip/csv/manifest).
    """
    payload = SubmitPayload(
        config_name=config_name,
        overrides=list(overrides or []),
        predictions=predictions,
        out_zip=out_zip,
        strict=strict,
        quiet=quiet,
        env=env or {},
    )

    repo_root = _find_repo_root()
    cfg = _compose_hydra_config(repo_root, payload)

    if payload.predictions is not None:
        cfg.submit.predictions = payload.predictions
    if payload.out_zip is not None:
        cfg.submit.out_zip = payload.out_zip

    # Validate schema (csv headers, shapes, types)
    _validate_predictions(cfg, repo_root, quiet=payload.quiet)

    # Create zip
    _package_submission(cfg, repo_root, quiet=payload.quiet)


# ---------------------------- Validation & Packaging -------------------------


def _validate_predictions(cfg: DictConfig, repo_root: Path, quiet: bool) -> None:
    preds_path = Path(cfg.submit.predictions)
    if not preds_path.exists():
        raise FileNotFoundError(f"Predictions file not found: {preds_path}")

    # Optional JSON Schema validation if present
    schema = repo_root / "schemas" / "submission.schema.json"
    if schema.exists():
        try:
            import jsonschema  # type: ignore
            import pandas as pd  # type: ignore

            data = pd.read_csv(preds_path)
            with schema.open("r") as f:
                submission_schema = json.load(f)
            jsonschema.validate(data.to_dict(orient="list"), submission_schema)
            if not quiet:
                sys.stderr.write("[submit] Predictions validated against JSON schema.\n")
        except Exception as e:
            raise RuntimeError(f"Submission schema validation failed: {e}") from e
    else:
        if not quiet:
            sys.stderr.write("[submit] No JSON schema found; skipping schema validation.\n")


def _package_submission(cfg: DictConfig, repo_root: Path, quiet: bool) -> None:
    out_zip = Path(cfg.submit.out_zip)
    out_zip.parent.mkdir(parents=True, exist_ok=True)

    manifest = {
        "predictions": str(Path(cfg.submit.predictions).resolve()),
        "created_at": __import__("time").time(),
    }

    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(cfg.submit.predictions, arcname=Path(cfg.submit.predictions).name)
        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    if not quiet:
        sys.stderr.write(f"[submit] Wrote package: {out_zip}\n")


# ---------------------------- Hydra / Utils ----------------------------------


def _compose_hydra_config(repo_root: Path, payload: SubmitPayload) -> DictConfig:
    config_dir = repo_root / "configs"
    if not config_dir.exists():  # pragma: no cover
        raise FileNotFoundError(f"Missing Hydra config directory: {config_dir}")

    with initialize(config_path=str(config_dir), version_base=None):
        try:
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)
        except Exception:
            if payload.strict:
                raise
            cfg = compose(config_name=payload.config_name, overrides=payload.overrides)

    _ensure_defaults(cfg)
    if not payload.quiet:
        sys.stderr.write("\n[hydra] Resolved submission config:\n")
        sys.stderr.write(OmegaConf.to_yaml(cfg, resolve=True) + "\n")
    return cfg


def _ensure_defaults(cfg: DictConfig) -> None:
    cfg.setdefault("submit", {})
    cfg.submit.setdefault("predictions", "outputs/predictions.csv")
    cfg.submit.setdefault("out_zip", "outputs/submission.zip")


def _find_repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in [here, *here.parents]:
        if (p / "pyproject.toml").exists() or (p / "setup.cfg").exists() or (p / ".git").exists():
            return p.parent if (p / "src").exists() and (p.name == "src") else p
    return here.parents[3]
