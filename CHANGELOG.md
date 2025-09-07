# 📜 Changelog — SpectraMind V50

All notable changes to this project will be documented here.  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## [Unreleased]

### 🚀 Added
* **Governance docs**: `SECURITY.md`, `CODE_OF_CONDUCT.md`, `CONTRIBUTING.md`, `LICENSE`.
* Baseline **`VERSION` file** (0.1.x series).
* Extended `dvc.yaml` with reproducible full pipeline:  
  `calibrate → train → predict → diagnose → submit`.
* `setup.cfg` with pytest, flake8, mypy, and Ruff lint/type rules.
* `pyproject.toml` with runtime + dev extras (`gpu`, `dev`).
* `requirements-kaggle.txt` (Kaggle-safe, pinned) and `requirements-dev.txt`.
* CI: added **artifact sweeper**, **docs build**, and **branch-protection auto-merge** workflows.
* Docs: added `ARCHITECTURE.md`, `docs/` with MkDocs config and diagrams.

### 🔄 Changed
* Repo promoted from **scaffold → production-grade** (per ADR-0001).
* Torch stack isolated into extras (`gpu`) for Kaggle compliance.
* CI enforces **CodeQL**, **pip-audit**, **Trivy**, and **SBOM generation** (CycloneDX + SPDX).
* CLI (`spectramind`) refactored to **Typer unified interface** with subcommands:  
  `calibrate`, `train`, `predict`, `diagnose`, `submit`.
* Dockerfile hardened: `python:3.10-slim`, rootless runtime, no unpinned system calls.

### ⚠️ Deprecated
* None.

### ❌ Removed
* None.

### 🛠️ Fixed
* Stable Hydra config composition (`configs/train.yaml`).
* DVC pipeline now persists processed artifacts in `data/processed/`.
* Kaggle notebooks auto-sync with repo; no duplicated pipeline logic.
* Fixed schema validation drift (`schemas/events.schema.json` and `schemas/submission.schema.json`).

---

## [0.1.1] — 2025-09-06

### 🚀 Added
* Introduced `scripts/bump_version.sh` for automated semantic versioning.

### 🔄 Changed
* Dependency upgrades: `pytest ≥8.1`, `ruff ≥0.6.8`, `mypy ≥1.11`.
* Docs enhanced: reproducibility, Kaggle runtime constraints, ADR workflow clarified.

---

## [0.1.0] — 2025-09-05

### 🚀 Added
* 🎉 Initial public release of **SpectraMind V50** scaffold.
* CLI (`spectramind`) with `calibrate`, `train`, `predict` commands.
* DVC integration for raw → processed data lineage.
* Initial configs (`configs/train.yaml`, `configs/env/`).
* ADR system bootstrapped (`ADR/0001-choose-hydra-dvc.md`).

---
