# 📜 Changelog — SpectraMind V50

All notable changes to this project will be documented here.  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

---

## [Unreleased]

### 🚀 Added
- JSON Schemas: `schemas/submission.tableschema.sample_id.json` (Tableschema with `sample_id` as canonical key); stricter enums/patterns and full worked examples.
- Integration tests: calibration chain (`tests/integration/test_calib_chain.py`) and CLI end-to-end smoke test (`tests/integration/test_end_to_end_cli.py`).
- Kaggle assets: finalized `kaggle/README.md`, notebook template wiring, and pinned `requirements-kaggle.txt`.
- Docs: expanded calibration internals (aperture/optimal, Horne-style trace), pipeline diagrams, and repo conventions.

### 🔄 Changed
- Trace modeling: multi-order ready center/width, NaN-safe math, stable denominators, background models (`column_median`, `row_poly`), and Torch-first execution.
- Photometry: batch-aware `[..., T, H, W]`, adaptive apertures, PSF-weighted optimal extraction with variance propagation, full NaN safety.
- CI: hardened workflows (matrix pinning, cache keys, deterministic pytest); pre-commit stack updated (ruff/black/isort/mypy/bandit/secrets).
- Schemas: tightened submission/events schema; added drift tests.

### 🛠️ Fixed
- Time-axis normalization and mask/variance alignment across calibration modules.
- PSF normalization edge cases (degenerate/empty masks) in optimal extraction.
- Docstring mismatches and dtype inconsistencies in Torch/NumPy paths.

### 🧪 Performance
- Vectorized photometry loops and reduced tensor allocations.
- NumPy fallback only for polyfit/smoothing (Torch-first everywhere else).

### 🔒 Security
- None in this cycle (all workflows passed CodeQL/Bandit).

### ⚠️ Deprecated
- None.

### ❌ Removed
- None.

---

## [0.1.2] — 2025-09-08
### 🚀 Added
- Calibration self-tests: lightweight CPU-safe checks in `photometry.py` and `trace.py`.
- Kaggle guardrails: runtime checks + permission tests (`tests/integration/test_kaggle_runtime_guardrails.py`).
- CI workflows: `ci.yml`, `kaggle_notebook_ci.yml`, `sbom-refresh.yml`, `artifact-sweeper.yml`.
- Pre-commit: mission-grade stack with autofix and repo-wide excludes.

### 🔄 Changed
- `src/spectramind/calib/photometry.py`: batch-aware outputs, NaN-safe reductions, adaptive apertures, robust sky annulus clipping, variance-aware errors.
- `src/spectramind/calib/trace.py`: normalized axes to `[..., T, Y, X]`, Horne-style optimal extraction with variance propagation, multi-order center/width, improved background models.
- CLI: unified Typer interface, clarified logging in `setup.cfg` and tests.
- Docs: calibration/extraction sections expanded; ADR cross-links.

### 🛠️ Fixed
- PSF weight normalization with masked pixels.
- Stable denominator guards in optimal extraction (`NaN`/`Inf` suppression).
- Dtype casts and device alignment in Torch paths.

### 🧪 Performance
- Vectorized inner loops; reduced allocations in per-frame apertures and PSF grids.

---

## [0.1.1] — 2025-09-06
### 🚀 Added
- `scripts/bump_version.sh` for automated semantic versioning.

### 🔄 Changed
- Dependency upgrades: `pytest ≥8.1`, `ruff ≥0.6.8`, `mypy ≥1.11`.
- Docs: reproducibility notes, Kaggle runtime constraints, ADR workflow clarity.

---

## [0.1.0] — 2025-09-05
### 🚀 Added
- 🎉 Initial public release of **SpectraMind V50** scaffold.
- CLI (`spectramind`) with `calibrate`, `train`, `predict`.
- DVC integration for raw → processed lineage.
- Initial configs (`configs/train.yaml`, `configs/env/`).
- ADR system bootstrapped (`ADR/0001-choose-hydra-dvc.md`).

---

## Conventions
- Use **Added / Changed / Fixed / Deprecated / Removed / Security / Performance** buckets.
- Dates in **YYYY-MM-DD** (UTC).
- Patch = bug fixes & safety; minor = features; major = breaking changes.
