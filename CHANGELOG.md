# 📜 Changelog — SpectraMind V50

All notable changes to this project will be documented here.  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

---

## [Unreleased]

### 🚀 Added
- **JSON Schemas**: `schemas/submission.tableschema.sample_id.json` (Tableschema with `sample_id` as the canonical key); stricter enums & patterns, full examples.
- **Integration tests**: Calibration chain & CLI E2E smoke (`tests/integration/test_calib_chain.py`, `tests/integration/test_end_to_end_cli.py`).
- **Kaggle assets**: Finalize `kaggle/README.md` and notebook template wiring; pinned `requirements-kaggle.txt`.
- **Docs**: Expanded `docs/` with calibration internals (aperture/optimal, trace Horne-style), pipeline diagrams, and repository conventions.

### 🔄 Changed
- **Trace modeling**: Multi-order ready center/width, NaN-safe math, stable denominators, background models (`column_median`, `row_poly`), and Torch-first execution.
- **Photometry**: Batch-aware `[..., T, H, W]`, adaptive apertures, PSF-weighted optimal extraction with variance propagation, full NaN safety.
- **CI**: Hardened workflows (matrix pinning, cache keys, deterministic pytest); pre-commit stack (ruff/black/isort/mypy/bandit/secrets).
- **Schemas**: Submission/events schema tightened; drift tests added.

### 🛠️ Fixed
- Time-axis normalization and mask/variance alignment across calib modules.
- PSF normalization edge cases (degenerate/empty masks) in optimal extraction.
- Minor docstring mismatches and dtype inconsistencies.

### ⚠️ Deprecated
- None.

### ❌ Removed
- None.

---

## [0.1.2] — 2025-09-08

### 🚀 Added
- **Calib self-tests**: Lightweight CPU-safe checks inside `photometry.py` and `trace.py`.
- **Guardrails (Kaggle)**: Runtime checks and permissions tests (`tests/integration/test_kaggle_runtime_guardrails.py`).
- **CI**: `ci.yml`, `kaggle_notebook_ci.yml`, `sbom-refresh.yml`, `artifact-sweeper.yml`.
- **Pre-commit**: Mission-grade stack with autofix and repo-wide excludes.

### 🔄 Changed
- **`src/spectramind/calib/photometry.py`**:  
  Batch-aware outputs, NaN-safe `_nansum`/`_nanmedian`, adaptive circular/elliptical apertures, robust sky annulus clipping, variance-aware errors.
- **`src/spectramind/calib/trace.py`**:  
  Robust axis normalization to `[..., T, Y, X]`, Horne-style optimal extraction with variance propagation, multi-order center/width, improved background modeling.
- **CLI**: Unified Typer interface remains stable; logging clarifications in `setup.cfg` and tests.
- **Docs**: Calibration and extraction sections expanded; ADR cross-links.

### 🛠️ Fixed
- PSF weight normalization in presence of masked pixels.
- Stable denominator guards in optimal extraction to avoid `NaN/Inf` propagation.
- Minor dtype casts and device moves in Torch paths.

### 🧪 Performance
- Vectorized inner loops; NumPy fallback only for polyfit/smoothing where necessary.
- Reduced allocations in per-frame apertures and PSF grids.

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

## Changelog conventions

- **Added / Changed / Fixed / Deprecated / Removed / Security / Performance** buckets.
- Dates are in **YYYY-MM-DD** (UTC).
- Patch releases focus on bug fixes & safety; minor releases may add features; major releases may break APIs.