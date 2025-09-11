# Changelog — SpectraMind V50

All notable changes to this project are documented here.  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

---

## [Unreleased]

### 🚀 Added
- **Submission Table Schema**
  - `schemas/submission.tableschema.sample_id.json` — Frictionless Table Schema with `sample_id` as the canonical key; `mu_000..mu_282` required; `sigma_000..sigma_282` required with `minimum: 0`; strict `dialect` and `encoding` for reproducible CSV I/O.
  - `schemas/submission_header.csv` — one-line header template to enforce correct column order in Kaggle submissions.
- **Integration tests**
  - `tests/integration/test_calib_chain.py`
  - `tests/integration/test_end_to_end_cli.py`
- **Kaggle assets**
  - `kaggle/README.md`, pinned `requirements-kaggle.txt`, and notebook templates wired to the CLI (`spectramind.py`).
- **Docs**
  - Expanded calibration internals in `src/spectramind/calib/`, pipeline diagrams in `assets/diagrams/`, and repo conventions in `docs/`.
- **Diagnostics**
  - FFT/UMAP analysis in `src/spectramind/diagnostics/spectral_analysis.py` and `src/spectramind/diagnostics/dimensionality.py` — see [ADR-0004].
  - Physics-informed checks in `src/spectramind/validators/physics.py` — see [ADR-0002].
  - Unified HTML/JSONL report generator `src/spectramind/diagnostics/reports.py` — see [ADR-0002] & [ADR-0004].
  - ADC quick-look in `src/spectramind/diagnostics/adc_diag.py` — see [ADR-0002].

### 🔄 Changed
- **Trace modeling** (`src/spectramind/calib/trace.py`): multi-order-ready center/width, NaN-safe math, stable denominators, background models (`column_median`, `row_poly`), Torch-first execution.
- **Photometry** (`src/spectramind/calib/photometry.py`): batch-aware `[..., T, H, W]`, adaptive apertures, PSF-weighted optimal extraction with variance propagation, full NaN safety.
- **CI workflows**: `.github/workflows/` hardened with matrix pinning, cache keys, deterministic pytest.
- **Pre-commit**: `.pre-commit-config.yaml` updated (ruff/black/isort/mypy/bandit/secrets).
- **Schemas**: tightened `schemas/submission.schema.json` and `schemas/events.schema.json`; added drift tests.
- **Diagnostics reporting** (`src/spectramind/diagnostics/reports.py`): improved JSON/HTML generator (inline CSS, titles, deterministic UTF-8) — see [ADR-0002].

### 🛠️ Fixed
- Time-axis normalization and mask/variance alignment across calibration modules (`src/spectramind/calib/*`).
- PSF normalization edge cases in `photometry.py`.
- Docstring mismatches and dtype inconsistencies (Torch vs NumPy).
- ADC calibration diagnostics: clamped NaN/Inf + guarded matplotlib fallback (`adc_diag.py`) — see [ADR-0002].

### 🧪 Performance
- Vectorized photometry loops & reduced tensor allocations (`photometry.py`).
- NumPy fallback only for `polyfit`; Torch-first everywhere else (`trace.py`).
- FFT analysis normalized per-sample, NaN-safe, batch-aware (`spectral_analysis.py`) — see [ADR-0004].

### 🔒 Security
- All new diagnostics modules passed Bandit + CodeQL scanning (`.github/workflows/*`).

### ⚠️ Deprecated
- _None._

### ❌ Removed
- _None._

---

## Release Links

<!-- Replace OWNER/REPO and version tags when you cut a release -->
[Unreleased]: https://github.com/OWNER/REPO/compare/vX.Y.Z...HEAD
[ADR-0002]: docs/adr/0002-composite-physics-informed-loss.md
[ADR-0004]: docs/adr/0004-dual-encoder-fusion-fgs1-airs.md
