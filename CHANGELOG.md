# Changelog — SpectraMind V50

All notable changes to this project are documented here.  
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html) and the
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) format.

---

## [Unreleased]

### 🚀 Added
- **Submission Table Schema**
  - `schemas/submission.tableschema.sample_id.json` — Frictionless Table Schema with `sample_id` as canonical key; `mu_000..mu_282` and `sigma_000..sigma_282` required (`sigma_* ≥ 0`); strict `dialect` and `encoding` for reproducible CSV I/O.
  - `schemas/submission_header.csv` — one-line header template enforcing column order for Kaggle submissions.
- **Preprocess presets** (Hydra)
  - `configs/preprocess/presets/fast.yaml` — CI/Kaggle budget; robust normalize; no detrend/binning/augment; NPZ export.
  - `configs/preprocess/presets/nominal.yaml` — balanced default; poly-detrend (FGS1), robust normalize, calib-or-fixed binning, Hann windows, light physics-safe augment.
  - `configs/preprocess/presets/strict.yaml` — research-grade; Savitzky–Golay detrend (FGS1 + AIRS), robust normalize, calib-strict binning, Hann + 50% overlap, Parquet+Zstd, deep assertions.
- **Preprocess methods**
  - `load.yaml` — split-aware; accepts Parquet/NPZ; paired channel guarantees; deterministic shuffling; temp-file filtering.
  - `mask.yaml` — NaN/Inf, saturation, robust spike/cosmic; dilation/min-run; mask coverage assertions; optional artifact export.
  - `normalize.yaml` — train-fit/apply-everywhere; mask-aware; per-sensor scope; exported stats (`scaler/`); manifest.
  - `window.yaml` — phase-aware centering with tolerance + sliding fallback; overlap-safe; label alignment; boundary mask dilation.
  - `pack.yaml` — fused + per-sensor tensors; union/per-sensor masks; metadata passthrough + coverage metrics; strict shape/bin checks.
  - `tokenize.yaml` — time & spectral positional encodings (sinusoid/rotary/learned), concat-guardrails, broadcast options.
  - `export.yaml` — compact NPZ/Parquet + `manifest.json` (key knobs captured).
- **Docs**
  - `configs/preprocess/ARCHITECTURE.md` — stage contracts, shapes, schemas, CLI patterns, safety gates (Mermaid diagram).
  - `configs/preprocess/README.md` — quickstart, env knobs, outputs, troubleshooting.
- **Makefile (mission-grade)**
  - Preset shortcuts: `make preprocess.{fast,nominal,strict} SPLIT=… OVERRIDES="k=v"`.
  - Security bundle: `make scan` (SBOM, pip-audit, YAML/MD lint, licenses, Trivy).
  - Release flow: `make bump|version|tag|push-tag|release`.
- **Integration tests**
  - `tests/integration/test_calib_chain.py`
  - `tests/integration/test_end_to_end_cli.py`
- **Kaggle assets**
  - `kaggle/README.md`, pinned `requirements-kaggle.txt`, and notebook templates wired to the CLI (`spectramind`).
- **Diagnostics**
  - FFT/UMAP analysis: `src/spectramind/diagnostics/spectral_analysis.py`, `src/spectramind/diagnostics/dimensionality.py` — see [ADR-0004].
  - Physics-informed checks: `src/spectramind/validators/physics.py` — see [ADR-0002].
  - Unified HTML/JSONL reports: `src/spectramind/diagnostics/reports.py` — see [ADR-0002], [ADR-0004].
  - ADC quick-look: `src/spectramind/diagnostics/adc_diag.py` — see [ADR-0002].

### 🔄 Changed
- **Trace modeling** (`src/spectramind/calib/trace.py`): multi-order-ready center/width, NaN-safe math, stable denominators, background models (`column_median`, `row_poly`), Torch-first execution.
- **Photometry** (`src/spectramind/calib/photometry.py`): batch-aware `[..., T, H, W]`, adaptive apertures, PSF-weighted optimal extraction with variance propagation, full NaN safety.
- **CI workflows**: `.github/workflows/*` hardened with matrix pinning, cache keys, deterministic pytest.
- **Pre-commit**: updated `.pre-commit-config.yaml` (ruff/mypy/bandit/secrets, sane excludes).
- **Schemas**: tightened `schemas/submission.schema.json` and `schemas/events.schema.json`; added drift tests.
- **Diagnostics reporting** (`reports.py`): improved HTML/JSON generator (inline CSS, titles, deterministic UTF-8) — see [ADR-0002].

### 🛠️ Fixed
- Time-axis normalization and mask/variance alignment across calibration modules (`src/spectramind/calib/*`).
- PSF normalization edge cases in `photometry.py`.
- Docstring mismatches and dtype inconsistencies (Torch vs NumPy).
- ADC calibration diagnostics: clamped NaN/Inf + guarded Matplotlib fallback (`adc_diag.py`) — see [ADR-0002].

### 🧪 Performance
- Vectorized photometry loops & reduced tensor allocations (`photometry.py`).
- NumPy fallback only for `polyfit`; Torch-first everywhere else (`trace.py`).
- FFT analysis normalized per-sample, NaN-safe, batch-aware (`spectral_analysis.py`) — see [ADR-0004].

### 🔒 Security
- All new diagnostics/modules passed Bandit + CodeQL scanning (`.github/workflows/*`).
- SBOM generation target (`make sbom`) with Syft/CycloneDX fallback; included in `make scan`.

### ⚠️ Deprecated
- _None._

### ❌ Removed
- _None._

---

## [0.1.0] — 2025-09-12

Initial public baseline of **SpectraMind V50** with CLI-first pipeline, Hydra configs, DVC hooks, diagnostics, and Kaggle integration.

### Added
- Core CLI entrypoints: `calibrate`, `preprocess`, `train`, `predict`, `diagnose`, `submit`.
- Base data/env/profile configs; JSONL logger; initial CI workflow set.

---

## Release Links

[Unreleased]: https://github.com/bartytime4life/SpectralMindV50/compare/v0.1.0...HEAD  
[0.1.0]: https://github.com/bartytime4life/SpectralMindV50/releases/tag/v0.1.0

[ADR-0002]: docs/adr/0002-composite-physics-informed-loss.md  
[ADR-0004]: docs/adr/0004-dual-encoder-fusion-fgs1-airs.md
```

### Guardrail (pre-commit): ensure compare links point to the right repo + exist

Append to `.pre-commit-config.yaml`:

```yaml
- repo: local
  hooks:
    - id: changelog-compare-links
      name: changelog compare links (owner/repo + Unreleased anchor present)
      language: system
      entry: bash -lc '
        set -euo pipefail;
        test -f CHANGELOG.md || exit 0;
        OWNER_REPO="bartytime4life/SpectralMindV50";
        grep -qE "^\[Unreleased\]:" CHANGELOG.md;
        awk -v or="$OWNER_REPO" "
          /^\[/ && /]: https:\/\/github\.com\// {
            if (\$0 !~ or) { print \"Wrong repo in link: \" \$0 > \"/dev/stderr\"; exit 1 }
          }" CHANGELOG.md
      '
      files: ^CHANGELOG\.md$
```

### Tip (when you cut v0.1.1)

After tagging `v0.1.1`, add a link:

```
[0.1.1]: https://github.com/bartytime4life/SpectralMindV50/releases/tag/v0.1.1
[Unreleased]: https://github.com/bartytime4life/SpectralMindV50/compare/v0.1.1...HEAD
```
