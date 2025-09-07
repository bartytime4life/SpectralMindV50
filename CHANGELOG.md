Here’s a **CHANGELOG.md** scaffold for your repo that follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/) conventions and semantic versioning. I aligned it with your SpectraMind V50 workflow (VERSION file, pyproject.toml, ADR governance).

---

# 📜 Changelog — SpectraMind V50

All notable changes to this project will be documented here.
This project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

---

## \[Unreleased]

### Added

* Initial **SECURITY.md**, **CODE\_OF\_CONDUCT.md**, **CONTRIBUTING.md**, and **LICENSE**.
* Baseline **`VERSION` file** (0.1.x series).
* `dvc.yaml` upgraded with reproducible calibrate → train → predict pipeline.
* `setup.cfg` with pytest, flake8, mypy, and Ruff lint rules.
* `pyproject.toml` with runtime + dev extras (`gpu`, `dev`).
* `requirements-kaggle.txt` and `requirements-dev.txt` with pinned versions.

### Changed

* Promoted repo from scaffold → production-grade (per ADR-0001).
* Moved Torch stack into **extras** (`gpu`) for Kaggle safety.
* CI now enforces CodeQL, pip-audit, SBOM scans.

### Deprecated

* None.

### Removed

* None.

### Fixed

* Stable config composition in Hydra (`configs/train.yaml`).
* DVC pipeline now persists processed data (`data/processed`).

---

## \[0.1.1] — 2025-09-06

### Added

* Introduced `VERSION` file bump script (`scripts/bump_version.sh`).

### Changed

* Upgraded dependencies (pytest ≥8.1, ruff ≥0.6.8, mypy ≥1.11).
* Enhanced docs: reproducibility and Kaggle constraints clarified.

---

## \[0.1.0] — 2025-09-05

### Added

* 🎉 Initial public release of **SpectraMind V50** scaffold.
* CLI (`spectramind`) with `calibrate`, `train`, `predict` subcommands.
* DVC integration for raw → processed data flow.
* Initial configs (`configs/train.yaml`, `configs/env/`).
* ADR system bootstrapped (`ADR/`).

---

Would you like me to also add a **GitHub Actions workflow (`.github/workflows/release.yml`)** that automatically:

1. Reads `VERSION`
2. Updates CHANGELOG on tag
3. Publishes release notes with artifacts (Docker, Kaggle submission package)?
