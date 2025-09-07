# 🤝 Contributing Guide — SpectraMind V50

We welcome contributions to **SpectraMind V50** (NeurIPS 2025 Ariel Data Challenge).
This repository is **mission-grade**: every commit must be reproducible, secure, and well-documented.

---

## 🚀 Quickstart (Local Dev)

1. Clone and install dev environment:

   ```bash
   git clone https://github.com/your-org/spectramind-v50.git
   cd spectramind-v50
   make dev
   ```

   This will install dependencies, setup pre-commit hooks, and prepare a reproducible Python environment.

2. Verify everything passes:

   ```bash
   pre-commit run --all-files
   pytest -q
   ```

   ✅ **All pre-commit hooks must pass before PR submission.**

---

## 🧪 Development Rules

* **Tests first**:
  Every new feature or bugfix requires tests (`pytest`) in `tests/`.
  Aim for **≥80% coverage**, including negative/error paths.

* **Stable interfaces**:
  Do not break CLI commands, config schema (`configs/`), or public APIs without:

  * An **ADR** (Architecture Decision Record) in `ADR/`
  * A version bump (`VERSION` file + `pyproject.toml`)

* **Docs required**:

  * Update `docs/guides/` if configs/CLI change.
  * Document new configs in `configs/*/ARCHITECTURE.md`.
  * Add diagrams in `assets/diagrams/` if architecture changes.

* **Hydra/DVC reproducibility**:

  * Never hardcode paths or params.
  * All experiment parameters must be overridable via Hydra configs.
  * Data changes must be tracked via DVC (`dvc add`).

---

## 📝 Architecture Decision Records (ADR)

* Any **non-trivial change** (model architecture, pipeline flow, CI/CD policy, security) requires an ADR.

* Place ADRs in `/ADR/` named as:

  ```
  ADR-YYYYMMDD-title.md
  ```

* Each ADR should capture:

  * Context
  * Decision
  * Alternatives considered
  * Consequences

---

## 🔒 Security & Compliance

* **No secrets in code**. Use `.env` or Kaggle secret manager.
* **Pinned dependencies** only (`requirements-kaggle.txt`, `requirements-dev.txt`).
* All contributions run through:

  * CodeQL
  * pip-audit
  * Trivy (Docker)
  * SBOM generation

---

## ✅ PR Checklist

* [ ] `make lint && make test` passes locally
* [ ] Added/updated tests
* [ ] Updated docs/configs/ADRs as needed
* [ ] Bumped version if breaking changes
* [ ] No hardcoded paths, secrets, or unpinned deps

---
