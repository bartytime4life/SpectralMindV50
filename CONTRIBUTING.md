# 🤝 Contributing Guide — SpectraMind V50

Welcome to **SpectraMind V50** — our **mission-grade, reproducible AI pipeline** for the \[NeurIPS 2025 Ariel Data Challenge].
This repo follows **NASA-grade engineering**: every commit must be reproducible, auditable, and scientifically credible.

---

## 🚀 Quickstart (Local Dev)

1. Clone and install the dev environment:

   ```bash
   git clone https://github.com/your-org/spectramind-v50.git
   cd spectramind-v50
   make dev
   ```

   This sets up a reproducible Python env, pre-commit hooks, Hydra configs, and DVC integration.

2. Verify:

   ```bash
   pre-commit run --all-files
   pytest -q --disable-warnings
   ```

   ✅ **All hooks & tests must pass before PR submission.**

---

## 🧪 Development Rules

* **Tests First**

  * Write tests in `tests/` before/with your code.
  * Target ≥80% coverage, covering positive + failure paths.

* **Stable Interfaces**

  * Never break CLI (`spectramind ...`), config schemas (`configs/*`), or Kaggle submission format without:

    * An **ADR** in `/ADR/`
    * Version bump (`VERSION`, `pyproject.toml`, `CHANGELOG.md`)

* **Documentation Required**

  * Update `docs/guides/` when CLI/configs change.
  * Add/update `configs/*/ARCHITECTURE.md`.
  * Place new diagrams in `assets/diagrams/` (Mermaid preferred).

* **Hydra/DVC Discipline**

  * No hardcoded paths or params.
  * All experiments reproducible via Hydra overrides + DVC stages.
  * Data tracked via DVC (`dvc add`), never Git.

* **Kaggle Safety**

  * No internet calls in runtime code.
  * Respect Kaggle 9h limit & 20GB RAM.
  * Test locally with `/kaggle/input/...` style paths.

---

## 📝 Architecture Decision Records (ADR)

* Required for **non-trivial changes** (models, calibration flow, CI/CD, security).

* File under `/ADR/` as:

  ```
  ADR-YYYYMMDD-title.md
  ```

* Must include: Context, Decision, Alternatives, Consequences.

---

## 🔒 Security & Compliance

* **No secrets in code.** Use `.env`, Kaggle secrets, or GitHub OIDC.
* **Pinned dependencies only** (`requirements-*.txt`).
* Contributions are scanned via CI:

  * CodeQL, pip-audit, Trivy (Docker), Syft/Grype SBOM.
* PRs failing security gates will be blocked.

---

## ✅ PR Checklist

* [ ] `make lint && make test` passes locally
* [ ] Tests added/updated
* [ ] Docs/configs/diagrams updated
* [ ] ADR + version bump if breaking change
* [ ] No secrets, hardcoded paths, or unpinned deps
* [ ] Submission validator (`tests/unit/test_submission_validator_property.py`) passes on `artifacts/submission.zip`

---

## 📦 Workflow Integration

* **CLI UX**: All pipeline stages (`calibrate`, `train`, `predict`, `diagnose`, `submit`) are invoked via `spectramind` CLI. Keep CLI thin; put business logic in `src/spectramind/*`.
* **DVC Pipeline**: Every stage (calibrate → preprocess → train → predict → diagnose → submit) declared in `dvc.yaml`.
* **Artifacts**: All outputs (checkpoints, predictions, reports) go under `artifacts/`, tracked by DVC.
* **Kaggle Notebooks**: Auto-generated wrappers in `notebooks/` — do not duplicate pipeline logic in notebooks.

---

## 🌌 Scientific Integrity

* Models must respect astrophysical constraints:

  * FGS1 channel dominates (×58 weight)
  * Spectra must remain smooth, non-negative, physically plausible
* Any physics-informed losses, priors, or symbolic checks must be documented in code + `ARCHITECTURE.md`.

---

### TL;DR

🚦 **If your PR is not reproducible, tested, documented, and Kaggle-safe — it will not be merged.**

---
