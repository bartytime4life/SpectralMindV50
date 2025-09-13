Here’s a hardened, cross-wired **`CONTRIBUTING.md`** you can drop at repo root. It fixes the loose connections to our CLI, Hydra/DVC pipeline, security policy, make targets, ADR naming, and Kaggle constraints — all consistent with the rest of SpectraMind V50.

---

# 🤝 Contributing Guide — SpectraMind V50

Welcome to **SpectraMind V50** — a **mission-grade, reproducible AI pipeline** for the NeurIPS 2025 **Ariel Data Challenge**.
This repo follows **NASA-grade engineering**: every change must be **reproducible, auditable, and scientifically credible**.

---

## 🚀 Quickstart (Local Dev)

1. Clone & bootstrap:

```bash
git clone https://github.com/your-org/spectramind-v50.git
cd spectramind-v50
make dev              # installs dev deps, pre-commit, typing/lint/test tools
pre-commit install
```

2. Verify the toolchain:

```bash
pre-commit run --all-files
pytest -q --disable-warnings
```

✅ All hooks & tests must pass **before** PR submission.

---

## 🧪 Development Rules

### Tests First

* Put tests in `tests/` and keep ≥ **80% coverage** (include failure/error paths).
* Unit tests for new code; integration tests for new stages or I/O.
* Run fast locally: prefer marking heavy tests with `@pytest.mark.slow`.

### Stable Interfaces

* Don’t break the public CLI (`spectramind …`), config shapes (`configs/**`), or **Kaggle submission** schema without:

  * an **ADR** in `/ADR/` (see below),
  * a **version bump** (`VERSION`, `pyproject.toml`, `CHANGELOG.md`).

### Documentation Required

* Update or add **docs** in `docs/guides/` when CLI/configs change.
* Keep `configs/**/ARCHITECTURE.md` in sync for changed groups.
* Put diagrams under `assets/diagrams/` (Mermaid preferred); keep SVGs committed.

### Hydra / DVC Discipline

* **No hardcoded paths** or “magic constants”.
* All experiments must be reproducible via **Hydra overrides** and **DVC**:

  * Pipeline stages: **calibrate → preprocess → train → predict → diagnose → submit** (`dvc.yaml`).
  * Deterministic artifact paths (no timestamps inside `data/**`, `artifacts/**`).
* Data is tracked with **DVC**, **never Git** (`dvc add ...`).

### Kaggle Safety

* **Offline kernels only**; no network calls in runtime code.
* Use inputs from `/kaggle/input/**` and write to `/kaggle/working/**`.
* Respect Kaggle limits: **≤ 9h** GPU, **≤ 30 GB** RAM.
* Prefer **pre-calibrated** datasets (see `package-precalibrated` stage) to save time.

---

## 🧭 Architecture Decision Records (ADR)

Create an ADR for any **non-trivial** changes (models, calibration flow, CI/CD, security, pipeline structure).

* Follow the repo’s existing naming convention:

  ```
  ADR/0002-physics-informed-losses.md
  ADR/0004-dual-encoder-fusion.md
  ```
* Include: **Context**, **Decision**, **Alternatives**, **Consequences**.
* Reference the ADR in your PR description and commit message.

---

## 🔒 Security & Compliance

* **No secrets** in code or Git history. Use `.env` (git-ignored), CI secrets, or Kaggle secrets.
* **Pinned dependencies** only (`requirements-kaggle.txt` for Kaggle).
* CI runs security gates (CodeQL, pip-audit, Trivy, Syft/Grype SBOM). Failing gates block PRs.
* Read the [Security Policy](SECURITY.md) for reporting procedures and supported versions.

Helpful local commands:

```bash
make check   # pre-commit, lint, types, tests (strict; mirrors CI)
make sbom    # CycloneDX SBOM → artifacts/sbom.json
make scan    # pip-audit + basic linters (non-failing locally)
```

---

## 🧰 CLI, Pipeline & Artifacts

### CLI UX

* All pipeline stages are driven via the **CLI** (keep CLI thin; put logic in `src/spectramind/*`):

  ```
  spectramind calibrate --in data/raw --out data/calibrated
  spectramind preprocess --in data/calibrated --out data/processed
  spectramind train
  spectramind predict --ckpt artifacts/train/ckpt.pt --out-csv artifacts/predictions/preds.csv
  spectramind diagnose --pred-csv artifacts/predictions/preds.csv --out-dir artifacts/diagnostics
  ```
* Our helper scripts can be used for ergonomics:

  * `bin/sm_train.sh` — training wrapper (profiles, resume, DVC track),
  * `bin/sm_submit.sh` — builds & validates `submission.zip`,
  * `bin/validate_submission.sh` — schema/physics checks,
  * `bin/sweep_artifacts.sh` — safe cleanup of old run dirs/caches.

### DVC Pipeline (reproducible)

* `dvc.yaml` defines **deterministic** outputs for each stage (no timestamps in data paths).
* Hydra run dirs are explicitly set per stage to avoid cache misses.
* Flip Hydra groups with DVC Experiments, e.g.:

  ```bash
  dvc exp run -S hydra.env=kaggle -S hydra.model=v50 -S hydra.training=fast_ci
  ```

### Artifacts

* Checkpoints, predictions, and reports live under `artifacts/**` and are tracked by DVC.
* **Submissions** must be validated via:

  ```bash
  bin/validate_submission.sh --csv outputs/submission/submission.csv --strict
  ```

  and packaged via:

  ```bash
  bin/sm_submit.sh --skip-predict
  ```

---

## 🌌 Scientific Integrity

* Models must respect astrophysical constraints:

  * **FGS1** channel’s outsized weight (\~×58),
  * Spectra **non-negative**, **smooth**, and **physically plausible**.
* Document any physics-informed losses/priors/symbolic checks in code **and** `ARCHITECTURE.md`.
* Keep run metadata & provenance (`manifest.json`, config snapshots, hashes) in `artifacts/**`.

---

## 🧾 Conventional Commits & Versioning

* Use **Conventional Commits** (`feat:`, `fix:`, `docs:`, `refactor:`, `chore:`…).
* Breaking changes require an **ADR** and a **version bump**.
* Bump & tag with:

  ```bash
  make version   # syncs VERSION → pyproject.toml, commits, creates tag vX.Y.Z
  ```

---

## ✅ PR Checklist

* [ ] `make lint && make test` (or `make check`) passes locally
* [ ] Tests added/updated; coverage ≥ 80%
* [ ] Docs/configs/diagrams updated as needed
* [ ] ADR + version bump if breaking change
* [ ] No secrets/hardcoded paths/unpinned deps
* [ ] `bin/validate_submission.sh` passes on your generated `submission.csv`
* [ ] DVC pipeline reproduces locally (`dvc repro`)
* [ ] SBOM generated for release candidates (`make sbom`)

---

## 🧪 Examples

**Run an experiment with alternate training profile:**

```bash
dvc exp run -S hydra.training=fast_ci
```

**Produce predictions & diagnostics:**

```bash
dvc repro predict
dvc repro diagnose
```

**Build & verify a submission:**

```bash
dvc repro submit     # or:
bin/sm_submit.sh     # validates, then zips to outputs/submission/submission.zip
```

---

### TL;DR

🚦 If your PR is **not reproducible, tested, documented, Kaggle-safe, and validated** — it will not be merged.

---
