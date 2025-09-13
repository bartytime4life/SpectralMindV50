# 🔐 Security Policy — SpectraMind V50

Mission-critical guardrails for a research-grade, competition-safe pipeline.

> TL;DR
>
> * Report privately to **[security@spectramind-v50.org](mailto:security@spectramind-v50.org)** (don’t open issues).
> * Use **pinned deps** and **offline** Kaggle kernels.
> * Generate an **SBOM** and run **`make scan`** before releases.
> * No secrets in repo. Use CI/Kaggle secrets.

---

## 📌 Supported Versions

We actively maintain the latest **`main`** branch and tagged releases (`0.x` is active dev).

| Version | Supported                                |
| ------: | :--------------------------------------- |
|  `main` | ✅                                        |
|   `0.x` | ✅ (active dev; breaking changes allowed) |
|  `<0.x` | ❌                                        |

> **Kaggle**: Submissions **must** use `requirements-kaggle.txt` and **disable internet**.

---

## 🛡️ Reporting a Vulnerability

**Do not** open a public issue. Contact us privately:

* Email: **[security@spectramind-v50.org](mailto:security@spectramind-v50.org)**
* or open a **Private Security Advisory** in GitHub (Security → Advisories → New draft)

Please include:

* Affected commit/tag (`git rev-parse --short HEAD`)
* Repro steps / minimal PoC
* Expected vs. actual behavior
* Impact (confidentiality / integrity / availability)
* Any logs, stack traces, screenshots

**Response targets**

* Acknowledgement: **≤ 72h**
* Initial remediation plan: **≤ 30 days** (severity-dependent)
* Credit in release notes (opt-in), unless anonymity requested
  (PGP available on request for encrypted follow-ups)

---

## 🔒 Security Principles

### 1) Dependency Hygiene

* **Runtime pins** for Kaggle in `requirements-kaggle.txt`; no ad-hoc `pip install` at runtime.
* **Dev/CI pins** in `requirements-dev.txt` (or `pyproject.toml` + lockfile if you standardize).
* Optional `constraints.txt` for resolver stability.
* Automated checks:

  * `pip-audit` (**CVE**s)
  * Dependabot (grouped PRs; reviewed & pinned)
  * **SBOM** produced per release (`make sbom`)

### 2) Supply Chain Protection

* **SBOM** (CycloneDX/SPDX) via Syft → track in `artifacts/sbom.json`.
* Optional Grype/Trivy scans for packages/containers.
* Reproducible builds:

  * Pinned **Python** minor version (3.11)
  * Pinned wheels where feasible
* Shell scripts: `bash -Eeuo pipefail`, no `curl | bash`, pinned SHAs for tools.

### 3) Code Quality & Safety

* **mypy** strict in core; relaxed in CLI/tests (see `setup.cfg`).
* **ruff/flake8** + recommended plugins (docstrings, bugbear, annotations).
* Tests cover:

  * misuse of CLI flags, invalid inputs
  * schema validation errors (submissions)
* **No secrets**: `.env` (git-ignored), CI secrets, or Kaggle secrets only.

### 4) Execution Environments

* **Kaggle**:

  * **Offline** kernels
  * ≤ **9h** GPU wallclock, ≤ **30 GB** RAM
  * Inputs: `/kaggle/input/...`
  * Outputs: `/kaggle/working/...`
  * Use **pre-packaged calibrated data** when available to save time/budget.

* **Local/CI**:

  * Reproducible via **Hydra + DVC**
  * Deterministic seeds
  * JSONL/TOML manifests recorded to `artifacts/`

* **Docker (optional)**:

  * Minimal base, non-root runtime
  * GPU runtime optional
  * Pinned OS packages

---

## 🛰️ Scope of Protection

Scope: **challenge data** processing and produced artifacts.

We protect:

* 🔒 **Dataset confidentiality** (per rules/licenses)
* 📑 **Traceability** of configs & calibration (Hydra snapshots; DVC lineage)
* 🧪 **Integrity** of results (schema-checked submissions; manifest checksums)
* 🔄 **Provenance** of artifacts (manifests, scaler stats, run metadata)

Out of scope (unless they materially affect safety): typos, style nits, non-security docs.

---

## 🛠️ Security Tooling (CI/CD)

**CI checks** (suggested jobs):

* **CodeQL**: static analysis (Python)
* **Trivy**: filesystem / IaC scans (and container images if used)
* **Syft + Grype**: SBOM + vuln scan
* **pip-audit**: Python dependency CVEs
* **Ruff / flake8**: lint / import safety
* **pre-commit**: YAML lint, secrets scan, notebook output strip

**Local helpers**

```bash
make sbom   # create CycloneDX SBOM → artifacts/sbom.json
make scan   # pip-audit + basic lint (non-failing locally)
make check  # pre-commit, lint, types, tests (strict; matches CI)
```

---

## 🔧 Reproducibility & Artifact Integrity

* **Hydra**: strict mode; per-stage deterministic `hydra.run.dir` (see `dvc.yaml`) so DVC caches properly.
* **DVC**: reproducible pipeline (calibrate → preprocess → train → predict → diagnose → submit), deterministic artifact paths (no timestamps in data paths).
* **Submission**:

  * Validate with `bin/validate_submission.sh` (header order, table schema, numeric types, physics sanity).
  * Package with `bin/sm_submit.sh`, which validates prior to zipping.
* **Manifests**:

  * Store run metadata (config hashes, data versions, model ckpt SHA) in `artifacts/*/manifest.json`.
  * Optionally add file checksums for `submission.csv`.

---

## ✅ Best Practices for Contributors

* Run `make check` before pushing (pre-commit, lint, types, tests).
* Use `make sbom` & `make scan` to sanity-check dependencies.
* Never commit `.env` or real secrets; use CI/Kaggle secrets.
* Kaggle notebooks:

  * **only** read `/kaggle/input/...` and write `/kaggle/working/...`
  * use pre-calibrated zip datasets to avoid runtime calibration where possible.
* Keep CLI (I/O) separate from core logic (clean architecture):

  * `src/spectramind/cli/…` is a thin shell; heavy lifting in `src/spectramind/*`.
* Document notable runs in `artifacts/` (e.g., `events.jsonl`, `manifest.json`) for auditability.

---

## 🔁 Versioning, Tags & Backports

* Bump version via `make version` (syncs `VERSION → pyproject.toml`, commits & tags).
* Security fixes land on `main`, then are **cherry-picked** to the latest stable tag when applicable.
* No guaranteed backports to `<0.x` unless severity warrants.

---

## 🤝 Responsible Disclosure

We follow coordinated disclosure:

* **Low-risk**: patch silently in `main`
* **Moderate/Critical**: out-of-band patch + advisory
* CVSS scoring used internally to prioritize
* Fixes tie back to commit hashes and (when applicable) DVC data versions
* Reporters credited (if desired) in release notes

---

## 📬 security.txt (optional)

Publish on your project/org site (e.g., GitHub Pages or org domain):

```
Contact: mailto:security@spectramind-v50.org
Preferred-Languages: en
Policy: https://github.com/<OWNER>/<REPO>/blob/main/SECURITY.md
Acknowledgements: https://github.com/<OWNER>/<REPO>/releases
```

---

## 📄 License & Legal

* Licensed under **MIT**. Submissions must comply with challenge rules and dataset licenses.
* By reporting vulnerabilities, you agree to act in good faith and avoid privacy violations or data exfiltration.

---

### Quick Checklist (copy into PR templates)

* [ ] Pinned deps only (`requirements-kaggle.txt` for Kaggle; no runtime installs).
* [ ] `make check` and `make scan` clean.
* [ ] SBOM regenerated (`make sbom`) and attached to release artifacts.
* [ ] No secrets committed; CI/Kaggle secrets used.
* [ ] Submission validated (`bin/validate_submission.sh`) before `submit`.
* [ ] DVC stages produce deterministic paths; no timestamped directories in data/ artifacts.

---
