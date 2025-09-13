Here’s a **drop-in `SECURITY.md`**—tight, GitHub-friendly, Kaggle-aware, and aligned with the repo’s tooling and CI. Replace your file with this.

````markdown
# 🔐 Security Policy — SpectraMind V50

Mission-critical guardrails for a research-grade, competition-safe pipeline.

---

## 📌 Supported Versions

We actively maintain the **latest `main` branch** and all tagged releases.

| Version | Supported                                |
|--------:|:-----------------------------------------|
| `main`  | ✅                                        |
| `0.x`   | ✅ (active dev; breaking changes allowed) |
| `<0.x`  | ❌                                        |

> **Kaggle submissions** must use **pinned dependencies** from `requirements-kaggle.txt` for reproducibility and supply-chain safety.

---

## 🛡️ Reporting a Vulnerability

If you believe you’ve found a security issue:

1. **Do not open a public GitHub Issue.**
2. Contact us privately:
   - Email: **security@spectramind-v50.org**
   - or use GitHub **Security Advisories** on this repository
3. Please include:
   - Affected commit/tag (`git rev-parse --short HEAD`)
   - Reproduction steps / PoC (minimal)
   - Expected vs. actual behavior
   - Impact assessment (confidentiality/integrity/availability)
   - Any logs, stack traces, or screenshots that help triage

**Response targets**
- Acknowledgement **≤ 72 hours**
- Initial remediation plan **≤ 30 days** (severity-dependent)
- Credit in release notes (opt-in), unless anonymity requested

Optional: if you prefer encryption, ask for our PGP key in your initial email.

---

## 🔒 Security Principles

### Dependency Hygiene
- Runtime pins in `requirements-kaggle.txt` (offline kernels; no `pip install` at runtime)
- Dev/CI pins in `requirements-dev.txt`; optional `constraints.txt` for resolver stability
- Automated checks in CI: **pip-audit** and Dependabot (PRs grouped & reviewed)

### Supply Chain Protection
- **SBOM** generation (CycloneDX/SPDX) via Syft; optional Grype scan
- Reproducible builds: pinned Python, pinned wheels where feasible
- Shell scripts run with `bash -Eeuo pipefail`; no unpinned curl|bash patterns

### Code Quality & Safety
- Type checks: **mypy** (strict for core, relaxed for CLI/tests)
- Linters: **ruff** (Black-compatible), **flake8** plug-ins as needed
- Tests cover error paths and misuse of CLI flags
- **No secrets** in repo; use `.env` (git-ignored), CI secrets, or Kaggle secrets

### Execution Environments
- **Kaggle** kernels: **offline**, ≤ **9h** GPU wallclock, ≤ **30 GB** RAM; outputs to `/kaggle/working`
- **Local/CI**: reproducible via Hydra + DVC; deterministic seeds; JSONL logs
- **Docker**: minimal base, non-root runtime; GPU optional; pinned OS packages when installed

---

## 🛰️ Scope of Protection

This repository processes **scientific challenge data** (FGS1 photometry + AIRS spectroscopy). We protect:

- 🔒 **Dataset confidentiality** (competition rules & licenses)
- 📑 **Traceability** of configs & calibration (Hydra snapshots; DVC lineage)
- 🧪 **Integrity** of results (schema-checked submissions; manifest checksums)
- 🔄 **Provenance** of artifacts (manifests, `scaler/` stats, and run metadata)

Out of scope (unless they materially affect safety): typos, stylistic nits, non-security doc issues.

---

## 🛠️ Security Tooling (CI/CD)

- **CodeQL** (static analysis for Python)
- **Trivy** (filesystem & IaC scanning; optional container image scan)
- **Syft + Grype** (SBOM + vulnerability scan)
- **pip-audit** (Python dependency CVEs)
- **Ruff / flake8** (lint, import safety)
- **pre-commit** (YAML lint, secrets scan, notebook output stripping)

Run locally:

```bash
make scan     # SBOM + pip-audit + basic linters (non-failing locally)
make sbom     # SBOM → artifacts/sbom.json
````

---

## 🤝 Responsible Disclosure

We follow coordinated disclosure practices:

* **Low-risk** issues (docs, warnings) → patched silently in `main`
* **Moderate/Critical** → out-of-band patch release + advisory
* CVSS scoring used internally to prioritize
* Fixes tie back to commit hashes and (when applicable) DVC data versions
* We credit reporters (if desired) in release notes

---

## ✅ Best Practices for Contributors

* Run `make check` before pushing (pre-commit, lint, types, tests)
* Use `make sbom` and `make scan` to sanity-check dependencies
* Never commit `.env` or real secrets; rely on CI/Kaggle secrets
* Kaggle notebooks should **only** read from `/kaggle/input/...` and write to `/kaggle/working`
* Keep CLI (I/O) separate from core logic (clean architecture)
* Document notable runs in `artifacts/` (e.g., `events.jsonl`, `manifest.json`) for auditability

---

## 📬 security.txt (optional)

Consider publishing a `.well-known/security.txt` in your project site or org:

```
Contact: mailto:security@spectramind-v50.org
Preferred-Languages: en
Policy: https://github.com/<OWNER>/<REPO>/blob/main/SECURITY.md
Acknowledgements: https://github.com/<OWNER>/<REPO>/releases
```

---

## 🧭 Versioning & Backports

* Security fixes land on `main`, then are **cherry-picked** to the latest stable tag if applicable.
* We do **not** guarantee backports to `<0.x` unless severity warrants.

---

## 📄 License & Legal

* Licensed under **MIT**. Submissions must comply with competition rules and dataset licenses.
* By reporting vulnerabilities, you agree to act in good faith and avoid privacy violations or data exfiltration.

---
