# 🔐 Security Policy — SpectraMind V50

## 📌 Supported Versions

We actively maintain the **latest `main` branch** and all tagged releases.

| Version | Supported                                |
| ------- | ---------------------------------------- |
| `main`  | ✅                                        |
| `0.x`   | ✅ (active dev, breaking changes allowed) |
| `<0.x`  | ❌                                        |

⚠️ Kaggle submissions must always use the **pinned dependencies** in `requirements-kaggle.txt` for reproducibility and security.

---

## 🛡️ Reporting a Vulnerability

If you discover a vulnerability:

1. **Do not open a public GitHub Issue.**
2. Email: **[security@spectramind-v50.org](mailto:security@spectramind-v50.org)**  
   or use GitHub’s [Security Advisories](https://docs.github.com/code-security/security-advisories/repository-security-advisories).
3. Include:
   * Repo commit hash
   * Steps to reproduce
   * Expected vs. actual behavior
   * Proof of concept (minimal)

We aim to **acknowledge within 72h** and provide a fix/mitigation plan **within 30 days**.

---

## 🔒 Security Principles

* **Dependency hygiene**
  * Runtime: pinned in `requirements-kaggle.txt`
  * Dev/CI: pinned in `requirements-dev.txt`
  * Enforced with optional `constraints.txt`
  * Automated scans via Dependabot + `pip-audit`

* **Supply chain protection**
  * SBOM (CycloneDX/SPDX) auto-generated via Syft/Grype
  * GitHub Actions enforce `--require-hashes` where feasible
  * No unpinned system calls in scripts

* **Code quality & safety**
  * Strict type-checking (`mypy --strict`)
  * Lint enforced (`ruff`, `flake8`)
  * Test coverage includes error paths & CLI misuse
  * No secrets hard-coded — use `.env`, Kaggle secrets, or CI secrets

* **Execution environments**
  * Kaggle kernels: no internet, ≤9h runtime enforced
  * Local/CI: Conda/Poetry lock files provided
  * Docker: minimal base image (`python:3.10-slim`), rootless runtime

---

## 🛰️ Scope of Protection

This repo processes **scientific challenge data** (FGS1 photometry + AIRS spectra):contentReference[oaicite:0]{index=0}.  
Protections focus on:

* 🔒 **Data confidentiality** of competition datasets
* 📑 **Traceability** of calibration/training configs (`configs/` are DVC-tracked)
* 🧪 **Integrity** against malicious PRs (pre-commit hooks + CI checks)

---

## 🛠️ Security Tooling (integrated in CI)

* GitHub **CodeQL** (static analysis)
* **Trivy** (container scan)
* **Syft + Grype** (SBOM + vuln scan)
* **pip-audit** (PyPI deps)
* **ruff / flake8** (lint + import safety)
* **pre-commit** (YAML lint, secrets-scan)

---

## 🤝 Responsible Disclosure

We follow [CVE](https://cve.mitre.org/) rules.  
* Low-risk issues (typos, Kaggle-only warnings) may be patched silently.  
* Critical issues may trigger **out-of-band releases** and coordinated advisories.

---
