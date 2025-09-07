# 🔐 Security Policy — SpectraMind V50

## 📌 Supported Versions

We actively maintain the **latest main branch** and tagged releases.

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
2. Email: **[security@spectramind-v50.org](mailto:security@spectramind-v50.org)** (or use GitHub’s [Security Advisories](https://docs.github.com/code-security/security-advisories/repository-security-advisories)).
3. Include:

   * Repo commit hash
   * Steps to reproduce
   * Expected vs. actual behavior
   * Any proof of concept (minimal)

We aim to acknowledge reports **within 72 hours** and provide a fix or mitigation plan **within 30 days**.

---

## 🔒 Security Principles

* **Dependency hygiene**

  * Runtime: pinned in `requirements-kaggle.txt`
  * Dev/CI: pinned in `requirements-dev.txt`
  * Resolved with optional `constraints.txt`
  * Automated scans via Dependabot + `pip-audit`

* **Supply chain protection**

  * SBOM (CycloneDX/SPDX) auto-generated via Syft/Grype
  * GitHub Actions enforce `--require-hashes` where feasible
  * No unpinned system calls in scripts

* **Code quality & safety**

  * Type-checked (`mypy` strict)
  * Lint enforced (`ruff`, `flake8`)
  * Test coverage includes error-paths and CLI misuse
  * Secrets never hard-coded — use `.env` or Kaggle secrets manager

* **Execution environments**

  * Kaggle kernels: no internet, ≤9h runtime enforced
  * Local/CI: Conda/Poetry lock files provided
  * Docker: minimal base image (`python:3.10-slim`), rootless runtime

---

## 🛰️ Scope of Protection

This repo handles **scientific challenge data** (FGS1 + AIRS). Security protections focus on:

* 🔒 Ensuring **data confidentiality** of competition datasets
* 📑 Maintaining **traceability** of calibration and training configs (`configs/` are DVC-tracked)
* 🧪 Preventing **malicious contributions** in pull requests (pre-commit hooks + CI checks)

---

## 🛠️ Security Tools Integrated

* GitHub CodeQL (static analysis)
* Trivy (container scan)
* Syft + Grype (SBOM + vuln scan)
* pip-audit (PyPI deps)
* ruff / flake8 (linting security rules, import safety)
* pre-commit hooks (YAML lint, secrets-scan)

---

## 🤝 Responsible Disclosure

We follow [CVE numbering authority rules](https://cve.mitre.org/) for disclosure.
Low-risk issues (e.g. typos, Kaggle-specific warnings) will be patched silently.
Critical issues may trigger an out-of-band release and coordinated advisory.

---
