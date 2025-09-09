# 🔐 Security Policy — SpectraMind V50

## 📌 Supported Versions

We actively maintain the **latest `main` branch** and all tagged releases.

| Version | Supported                                |
| ------- | ---------------------------------------- |
| `main`  | ✅                                        |
| `0.x`   | ✅ (active dev, breaking changes allowed) |
| `<0.x`  | ❌                                        |

⚠️ Kaggle submissions must **always use pinned dependencies** in  
`requirements-kaggle.txt` for reproducibility and security:contentReference[oaicite:0]{index=0}.

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
   * Minimal proof of concept

We aim to **acknowledge within 72h** and provide a fix/mitigation plan **within 30 days**.

---

## 🔒 Security Principles

### Dependency Hygiene
* Runtime: pinned in `requirements-kaggle.txt`  
* Dev/CI: pinned in `requirements-dev.txt`  
* Enforced with optional `constraints.txt`  
* Automated scans: Dependabot + `pip-audit`:contentReference[oaicite:1]{index=1}

### Supply Chain Protection
* SBOM (CycloneDX/SPDX) auto-generated via **Syft/Grype**:contentReference[oaicite:2]{index=2}
* GitHub Actions enforce `--require-hashes` where feasible
* No unpinned system calls in scripts (`bash -Eeuo pipefail`):contentReference[oaicite:3]{index=3}

### Code Quality & Safety
* Strict typing (`mypy --strict`):contentReference[oaicite:4]{index=4}
* Lint enforced (`ruff`, `flake8`):contentReference[oaicite:5]{index=5}
* Tests cover error paths & CLI misuse
* No secrets in repo — use `.env`, Kaggle secrets, or CI secrets:contentReference[oaicite:6]{index=6}

### Execution Environments
* Kaggle kernels: **offline, ≤9h runtime** enforced:contentReference[oaicite:7]{index=7}
* Local/CI: reproducible via Hydra + DVC:contentReference[oaicite:8]{index=8}
* Docker: minimal base (`python:3.10-slim`), non-root runtime:contentReference[oaicite:9]{index=9}

---

## 🛰️ Scope of Protection

This repo processes **scientific challenge data** (FGS1 photometry + AIRS spectra).  
Protections focus on:

* 🔒 **Data confidentiality** of competition datasets  
* 📑 **Traceability** of configs & calibration (`configs/` are DVC-tracked):contentReference[oaicite:10]{index=10}  
* 🧪 **Integrity** against malicious PRs (pre-commit hooks + CI checks):contentReference[oaicite:11]{index=11}

---

## 🛠️ Security Tooling (CI/CD Integrated)

* GitHub **CodeQL** (static analysis):contentReference[oaicite:12]{index=12}
* **Trivy** (container & IaC scan):contentReference[oaicite:13]{index=13}
* **Syft + Grype** (SBOM + vuln scan):contentReference[oaicite:14]{index=14}
* **pip-audit** (Python deps):contentReference[oaicite:15]{index=15}
* **Ruff / flake8** (lint + import safety):contentReference[oaicite:16]{index=16}
* **pre-commit** (YAML lint, secrets scan, nbstripout):contentReference[oaicite:17]{index=17}

---

## 🤝 Responsible Disclosure

We follow [CVE](https://cve.mitre.org/) rules:

* Low-risk issues (typos, Kaggle-only warnings) → patched silently.  
* Critical issues → **out-of-band release** + coordinated advisory.  
* All fixes are linked to commit hashes + DVC data versions for full reproducibility:contentReference[oaicite:18]{index=18}.

---

## ✅ Best Practices for Contributors

* Run `make check` before pushing (enforces lint, type, tests, schema).  
* Use `make sbom` + `make scan` to locally validate dependencies.  
* Never push `.env` or real secrets — repo CI enforces `.gitignore` + secret scan.  
* If working on Kaggle notebooks, **only import from `/kaggle/input/...`** (competition data mount):contentReference[oaicite:19]{index=19}.  
* Follow clean architecture patterns (separation of CLI vs. logic):contentReference[oaicite:20]{index=20}.  
* Document configs and pipeline runs in `artifacts/run_events.jsonl` for audit traceability:contentReference[oaicite:21]{index=21}.

---
