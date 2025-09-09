# 🔐 Security Policy — SpectraMind V50

> NASA-grade rigor for a research-grade codebase. Security is a first-class, testable requirement.

## 📌 Supported Versions

We actively maintain the **latest `main` branch** and all tagged releases.

| Version | Supported                                |
| ------: | ---------------------------------------- |
|  `main` | ✅                                        |
|   `0.x` | ✅ (active dev, breaking changes allowed) |
|  `<0.x` | ❌                                        |

> **Kaggle**: Submissions **must** use pinned dependencies in `requirements-kaggle.txt` for reproducibility and supply-chain safety.

---

## 🛡️ Report a Vulnerability

1. **Do not open a public issue.**

2. Contact the security team:

   * Email: **[security@spectramind-v50.org](mailto:security@spectramind-v50.org)**
   * (Preferred) **GitHub Security Advisory** (private fork advisory)
   * Optional: encrypt with our PGP (placeholder)
     **PGP fingerprint:** `AAAA BBBB CCCC DDDD EEEE FFFF 1111 2222 3333 4444`
     **PGP key URL:** `/docs/keys/security.asc`

3. Include:

   * Commit hash / release tag
   * Repro steps, expected vs. actual
   * Minimal PoC
   * Environment (OS, Python, Docker/Kaggle)
   * Impact assessment (confidentiality/integrity/availability)

**SLA:** Acknowledge **≤72h**, initial fix/mitigation plan **≤30 days** (expedited for High/Critical).

---

## 🧭 Triage & Severity

We use **CVSS v3.1** (or v4.0 when applicable).

| Severity | Target Acknowledge | Target Fix/Advisory |
| -------- | ------------------ | ------------------- |
| Critical | ≤24h               | ≤7 days             |
| High     | ≤48h               | ≤14 days            |
| Medium   | ≤72h               | ≤30 days            |
| Low      | ≤5 days            | Next release        |

Embargoes are honored for coordinated disclosures. We may ship **out-of-band releases** for Critical issues.

---

## 🔒 Security Principles

**1) Dependency hygiene**

* **Kaggle runtime:** fully pinned in `requirements-kaggle.txt`.
* **Dev/CI:** pinned in `requirements-dev.txt`; optional `constraints.txt` enforced in CI.
* **Lock capture:** `make freeze` emits `requirements.lock.txt`.
* **Audits:** `pip-audit` in CI; Dependabot for ecosystem bumping.

**2) Supply-chain protection**

* **SBOM:** CycloneDX JSON via Syft; stored at `artifacts/sbom.json` (`make sbom`).
* **Vuln scan:** Grype/Trivy in CI for filesystem/Docker.
* **Provenance:** Attest builds (planned SLSA-2+) and tag immutability (`make tag` requires clean tree).
* **Action hardening:** Pin GitHub Actions by **commit SHA**, least-privilege `permissions:` in workflows, and `concurrency:` to avoid race conditions.

**3) Code quality & safety**

* **Type safety:** `mypy` (strict subsets), \*\* Ruff\*\* for lint/format, **Bandit** (optional) for security lint.
* **Tests:** Unit + integration; error paths, CLI misuse, schema validations; coverage gates in CI.
* **No secrets in repo:** Use `.env` (local only), GitHub Actions Secrets, Kaggle Secrets. Pre-commit secret scanning enabled.
* **Schema discipline:** JSON schema for submissions & events; `check-jsonschema` in CI.

**4) Execution environments**

* **Kaggle:** Offline by default; 9h max; strict writes to `/kaggle/working`.
* **Local/CI:** Python **3.11** (min), venv managed by `make env`.
* **Docker:** Minimal base image (`python:3.11-slim`), non-root runtime, user-mapped IDs, read-only FS where feasible.

---

## 🛰️ Scope of Protection

This project processes **AIRS spectroscopy (283 channels)** and **FGS1 photometry** data for research/competition use. Protections focus on:

* **Data confidentiality** (competition datasets, Kaggle private assets).
* **Traceability** (DVC-tracked `data/` and Hydra configs in `configs/`).
* **Integrity** (guard rails on PRs: pre-commit, schema checks, tests, linters, SBOM/vuln scans).

**Out of scope (typical)**

* Non-security bugs (typos, docs); harmless Kaggle warnings; third-party model performance claims.

---

## 🧪 CI/CD Security Tooling (enforced)

* **CodeQL** (static analysis)
* **Trivy** (FS & config scans; Docker images)
* **Syft → CycloneDX** (SBOM) + **Grype** (SBOM vuln scan)
* **pip-audit** (PyPI CVEs)
* **Ruff / mypy / mdformat / yamllint**
* **pre-commit** (secrets, nbstripout, JSON/YAML/TOML sanity)

> One-shot local parity: `make ci` mirrors CI checks.

---

## 🔑 Secrets & Credentials

* Never commit secrets.
* Prefer **OIDC-to-cloud** or per-job **scoped GitHub secrets**.
* **Kaggle:** use Dataset or Secret store (no inline tokens in notebooks).
* Rotate credentials on role/team changes or after incidents.
* Git history is immutable; treat leaked secrets as compromised.

---

## 🧰 Secure Coding Guidelines

* Validate all external inputs (file paths, JSON, env vars).
* Use safe temp dirs (`tempfile`), avoid `shell=True`.
* Prefer hash-pinned downloads; verify checksums when fetching artifacts.
* Avoid dynamic `eval/exec`; restrict plugin loading to vetted entry points.
* Log minimal PII; redact secrets in logs; prefer structured JSON logs.
* Keep numerical routines robust: NaN/Inf guards, bounds checks, explicit dtype handling.

---

## 🧷 Kaggle-Specific Guardrails

* Runtime checks ensure Kaggle vs. local CI behavior (see tests under `tests/integration/`).
* No network calls unless the challenge explicitly allows (and then only via Kaggle Datasets).
* Deterministic seeds; CPU/GPU parity checks; filesystem write checks to `/kaggle/working`.

---

## 🧯 Incident Response

1. **Triage** (within SLA): confirm repro, assess impact & affected versions.
2. **Mitigation**: short-term config flags, feature gates, or revocation.
3. **Fix**: patch + tests + changelog; bump semver; build SBOM.
4. **Advisory**: private GH advisory → coordinated disclosure → public release notes.
5. **Post-mortem**: timeline, root cause, preventive actions (added tests/linters/rules).

---

## 📝 Coordinated Disclosure

We follow CVE best practices. If a reporter requests an embargo, we’ll coordinate timelines and credit contributors. We publish:

* Affected versions
* CVSS vector & score
* Workarounds and permanent fixes
* SBOM/VEX (if applicable)

---

## 🧾 security.txt (copy-paste to `/.well-known/security.txt`)

```
Contact: mailto:security@spectramind-v50.org
Encryption: https://spectramind-v50.org/docs/keys/security.asc
Preferred-Languages: en
Policy: https://github.com/<org>/<repo>/blob/main/SECURITY.md
Hiring: https://spectramind-v50.org/careers
Canonical: https://spectramind-v50.org/.well-known/security.txt
```

---

## ✅ Quick Commands

* Local parity checks: `make ci`
* SBOM: `make sbom` → `artifacts/sbom.json`
* Full scan bundle: `make scan`
* Lock environment: `make freeze`

---

### Appendix: Workflow Hardening Checklist

* [ ] All Actions pinned by **commit SHA**
* [ ] `permissions:` set to **least privilege** (e.g., `contents: read`)
* [ ] `concurrency:` groups to prevent overlapping writes
* [ ] Protected branches with required checks (`ci`, `sbom`, `scan`)
* [ ] Mandatory code owners for sensitive paths (`src/spectramind/calib/`, `scripts/`, `.github/workflows/`)
* [ ] Pre-commit: secrets scan, nbstripout, Ruff, mypy (where practical)
* [ ] Docker run non-root, read-only FS, and no host networking in CI

---
