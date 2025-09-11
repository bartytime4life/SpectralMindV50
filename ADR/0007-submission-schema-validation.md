# ADR 0007 — Submission Schema & Validation

> **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge  
> **Status:** ✅ Accepted | **Date:** 2025-09-07  
> **Tags:** `submission` · `schema` · `validation` · `kaggle` · `ci`  
> **Owners:** ML/Infra WG (Lead: Andy Barta), Data Ops, CI/CD WG

---

## 1) Context

The Ariel Data Challenge leaderboard requires predictions in a **CSV submission file**:

- Each row = 1 observation (id).
- Columns = `id`, 283 mu values, 283 sigma values.
- Evaluation metric: Gaussian Log-Likelihood (GLL) → punishes overconfident sigma and misaligned mu.

Risks without schema enforcement:

- Wrong column count or header names → Kaggle rejects submission.
- Negative sigma values → physically invalid, leaderboard penalty.
- Floating-point formatting issues → silent errors.
- CI vs Kaggle mismatch → local passes, submission fails.

We need **strict schema validation** in CI and before packaging.

---

## 2) Decision

Adopt a **formal submission schema** + validation pipeline:

- **JSON Schema** (`schemas/submission.schema.json`) defines:
  - `id: string`
  - `mu: array[number], length=283`
  - `sigma: array[number ≥ 0], length=283`
- **Validation tooling:**
  - Python: `jsonschema` + pandas wrapper.
  - CI: schema validation step on all candidate submissions.
  - CLI: `spectramind submit --validate` runs schema check before packaging.
- **Kaggle safety:**
  - `bin/kaggle-boot.sh` includes validation pre-run.
  - Fails early if schema mismatch.

---

## 3) Drivers

- **Safety** — prevent wasted leaderboard submissions.
- **Scientific credibility** — ensure sigma ≥ 0, mu length correct.
- **Reproducibility** — schema + CI tests guarantee outputs are stable across runs.
- **Velocity** — developers get immediate feedback before submitting to Kaggle.

---

## 4) Alternatives

| Option                             | Pros                   | Cons                        |
|------------------------------------|------------------------|-----------------------------|
| Rely on Kaggle feedback            | Simple                 | Too late, wasted runs       |
| Ad-hoc `assert` checks             | Lightweight            | Inconsistent, easy to miss  |
| **Chosen: JSON Schema + CI gates** | Rigorous, standardized | Requires maintaining schema |

---

## 5) Architecture

```mermaid
flowchart TD
  A["Model Predictions (mu/sigma)"] --> B["spectramind submit --validate"]
  B --> C["JSON Schema Validation (schemas/submission.schema.json)"]
  C -->|pass| D["Package submission.zip"]
  C -->|fail| E["Block CI / Kaggle submit"]
  D --> F["Kaggle Leaderboard"]
````

---

## 6) Implementation Plan

1. **Schema file**

   * `schemas/submission.schema.json` with strict rules.

2. **Validation library**

   * `src/spectramind/utils/schema.py` → wrapper for `jsonschema` + pandas.

3. **CLI integration**

   * `spectramind submit --validate` → validates CSV against schema.

4. **CI step**

   * `.github/workflows/ci.yml` job runs validation on `artifacts/submission.csv`.

5. **Kaggle integration**

   * `bin/kaggle-boot.sh` calls schema validation before submit.

---

## 7) Risks & Mitigations

| Risk                                        | Mitigation                                      |
| ------------------------------------------- | ----------------------------------------------- |
| Schema drift (columns updated by challenge) | Weekly drift job compares Kaggle docs vs schema |
| Extra runtime overhead                      | Validation is O(N) in rows; negligible          |
| Developer bypasses validation               | Pre-commit hook ensures schema check            |

---

## 8) Consequences

* ✅ Submissions guaranteed valid before upload.
* ✅ Prevents leaderboard downtime from invalid CSVs.
* ✅ Aligned with ADR-0006 reproducibility (schema as contract).
* ❌ Requires maintaining schema when Kaggle changes rules.

---

## 9) Compliance Gates (CI)

* [ ] All submissions validated against `schemas/submission.schema.json`.
* [ ] CI blocks merges if validation fails.
* [ ] `spectramind submit --validate` returns 0 on success, non-0 on failure.
* [ ] Kaggle runner auto-checks schema before upload.

---

## 10) References

* Submission schema file
* SpectraMind repo blueprint & scaffold
* Reproducibility standards ADR-0006
* Scientific context: physical plausibility of spectra

---

````

### Why this Mermaid block works on GitHub
- Fence starts with exactly ` ```mermaid ` and ends with exactly three backticks on their own lines.
- No HTML line breaks; label uses plain ASCII (`mu/sigma`) to avoid parser quirks.
- No extra prose inside the fenced block.
- Straight quotes only.
