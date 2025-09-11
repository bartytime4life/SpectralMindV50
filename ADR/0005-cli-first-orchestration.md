# ADR 0005 — CLI-First Orchestration

> **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge  
> **Status:** ✅ Accepted | **Date:** 2025-09-07  
> **Tags:** `cli` · `orchestration` · `hydra` · `reproducibility` · `ux`  
> **Owners:** Infra WG (Lead: Andy Barta), ML/Infra, Tooling WG

---

## 🔎 1) Context

SpectraMind V50 is a **mission-grade AI pipeline** with the stages  
`calibrate → train → predict → diagnose → submit`.

**Challenges (pre-ADR):**
- Scripts scattered (`train.py`, `predict.py`, `notebooks/`) → inconsistent UX.
- Hydra configs are powerful but hard to invoke uniformly without a wrapper.
- Kaggle runtime requires **internet-disabled**, single-command reproducibility.
- CI/CD (GitHub Actions) needs uniform entrypoints to run all stages safely.

We need a **single, discoverable CLI** that exposes *all* pipeline functions with reproducible defaults.

---

## ✅ 2) Decision

Adopt a **CLI-first orchestration layer**:

- **Framework:** [Typer] (Python 3.10+) — modern, type-hinted, great help UX.
- **Single entrypoint:** `spectramind` with subcommands for all stages.
- **Hydra integration:** every subcommand accepts `--config-name` + overrides (e.g. `+foo=bar`).
- **Rich UX:** `--help`, tab completion, colorized errors (via [Rich]).
- **Determinism/Safety:** seeds, strict Hydra mode, Kaggle offline guards.

---

## 🎯 3) Drivers

- **Reproducibility** — one entrypoint across local dev, CI, and Kaggle.
- **Velocity** — fast override-driven experiments (`spectramind train +loss=nonneg`).
- **Auditability** — every invocation logs JSONL manifests with config hash.
- **Safety** — fail-loud defaults in CI; offline enforcement for Kaggle.
- **UX** — discoverable, consistent, low-friction for collaborators.

---

## 🔁 4) Alternatives Considered

| Option                                             | Pros                               | Cons                                       |
|---------------------------------------------------|------------------------------------|--------------------------------------------|
| Ad-hoc Python scripts (`train.py`, `predict.py`)  | Simple                             | Fragmented UX, poor discoverability        |
| Bash wrappers around Hydra                        | Easy CI bootstrap                  | Fragile, limited UX/portability            |
| Full workflow managers (Airflow, Prefect)         | Powerful DAGs                      | Overkill, heavy infra, **not Kaggle-safe** |
| **Chosen: Typer CLI + Hydra**                     | Lightweight, modern, Kaggle-safe   | Requires disciplined design                 |

---

## 🧩 5) Architecture

```mermaid
flowchart TD
  A["User / CI / Kaggle"] --> B["spectramind CLI (Typer)"]
  B -->|compose| C["Hydra Configs (configs/*)"]
  C --> D["Pipeline Modules (src/spectramind/*)"]
  D --> E["Artifacts (DVC-tracked)"]
  E --> F["Submission (283 μ/σ bins)"]
  B -.->|logs| G["Run Manifest (JSONL, config hash)"]
````

---

## 🛠 6) Implementation Plan

1. **CLI module:** `src/spectramind/cli.py`

   * Subcommands: `calibrate`, `train`, `predict`, `diagnose`, `submit`.
   * Delegates to pipeline functions (e.g. `spectramind.train.train:main`).

2. **Hydra binding:**

   * `--config-name` and arbitrary overrides (`+key=value`).
   * Snapshot resolved config to `artifacts/runs/.../config.snapshot.yaml`.

3. **UX polish:**

   * Rich-powered help/errors, colored tables.
   * Shell auto-completion (bash/zsh/fish).

4. **Compliance hooks:**

   * Each command appends to `events.jsonl` (UTC timestamp, seed, CWD, command line, config snapshot).
   * Kaggle mode enforces offline flags and reproducible seeds.

---

## 🧯 7) Risks & Mitigations

| Risk                       | Mitigation                                                          |
| -------------------------- | ------------------------------------------------------------------- |
| CLI bloat (too many flags) | Keep flags minimal; push complexity into Hydra configs + overrides  |
| CI vs local drift          | Pin Hydra/OmegaConf; CI smoke tests assert parity                   |
| Kaggle runtime quirks      | `bin/kaggle-boot.sh` sets offline/deterministic env                 |
| Override confusion         | Clear docs (`docs/guides/hydra.md`), `spectramind --help`, examples |

---

## 📌 8) Consequences

* ✅ Unified entrypoint → reproducible runs everywhere.
* ✅ Seamless Hydra integration & config snapshots.
* ✅ Better onboarding and CI uniformity.
* ❗ Discipline required: all workflows must route through the CLI (no hidden scripts).

---

## ✅ 9) Compliance Gates (CI)

* [ ] `spectramind --help` runs without error (smoke).
* [ ] Subcommands `calibrate/train/predict/diagnose/submit` expose `--help`.
* [ ] Config snapshot + JSONL manifest emitted per run.
* [ ] Pre-commit hook ensures new modules are reachable from CLI.

*Minimum test file:* `tests/integration/test_cli_smoke.py`

---

## 📚 10) References

* Repo Scaffold & Blueprint
* AI Research Notebook & Upgrade Guide
* ADR-0001 (Hydra + DVC)
* ADR-0003 (CI ↔ CUDA parity)
* GitHub Mermaid CLI diagrams

**Related ADRs:**

* [ADR-0002 — Composite Physics-Informed Loss]
* [ADR-0004 — Dual Encoder Fusion (FGS1 + AIRS)]

[Typer]: https://typer.tiangolo.com
[Rich]: https://rich.readthedocs.io
[ADR-0002 — Composite Physics-Informed Loss]: 0002-composite-physics-informed-loss.md
[ADR-0004 — Dual Encoder Fusion (FGS1 + AIRS)]: 0004-dual-encoder-fusion-fgs1-airs.md

```
