# 📚 Architecture Decision Records (ADRs)

This directory contains **mission-grade ADRs** for SpectraMind V50.  
Each ADR documents context, decision, consequences, and compliance gates.

---

## Index

* **ADR-0001 — Choose Hydra + DVC for Config/Lineage** ✅  
* **ADR-0002 — Physics-Informed Losses** ✅  
* **ADR-0003 — CI ↔ CUDA Parity** ✅  
* **ADR-0004 — Dual Encoder Fusion (FGS1 + AIRS)** ✅  
* **ADR-0005 — CLI-First Orchestration** ✅  
* **ADR-0006 — Reproducibility Standards** ✅  
* **ADR-0007 — Submission Schema & Validation** ✅  
* **ADR-0008 — Model Explainability & Diagnostics (FFT, UMAP, SHAP, lineage reports)** 🚧 Planned  
* **ADR-0009 — Artifact Retention & Governance (cleanup, archive, provenance policy)** 🚧 Planned  

---

## Conventions

- Numbered sequentially: `ADR/000X-<slug>.md`  
- Status legend: ✅ Accepted, 🚧 Draft, ❌ Superseded  
- Each ADR includes: Status, Context, Decision, Drivers, Alternatives, Risks, Compliance Gates, References.  

---

## ADR Dependency Graph

```mermaid
flowchart TD
  A["ADR-0001 Hydra + DVC"] --> B["ADR-0005 CLI-First Orchestration"]
  B --> C["ADR-0006 Reproducibility Standards"]
  C --> D["ADR-0007 Submission Schema & Validation"]

  A --> E["ADR-0002 Physics-Informed Losses"]
  E --> F["ADR-0004 Dual Encoder Fusion"]

  A --> G["ADR-0003 CI ↔ CUDA Parity"]
  G --> C
