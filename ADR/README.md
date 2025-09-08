# 📚 Architecture Decision Records (ADRs)

This directory contains mission-grade architecture decision records for **SpectraMind V50**.  
Each ADR captures the **context, decision, consequences, and compliance gates** for a major repo choice.  

---

## Index

* **ADR-0001 — Choose Hydra + DVC for Config/Lineage**  
  * Decision to adopt Hydra for hierarchical configs and DVC for pipeline lineage & artifacts.  

* **ADR-0002 — Physics-Informed Losses**  
  * Composite loss engine: GLL baseline + smoothness, nonnegativity, band coherence, calibration priors.  

* **ADR-0003 — CI ↔ CUDA Parity**  
  * Enforce strict parity between CI and Kaggle CUDA runtime (pinned Torch/cuDNN, determinism).  

* **ADR-0004 — Dual Encoder Fusion (FGS1 + AIRS)**  
  * Adopt dual-encoder architecture: FGS1 via Mamba SSM, AIRS via CNN/GNN, fused with cross-attention, heteroscedastic decoder.  

* **ADR-0005 — CLI-First Orchestration**  
  * Typer-based CLI as the single entrypoint (`spectramind`) binding Hydra configs, enforcing determinism, and standardizing UX across local, CI, and Kaggle.  

* **ADR-0006 — Reproducibility Standards**  
  * Mandate run manifests, config snapshots, DVC artifact lineage, SBOM generation, and determinism guardrails to ensure mission-grade reproducibility.  

* **ADR-0007 — Submission Schema & Validation**  
  * Define strict JSON schema for submission CSVs (id, 283 μ, 283 σ ≥ 0) and enforce validation via CLI, CI, and Kaggle bootstrap.  

---

## Conventions

* Files are numbered sequentially: `ADR/000X-<slug>.md`.  
* Each ADR includes: Status, Context, Decision, Drivers, Alternatives, Risks, Compliance Gates, References.  
* Statuses: ✅ Accepted, 🚧 Draft, ❌ Superseded.  

---

## Next Planned ADRs

* ADR-0008 — [TBD: Model Explainability & Diagnostics (FFT/UMAP, SHAP, lineage reports)]  
* ADR-0009 — [TBD: Artifact Retention & Governance (cleanup, archive, provenance policy)]  

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
