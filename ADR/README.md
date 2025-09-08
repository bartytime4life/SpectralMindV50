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

---

## Conventions

* Files are numbered sequentially: `ADR/000X-<slug>.md`.  
* Each ADR includes: Status, Context, Decision, Drivers, Alternatives, Risks, Compliance Gates, References.  
* Statuses: ✅ Accepted, 🚧 Draft, ❌ Superseded.  

---

## Next Planned ADRs

* ADR-0005 — CLI-First Orchestration (Typer + Hydra overrides)  
* ADR-0006 — Reproducibility Standards (run manifests, SBOM, JSONL lineage)  
* ADR-0007 — Submission Schema & Validation (JSON schema, CI gates)  

---
