# ADR-0009 — Artifact Retention & Governance (Cleanup, Archive, Provenance Policy)

**Status:** 🚧 Draft  
**Date:** 2025-09-13  
**Author:** SpectraMind V50 Team  

---

## Context

SpectraMind V50 produces a wide range of artifacts across the pipeline:
- Raw and calibrated cubes (FGS1 + AIRS)
- Preprocessed tensors
- Model checkpoints
- Diagnostics reports (FFT, UMAP, SHAP, lineage manifests)
- Submission bundles (`submission.csv`, `submission.zip`)

Without a governance policy, artifacts can **proliferate uncontrollably**, leading to:
- Disk exhaustion in Kaggle / CI environments
- Confusion about which checkpoint or submission is canonical
- Loss of reproducibility if artifacts are overwritten or missing

Our repository design already integrates DVC for versioning:contentReference[oaicite:3]{index=3} and lineage manifests:contentReference[oaicite:4]{index=4}, but we need **formal rules for retention, cleanup, and archival**.

---

## Decision

We adopt a **tiered artifact retention & governance policy**:

1. **Ephemeral Artifacts**  
   - CI scratch runs, intermediate tensors, and debug plots  
   - Auto-cleaned after job completion (`bin/artifact_sweeper.sh` + GitHub Actions):contentReference[oaicite:5]{index=5}  
   - Not DVC-tracked  

2. **Retained Artifacts (short-term)**  
   - Model checkpoints, validation metrics, diagnostics reports for active experiments  
   - Retained in DVC cache for **30 days**  
   - Tagged with run hash and Hydra config snapshot  

3. **Archived Artifacts (long-term)**  
   - Accepted submissions, release models, milestone diagnostics  
   - Stored under `artifacts/` and DVC-pushed to remote storage  
   - Linked to a Git tag + ADR reference in CHANGELOG  

4. **Provenance Reports**  
   - Every retained/archived artifact must include:  
     - Hydra config snapshot (`configs_snapshot.yaml`)  
     - DVC hash lineage (`dvc.lock` entry)  
     - Git commit SHA  
     - JSONL run manifest (`events.jsonl`)  
   - Reports rendered as `report.html` and committed under `outputs/diagnostics/`:contentReference[oaicite:6]{index=6}

---

## Drivers

- **NASA-grade reproducibility:** Ensure every submission can be traced to configs, code, and data:contentReference[oaicite:7]{index=7}  
- **Storage constraints:** Kaggle kernels have 20GB limits; GitHub CI has ephemeral runners  
- **Scientific credibility:** Archived results must remain inspectable years later (ESA Ariel launch ~2029):contentReference[oaicite:8]{index=8}  
- **Team collaboration:** Standardized retention reduces ambiguity across contributors  

---

## Alternatives Considered

- **Retain everything:** Unsustainable; storage and runtime limits exceeded.  
- **Manual cleanup only:** Error-prone; risks deleting canonical artifacts.  
- **External storage only (no DVC):** Breaks Git/DVC integration; weakens reproducibility.  

---

## Risks

- **Over-aggressive cleanup:** May delete artifacts still needed; mitigated with retention window + archive tier.  
- **Remote storage failures:** Kaggle offline runtime cannot pull; mitigated by packaging essentials as Kaggle datasets.  
- **Governance overhead:** Requires discipline in tagging/archiving; mitigated with CI automation.  

---

## Compliance Gates

- CI/CD must run `bin/artifact_sweeper.sh --dry-run` weekly to list stale artifacts.  
- Any submission merged to `main` must include an **archived bundle** under DVC.  
- Provenance manifests must be validated against `schemas/events.schema.json`.  
- Release CI fails if an ADR-referenced artifact is missing its provenance.  

---

## References

- SpectraMind V50 Repository Design:contentReference[oaicite:9]{index=9}:contentReference[oaicite:10]{index=10}  
- Audit: Missing Components & Recommendations:contentReference[oaicite:11]{index=11}  
- AI Research Notebook & Upgrade Guide:contentReference[oaicite:12]{index=12}  
- Scientific Context: Exoplanet Spectroscopy:contentReference[oaicite:13]{index=13}  
- Kaggle Notebooks & Governance Practices:contentReference[oaicite:14]{index=14}

---
