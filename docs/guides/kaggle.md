# 🏆 Kaggle Integration Guide — SpectraMind V50

This guide explains how to run the **SpectraMind V50** pipeline in the **Kaggle** environment for the NeurIPS 2025 Ariel Data Challenge.  
It covers environment setup, dataset access, notebook patterns, and CI integration.

---

## 🚀 Kaggle Runtime Basics

- **Offline only**: Kaggle kernels run with **no Internet**. All inputs must come from uploaded datasets.
- **Fixed compute**: ~9h runtime budget; GPUs (T4, P100, A100 depending on quota) and 16–30GB RAM.
- **File layout**:
  - `/kaggle/input/` → competition & attached datasets
  - `/kaggle/working/` → writable scratch space (persists only for notebook session)
  - `/kaggle/output/` → final artifacts (CSV/ZIP) for submission

---

## 📂 Repository–Kaggle Workflow

We treat **GitHub as source of truth** and **Kaggle as execution host**:contentReference[oaicite:3]{index=3}.

1. **Code lives in GitHub** (`src/spectramind/`).
2. **Thin Kaggle notebooks** (train / inference) import and call library functions.
3. **ArielSensorArray CLI** auto-generates these notebooks from templates whenever configs change:contentReference[oaicite:4]{index=4}.
4. **GitHub Actions** can sync notebooks to Kaggle via API on release.

This ensures **no code duplication**: Kaggle notebooks simply wrap the CLI, keeping logic consistent.

---

## 🧰 Environment Setup

Our Kaggle-safe dependencies are in `requirements-kaggle.txt`.  
Use the bootstrap script:

```bash
!bash bin/kaggle-boot.sh
````

It will:

* Install pinned packages (Torch stack in `--extra-index-url` if needed).
* Add extras (`pyg`, `umap-learn`, etc.) if supported by Kaggle’s base image.
* Verify CUDA availability.

---

## 📊 Data Access

* Kaggle competition dataset is automatically available under `/kaggle/input/neurips-2025-ariel-data-challenge/`.
* **Do not commit raw data** to GitHub; all large data tracked by DVC.
* Use symlinks for consistent paths (see [`kaggle/input_symlinks.md`](../../kaggle/input_symlinks.md)).

Example in config:

```yaml
data:
  root: /kaggle/input/neurips-2025-ariel-data-challenge
  fgs1: ${data.root}/fgs1
  airs: ${data.root}/airs
```

---

## 📒 Notebook Types

Following Kaggle best practices:

* **Competition notebooks**:

  * End-to-end training → CSV submission.
  * Must run <9h with deterministic seeds.
* **Inference notebooks**:

  * Load pre-trained checkpoint, run prediction only.
* **EDA notebooks** (optional):

  * Data visualization, storytelling, not for submission.

We adopt the **two-kernel pattern**:
`train.ipynb` produces model → `predict.ipynb` generates submission.

---

## 📐 Metric Awareness

* Official metric: **Gaussian Log-Likelihood (GLL)**.
* The **FGS1 channel is weighted \~58× more than a single spectrometer channel** in scoring.
* Strategy:

  * Prioritize accurate white-light transit fits (FGS1).
  * Ensure realistic uncertainty (σ) — overconfidence is penalized.

---

## ✅ Validation & Guardrails

* Run **`notebooks/99_submission_check.ipynb`** before Kaggle upload:

  * Validates shape (283 μ and 283 σ per row).
  * Checks schema compliance (`schemas/submission.schema.json`).
* Use **`scripts/package_submission.sh`** to create ZIP bundles with manifest.

---

## 🔄 CI / CD Integration

* `.github/workflows/kaggle_notebook_ci.yml` runs lightweight dry-runs to ensure Kaggle-compatibility:

  * No Internet calls.
  * Memory/speed budget estimation.
  * Fast-fail with dummy data.
* On release, CI can auto-push notebooks + submission dataset to Kaggle.

---

## 📌 Tips & Tricks

* Pin library versions; Kaggle base images update silently.
* Keep outputs minimal (`artifacts/submission.csv`, logs).
* Use **Markdown storytelling + consistent plots** in EDA notebooks for readability.
* Always document which GitHub commit produced the notebook (use `Lineage` manifest).

---

## 📚 References

* [Kaggle Notebooks Expert Guide (Ariel Challenge)]: contentReference[oaicite:10]{index=10}
* [Advanced Spectrum Extraction Techniques]: contentReference[oaicite:11]{index=11}
* [SpectraMind V50 GitHub–Kaggle CLI Integration]: contentReference[oaicite:12]{index=12}

```
