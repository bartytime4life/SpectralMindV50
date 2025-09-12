# 📘 Kaggle Integration — SpectraMind V50

Kaggle-ready assets for the **NeurIPS 2025 Ariel Data Challenge** (FGS1 + AIRS).
Designed for **zero-internet** kernels, strict reproducibility, and fast iteration.

---

## 📂 Contents

* `notebook_template.ipynb` — lightweight, submission-ready starter (auto-detects Kaggle vs local; zero-internet).
* *(optional)* `kaggle_train_smv50.ipynb` — training scaffold (Kaggle-safe; writes `outputs/train_manifest.json`).
* *(optional)* `kaggle_predict_smv50.ipynb` — inference scaffold (produces `outputs/submission.csv`).
* *(optional)* `smv50_kaggle_error_analysis.ipynb` — post-hoc μ/σ error & uncertainty diagnostics.
* `README.md` — this document.

> Keep heavy work in your **library** (e.g., `src/spectramind/`) and DVC stages; notebooks remain thin, reproducible drivers.

---

## 🚀 Usage on Kaggle

### 1) Attach **code & config**

* Export your repository as a **Kaggle Dataset** (code only), e.g. `spectramind-v50`.
* In the Kaggle Notebook → **Add Data**, attach your code dataset. It will mount at:

  ```
  /kaggle/input/spectramind-v50/
  ```
* The template auto-adds `sys.path` for `/kaggle/input/spectramind-v50/src` if present (no pip, no internet).

### 2) Attach **competition data**

* Add **Competition Data** → **Ariel Data Challenge 2025**. It will mount at:

  ```
  /kaggle/input/ariel-data-challenge-2025/
  ```

### 3) (Optional) Map inputs to your repo layout (zero-copy)

If you unzip/import your repo under `/kaggle/working/spectramind-v50`, you can symlink inputs:

```bash
REPO=/kaggle/working/spectramind-v50
mkdir -p "$REPO/data/raw" "$REPO/data/interim" "$REPO/data/processed" "$REPO/data/external" "$REPO/artifacts" "$REPO/models"
ln -sfn /kaggle/input/ariel-data-challenge-2025  "$REPO/data/raw/adc2025"
echo "Symlinked Kaggle inputs under $REPO/data/raw/adc2025"
```

> 🔒 `/kaggle/input/*` is **read-only**. Always create symlinks inside `/kaggle/working/…`.

### 4) Run the **template**

Open and run `notebook_template.ipynb`. It will:

* Detect Kaggle vs local; create `outputs/`.
* Write a config snapshot → `outputs/config_snapshot.json`.
* (Optional) import your package if present: `/kaggle/input/spectramind-v50/src/spectramind`.
* (Optional) call a `notebook_predict()` hook if your repo exposes it.
* If a `submission.csv` is produced, it zips to `submission.zip` for leaderboard upload.

> No network calls; no `pip install`. Use the Kaggle base image + attached **code datasets** only. If you need third-party packages not in the base image, **vendor the wheels** in your dataset and install from **local paths**.

---

## 🧭 End-to-End Workflow (Two-Kernel Pattern)

* **Training kernel** (`kaggle_train_smv50.ipynb`)

  * Reads `/kaggle/input/ariel-data-challenge-2025/train.csv` (guarded).
  * Calls your library hooks (e.g., `spectramind.cli_hooks.notebook_train(config)`), saves:

    * `outputs/train_metrics.json`
    * `outputs/train_manifest.json`
    * artifact checkpoint(s) under `artifacts/` or as Notebook Output.

* **Inference kernel** (`kaggle_predict_smv50.ipynb`)

  * Attach your **trained artifacts** as a dataset.
  * Calls `spectramind.cli_hooks.notebook_predict(config)` and writes:

    * `outputs/submission.csv`
    * `outputs/predict_manifest.json`
  * Zips to `submission.zip`.

* **Error analysis kernel** (`smv50_kaggle_error_analysis.ipynb`)

  * Loads GT (if available, e.g., train folds) & predictions; computes MAE/RMSE, approx GLL, reliability curves, coverage, PIT hist.
  * Exports:

    * `outputs/error_summary.json`
    * `outputs/per_bin_rmse.csv`

> This separation keeps each kernel small, fast, and compliant with Kaggle’s **0-internet / 9-hour** constraints.

---

## 🛡️ Guardrails

* **Zero internet**: All code and dependencies must come from Kaggle’s base image or attached datasets.
* **Symlinks**: Only under `/kaggle/working`. Recreate symlinks on startup (they’re not guaranteed to persist if you export outputs).
* **Pinned runtime**: Prefer the Kaggle base image where possible. If you must install, vendor wheels into your code dataset and install from local paths.
* **Reproducibility**: The template always writes `outputs/config_snapshot.json`; training & inference notebooks also write manifests.

---

## 🔧 Config tip (Hydra)

Create a Kaggle profile (e.g., `configs/data/kaggle.yaml`) and reference the repo root:

```yaml
repo: /kaggle/working/spectramind-v50

data:
  raw_dir:       ${repo}/data/raw/adc2025
  interim_dir:   ${repo}/data/interim
  processed_dir: ${repo}/data/processed

external:
  axis_dir:      ${repo}/data/external/axis

artifacts_dir:    ${repo}/artifacts
models_dir:       ${repo}/models
```

---

## ✅ Quick Checklist

* [ ] Attached code dataset: `/kaggle/input/spectramind-v50/` (contains `src/spectramind`)
* [ ] Attached competition data: `/kaggle/input/ariel-data-challenge-2025/`
* [ ] Optional symlink mounts under `/kaggle/working/spectramind-v50/data/...`
* [ ] `outputs/config_snapshot.json` written
* [ ] For inference: `outputs/submission.csv` created and zipped to `submission.zip`
* [ ] For diagnostics: `outputs/error_summary.json` + `outputs/per_bin_rmse.csv`

---

## ❓ FAQ

**Q: Why symlink, not copy?**
A: Copying large datasets wastes time and disk. Symlinks are instant and point to the canonical source in `/kaggle/input`.

**Q: Will symlinks persist if I export Notebook outputs?**
A: Not guaranteed. Recreate links each run (first cell). Treat them as runtime mounts.

**Q: I need extra Python packages. How do I install them without internet?**
A: Vendor the wheels in your code dataset and install from local paths, e.g.:

```bash
pip install /kaggle/input/spectramind-v50/wheels/<pkg>-<ver>-py3-none-any.whl
```

**Q: Where do I put my training checkpoints?**
A: Save them under `/kaggle/working/spectramind-v50/artifacts` and then **publish them as a dataset** for the inference kernel.

---

**Happy building & good luck on the leaderboard!** 🚀
