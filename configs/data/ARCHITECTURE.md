# 📊 Data Config Architecture — SpectraMind V50

Data configs (`configs/data/*.yaml`) define **dataset sources, profiles, and split rules** for the
NeurIPS 2025 Ariel Data Challenge. They let the same pipeline run across **local, Kaggle,
and CI** environments without changing code (Hydra composition all the way).

---

## 📂 Directory Layout

```

configs/data/
├── default.yaml      # Local/CI default (DVC-friendly, debug-ready)
├── kaggle.yaml       # Kaggle runtime dataset profile (mount paths, tuned loaders)
├── nominal.yaml      # Canonical “full” profile for local/CI experiments
├── debug.yaml        # Tiny deterministic overlay (2% subsets, tiny loaders)
├── splits.yaml       # Train/val/test partition rules (holdout/kfold/timeseries/precomputed)
└── ARCHITECTURE.md   # (this document)

````

> Tip: You can stack configs, e.g. `+data=nominal,data/debug` to run a tiny slice of nominal.

---

## ⚙️ Design Principles

1. **Separation of concerns**
   - `env/*` chooses **where** files live (local vs Kaggle vs CI).
   - `data/*` chooses **what** files/splits to use (paths, loaders, stratify/group).
   - Code never hard-codes paths — everything flows through Hydra.

2. **Kaggle safety**
   - Inputs are mounted under `/kaggle/input/<dataset>/` and writes go to `/kaggle/working`.
   - Hidden test labels are not accessible in submissions; configs must not assume labels exist.

3. **Reproducibility**
   - Splits are deterministic (seeded) and can be exported (CSV/Parquet) for DVC/traceability.
   - Config snapshots can be logged by the pipeline for provenance.

4. **Flexibility**
   - Multiple data profiles can coexist (`nominal`, `default`, `kaggle`, `debug`).
   - Switch via overrides:
     ```bash
     python -m spectramind.train +env=local +data=nominal            # local full
     python -m spectramind.train +env=local +data=nominal,data/debug  # local tiny
     python -m spectramind.train +env=kaggle +data=kaggle             # kaggle
     ```

---

## 🧩 Key Configs (what they expose)

### `default.yaml` (Local/CI default)
- **`dataset.*`**
  - `root`: `${paths.data_dir}` from env (DVC-friendly).
  - `splits.train|valid|test`: CSV paths (e.g. `${root}/train.csv`).
  - `n_bins`: spectral bin count (must match `model.num_bins`).
  - `fgs1_bin_idx`, `fgs1_bin_scale`: white-light bin index and the 58× weighting for the metric/loss.
- **`loader.*`**
  - Dataloader knobs: `batch_size`, `num_workers`, `persistent_workers`, `prefetch_factor`, `pin_memory`, `drop_last`.
  - Optional sampler setup (`none|distributed`) with `seed`.
- **`debug.*`**
  - Guardrails: path/header checks, preview row printing, fail-fast switches.

### `kaggle.yaml` (Kaggle)
- Targets the `ArielDataModule` directly and pins canonical Kaggle paths:
````

root: /kaggle/input/ariel-data-challenge-2025
train\_csv/test\_csv/...: \${.root}/...

````
- Conservative DataLoader settings for Kaggle containers
(e.g., `num_workers=2`, `persistent_workers=false`, `prefetch_factor=2`).
- Same spectral/metric keys (`n_bins`, `fgs1_bin_scale`) so training aligns with the loss.

### `nominal.yaml` (Local canonical full profile)
- Mirrors Kaggle’s file layout but under `${paths.data_dir}` (DVC/local mirror of the Kaggle dataset).
- Exposes the same spectral/metric and loader keys as `kaggle.yaml`.

### `debug.yaml` (Overlay)
- Tiny, deterministic subset for fast iteration:
- `subset.enable=true`, `train/val/test_frac=0.02`, `seed=123`.
- Small batches (`8`), `num_workers=0`, no `persistent_workers`.
- Works as a stackable overlay on top of any profile:
```bash
python -m spectramind.train +env=local +data=nominal,data/debug
````

### `splits.yaml` (Strategy)

* `scheme`: `holdout | kfold | timeseries | precomputed`
* `columns`: `{id, group, stratify, time}` — resolve to actual metadata columns.
* **KFold/holdout priority**: `group` → GroupKFold/blocked holdout (no group leakage); else `stratify`; else plain.
* **Timeseries**: walk-forward split; guards against future-leak (train\_time\_max < val\_time\_min).
* **Precomputed**: read `{id, fold}` or `{id, split}` from CSV/Parquet.
* **Constraints**: disjoint groups across splits, enforce min per fold, optional strata balancing.
* **Export**: write assigned splits to an artifacts dir (CSV/Parquet) with scheme/seed in filename.
* **Diagnostics**: class balance, group overlap, time windows, and a small preview.

---

## 🔄 Typical Usage

**Local (full):**

```bash
python -m spectramind.train +env=local +data=nominal
```

**Local (tiny & fast):**

```bash
python -m spectramind.train +env=local +data=nominal,data/debug
```

**Kaggle notebook:**

```bash
python -m spectramind.train +env=kaggle +data=kaggle
```

**Custom splits (5-fold, stratified):**

```bash
python -m spectramind.train +env=local +data=nominal \
  +data.splits.scheme=kfold +data.splits.kfold.n_splits=5 \
  +data.splits.columns.stratify=target_bucket
```

---

## 🧪 Validation & Guardrails

* **Path checks**: `debug.validate_paths=true` verifies all declared files exist.
* **Header checks**: `debug.strict_headers=true` enforces minimal headers (e.g., `sample_id`).
* **Preview**: `debug.print_first_rows=N` prints first N rows per split to help catch schema surprises.
* **Fail fast**: `debug.fail_on_missing=true` raises immediately on any missing input.

---

## 📌 Notes & Pitfalls

* **FGS1 vs AIRS**: both sensor channels share metadata (ADC, axis info) — configs ensure these are present for the encoders and loss.
* **Hidden test**: on Kaggle submissions, `test.csv` contains only IDs; pipeline must not assume labels.
* **Workers**: use `num_workers=0` for **fast boot** in debug/CI; bump for throughput in long runs.
* **Persistent workers**: great for long local runs; set `false` in Kaggle/CI to avoid worker hangups on restarts.

---

## ➕ Extending

* Add `augmented.yaml` for synthetic datasets.
* Add `hpc.yaml` for multi-node clusters (different paths and larger `num_workers`).
* Add `reduced.yaml` for experiments with fewer wavelengths (keep `n_bins` in sync with the model).

---

✅ With this structure, **SpectraMind V50** cleanly abstracts dataset handling across local, Kaggle, and CI,
**enforces leak-safety**, and supports **reproducible** experiments with minimal friction.

```
