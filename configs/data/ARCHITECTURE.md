# 📊 Data Config Architecture — SpectraMind V50

Data configs (`configs/data/*.yaml`) define **dataset sources, splits, and profiles** for the
NeurIPS 2025 Ariel Data Challenge. They allow the same pipeline to run across Kaggle,
local, and CI environments without modifying code.

---

## 📂 Directory Layout

```

configs/data/
├── kaggle.yaml        # Kaggle runtime dataset paths
├── nominal.yaml       # Default local dataset profile
├── splits.yaml        # Train/val/test partition rules
└── ARCHITECTURE.md    # (this document)

````

---

## ⚙️ Design Principles

1. **Separation of Concerns**  
   - `env/` handles compute/runtime (local vs Kaggle).  
   - `data/` handles dataset paths, profiles, and splits.  
   - No hard-coded paths in code — always via Hydra config:contentReference[oaicite:3]{index=3}.

2. **Kaggle Safety**  
   - Kaggle input paths fixed to `/kaggle/input/ariel-data-challenge-2025/`:contentReference[oaicite:4]{index=4}.  
   - Hidden test data is automatically present during submissions; configs must not assume labels.  

3. **Reproducibility**  
   - Splits are deterministic (seeded) and logged via `config_snapshot.schema.json`.  
   - DVC can track any large local data for debugging:contentReference[oaicite:5]{index=5}.

4. **Flexibility**  
   - Multiple dataset profiles can coexist (`nominal.yaml`, `debug.yaml`, etc.).  
   - Hydra overrides make switching trivial:  
     ```bash
     python -m spectramind.train data=kaggle
     ```

---

## 📝 Key Configs

### `kaggle.yaml`
```yaml
data:
  root: /kaggle/input/ariel-data-challenge-2025
  train_csv: ${data.root}/train.csv
  test_csv: ${data.root}/test.csv
  train_star_info: ${data.root}/train_star_info.csv
  test_star_info: ${data.root}/test_star_info.csv
  adc_info: ${data.root}/adc_info.csv
  axis_info: ${data.root}/axis_info.parquet
  sample_submission: ${data.root}/sample_submission.csv
  raw_train_dir: ${data.root}/train
  raw_test_dir: ${data.root}/test
````

➡ Mirrors Kaggle dataset mount.

### `nominal.yaml`

Defines equivalent paths for local dev (DVC-tracked mirrors of Kaggle structure).

### `splits.yaml`

```yaml
splits:
  seed: 1337
  train_frac: 0.8
  val_frac: 0.2
  stratify: planet_id
```

➡ Ensures reproducible train/val partitions.

---

## 🔄 Typical Usage

* Local debug:

  ```bash
  python -m spectramind.train data=nominal env=local
  ```
* Kaggle notebook:

  ```bash
  python -m spectramind.train data=kaggle env=kaggle
  ```

---

## 📌 Notes

* **FGS1 vs AIRS**: Both sensor channels share metadata (ADC, axis info). Data configs ensure these files are always available to encoders.
* **Hidden Test Handling**: Only IDs, no labels, are exposed in `test.csv` during Kaggle runs.
* **Future extension**: Add `augmented.yaml` for synthetic datasets or `hpc.yaml` for supercomputing clusters.

---

✅ With this structure, **SpectraMind V50 data configs cleanly abstract dataset handling across local, Kaggle, and CI environments, ensuring reproducibility and Kaggle-compliance.**

```
