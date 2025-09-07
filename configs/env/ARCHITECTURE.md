# 🌍 Environment Config Architecture — SpectraMind V50

Environment configs (`configs/env/*.yaml`) define **runtime context** for the SpectraMind V50
pipeline. They allow the same codebase to run consistently across:

- 🖥️ Local development (workstation, conda/venv, GPUs/CPUs)
- 📦 Kaggle competition environment (restricted, offline, ephemeral)
- 🔄 CI/CD workflows (GitHub Actions, offline Docker runners, DVC checks)

---

## 📂 Directory Layout

```

configs/env/
├── local.yaml        # Local dev environment
├── kaggle.yaml       # Kaggle competition runtime
├── ci.yaml           # GitHub Actions / CI runners
├── debug.yaml        # Lightweight fast-fail debugging
└── ARCHITECTURE.md   # (this document)

````

---

## ⚙️ Design Principles

1. **Hydra-first**  
   All environment settings are Hydra-composable. Each YAML file provides overrides for paths,
   logging, and compute.

2. **Reproducibility**  
   Every run’s environment config is hashed and logged (`config_snapshot.schema.json`), ensuring
   provenance of results:contentReference[oaicite:2]{index=2}.

3. **Kaggle Safety**  
   - `offline: true` for loggers (no WANDB/HTTP calls).  
   - Data paths fixed under `/kaggle/input/ariel-data-challenge-2025/`.  
   - Strict reliance on attached Kaggle datasets:contentReference[oaicite:3]{index=3}.

4. **Separation of Concerns**  
   - `env/` only describes environment/runtime.  
   - Data configs (`configs/data/`) describe datasets.  
   - Model/training/loss configs remain unchanged across environments.

---

## 📝 Key Fields

Each env YAML may define:

```yaml
# Example (local.yaml)
env:
  name: "local"
  debug: false
  device: "cuda:0"       # "cpu" or "cuda:N"
  num_workers: 4
  dvc_remote: "local"    # DVC remote name
  output_dir: "./outputs"
  log_dir: "./logs"
  offline: false         # overrides logger configs
````

```yaml
# Example (kaggle.yaml)
env:
  name: "kaggle"
  debug: false
  device: "cuda:0"
  num_workers: 2
  dvc_remote: null       # Kaggle has no DVC remote access
  output_dir: "/kaggle/working/outputs"
  log_dir: "/kaggle/working/logs"
  offline: true          # force offline logging
```

---

## 🔄 Typical Usage

Select an environment via Hydra overrides:

```bash
# Local run
python -m spectramind.train env=local

# Kaggle run
python -m spectramind.train env=kaggle
```

The environment config composes into the global Hydra config tree, ensuring that all
paths/loggers/devices are adjusted without code changes.

---

## 📌 Notes

* Always commit environment YAMLs, never hard-code paths in code.
* DVC integration: CI/CD runs (`ci.yaml`) ensure `dvc.yaml` stages execute in offline mode.
* Future extension: add `slurm.yaml` or `hpc.yaml` for supercomputing deployments.

---

✅ With this architecture, **SpectraMind V50 environment configs are modular, Kaggle-ready, and fully reproducible across runs.**

```
