# Profiles — SpectraMind V50

Runtime overlays you can stack on top of the base training configs.

## Usage

```bash
# Kaggle GPU (T4/L4/16GB), mixed precision, offline logging
spectramind train +defaults='[/profiles/kaggle]'

# Local development (fast feedback, chatty UI)
spectramind train +defaults='[/profiles/local_dev]'

# GitHub Actions / Kaggle CI smoke (2 epochs, FP32)
spectramind train +defaults='[/profiles/ci_fast]'

# Ampere (A100/L4/4090): bf16 + TF32 fast kernels
spectramind train +defaults='[/profiles/ampere_bf16]'

# Full production run (longer schedule)
spectramind train +defaults='[/profiles/full]'

# Priors OFF (GLL only) ablation suite
spectramind train +defaults='[/profiles/ablation_priors_off]'
