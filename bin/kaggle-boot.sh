#!/usr/bin/env bash
set -euo pipefail
REQ=${1:-requirements-kaggle.txt}
python -m pip install -U pip wheel
pip install -r "$REQ"
python - <<'PY'
import torch
print("Torch:", torch.__version__, "CUDA:", torch.cuda.is_available())
PY
