# Environment & Parity

- Targets: Local dev (CPU/GPU), Docker (CUDA 12.1), Kaggle.
- Determinism: `PYTHONHASHSEED=0`; consider `CUBLAS_WORKSPACE_CONFIG` for strict.
- ROCm: prefer pip ROCm wheels; keep kernels deterministic off.
