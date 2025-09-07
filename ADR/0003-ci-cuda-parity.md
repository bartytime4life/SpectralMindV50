# ADR 0003 — CI ↔ CUDA Parity

* **Status:** ✅ Accepted  
* **Date:** 2025-09-06  
* **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge  
* **Tags:** ci, cuda, reproducibility, environment, testing  
* **Owners:** Infra WG (Lead: Andy Barta), ML/Infra, CI/CD Ops  

---

## 1. Context

SpectraMind V50 relies on GPU-accelerated PyTorch models for multi-sensor fusion (FGS1 + AIRS).  
To ensure **mission-grade reproducibility**, experiments must run **identically** across:

- Local dev (Linux workstation, CUDA drivers vary).  
- CI runners (GitHub-hosted or self-hosted).  
- Kaggle (fixed CUDA runtime, internet disabled, 9h wallclock).  

Without strict CUDA parity, we risk:

- Silent kernel mismatches (e.g. cuDNN, PTX → SASS compile).  
- CI “green” jobs that don’t match Kaggle runtime.  
- Non-deterministic results (different seeds, fused ops).  

---

## 2. Decision

Enforce **CUDA parity between CI and target runtime (Kaggle)** by:

1. **Pinning PyTorch/CUDA stack** to Kaggle’s published runtime (e.g. `pytorch==2.3.1+cu121`).  
2. **Using containers in CI** (`Dockerfile` builds pinned to Kaggle CUDA).  
3. **Adding parity tests** in CI:
   - `spectramind doctor --cuda` checks driver, runtime, seeds, determinism.  
   - Ensure identical `torch.version.cuda`, `torch.backends.cudnn.version()`, and `nvcc --version`.  

---

## 3. Drivers

- **Reproducibility** — CI builds must mimic Kaggle exactly.  
- **Safety** — Fail CI early if CUDA mismatches.  
- **Iteration speed** — Prevent “works in CI, fails on Kaggle” cycles.  
- **Auditability** — Run manifests include CUDA versions.  

---

## 4. Alternatives

1. **Loose pinning (major.minor only)**  
   - ✅ Less overhead.  
   - ❌ Breaks parity when Kaggle hot-patches runtime.  

2. **Matrix test across CUDA versions**  
   - ✅ Ensures broad coverage.  
   - ❌ Too costly; irrelevant to Kaggle competition.  

3. **Rely only on PyTorch wheels**  
   - ✅ Simpler setup.  
   - ❌ Wheel updates drift from Kaggle environment.  

**Chosen:** strict parity with Kaggle CUDA.  

---

## 5. Scope

- **In-scope:**  
  - CI jobs (`ci.yml`) with CUDA parity check.  
  - `spectramind doctor` parity command.  
  - Dockerfile pinned to Kaggle CUDA runtime.  
- **Out-of-scope:**  
  - Supporting non-Kaggle CUDA builds.  
  - Multi-backend support (ROCm, CPU-only) — tracked separately.  

---

## 6. Architecture

```mermaid
flowchart TD
  A["GitHub CI Runner"] --> B["Dockerfile (PyTorch + CUDA pinned to Kaggle)"]
  B --> C["Parity Check (spectramind doctor --cuda)"]
  C -->|compare| D["Kaggle Runtime Specs"]
  D --> E["Pass: identical CUDA stack"]
  C --> F["Fail: mismatch → block merge"]
````

---

## 7. Implementation Plan

* **Dockerfile**

  * Base: `nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04`.
  * Install pinned PyTorch wheel (`pip install torch==2.3.1+cu121`).

* **CI (ci.yml)**

  * Add job `cuda-parity`:

    ```bash
    spectramind doctor --cuda --fail-on-mismatch
    ```

* **spectramind doctor**

  * Print + log:

    * CUDA driver version.
    * `torch.version.cuda`.
    * `torch.backends.cudnn.version()`.
    * `nvcc --version`.

* **Run manifests**

  * Include CUDA + cuDNN versions for audit trail.

---

## 8. Risks & Mitigations

* **Kaggle runtime updates silently**
  → Add scheduled CI job to fetch Kaggle runtime specs, alert on drift.

* **Self-hosted runners missing GPU**
  → CI parity job marked “required but skippable” if no GPU.

* **Overhead of container builds**
  → Use caching layers in GitHub Actions.

---

## 9. Consequences

* ✅ CI builds and Kaggle submissions aligned.
* ✅ Reproducible GPU kernels across environments.
* ❌ More discipline: updates require bumping Dockerfile + parity tests.

---

## 10. Compliance Gates (CI)

* [ ] `spectramind doctor --cuda` passes.
* [ ] `torch.version.cuda` matches Kaggle.
* [ ] `cuDNN` version matches Kaggle.
* [ ] Run manifest includes CUDA + cuDNN.

---

## 11. Revisit Triggers

* Kaggle upgrades CUDA base image.
* ESA/NASA release new GPU requirements.
* Competition adds CPU-only track.

---

## 12. References

* Kaggle: [Competition Docker envs](https://www.kaggle.com/docs/competitions#docker-images).
* PyTorch CUDA wheels matrix.
* Internal: `Dockerfile`, `.github/workflows/ci.yml`, `spectramind doctor`.

---

```
