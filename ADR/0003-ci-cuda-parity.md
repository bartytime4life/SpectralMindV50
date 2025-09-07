# ADR 0003 — CI ↔ CUDA Parity

* **Status:** ✅ Accepted
* **Date:** 2025-09-06
* **Project:** SpectraMind V50 — NeurIPS 2025 Ariel Data Challenge
* **Tags:** ci, cuda, reproducibility, environment, testing
* **Owners:** Infra WG (Lead: Andy Barta), ML/Infra, CI/CD Ops

---

## 1) Context

SpectraMind V50 trains GPU-accelerated PyTorch models for multi-sensor fusion (FGS1 + AIRS). Our experiments must execute **bit-for-bit consistently** across:

* **Local dev** (Linux; heterogeneous driver stacks).
* **CI** (GitHub-hosted and self-hosted GPU runners).
* **Kaggle** (fixed CUDA runtime, internet disabled during fit, 9h wallclock).

Without strict CUDA parity, we risk **silently different kernels** (e.g., cuDNN heuristics, TF32 enabling, PTX→SASS JIT drift), leading to **“green in CI, broken on Kaggle”** and non-deterministic outputs.

---

## 2) Decision

Enforce **strict CUDA parity with Kaggle** in CI and container builds:

1. **Pin the PyTorch/CUDA stack** to Kaggle’s published runtime (e.g., `torch==2.3.1+cu121`, `torchvision==0.18.1+cu121`, `torchaudio==2.3.1+cu121`, cuDNN 8.x).
2. **Build & run CI jobs in a pinned CUDA container** (`nvidia/cuda:12.1.1xx-cudnn8-runtime-ubuntu22.04`) that mirrors Kaggle.
3. **Gate merges on a CUDA parity check**:

   * `spectramind doctor --cuda --fail-on-mismatch` compares:

     * `torch.version.cuda`, `torch.backends.cudnn.version()`, `torch.__version__`.
     * Driver/runtime (`nvidia-smi`, `/usr/local/cuda/version.json`).
     * `nvcc --version` if present.
   * Fails CI if mismatched.

Additionally, we **standardize determinism knobs** across environments (see §7).

---

## 3) Drivers

* **Reproducibility** — identical kernels & math paths in CI and Kaggle.
* **Safety** — fail early on environment drift.
* **Velocity** — stop chasing environment bugs late in the cycle.
* **Auditability** — run manifests include CUDA/cuDNN/PyTorch triplet.

---

## 4) Alternatives

| Option                                 | Pros              | Cons                                       |
| -------------------------------------- | ----------------- | ------------------------------------------ |
| **Loose pinning** (`2.3.*`, `cu12*`)   | Lower maintenance | Parity breaks on Kaggle hot-patches        |
| **CUDA matrix** (e.g., 11.8/12.1/12.4) | Wider coverage    | Expensive; not aligned to Kaggle target    |
| **Rely on upstream PyTorch defaults**  | Simpler           | Wheels drift from Kaggle image composition |

**Chosen:** strict pinning to Kaggle.

---

## 5) Scope

* **In-scope:** pinned CUDA container, `cuda-parity` CI job, `spectramind doctor --cuda`, run-manifest additions.
* **Out-of-scope:** ROCm/CPU-only backends (tracked elsewhere), legacy CUDA support matrices.

---

## 6) Architecture

```mermaid
flowchart TD
  A[GitHub Actions GPU Runner] --> B[Docker build & run\n(nvidia/cuda:12.1 + cuDNN8)]
  B --> C[spectramind doctor --cuda]
  D[Kaggle Runtime Spec\n(pinned versions)] --> C
  C -->|identical| E{Gate pass}
  C -->|mismatch| F{Gate fail}
  E --> G[Merge allowed]
  F --> H[Block merge + alert]

```

---

## 7) Implementation Plan

### 7.1 Dockerfile (CI image)

```Dockerfile
# ./Dockerfile.ci.cuda
# Pinned to Kaggle CUDA 12.1 + cuDNN 8, Ubuntu 22.04
FROM nvidia/cuda:12.1.105-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    FORCE_CUDA=1 \
    LC_ALL=C.UTF-8 LANG=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-dev python3-venv \
      git curl ca-certificates build-essential jq pciutils \
      && rm -rf /var/lib/apt/lists/*

# Ensure python/pip defaults
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3 1 && \
    python -m pip install --upgrade pip setuptools wheel

# Install pinned torch stack (match Kaggle)
# NOTE: keep versions in sync with requirements-kaggle.txt and kaggle-boot.sh
RUN pip install --extra-index-url https://download.pytorch.org/whl/cu121 \
      torch==2.3.1+cu121 torchvision==0.18.1+cu121 torchaudio==2.3.1+cu121

# (Optional) minimal runtime deps for repo
COPY requirements-kaggle.txt /tmp/requirements-kaggle.txt
RUN pip install -r /tmp/requirements-kaggle.txt || true

# App user (optional)
RUN useradd -ms /bin/bash runner
USER runner
WORKDIR /workspace

# Default: show versions for debugging
CMD python - <<'PY'
import torch, sys
print("Torch:", torch.__version__, "CUDA:", torch.version.cuda)
print("cuDNN:", torch.backends.cudnn.version())
print(sys.version)
PY
```

### 7.2 CI: cuda-parity job (GitHub Actions)

```yaml
# .github/workflows/ci.yml (excerpt)
name: ci
on:
  push: { branches: [ main ] }
  pull_request:

jobs:
  cuda-parity:
    name: CUDA Parity (Kaggle)
    runs-on: ubuntu-22.04
    permissions: { contents: read }
    # Requires a GPU runner / or uses docker + --gpus (self-hosted recommended)
    container:
      image: ghcr.io/your-org/spectramind-ci-cuda:12.1.105  # build & push from Dockerfile.ci.cuda
      options: --gpus all --ipc=host
    steps:
      - uses: actions/checkout@v4
      - name: Install package (editable)
        run: pip install -e .
      - name: Doctor CUDA parity
        run: |
          spectramind doctor --cuda --fail-on-mismatch --emit-json parity_report.json
      - name: Upload parity report
        if: always()
        uses: actions/upload-artifact@v4
        with:
          name: cuda-parity
          path: parity_report.json
```

> **Notes**
> • On GitHub-hosted runners, GPU is not yet universally available; use **self-hosted NVIDIA runners** (or GHES w/ GPU) to enable `--gpus all`.
> • As a fallback, mark the job **required but skippable** when no GPU is present (see Risks & Mitigations).

### 7.3 CLI parity command

```python
# src/spectramind/cli/doctor.py (excerpt)
import json, subprocess, sys, os, platform
import typer

app = typer.Typer()

def _sh(cmd):
    try:
        return subprocess.check_output(cmd, shell=True, stderr=subprocess.STDOUT, text=True).strip()
    except subprocess.CalledProcessError as e:
        return f"ERR: {e.output.strip()}"

@app.command("doctor")
def doctor_cuda(
    cuda: bool = typer.Option(False, "--cuda", help="Include CUDA parity checks"),
    fail_on_mismatch: bool = typer.Option(False, "--fail-on-mismatch"),
    emit_json: str = typer.Option("", "--emit-json", help="Write JSON report to path"),
):
    import torch

    report = {
        "host": {
            "os": platform.platform(),
            "python": sys.version.split()[0],
        },
        "torch": {
            "version": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        },
        "nvidia": {},
        "cuda_runtime": {},
        "nvcc": {},
        "determinism": {}
    }

    if cuda:
        report["nvidia"]["smi"] = _sh("nvidia-smi || true")
        report["cuda_runtime"]["version_json"] = _sh("cat /usr/local/cuda/version.json 2>/dev/null || true")
        report["cuda_runtime"]["version_txt"] = _sh("cat /usr/local/cuda/version.txt 2>/dev/null || true")
        report["nvcc"]["version"] = _sh("nvcc --version || true")

        # Determinism knobs (our policy)
        report["determinism"] = {
            "torch.use_deterministic_algorithms": True,
            "cudnn.deterministic": True,
            "cudnn.benchmark": False,
            "float32_matmul_precision": "highest",
            "tf32": {
                "matmul": os.environ.get("NVIDIA_TF32_OVERRIDE", "0"),
                "allow_tf32": getattr(torch.backends.cuda, "matmul", None) and getattr(torch.backends.cuda.matmul, "allow_tf32", None),
            }
        }

    # Kaggle target spec (keep updated)
    target = {
        "torch": { "version": "2.3.1+cu121", "cuda": "12.1" },
        "cudnn": { "major": 8 },  # allow any 8.x
    }

    mismatches = []

    # Compare torch/cuda
    if report["torch"]["version"] != target["torch"]["version"]:
        mismatches.append(f'torch.version {report["torch"]["version"]} != {target["torch"]["version"]}')
    if not (report["torch"]["cuda"] or "").startswith(target["torch"]["cuda"]):
        mismatches.append(f'torch.version.cuda {report["torch"]["cuda"]} != {target["torch"]["cuda"]}.x')

    # Compare cuDNN major
    cudnn_v = report["torch"]["cudnn"]
    if isinstance(cudnn_v, int) and cudnn_v // 10000 != target["cudnn"]["major"]:
        mismatches.append(f'cuDNN major {(cudnn_v // 10000) if cudnn_v else None} != {target["cudnn"]["major"]}')

    ok = not mismatches
    report["result"] = { "ok": ok, "mismatches": mismatches }

    if emit_json:
        with open(emit_json, "w") as f:
            json.dump(report, f, indent=2)

    # Human output
    typer.echo(json.dumps(report["result"], indent=2))
    if fail_on_mismatch and not ok:
        raise typer.Exit(code=1)
```

### 7.4 Determinism policy (applied in train/predict)

```python
# src/spectramind/utils/determinism.py
import os, random, numpy as np, torch

def enforce(seed: int = 1337):
    os.environ.setdefault("PYTHONHASHSEED", str(seed))
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    os.environ.setdefault("NVIDIA_TF32_OVERRIDE", "0")  # disable TF32

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.use_deterministic_algorithms(True, warn_only=False)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For matmul precision parity across envs
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
```

### 7.5 Run Manifest (augmented)

We add CUDA fingerprints to every run’s manifest (e.g., `artifacts/run_manifest.json`):

```json
{
  "run_id": "2025-09-06T21:03:11Z_abc123",
  "env": {
    "torch": "2.3.1+cu121",
    "torch_cuda": "12.1",
    "cudnn": 8905,
    "python": "3.10.14",
    "nvidia_smi": "Driver Version: 535.161.08 CUDA Version: 12.2",
    "cuda_version_txt": "CUDA Version 12.1.1"
  },
  "git": { "commit": "deadbeef", "dirty": false },
  "seeds": 1337,
  "determinism": { "cudnn_deterministic": true, "cudnn_benchmark": false }
}
```

The manifest is emitted by the pipeline entry-points (calibrate/train/predict/submit).

---

## 8) Risks & Mitigations

| Risk                                | Mitigation                                                                                                                                                 |
| ----------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Kaggle runtime silently updates** | Scheduled weekly CI job that **fetches Kaggle image specs** (documented versions), diffs against pinned JSON, opens an alert PR if drift detected.         |
| **No GPU on runner**                | Mark `cuda-parity` as **required but skippable** if `nvidia-smi` absent; provide a separate **pre-merge label** requiring GPU path before merge to `main`. |
| **Container build overhead**        | Layered Dockerfile, build-kit cache to GHCR, reuse image across jobs (pull once), limit dependency footprint.                                              |
| **nvcc not installed**              | Parity accepts absence on Kaggle; we verify via `torch.version.cuda` and `/usr/local/cuda/version*`.                                                       |
| **cuDNN patch-level drift**         | Gate on **major** (8.x) and (optionally) minor; when Kaggle updates, bump pins in one PR.                                                                  |

---

## 9) Consequences

* ✅ CI builds **mirror Kaggle**; no last-minute environment surprises.
* ✅ Deterministic training & evaluation across environments (within known PyTorch limits).
* ❌ Slightly higher maintenance: version bumps now require Dockerfile + parity update + requirement sync.

---

## 10) Compliance Gates (CI)

* [ ] `spectramind doctor --cuda --fail-on-mismatch` passes.
* [ ] `torch.__version__` **exactly** matches target (e.g., `2.3.1+cu121`).
* [ ] `torch.version.cuda` startswith `12.1`.
* [ ] `torch.backends.cudnn.version()` major == 8.
* [ ] Run manifest contains torch/cuda/cuDNN fingerprints.

---

## 11) Revisit Triggers

* Kaggle publishes a new competitions base image (CUDA / cuDNN upgrades).
* NVIDIA policy changes (TF32 defaults, cuDNN determinism semantics).
* Competition adds CPU-only track or alternative backends.

---

## 12) References

* Kaggle — Competition Docker envs & base images.
* PyTorch — CUDA wheels matrix for 2.3.1 / cu121.
* Internal — `Dockerfile.ci.cuda`, `.github/workflows/ci.yml`, `spectramind doctor`, `requirements-kaggle.txt`.

---

### Appendix A — Makefile Targets (optional)

```makefile
.PHONY: image.push cuda.parity

image.push:
	docker buildx build -f Dockerfile.ci.cuda -t ghcr.io/your-org/spectramind-ci-cuda:12.1.105 --push .

cuda.parity:
	spectramind doctor --cuda --fail-on-mismatch --emit-json parity_report.json
	jq . parity_report.json
```

### Appendix B — requirements-kaggle.txt (pinned)

```
# Match Kaggle runtime
--extra-index-url https://download.pytorch.org/whl/cu121
torch==2.3.1+cu121
torchvision==0.18.1+cu121
torchaudio==2.3.1+cu121

# Core
numpy==1.26.4
pandas==2.2.2
scipy==1.13.1
typer==0.12.3
rich==13.7.1
jsonschema==4.23.0

# (avoid network-fetching extras in Kaggle notebooks)
```

### Appendix C — Kaggle Notebook Bootstrap

At the top of Kaggle notebooks:

```bash
%run bin/kaggle-boot.sh
!python -c "import torch; print(torch.__version__, torch.version.cuda, __import__('torch').backends.cudnn.version())"
```

This ensures notebook and CI stacks remain aligned.

---
