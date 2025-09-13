# syntax=docker/dockerfile:1.7-labs
################################################################################
# SpectraMind V50 — CUDA Runtime Image (Prod, Upgraded)
# - CUDA 12.1 + cuDNN 8 (Kaggle-compatible cu121 wheels)
# - Python 3.11 by default (flip to 3.10 for Kaggle-notebook parity)
# - Hermetic wheelhouse optional; otherwise editable install
# - Non-root + tini + healthcheck; lean, cache-friendly layers
################################################################################

ARG CUDA_VERSION=12.1.1
ARG UBUNTU_VERSION=22.04
ARG PYVER=3.11             # set to 3.10 for Kaggle notebook parity
ARG TORCH_VER=2.3.1
ARG TVISION_VER=0.18.1
ARG TAUDIO_VER=2.3.1
ARG INSTALL_TORCH=1        # 1: install torch stack; 0: skip (e.g., CPU image)

############################
# Base CUDA runtime
############################
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# ---------- metadata ----------
ARG VCS_REF=""
ARG BUILD_DATE=""
LABEL org.opencontainers.image.title="SpectraMind V50" \
      org.opencontainers.image.description="NeurIPS 2025 Ariel Data Challenge — physics-informed, neuro-symbolic pipeline" \
      org.opencontainers.image.source="https://github.com/your-org/spectramind-v50" \
      org.opencontainers.image.licenses="MIT" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}"

# ---------- env ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COLOR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    PIP_REQUIRE_VIRTUALENV=1 \
    UV_NO_TELEMETRY=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin \
    MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    NCCL_LAUNCH_MODE=GROUP \
    NCCL_P2P_LEVEL=SYS \
    TORCH_USE_CUDA_DSA=0 \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    NCCL_IB_DISABLE=1

# Make shell strict by default
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ---------- base OS ----------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
 && apt-get install -y --no-install-recommends \
      ca-certificates curl git git-lfs jq tini \
      build-essential pkg-config \
      graphviz \
      libffi-dev libssl-dev \
      libgl1 libglib2.0-0 \
      locales software-properties-common \
 && git lfs install --skip-repo \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ---------- Python (3.11 default via deadsnakes; 3.10 = system) ----------
RUN if [[ "${PYVER}" == "3.10" ]]; then \
      apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-distutils python3-dev && \
      rm -rf /var/lib/apt/lists/* && PYBIN=python3; \
    else \
      add-apt-repository -y ppa:deadsnakes/ppa && apt-get update && \
      apt-get install -y --no-install-recommends \
        "python${PYVER}" "python${PYVER}-venv" "python${PYVER}-dev" python3-pip && \
      rm -rf /var/lib/apt/lists/* && PYBIN="python${PYVER}"; \
    fi \
 && ${PYBIN} -m venv "${VIRTUAL_ENV}" \
 && python -m pip install -U --no-cache-dir pip==24.2 setuptools==72.2.0 wheel==0.44.0 \
 && python -m pip config set global.no-python-version-warning true

WORKDIR /app

# Optional Torch stack (CUDA 12.1 channel); safe to skip for CPU images built from ubuntu
RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ "${INSTALL_TORCH}" == "1" ]]; then \
      python -m pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cu121 \
        "torch==${TORCH_VER}" "torchvision==${TVISION_VER}" "torchaudio==${TAUDIO_VER}"; \
    else \
      echo "Skipping Torch install (INSTALL_TORCH=${INSTALL_TORCH})"; \
    fi

# ---------- non-root user ----------
ARG APP_USER=spectra
ARG APP_UID=10001
RUN useradd -m -u ${APP_UID} -s /bin/bash ${APP_USER}

# ==============================================================================
# Builder: optional wheelhouse (hermetic / offline-friendly)
#   Build with:  docker build --target builder --build-arg BUILD_WHEELS=1 .
# ==============================================================================
FROM runtime AS builder
ARG INSTALL_EXTRAS=""
ARG BUILD_WHEELS=0

# Copy minimal metadata first (better cache)
COPY --link pyproject.toml README.md ./
# Copy sources and runtime assets needed by the wheel (package_data)
COPY --link src ./src
COPY --link configs ./configs
COPY --link schemas ./schemas

# Produce local wheels (project + deps) when requested
RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ "${BUILD_WHEELS}" == "1" ]]; then \
      echo ">> Building wheelhouse (extras='${INSTALL_EXTRAS}')"; \
      mkdir -p /opt/wheels; \
      if [[ -z "${INSTALL_EXTRAS}" ]]; then \
        python -m pip wheel --wheel-dir=/opt/wheels . ; \
      else \
        python -m pip wheel --wheel-dir=/opt/wheels ".[${INSTALL_EXTRAS}]" ; \
      fi ; \
    else \
      echo ">> Skipping wheel build (BUILD_WHEELS=0)"; \
    fi

############################
# Final runtime image
############################
FROM runtime AS final
ARG INSTALL_EXTRAS=""
ARG BUILD_WHEELS=0

# Copy metadata needed for pip editable/wheel installs (late = better cache)
COPY --link pyproject.toml README.md ./

# If a wheelhouse was built, copy it and install hermetically; else from index
COPY --from=builder /opt/wheels /opt/wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    if [[ -d /opt/wheels && "$(ls -A /opt/wheels || true)" ]]; then \
      echo ">> Installing from local wheelhouse (no network)"; \
      python -m pip install --no-index --find-links=/opt/wheels /opt/wheels/*.whl || \
      ( echo ">> Fallback: install package (extras) from wheelhouse"; \
        if [[ -z "${INSTALL_EXTRAS}" ]]; then \
          python -m pip install --no-index --find-links=/opt/wheels spectramind-v50; \
        else \
          python -m pip install --no-index --find-links=/opt/wheels "spectramind-v50[${INSTALL_EXTRAS}]"; \
        fi ); \
    else \
      echo ">> Installing from index (editable)"; \
      if [[ -z "${INSTALL_EXTRAS}" ]]; then \
        python -m pip install -e . ; \
      else \
        python -m pip install -e ".[${INSTALL_EXTRAS}]" ; \
      fi ; \
    fi

# App code & configs (late copy = better cache reuse for deps)
COPY --link src ./src
COPY --link configs ./configs
COPY --link schemas ./schemas
# Optional: COPY assets if you ship them outside package
# COPY --link assets ./assets

# Pre-compile bytecode (best-effort)
RUN python -m compileall -q /app/src || true

# ---------- healthcheck ----------
HEALTHCHECK --interval=1m --timeout=10s --start-period=15s --retries=3 \
  CMD spectramind --help >/dev/null || exit 1

# ---------- runtime user ----------
USER ${APP_USER}

# PID 1 reaping via Tini; default to CLI help
ENTRYPOINT ["/usr/bin/tini", "--", "spectramind"]
CMD ["--help"]
```

### What I fixed/added

* **Python version switch (3.11 ↔ 3.10):** `ARG PYVER` + deadsnakes for 3.11 on 22.04. Keeps repo/type-check parity but lets you mirror Kaggle notebooks when needed.
* **Deterministic Torch cu121 (optional):** `INSTALL_TORCH=1` with pinned `torch/vision/audio` via cu121 index. Flip to `0` if you build a CPU variant.
* **Cache-savvy layers:** `--mount=type=cache` for APT and pip; `COPY --link` to reduce layer churn.
* **Hardened runtime:** non-root user, `tini`, strict shell, locale set, NCCL defaults safe for CI/containers.
* **Hermetic/offline mode:** optional **wheelhouse** stage; falls back to editable install when not built.
* **Healthcheck wired to CLI:** ensures the image boots the `spectramind` entrypoint.

### Build & smoke-test

```bash
# GPU runtime w/ Python 3.11 + Torch cu121
docker build -f Dockerfile.runtime.cuda -t spectramind:cu121 \
  --build-arg PYVER=3.11 \
  --build-arg INSTALL_TORCH=1 .

# GPU runtime (Kaggle-notebook parity Python 3.10)
docker build -f Dockerfile.runtime.cuda -t spectramind:cu121-py310 \
  --build-arg PYVER=3.10 \
  --build-arg INSTALL_TORCH=1 .

# Hermetic build (wheelhouse + extras, e.g., [gpu])
docker build -f Dockerfile.runtime.cuda -t spectramind:cu121-wheels \
  --target builder --build-arg BUILD_WHEELS=1 --build-arg INSTALL_EXTRAS=gpu .
docker build -f Dockerfile.runtime.cuda -t spectramind:cu121-prod \
  --build-arg BUILD_WHEELS=1 --build-arg INSTALL_EXTRAS=gpu .

# Run (expects NVIDIA runtime)
docker run --rm --gpus all spectramind:cu121 --help
```
