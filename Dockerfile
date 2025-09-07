# syntax=docker/dockerfile:1.7-labs
################################################################################
# SpectraMind V50 — CUDA Runtime Image (Prod)
################################################################################
ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COLOR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    UV_NO_TELEMETRY=1 \
    # venv location; keep it stable for caching
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH

# Core OS deps + tiny init for signal handling; keep slim and deterministic
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-distutils python3-dev \
      build-essential pkg-config \
      git git-lfs curl ca-certificates \
      graphviz tini \
      libffi-dev libssl-dev \
      # Useful for many Python wheels (opencv/scipy/numba/etc.)
      libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && git lfs install

# Create dedicated non-root user early for better layer reuse
ARG APP_USER=spectra
ARG APP_UID=10001
RUN useradd -m -u ${APP_UID} -s /bin/bash ${APP_USER}

# Create an isolated venv and upgrade tooling
RUN python3 -m venv "${VIRTUAL_ENV}" \
 && python -m pip install -U pip setuptools wheel

# Optional: set CUDA env knobs that often improve reliability/perf
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    # PyTorch / NCCL often benefit from these defaults:
    NCCL_LAUNCH_MODE=GROUP \
    NCCL_P2P_LEVEL=SYS \
    TORCH_USE_CUDA_DSA=0

WORKDIR /app

# ------------------------------------------------------------------------------
# Dependency resolution layer
# Copy *only* files that affect dependency graph to maximize cache hits.
# If you keep constraints/locks (e.g., requirements-*.txt), add them here.
# ------------------------------------------------------------------------------
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml README.md ./
# If you maintain constraints, uncomment next line and add the file.
# COPY --chown=${APP_USER}:${APP_USER} constraints.txt ./

# Allow optional extras at build time (e.g., INSTALL_EXTRAS=gpu,dev)
ARG INSTALL_EXTRAS=""
# If your build needs private GitHub tokens, pass them as build args or use
# GitHub Actions OIDC to pre-resolve wheels in a separate stage.

RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -z "${INSTALL_EXTRAS}" ]; then \
      python -m pip install -e . ; \
    else \
      python -m pip install -e ".[${INSTALL_EXTRAS}]" ; \
    fi

# ------------------------------------------------------------------------------
# App code & configs
# ------------------------------------------------------------------------------
COPY --chown=${APP_USER}:${APP_USER} src ./src
COPY --chown=${APP_USER}:${APP_USER} configs ./configs
COPY --chown=${APP_USER}:${APP_USER} schemas ./schemas

# Optionally pre-compile bytecode for slightly faster cold starts
RUN python -m compileall -q /app/src || true

# Drop privileges
USER ${APP_USER}

# Tini as PID 1 for clean sigterm/child reaping
ENTRYPOINT ["/usr/bin/tini", "--", "spectramind"]
CMD ["--help"]

################################################################################
# (Optional) CPU-only base for CI or dev shells (no CUDA required)
################################################################################
# Uncomment to build a smaller CPU image (handy for CI)
# FROM ubuntu:${UBUNTU_VERSION} AS cpu
# ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
# RUN --mount=type=cache,target=/var/cache/apt \
#     apt-get update && apt-get install -y --no-install-recommends \
#       python3 python3-venv python3-distutils python3-dev \
#       build-essential pkg-config git git-lfs curl ca-certificates graphviz tini \
#       libffi-dev libssl-dev libgl1 libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/* && git lfs install
# WORKDIR /app
# RUN python3 -m venv "${VIRTUAL_ENV}" && python -m pip install -U pip setuptools wheel
# COPY pyproject.toml README.md ./
# ARG INSTALL_EXTRAS=""
# RUN --mount=type=cache,target=/root/.cache/pip \
#     if [ -z "${INSTALL_EXTRAS}" ]; then python -m pip install -e . ; \
#     else python -m pip install -e ".[${INSTALL_EXTRAS}]" ; fi
# COPY src ./src
# COPY configs ./configs
# COPY schemas ./schemas
# RUN python -m compileall -q /app/src || true
# USER 1000
# ENTRYPOINT ["/usr/bin/tini", "--", "spectramind"]
# CMD ["--help"]
