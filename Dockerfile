# syntax=docker/dockerfile:1.7-labs
################################################################################
# SpectraMind V50 — CUDA Runtime Image (Prod)
################################################################################
ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04
FROM nvidia/cuda:${CUDA_VERSION}-cudnn8-runtime-ubuntu${UBUNTU_VERSION} AS runtime

# ---------- metadata ----------
LABEL org.opencontainers.image.title="SpectraMind V50"
LABEL org.opencontainers.image.description="NeurIPS 2025 Ariel Data Challenge — physics-informed, neuro-symbolic pipeline"
LABEL org.opencontainers.image.source="https://github.com/your-org/spectramind-v50"
LABEL org.opencontainers.image.licenses="MIT"

# ---------- env ----------
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_COLOR=1 \
    PIP_DEFAULT_TIMEOUT=100 \
    UV_NO_TELEMETRY=1 \
    HF_HUB_DISABLE_TELEMETRY=1 \
    # venv location; keep it stable for caching
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:$PATH \
    # Avoid matplotlib trying to write in HOME when read-only
    MPLCONFIGDIR=/tmp/matplotlib

# CUDA / NCCL reliability & perf
ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    NCCL_LAUNCH_MODE=GROUP \
    NCCL_P2P_LEVEL=SYS \
    TORCH_USE_CUDA_DSA=0

# Make shell strict by default
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ---------- base OS ----------
# Core OS deps + tiny init for signal handling; keep slim and deterministic
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
 && apt-get install -y --no-install-recommends \
      python3 python3-venv python3-distutils python3-dev \
      build-essential pkg-config \
      git git-lfs curl ca-certificates \
      graphviz tini \
      libffi-dev libssl-dev \
      # Useful for many Python wheels (opencv/scipy/numba/etc.)
      libgl1 libglib2.0-0 \
      # optional: locale (avoid LC warnings)
      locales \
 && rm -rf /var/lib/apt/lists/* \
 && git lfs install --skip-repo \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen en_US.UTF-8
ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8

# ---------- non-root user ----------
ARG APP_USER=spectra
ARG APP_UID=10001
RUN useradd -m -u ${APP_UID} -s /bin/bash ${APP_USER}

# ---------- python venv ----------
RUN python3 -m venv "${VIRTUAL_ENV}" \
 && python -m pip install -U pip setuptools wheel \
 && python -m pip config set global.no-python-version-warning true

WORKDIR /app

# ------------------------------------------------------------------------------
# Dependency resolution layer
# Copy *only* files that affect dependency graph to maximize cache hits.
# If you keep constraints/locks (e.g., constraints.txt or requirements-*.txt), add them here.
# ------------------------------------------------------------------------------
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml README.md ./
# COPY --chown=${APP_USER}:${APP_USER} constraints.txt ./

# Allow optional extras at build time (e.g., INSTALL_EXTRAS=gpu,dev)
ARG INSTALL_EXTRAS=""
# You can pre-build wheels, then install, to make this more reproducible/offline-friendly.
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
# optional diagrams/assets if packaged
# COPY --chown=${APP_USER}:${APP_USER} assets ./assets

# Bytecode pre-compile (best-effort)
RUN python -m compileall -q /app/src || true

# ---------- healthcheck ----------
# Check CLI is importable and prints help within 10s.
HEALTHCHECK --interval=1m --timeout=10s --start-period=15s --retries=3 \
  CMD spectramind --help >/dev/null || exit 1

# ---------- runtime user ----------
USER ${APP_USER}

# Tini as PID 1 for clean sigterm/child reaping
ENTRYPOINT ["/usr/bin/tini", "--", "spectramind"]
CMD ["--help"]

################################################################################
# (Optional) CPU-only base for CI or dev shells (no CUDA required)
################################################################################
# FROM ubuntu:${UBUNTU_VERSION} AS cpu
# LABEL org.opencontainers.image.title="SpectraMind V50 (CPU)"
# ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH \
#     MPLCONFIGDIR=/tmp/matplotlib
# SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
# RUN --mount=type=cache,target=/var/cache/apt \
#     apt-get update -y && apt-get install -y --no-install-recommends \
#       python3 python3-venv python3-distutils python3-dev \
#       build-essential pkg-config git git-lfs curl ca-certificates graphviz tini \
#       libffi-dev libssl-dev libgl1 libglib2.0-0 locales \
#     && rm -rf /var/lib/apt/lists/* && git lfs install --skip-repo \
#     && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen en_US.UTF-8
# ENV LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8
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
