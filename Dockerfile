# syntax=docker/dockerfile:1.7-labs
################################################################################
# SpectraMind V50 — CUDA Runtime Image (Prod, Upgraded)
################################################################################

ARG CUDA_VERSION=12.1.0
ARG UBUNTU_VERSION=22.04

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
    PATH=/opt/venv/bin:$PATH \
    MPLCONFIGDIR=/tmp/matplotlib \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    CUDA_MODULE_LOADING=LAZY \
    NCCL_LAUNCH_MODE=GROUP \
    NCCL_P2P_LEVEL=SYS \
    TORCH_USE_CUDA_DSA=0

# Make shell strict by default
SHELL ["/bin/bash", "-euo", "pipefail", "-c"]

# ---------- base OS ----------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y \
 && apt-get install -y --no-install-recommends \
      python3 python3-pip python3-venv python3-distutils python3-dev \
      build-essential pkg-config \
      git git-lfs curl ca-certificates \
      graphviz tini \
      libffi-dev libssl-dev \
      libgl1 libglib2.0-0 \
      locales \
 && git lfs install --skip-repo \
 && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen \
 && locale-gen en_US.UTF-8 \
 && rm -rf /var/lib/apt/lists/*
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

# ==============================================================================
# Builder: produce wheelhouse for hermetic install (optional)
#   Build with:  docker build --target builder --build-arg BUILD_WHEELS=1 .
# ==============================================================================
FROM runtime AS builder
ARG INSTALL_EXTRAS=""
ARG BUILD_WHEELS=0

# Copy minimal metadata first (better cache)
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml README.md ./
# Copy sources and runtime assets needed by the wheel (package_data)
COPY --chown=${APP_USER}:${APP_USER} src ./src
COPY --chown=${APP_USER}:${APP_USER} configs ./configs
COPY --chown=${APP_USER}:${APP_USER} schemas ./schemas

# Produce local wheels (project + deps) when requested
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ "${BUILD_WHEELS}" = "1" ]; then \
      echo ">> Building wheelhouse (extras='${INSTALL_EXTRAS}')"; \
      mkdir -p /opt/wheels; \
      if [ -z "${INSTALL_EXTRAS}" ]; then \
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

# Copy metadata needed for pip editable installs
COPY --chown=${APP_USER}:${APP_USER} pyproject.toml README.md ./

# If a wheelhouse was built, copy it and install hermetically; else from index
COPY --from=builder /opt/wheels /opt/wheels
RUN --mount=type=cache,target=/root/.cache/pip \
    if [ -d /opt/wheels ] && [ "$(ls -A /opt/wheels || true)" ]; then \
      echo ">> Installing from local wheelhouse (no network)"; \
      # Try direct wheels; if multiple, install spectramind-v50 with extras to ensure entrypoints
      python -m pip install --no-index --find-links=/opt/wheels /opt/wheels/*.whl || \
      ( echo ">> Fallback: install package (extras) from wheelhouse"; \
        if [ -z "${INSTALL_EXTRAS}" ]; then \
          python -m pip install --no-index --find-links=/opt/wheels spectramind-v50; \
        else \
          python -m pip install --no-index --find-links=/opt/wheels "spectramind-v50[${INSTALL_EXTRAS}]"; \
        fi ); \
    else \
      echo ">> Installing from index (editable)"; \
      if [ -z "${INSTALL_EXTRAS}" ]; then \
        python -m pip install -e . ; \
      else \
        python -m pip install -e ".[${INSTALL_EXTRAS}]" ; \
      fi ; \
    fi

# App code & configs (late copy = better cache reuse for deps)
COPY --chown=${APP_USER}:${APP_USER} src ./src
COPY --chown=${APP_USER}:${APP_USER} configs ./configs
COPY --chown=${APP_USER}:${APP_USER} schemas ./schemas
# Optional: COPY assets if you ship them outside package
# COPY --chown=${APP_USER}:${APP_USER} assets ./assets

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

################################################################################
# (Optional) CPU-only base for CI/dev (no CUDA required)
################################################################################
# FROM ubuntu:${UBUNTU_VERSION} AS cpu
# LABEL org.opencontainers.image.title="SpectraMind V50 (CPU)"
# ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 \
#     VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH MPLCONFIGDIR=/tmp/matplotlib
# SHELL ["/bin/bash", "-euo", "pipefail", "-c"]
# RUN --mount=type=cache,target=/var/cache/apt \
#     apt-get update -y && apt-get install -y --no-install-recommends \
#       python3 python3-pip python3-venv python3-distutils python3-dev \
#       build-essential pkg-config git git-lfs curl ca-certificates graphviz tini \
#       libffi-dev libssl-dev libgl1 libglib2.0-0 locales \
#   && git lfs install --skip-repo \
#   && sed -i -e 's/# en_US.UTF-8 UTF-8/en_US.UTF-8 UTF-8/' /etc/locale.gen && locale-gen en_US.UTF-8 \
#   && rm -rf /var/lib/apt/lists/*
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