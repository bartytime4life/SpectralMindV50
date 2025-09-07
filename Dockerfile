FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1
RUN apt-get update && apt-get install -y --no-install-recommends \    git curl ca-certificates build-essential graphviz && \    rm -rf /var/lib/apt/lists/*
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3
WORKDIR /app
COPY pyproject.toml README.md ./
RUN pip install -e .
COPY src ./src
COPY configs ./configs
COPY schemas ./schemas
ENTRYPOINT ["spectramind"]
CMD ["--help"]
