# TorchGenomics Dockerfile
# Build:
#   CPU:  docker build -t torchgenomics .
#   GPU:  docker build -t torchgenomics:gpu --build-arg BASE_IMAGE=pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime .

ARG BASE_IMAGE=python:3.11-slim

# ---------- Stage 1: build native extensions ----------
FROM ${BASE_IMAGE} AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Install build dependencies first (layer caching)
COPY pyproject.toml setup.py ./
RUN pip install --no-cache-dir "torch>=2.0" "pybind11>=2.11" setuptools wheel numpy

# Copy source and build
COPY torchgenomics/ torchgenomics/
COPY csrc/ csrc/
RUN pip wheel --no-deps --wheel-dir /wheels .

# ---------- Stage 2: runtime ----------
FROM ${BASE_IMAGE} AS runtime

WORKDIR /app

# Install the built wheel
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# Install optional dependencies
RUN pip install --no-cache-dir zarr h5py pyarrow seaborn || true

# Copy tests and scripts for optional use
COPY tests/ tests/
COPY bench/ bench/
COPY scripts/ scripts/

# Verify installation
RUN python -c "import torchgenomics; print(f'TorchGenomics v{torchgenomics.__version__} installed successfully')"

ENTRYPOINT ["torchgenomics"]
CMD ["--help"]
