# ── Base image ────────────────────────────────────────────────────────────────
# nvidia/cuda provides libcuda.so, libcublas, cuDNN — required by PyTorch + faster-whisper.
# 12.1.1 matches the CUDA version used in the Cog build (cog/cog.yaml).
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

# ── System packages ───────────────────────────────────────────────────────────
# Ubuntu 22.04 (jammy) doesn't ship python3.12 in its default repos.
# The deadsnakes PPA provides it.
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        software-properties-common \
        curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y --no-install-recommends \
        python3.12 \
        python3.12-dev \
        python3.12-venv \
        ffmpeg \
    && curl -sS https://bootstrap.pypa.io/get-pip.py | python3.12 \
    && rm -rf /var/lib/apt/lists/*

# Make python3.12 the default python / python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.12 1 \
 && update-alternatives --install /usr/bin/pip     pip     /usr/local/bin/pip3.12 1

WORKDIR /app

# ── Python dependencies ───────────────────────────────────────────────────────
# Copy requirements first so Docker can cache this layer separately from source.
COPY core/requirements.txt core/requirements.txt
COPY api/requirements.txt  api/requirements.txt
RUN pip3 install --no-cache-dir -r api/requirements.txt

# ── Application source ────────────────────────────────────────────────────────
COPY core/  core/
COPY api/   api/

# ── Pre-built model cache ─────────────────────────────────────────────────────
# model_cache/ contains the whisper and pyannote models baked into the image.
# This avoids downloading models at runtime, matching the Cog approach.
# If the cache is large, consider mounting it as a volume instead.
COPY model_cache/ model_cache/

# ── Runtime config ────────────────────────────────────────────────────────────
# Override DEVICE and COMPUTE_TYPE at `docker run` time if needed.
ENV DEVICE=cuda
ENV COMPUTE_TYPE=int8

EXPOSE 8000

# ── Server ────────────────────────────────────────────────────────────────────
# --workers 1: the pipeline loads models into GPU VRAM once per worker.
#   Multiple workers = multiple model copies = OOM on a single GPU.
#   Concurrency is handled by the job queue in server.py.
#   For multi-GPU: run one container per GPU with CUDA_VISIBLE_DEVICES=N.
CMD ["uvicorn", "api.server:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
