# Base image shared by every service. Heavy build (CUDA + PyTorch +
# sam_audio's deps) lives here; per-service images just copy code on top.
#
# Built once with:
#   docker build -f docker/base.Dockerfile -t sam-audio-base:latest .

FROM pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 
# L40 is sm_89; pin so kernels JIT only once

RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
        ffmpeg libsndfile1 \
        ca-certificates curl \
    && rm -rf /var/lib/apt/lists/* 

# PyTorch first so the next layer can resolve dacvae and friends against it
RUN python -m pip install --upgrade pip --break-system-packages  

# sam_audio source + its git-pinned deps. We install with --no-deps for the
# packages we don't want (perception-models pulls a lot of vision stuff we
# don't use in this pipeline) and let the rest resolve.
#
# NOTE: this assumes the sam_audio source tree is mounted at /src/sam-audio
# during build. The compose file handles that.
COPY sam_audio_src /src/sam-audio
RUN cd /src/sam-audio && python -m pip install  --break-system-packages .

# Service-layer deps (FastAPI, async batching helpers)
RUN python -m pip install  --break-system-packages \
    "fastapi==0.115.*" \
    "uvicorn[standard]==0.32.*" \
    "httpx==0.27.*" \
    "pydantic==2.9.*" 

# Default HF cache mount point. Override with -v on the host.
ENV HF_HOME=/cache/huggingface \
    TRANSFORMERS_CACHE=/cache/huggingface \
    HF_HUB_CACHE=/cache/huggingface

WORKDIR /app
