FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/hf_cache \
    HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    TRANSFORMERS_CACHE=/hf_cache/transformers \
    WEB_PRELOAD_MODELS=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    git \
    curl \
    ca-certificates \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt /app/requirements.txt

RUN python3 -m pip install --upgrade pip setuptools wheel && \
    python3 -m pip install --index-url https://download.pytorch.org/whl/cu121 \
        torch torchvision torchaudio && \
    python3 -m pip install -r requirements.txt

COPY . /app

RUN mkdir -p /hf_cache

EXPOSE 8000

CMD ["python3", "-m", "uvicorn", "web.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
