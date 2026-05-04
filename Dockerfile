FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TOKENIZERS_PARALLELISM=false \
    HF_HOME=/hf_cache \
    HUGGINGFACE_HUB_CACHE=/hf_cache/hub \
    TRANSFORMERS_CACHE=/hf_cache/transformers \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_SERVER_PORT=7860 \
    ANSWER_REWRITE_MODEL_ID=Qwen/Qwen2.5-1.5B-Instruct \
    ANSWER_REWRITE_USE_4BIT=1

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

EXPOSE 7860

CMD ["python3", "app.py"]
