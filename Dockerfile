# CUDA 12.2 comme demandé
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Métadonnées
LABEL maintainer="whisper-diarization-gpu"
LABEL version="5.1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Configuration CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Installation système de base
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Liens symboliques CUDA
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.11.0 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/lib/x86_64-linux-gnu/libcudart.so.11.0 && \
    ldconfig

RUN python -m pip install --upgrade pip setuptools wheel

# 1. NumPy 1.x obligatoire
RUN pip install --no-cache-dir "numpy>=1.21.0,<2.0.0"

# 2. PyTorch avec CUDA
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchaudio==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# 3. Dependencies de base
RUN pip install --no-cache-dir \
    paramiko \
    cryptography \
    faster-whisper>=1.1.0

# 4. Web framework
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0 \
    runpod>=1.6.2

# 5. Repos Git du projet
RUN pip install --no-cache-dir \
    git+https://github.com/MahmoudAshraf97/demucs.git \
    git+https://github.com/oliverguhr/deepmultilingualpunctuation.git \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# 6. CRITICAL: Versions exactes pour compatibilité NeMo
RUN pip install --no-cache-dir \
    "huggingface_hub==0.20.3" \
    "transformers==4.37.2" \
    "tokenizers==0.15.2" \
    nltk \
    wget

# 7. Dependencies PyTorch Lightning compatibles
RUN pip install --no-cache-dir \
    "pytorch-lightning==2.1.4" \
    "torchmetrics==1.2.1" \
    "omegaconf==2.3.0" \
    "hydra-core==1.3.2"

# 8. Installation NeMo version stable
RUN echo "Installing NeMo 1.22.0 (stable version)..." && \
    pip install --no-cache-dir "nemo-toolkit[asr]==1.22.0" && \
    echo "✅ NeMo 1.22.0 installed successfully"

# Test des imports
RUN python -c "import torch; import faster_whisper; import runpod; print('✅ Core imports OK')"
RUN python -c "import nemo; print('✅ NeMo version:', nemo.__version__)"
RUN python -c "from nemo.collections.asr.models.msdd_models import NeuralDiarizer; print('✅ NeuralDiarizer imported successfully')"

WORKDIR /app
COPY . .

RUN mkdir -p /tmp/whisper-cache /tmp/audio-temp /app/temp_outputs
RUN chmod +x main.py

ENV WHISPER_CACHE_DIR=/tmp/whisper-cache
ENV TEMP_AUDIO_DIR=/tmp/audio-temp

HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD python -c "import torch; import runpod; print('Health: OK')" || exit 1

EXPOSE 8000
CMD ["python", "main.py"]
