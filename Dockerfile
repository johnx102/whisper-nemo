# Image de base identique à votre pod fonctionnel
FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Configuration CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Optimisations GPU identiques à votre setup
ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0 7.5 8.0 8.6+PTX"
ENV FORCE_CUDA="1"

# Installation système - EXACTEMENT comme votre script
RUN apt-get update && apt-get install -y \
    ffmpeg \
    python3-venv \
    python3-pip \
    git \
    curl \
    python3.10 \
    python3.10-dev \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python
RUN python -m pip install --upgrade pip

# Installation PyTorch - MÊME VERSION que votre setup
RUN pip install --no-cache-dir \
    torch \
    torchvision \
    torchaudio \
    --index-url https://download.pytorch.org/whl/cu118

# Installation packages - EXACTEMENT comme votre script
RUN pip install --no-cache-dir \
    openai-whisper==20240930 \
    pyannote.audio \
    huggingface_hub \
    numpy

# Ajout des dépendances serverless
RUN pip install --no-cache-dir \
    runpod>=1.6.2 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0

WORKDIR /app

# Copier notre code
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p /tmp/whisper-cache /tmp/audio-temp

# Variables d'environnement
ENV WHISPER_CACHE_DIR=/tmp/whisper-cache
ENV TEMP_AUDIO_DIR=/tmp/audio-temp

# Test des imports critiques
RUN python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
RUN python -c "import whisper; print('Whisper: OK')"
RUN python -c "from pyannote.audio import Pipeline; print('PyAnnote: OK')"

EXPOSE 8000

CMD ["python", "main.py"]
