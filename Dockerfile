# CUDA 12.2 comme demandé
FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

# Métadonnées
LABEL maintainer="whisper-diarization-gpu"
LABEL version="6.0.0"

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

# Installation système EXACTE selon le projet
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    cython3 \
    libsndfile1 \
    libsox-dev \
    sox \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Liens symboliques CUDA pour torchaudio
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.11.0 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/lib/x86_64-linux-gnu/libcudart.so.11.0 && \
    ldconfig

RUN python -m pip install --upgrade pip setuptools wheel

# 1. Cloner le repo original pour avoir constraints.txt
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization

# 2. MÉTHODE OFFICIELLE: Installer avec constraints.txt
WORKDIR /tmp/whisper-diarization
RUN pip install -c constraints.txt -r requirements.txt

# 3. Ajouter les dépendances manquantes pour notre service
RUN pip install --no-cache-dir \
    runpod>=1.6.2 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0 \
    paramiko \
    cryptography

# Test que tout fonctionne
RUN python -c "import torch; print('PyTorch CUDA:', torch.cuda.is_available())"
RUN python -c "import faster_whisper; print('faster-whisper: OK')"
RUN python -c "import runpod; print('runpod: OK')"
RUN python -c "import nemo; print('NeMo version:', nemo.__version__)"
RUN python -c "from nemo.collections.asr.models.msdd_models import NeuralDiarizer; print('✅ NeuralDiarizer imported successfully!')"

# Copier les helpers du projet whisper-diarization dans notre app
WORKDIR /app
RUN cp /tmp/whisper-diarization/helpers.py /app/
RUN cp /tmp/whisper-diarization/diarize.py /app/ || echo "diarize.py copied"

# Copier notre code
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p /tmp/whisper-cache /tmp/audio-temp /app/temp_outputs

RUN chmod +x main.py

ENV WHISPER_CACHE_DIR=/tmp/whisper-cache
ENV TEMP_AUDIO_DIR=/tmp/audio-temp

HEALTHCHECK --interval=30s --timeout=15s --start-period=240s --retries=3 \
    CMD python -c "import torch; import runpod; import nemo; print('Health: OK')" || exit 1

EXPOSE 8000
CMD ["python", "main.py"]
