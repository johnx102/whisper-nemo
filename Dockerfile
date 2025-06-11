# Dockerfile ultra-simplifié pour whisper-diarization
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    curl \
    cython3 \
    && rm -rf /var/lib/apt/lists/*

# Python par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Pip
RUN python -m pip install --upgrade pip setuptools wheel cython

# Répertoire de travail
WORKDIR /app

# Cloner whisper-diarization
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization
RUN cp /tmp/whisper-diarization/*.py /app/ || true
RUN cp -r /tmp/whisper-diarization/config /app/ || mkdir -p /app/config

# Copier nos fichiers
COPY requirements.txt /app/requirements.txt
COPY constraints.txt /app/constraints.txt
COPY main.py /app/main.py

# Installation des dépendances
RUN pip install -c constraints.txt -r requirements.txt

# Répertoires
RUN mkdir -p temp_outputs outputs /models/cache
RUN chmod -R 777 temp_outputs outputs /models/cache

# Variables d'environnement
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache
ENV TORCH_HOME=/models/cache

# Script de démarrage
COPY <<EOF /app/start.sh
#!/bin/bash
echo "🚀 Starting Whisper Diarization Service..."
mkdir -p temp_outputs outputs /models/cache
chmod -R 777 temp_outputs outputs /models/cache 2>/dev/null || true
echo "🎯 Starting application..."
exec python -u main.py
EOF

RUN chmod +x /app/start.sh

# Health check simple
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import torch; print('healthy')" || exit 1

EXPOSE 8000
CMD ["/app/start.sh"]
