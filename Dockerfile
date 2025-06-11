# CUDA 12.2 comme demandé
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Métadonnées
LABEL maintainer="whisper-diarization-gpu"
LABEL version="4.0.0"

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# Configuration CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# FIX CRITIQUE: Variables pour forcer la compatibilité CUDA
ENV CUDA_VERSION=12.2
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0"

# Mise à jour système et installation des dépendances de base
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    git \
    wget \
    curl \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    build-essential \
    pkg-config \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Créer lien symbolique pour python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# WORKAROUND CRITIQUE: Créer les liens symboliques nécessaires pour torchaudio
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.11.0 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/lib/x86_64-linux-gnu/libcudart.so.11.0 && \
    ldconfig

# Dossier de travail
WORKDIR /app

# Installation de PyTorch AVANT les autres packages (ordre important!)
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchaudio==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Test que PyTorch fonctionne
RUN python -c "import torch; print('PyTorch OK:', torch.cuda.is_available())"

# Installation de faster-whisper d'abord
RUN pip install --no-cache-dir faster-whisper>=1.1.0

# Installation des dépendances Git personnalisées
RUN pip install --no-cache-dir \
    git+https://github.com/MahmoudAshraf97/demucs.git \
    git+https://github.com/oliverguhr/deepmultilingualpunctuation.git \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# Installation des autres dépendances
RUN pip install --no-cache-dir \
    nltk \
    wget \
    runpod>=1.6.2 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0

# Installation de NeMo EN DERNIER (cause le plus de problèmes)
RUN pip install --no-cache-dir nemo-toolkit[asr]==2.0.0rc0 || \
    pip install --no-cache-dir nemo-toolkit[asr] || \
    echo "Warning: NeMo installation failed, basic diarization only"

# Test final
RUN python -c "import torch; import torchaudio; print('All imports OK')" || \
    echo "Warning: Some imports failed"

# Copier les fichiers du projet
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p /tmp/whisper-cache /tmp/audio-temp /app/temp_outputs

# Permissions
RUN chmod +x main.py

# Variables d'environnement pour l'application
ENV WHISPER_CACHE_DIR=/tmp/whisper-cache
ENV TEMP_AUDIO_DIR=/tmp/audio-temp

# Health check amélioré
HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
    CMD python -c "import torch; print('CUDA:', torch.cuda.is_available())" || exit 1

# Exposer le port FastAPI
EXPOSE 8000

# Commande par défaut
CMD ["python", "main.py"]
