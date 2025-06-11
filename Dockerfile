# Dockerfile pour whisper-diarization avec RunPod
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    python3.10-venv \
    build-essential \
    cmake \
    pkg-config \
    libffi-dev \
    libssl-dev \
    ffmpeg \
    libsndfile1 \
    libsox-dev \
    sox \
    git \
    wget \
    curl \
    cython3 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Définir Python 3.10 comme défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Mettre à jour pip
RUN python -m pip install --upgrade pip setuptools wheel

# Installer Cython d'abord (requis pour certains packages)
RUN pip install --no-cache-dir cython

# Créer le répertoire de travail
WORKDIR /app

# Cloner le repository whisper-diarization
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization

# Copier les fichiers nécessaires du projet
COPY --from=/tmp/whisper-diarization/requirements
