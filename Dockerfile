# Dockerfile simplifié pour whisper-diarization avec RunPod
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

# Installer Cython d'abord
RUN pip install --no-cache-dir cython

# Créer le répertoire de travail
WORKDIR /app

# Cloner le repository whisper-diarization
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization

# Copier les fichiers Python du projet original
RUN cp /tmp/whisper-diarization/*.py /app/ || true
RUN cp -r /tmp/whisper-diarization/config /app/ 2>/dev/null || mkdir -p /app/config

# Copier nos fichiers personnalisés
COPY requirements.txt /app/requirements.txt
COPY constraints.txt /app/constraints.txt
COPY main.py /app/main.py

# Installation des dépendances avec contraintes
RUN pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements.txt

# Créer les répertoires nécessaires
RUN mkdir -p /app/temp_outputs \
    && mkdir -p /app/outputs \
    && mkdir -p /models/cache \
    && chmod -R 777 /app/temp_outputs \
    && chmod -R 777 /app/outputs \
    && chmod -R 777 /models

# Variables d'environnement pour les modèles
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV CUDA_VISIBLE_DEVICES=0
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:256
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Cache directories
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache
ENV TORCH_HOME=/models/cache

# Test des imports critiques au build
RUN python -c "
try:
    import torch
    print(f'✅ PyTorch: {torch.__version__}')
    print(f'✅ CUDA available: {torch.cuda.is_available()}')
    
    import faster_whisper
    print('✅ faster-whisper imported')
    
    import nemo
    print('✅ NeMo imported')
    
    import runpod
    print('✅ RunPod imported')
    
    print('🎉 All critical imports successful!')
    
except Exception as e:
    print(f'❌ Import error: {e}')
    import sys
    sys.exit(1)
"

# Script de démarrage simple
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Whisper Diarization Service..."\n\
\n\
# Créer les répertoires\n\
mkdir -p /app/temp_outputs /app/outputs /models/cache\n\
chmod -R 777 /app/temp_outputs /app/outputs /models/cache 2>/dev/null || true\n\
\n\
# Vérification rapide\n\
echo "🔍 Quick dependency check..."\n\
python -c "import torch, faster_whisper, nemo, runpod; print(\"✅ Dependencies OK\")" || exit 1\n\
\n\
echo "🎯 Starting main application..."\n\
exec python -u main.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=10s --retries=3 \
    CMD python -c "import torch, faster_whisper, nemo; print('healthy')" || exit 1

# Exposer le port
EXPOSE 8000

# Point d'entrée
CMD ["/app/start.sh"]
