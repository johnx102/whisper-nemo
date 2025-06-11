# Dockerfile pour votre fork - CPU uniquement
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Forcer CPU uniquement - pas de GPU
ENV CUDA_VISIBLE_DEVICES=""
ENV OMP_NUM_THREADS=4
ENV TOKENIZERS_PARALLELISM=false

# Installer dépendances système
RUN apt-get update && apt-get install -y \
    python3.10 python3.10-dev python3-pip \
    build-essential cmake pkg-config \
    ffmpeg libsndfile1 libsox-dev sox \
    git wget curl cython3 \
    && rm -rf /var/lib/apt/lists/*

# Python par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Pip
RUN python -m pip install --upgrade pip setuptools wheel cython

WORKDIR /app

# Copier TOUS vos fichiers du fork (pas de clone)
COPY . /app/

# PyTorch CPU uniquement - pas de CUDA
RUN pip install torch==2.1.0+cpu torchaudio==2.1.0+cpu --index-url https://download.pytorch.org/whl/cpu

# NumPy/Numba compatibles
RUN pip install "numpy>=1.21.0,<2.0.0" "numba>=0.56.0,<0.60.0"

# Dependencies depuis votre requirements.txt mais en mode CPU
RUN pip install \
    faster-whisper>=1.1.0 \
    runpod \
    fastapi \
    uvicorn \
    aiohttp \
    aiofiles \
    pydantic \
    nltk \
    cython

# Essayer d'installer NeMo CPU (optionnel)
RUN pip install nemo_toolkit[asr] || echo "NeMo installation failed - continuing without it"

# Répertoires
RUN mkdir -p temp_outputs outputs /models/cache
RUN chmod -R 777 temp_outputs outputs /models/cache

# Variables d'environnement
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache

# Script de démarrage simple
RUN echo '#!/bin/bash\n\
echo "Starting Whisper Diarization Service (CPU Mode)..."\n\
echo "Device: CPU"\n\
echo "Threads: 4"\n\
mkdir -p temp_outputs outputs /models/cache\n\
chmod -R 777 temp_outputs outputs /models/cache 2>/dev/null || true\n\
echo "Starting main application..."\n\
exec python -u main.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Test imports
RUN python -c "import torch; print('PyTorch CPU:', torch.__version__)"
RUN python -c "import faster_whisper; print('faster-whisper: OK')"
RUN python -c "import runpod; print('RunPod: OK')"

EXPOSE 8000
CMD ["/app/start.sh"]
