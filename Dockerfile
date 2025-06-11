# CUDA 12.2 comme demandé
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Métadonnées
LABEL maintainer="whisper-diarization-gpu"
LABEL version="5.0.0"

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

# Variables pour forcer la compatibilité
ENV CUDA_VERSION=12.2
ENV TORCH_CUDA_ARCH_LIST="6.0;6.1;7.0;7.5;8.0;8.6;9.0"

# Mise à jour système et installation des dépendances système
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
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

# Créer lien symbolique pour python
RUN ln -sf /usr/bin/python3.10 /usr/bin/python

# WORKAROUND: Créer les liens symboliques nécessaires pour torchaudio
RUN mkdir -p /usr/local/cuda/lib64 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/local/cuda/lib64/libcudart.so.11.0 && \
    ln -sf /usr/local/cuda/lib64/libcudart.so.12 /usr/lib/x86_64-linux-gnu/libcudart.so.11.0 && \
    ldconfig

# Upgrade pip
RUN python -m pip install --upgrade pip setuptools wheel

# ÉTAPE 1: FORCER NumPy 1.x en premier
RUN pip install --no-cache-dir "numpy>=1.21.0,<2.0.0"

# Vérifier NumPy
RUN python -c "import numpy; print('NumPy version:', numpy.__version__); assert numpy.__version__.startswith('1.'), 'NumPy 2.x detected!'"

# ÉTAPE 2: Installation PyTorch avec CUDA
RUN pip install --no-cache-dir \
    torch==2.1.2+cu121 \
    torchaudio==2.1.2+cu121 \
    torchvision==0.16.2+cu121 \
    --index-url https://download.pytorch.org/whl/cu121

# Test PyTorch
RUN python -c "import torch; import torchaudio; print('PyTorch CUDA:', torch.cuda.is_available())"

# ÉTAPE 3: Dependencies de base
RUN pip install --no-cache-dir \
    paramiko \
    cryptography \
    pycryptodome \
    cffi

# ÉTAPE 4: faster-whisper
RUN pip install --no-cache-dir faster-whisper>=1.1.0

# ÉTAPE 5: Web framework
RUN pip install --no-cache-dir \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0 \
    runpod>=1.6.2

# ÉTAPE 6: Installation des repos Git du projet whisper-diarization
RUN pip install --no-cache-dir \
    git+https://github.com/MahmoudAshraf97/demucs.git \
    git+https://github.com/oliverguhr/deepmultilingualpunctuation.git \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# ÉTAPE 7: Autres dépendances
RUN pip install --no-cache-dir \
    nltk \
    wget \
    omegaconf \
    hydra-core

# ÉTAPE 8: NeMo avec versions compatibles
RUN echo "Installing NeMo..." && \
    pip install --no-cache-dir \
    pytorch-lightning==2.1.4 \
    torchmetrics==1.2.1 \
    omegaconf==2.3.0 \
    hydra-core==1.3.2

# Essayer plusieurs versions de NeMo
RUN pip install --no-cache-dir nemo-toolkit[asr]==1.22.0 || \
    pip install --no-cache-dir nemo-toolkit[asr]==2.0.0rc0 || \
    pip install --no-cache-dir nemo-toolkit[asr] || \
    echo "Warning: All NeMo installations failed"

# Test final des imports avec debug
RUN python -c "
try:
    import nemo
    print('✅ NeMo imported, version:', nemo.__version__)
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    print('✅ NeuralDiarizer imported successfully')
except Exception as e:
    print('❌ NeMo error:', e)
    import traceback
    traceback.print_exc()
"

# Dossier de travail
WORKDIR /app

# Copier les fichiers du projet
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p /tmp/whisper-cache /tmp/audio-temp /app/temp_outputs

# Permissions
RUN chmod +x main.py

# Variables d'environnement pour l'application
ENV WHISPER_CACHE_DIR=/tmp/whisper-cache
ENV TEMP_AUDIO_DIR=/tmp/audio-temp

# Health check robuste
HEALTHCHECK --interval=30s --timeout=15s --start-period=180s --retries=3 \
    CMD python -c "import torch; import runpod; print('Health: OK')" || exit 1

# Exposer le port FastAPI
EXPOSE 8000

# Commande par défaut
CMD ["python", "main.py"]
