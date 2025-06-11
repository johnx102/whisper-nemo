# Dockerfile avec PyTorch compatible cuDNN 8
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# FORCER cuDNN 8 (pas 9)
ENV CUDNN_VERSION=8

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

# Cloner projet
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization
RUN cp /tmp/whisper-diarization/*.py /app/ 2>/dev/null || true
RUN cp -r /tmp/whisper-diarization/config /app/ 2>/dev/null || mkdir -p /app/config

# Copier fichiers
COPY requirements.txt /app/requirements.txt
COPY constraints.txt /app/constraints.txt
COPY main.py /app/main.py

# CRITIQUE: PyTorch version compatible cuDNN 8
# Utiliser PyTorch 2.0.1 qui est compatible cuDNN 8
RUN pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# NumPy/Numba compatibles
RUN pip install "numpy>=1.21.0,<2.0.0" "numba>=0.56.0,<0.60.0"

# Autres dépendances
RUN pip install -c constraints.txt -r requirements.txt

# Répertoires
RUN mkdir -p temp_outputs outputs /models/cache
RUN chmod -R 777 temp_outputs outputs /models/cache

# Variables d'environnement
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache

# Variables pour cuDNN 8 compatibility
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Variables pour forcer cuDNN 8
ENV CUDNN_PATH=/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Test PyTorch avec cuDNN 8
RUN python -c "import torch; print('PyTorch:', torch.__version__)"
RUN python -c "import torch; print('cuDNN version:', torch.backends.cudnn.version())"

# Tests autres modules
RUN python -c "import faster_whisper"
RUN python -c "import runpod"

# Script de démarrage avec détection cuDNN
COPY <<EOF /app/start.sh
#!/bin/bash
echo "Starting Whisper Diarization Service..."

# Vérifier cuDNN version
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'cuDNN enabled: {torch.backends.cudnn.enabled}')
    cudnn_version = torch.backends.cudnn.version()
    print(f'cuDNN version: {cudnn_version}')
    
    # Forcer CPU si cuDNN version problématique
    if cudnn_version >= 9000:
        print('WARNING: cuDNN 9+ detected, forcing CPU mode for compatibility')
        import os
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        with open('/tmp/force_cpu', 'w') as f:
            f.write('cudnn9_incompatible')
    else:
        print('cuDNN 8 detected - GPU mode OK')
"

# Appliquer le mode CPU si nécessaire
if [ -f "/tmp/force_cpu" ]; then
    echo "Forcing CPU mode due to cuDNN incompatibility"
    export CUDA_VISIBLE_DEVICES=""
fi

mkdir -p temp_outputs outputs /models/cache
chmod -R 777 temp_outputs outputs /models/cache 2>/dev/null || true

echo "Starting main application..."
exec python -u main.py
EOF

RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
