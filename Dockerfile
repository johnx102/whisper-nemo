# Dockerfile avec version CUDA compatible
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Variables cuDNN 8 (pas 9)
ENV CUDNN_PATH=/usr/local/cuda/lib64
ENV LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH

# FORCE cuDNN 8 compatibility
ENV CUDNN_VERSION=8

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    --no-install-recommends \
    python3.10 \
    python3.10-dev \
    python3-pip \
    build-essential \
    cmake \
    pkg-config \
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

# Python par défaut
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1

# Pip
RUN python -m pip install --upgrade pip setuptools wheel cython

# Répertoire de travail
WORKDIR /app

# Cloner whisper-diarization
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization

# Copier les fichiers Python du projet original
RUN cp /tmp/whisper-diarization/*.py /app/ 2>/dev/null || true
RUN cp -r /tmp/whisper-diarization/config /app/ 2>/dev/null || mkdir -p /app/config

# Copier nos fichiers personnalisés
COPY requirements.txt /app/requirements.txt
COPY constraints.txt /app/constraints.txt
COPY main.py /app/main.py

# Installation PyTorch avec CUDA 12.1 compatible
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cu121

# Installation des dépendances critiques
RUN pip install --no-cache-dir \
    "numpy>=1.21.0,<2.0.0" \
    "numba>=0.56.0,<0.60.0"

# Installation du reste avec contraintes
RUN pip install --no-cache-dir -c /app/constraints.txt -r /app/requirements.txt

# Répertoires
RUN mkdir -p /app/temp_outputs \
    && mkdir -p /app/outputs \
    && mkdir -p /models/cache \
    && chmod -R 777 /app/temp_outputs \
    && chmod -R 777 /app/outputs \
    && chmod -R 777 /models

# Variables d'environnement optimisées
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache
ENV TORCH_HOME=/models/cache

# Variables pour forcer CPU en cas de problème cuDNN
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.6
ENV CUDA_LAUNCH_BLOCKING=1
ENV OMP_NUM_THREADS=1
ENV TOKENIZERS_PARALLELISM=false

# Test PyTorch avec gestion cuDNN
RUN python -c "\
import torch;\
print('PyTorch version:', torch.__version__);\
print('CUDA available:', torch.cuda.is_available());\
if torch.cuda.is_available():\
    try:\
        print('cuDNN enabled:', torch.backends.cudnn.enabled);\
        print('cuDNN version:', torch.backends.cudnn.version());\
        x = torch.randn(10, 10).cuda();\
        y = torch.randn(10, 10).cuda();\
        z = torch.mm(x, y);\
        print('CUDA operations: OK');\
    except Exception as e:\
        print('CUDA test failed:', e);\
        print('Will fallback to CPU mode');\
"

# Test des autres imports
RUN python -c "import faster_whisper; print('faster-whisper: OK')"
RUN python -c "import runpod; print('RunPod: OK')"

# Script de démarrage avec fallback CPU
RUN echo '#!/bin/bash\n\
echo "Starting Whisper Diarization Service..."\n\
\n\
# Creer les repertoires\n\
mkdir -p /app/temp_outputs /app/outputs /models/cache\n\
chmod -R 777 /app/temp_outputs /app/outputs /models/cache 2>/dev/null || true\n\
\n\
# Test cuDNN et fallback CPU si probleme\n\
echo "Testing CUDA/cuDNN compatibility..."\n\
python -c "\n\
import torch\n\
import os\n\
print(f\"CUDA available: {torch.cuda.is_available()}\")\n\
if torch.cuda.is_available():\n\
    try:\n\
        print(f\"GPU: {torch.cuda.get_device_name()}\")\n\
        print(f\"cuDNN enabled: {torch.backends.cudnn.enabled}\")\n\
        print(f\"cuDNN version: {torch.backends.cudnn.version()}\")\n\
        # Test simple operation\n\
        x = torch.randn(10, 10).cuda()\n\
        y = torch.mm(x, x)\n\
        print(\"GPU operations: OK\")\n\
    except Exception as e:\n\
        print(f\"GPU test failed: {e}\")\n\
        print(\"Setting CUDA_VISIBLE_DEVICES=\\\"\\\" to force CPU mode\")\n\
        os.environ[\"CUDA_VISIBLE_DEVICES\"] = \"\"\n\
        with open(\"/tmp/force_cpu\", \"w\") as f:\n\
            f.write(\"1\")\n\
else:\n\
    print(\"Using CPU mode\")\n\
"\n\
\n\
# Force CPU si fichier existe\n\
if [ -f "/tmp/force_cpu" ]; then\n\
    echo "Forcing CPU mode due to GPU issues"\n\
    export CUDA_VISIBLE_DEVICES=""\n\
fi\n\
\n\
# Test dependencies\n\
echo "Quick dependency check..."\n\
python -c "\n\
try:\n\
    import faster_whisper\n\
    print(\"faster-whisper: OK\")\n\
    import runpod\n\
    print(\"RunPod: OK\")\n\
    print(\"Dependencies OK\")\n\
except Exception as e:\n\
    print(f\"Dependency error: {e}\")\n\
    exit(1)\n\
" || exit 1\n\
\n\
echo "Starting main application..."\n\
exec python -u main.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD python -c "import torch, faster_whisper; print('healthy')" || exit 1

EXPOSE 8000
CMD ["/app/start.sh"]
