# Dockerfile corrigé avec cuDNN pour whisper-diarization
FROM nvidia/cuda:12.2.2-cudnn8-devel-ubuntu22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# CORRECTION CUDNN - Variables critiques
ENV CUDNN_PATH=/usr/lib/x86_64-linux-gnu
ENV LD_LIBRARY_PATH=$CUDNN_PATH:$LD_LIBRARY_PATH

# Installer les dépendances système + cuDNN
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
    libcudnn8 \
    libcudnn8-dev \
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

# CORRECTION: Installation des dépendances avec versions strictes
RUN pip install --no-cache-dir \
    "numpy>=1.21.0,<2.0.0" \
    "numba>=0.56.0,<0.60.0"

# Installation du reste avec contraintes
RUN pip install --no-cache-dir -c constraints.txt -r requirements.txt

# Répertoires
RUN mkdir -p temp_outputs outputs /models/cache
RUN chmod -R 777 temp_outputs outputs /models/cache

# Variables d'environnement optimisées
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache
ENV TORCH_HOME=/models/cache

# CORRECTION: Variables pour éviter les crashes mémoire
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512
ENV CUDA_LAUNCH_BLOCKING=1

# Script de démarrage corrigé
RUN echo '#!/bin/bash\n\
echo "🚀 Starting Whisper Diarization Service..."\n\
\n\
# Vérifier cuDNN\n\
echo "🔍 Checking cuDNN..."\n\
if [ -f "/usr/lib/x86_64-linux-gnu/libcudnn.so.8" ]; then\n\
    echo "✅ cuDNN found"\n\
    ls -la /usr/lib/x86_64-linux-gnu/libcudnn*\n\
else\n\
    echo "❌ cuDNN not found"\n\
fi\n\
\n\
# Créer les répertoires\n\
mkdir -p temp_outputs outputs /models/cache\n\
chmod -R 777 temp_outputs outputs /models/cache 2>/dev/null || true\n\
\n\
# Test rapide des dépendances\n\
echo "🔍 Quick dependency check..."\n\
python -c "\n\
try:\n\
    import torch\n\
    print(f\"PyTorch: {torch.__version__}\")\n\
    print(f\"CUDA available: {torch.cuda.is_available()}\")\n\
    if torch.cuda.is_available():\n\
        print(f\"GPU: {torch.cuda.get_device_name()}\")\n\
        print(f\"cuDNN enabled: {torch.backends.cudnn.enabled}\")\n\
        print(f\"cuDNN version: {torch.backends.cudnn.version()}\")\n\
    import faster_whisper\n\
    print(\"faster-whisper: OK\")\n\
    import nemo\n\
    print(\"NeMo: OK\")\n\
    import runpod\n\
    print(\"RunPod: OK\")\n\
    print(\"✅ All dependencies OK\")\n\
except Exception as e:\n\
    print(f\"❌ Dependency error: {e}\")\n\
    exit(1)\n\
" || exit 1\n\
\n\
echo "🎯 Starting main application..."\n\
exec python -u main.py' > /app/start.sh

RUN chmod +x /app/start.sh

# Health check amélioré
HEALTHCHECK --interval=30s --timeout=30s --start-period=15s --retries=3 \
    CMD python -c "import torch; print(f'GPU: {torch.cuda.is_available()}'); print('healthy')" || exit 1

EXPOSE 8000
CMD ["/app/start.sh"]
