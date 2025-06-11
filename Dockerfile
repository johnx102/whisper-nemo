# Dockerfile GPU robuste avec gestion cuDNN
FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=$CUDA_HOME/bin:$PATH
ENV LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# Variables pour GPU
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

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

# Copier vos fichiers du fork
COPY . /app/

# PyTorch compatible CUDA 11.8 + cuDNN 8
RUN pip install torch==2.0.1+cu118 torchaudio==2.0.2+cu118 --index-url https://download.pytorch.org/whl/cu118

# NumPy/Numba compatibles
RUN pip install "numpy>=1.21.0,<2.0.0" "numba>=0.56.0,<0.60.0"

# Dependencies du projet
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

# NeMo compatible
RUN pip install nemo_toolkit[asr] || echo "NeMo failed - will work without it"

# Autres dependencies du projet original
RUN pip install \
    deepmultilingualpunctuation \
    || echo "Punctuation model failed - will work without it"

# Répertoires
RUN mkdir -p temp_outputs outputs /models/cache pred_rttms
RUN chmod -R 777 temp_outputs outputs /models/cache pred_rttms

# Variables d'environnement GPU
ENV HF_HOME=/models/cache
ENV TRANSFORMERS_CACHE=/models/cache
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Variables pour éviter les problèmes cuDNN
ENV CUDNN_DETERMINISTIC=1
ENV CUBLAS_WORKSPACE_CONFIG=:4096:8

# Test PyTorch GPU
RUN python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"

# Test autres imports
RUN python -c "import faster_whisper; print('faster-whisper: OK')"
RUN python -c "import runpod; print('RunPod: OK')"

# Script de démarrage avec test GPU
RUN echo '#!/bin/bash\n\
echo "Starting Whisper Diarization Service (GPU Mode)..."\n\
\n\
# Test GPU\n\
python -c "\n\
import torch\n\
print(f\"PyTorch: {torch.__version__}\")\n\
print(f\"CUDA available: {torch.cuda.is_available()}\")\n\
if torch.cuda.is_available():\n\
    print(f\"GPU: {torch.cuda.get_device_name()}\")\n\
    print(f\"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB\")\n\
    try:\n\
        print(f\"cuDNN version: {torch.backends.cudnn.version()}\")\n\
        # Test GPU rapide\n\
        x = torch.randn(100, 100).cuda()\n\
        y = torch.mm(x, x)\n\
        del x, y\n\
        torch.cuda.empty_cache()\n\
        print(\"GPU test: OK\")\n\
    except Exception as e:\n\
        print(f\"GPU test failed: {e}\")\n\
        print(\"Service will handle this gracefully\")\n\
else:\n\
    print(\"No GPU detected\")\n\
"\n\
\n\
mkdir -p temp_outputs outputs /models/cache pred_rttms\n\
chmod -R 777 temp_outputs outputs /models/cache pred_rttms 2>/dev/null || true\n\
\n\
echo "Starting main application..."\n\
exec python -u main.py' > /app/start.sh

RUN chmod +x /app/start.sh

EXPOSE 8000
CMD ["/app/start.sh"]
