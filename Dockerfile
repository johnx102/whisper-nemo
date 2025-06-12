# Garder CUDA 12.3.2 cuDNN 9 existant (fonctionne)
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

# Métadonnées
LABEL maintainer="whisper-diarization-gpu"
LABEL version="6.1.0"

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1

# Configuration CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Installation système EXACTE selon le projet
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-dev \
    python3-pip \
    git \
    wget \
    curl \
    ffmpeg \
    cython3 \
    libsndfile1 \
    libsox-dev \
    sox \
    build-essential \
    pkg-config \
    libssl-dev \
    libffi-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.10 /usr/bin/python

RUN python -m pip install --upgrade pip setuptools wheel

# WORKAROUND cuDNN: Installer PyTorch avec une version compatible
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# 1. Cloner le repo original et installer manuellement pour éviter conflits PyTorch
RUN git clone https://github.com/MahmoudAshraf97/whisper-diarization.git /tmp/whisper-diarization

# 2. Installer les dépendances une par une pour éviter les conflits
RUN pip install --no-cache-dir \
    faster-whisper>=1.1.0 \
    nltk \
    wget \
    omegaconf \
    hydra-core

# 3. Installer NeMo compatible avec PyTorch 2.0
RUN pip install --no-cache-dir \
    nemo-toolkit[asr]==1.20.0 || \
    echo "Warning: NeMo 1.20.0 failed, trying fallback"

# 4. Installer les repos Git spécifiques
RUN pip install --no-cache-dir \
    git+https://github.com/MahmoudAshraf97/demucs.git \
    git+https://github.com/oliverguhr/deepmultilingualpunctuation.git \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git

# 5. Web framework et API
RUN pip install --no-cache-dir \
    paramiko \
    cryptography \
    runpod>=1.6.2 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0

# Test que tout fonctionne avec PyTorch 2.0
RUN python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
RUN python -c "import faster_whisper; print('faster-whisper: OK')"
RUN python -c "import runpod; print('runpod: OK')"
RUN python -c "import nemo; print('NeMo version:', nemo.__version__)" || echo "NeMo test failed"
RUN python -c "from nemo.collections.asr.models.msdd_models import NeuralDiarizer; print('✅ NeuralDiarizer imported!')" || echo "MSDD import failed"

# Copier les helpers du projet whisper-diarization dans notre app
WORKDIR /app
RUN cp /tmp/whisper-diarization/helpers.py /app/
RUN cp /tmp/whisper-diarization/diarize.py /app/ || echo "diarize.py copied"

# Copier notre code
COPY . .

# Créer les dossiers nécessaires
RUN mkdir -p /tmp/whisper-cache /tmp/audio-temp /app/temp_outputs

RUN chmod +x main.py

ENV WHISPER_CACHE_DIR=/tmp/whisper-cache
ENV TEMP_AUDIO_DIR=/tmp/audio-temp

HEALTHCHECK --interval=30s --timeout=15s --start-period=240s --retries=3 \
    CMD python -c "import torch; import runpod; import nemo; print('Health: OK')" || exit 1

EXPOSE 8000
CMD ["python", "main.py"]
