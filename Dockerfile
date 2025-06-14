# Retour à l'image qui fonctionnait
FROM nvidia/cuda:12.3.2-cudnn9-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1

# Configuration CUDA
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Installation système de base
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

# ÉTAPE 1: Forcer NumPy 1.x et versions compatibles
RUN pip install --no-cache-dir \
    "numpy>=1.21.0,<2.0.0" \
    "huggingface_hub==0.20.3" \
    "transformers==4.37.2" \
    "tokenizers==0.15.2"

# ÉTAPE 2: PyTorch compatible avec cuDNN 9.0
RUN pip install --no-cache-dir \
    torch==2.0.1+cu118 \
    torchaudio==2.0.2+cu118 \
    torchvision==0.15.2+cu118 \
    --index-url https://download.pytorch.org/whl/cu118

# ÉTAPE 3: Dependencies de base
RUN pip install --no-cache-dir \
    faster-whisper>=1.1.0 \
    omegaconf \
    hydra-core \
    nltk \
    wget

# ÉTAPE 4: Essayer NeMo version stable
RUN pip install --no-cache-dir nemo-toolkit[asr]==2.0.0
  
# ÉTAPE 5: Repos Git nécessaires (installation conditionnelle)
RUN pip install --no-cache-dir \
    git+https://github.com/MahmoudAshraf97/demucs.git || echo "Demucs failed" && \
    pip install --no-cache-dir \
    git+https://github.com/oliverguhr/deepmultilingualpunctuation.git || echo "Punctuation failed" && \
    pip install --no-cache-dir \
    git+https://github.com/MahmoudAshraf97/ctc-forced-aligner.git || echo "CTC aligner failed"

# ÉTAPE 6: Web framework
RUN pip install --no-cache-dir \
    paramiko \
    cryptography \
    runpod>=1.6.2 \
    fastapi>=0.104.0 \
    uvicorn[standard]>=0.24.0 \
    aiohttp>=3.9.0 \
    pydantic>=2.5.0

# Télécharger juste helpers.py du projet original
RUN wget -O /tmp/helpers.py https://raw.githubusercontent.com/MahmoudAshraf97/whisper-diarization/main/helpers.py || \
    echo "Could not download helpers.py"

WORKDIR /app

# Copier notre code
COPY . .

# Copier helpers.py si disponible
RUN cp /tmp/helpers.py /app/ || echo "No helpers.py to copy"

# Créer les dossiers nécessaires
RUN mkdir -p /tmp/whisper-cache /tmp/audio-temp /app/temp_outputs

RUN chmod +x main.py

ENV WHISPER_CACHE_DIR=/tmp/whisper-cache
ENV TEMP_AUDIO_DIR=/tmp/audio-temp

# Test final simple
RUN python -c "import torch; print('PyTorch:', torch.__version__, 'CUDA:', torch.cuda.is_available())"
RUN python -c "import faster_whisper; print('faster-whisper: OK')"
RUN python -c "import runpod; print('runpod: OK')"

HEALTHCHECK --interval=30s --timeout=15s --start-period=240s --retries=3 \
    CMD python -c "import torch; import runpod; print('Health: OK')" || exit 1

EXPOSE 8000
CMD ["python", "main.py"]
