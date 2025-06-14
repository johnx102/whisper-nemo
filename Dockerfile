# Image de base avec CUDA support optimisée pour serverless
FROM runpod/pytorch:2.8.0-py3.11-cuda12.8.1-cudnn-devel-ubuntu22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_VISIBLE_DEVICES=0
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Mise à jour et installation des dépendances système
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    ffmpeg \
    git \
    wget \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Mise à jour pip
RUN python3 -m pip install --upgrade pip

# Créer répertoire de travail
WORKDIR /app

# Copier requirements
COPY requirements.txt .

# Installation des dépendances Python avec optimisations
RUN pip install -r requirements.txt

# Copier le code source
COPY main.py .

# Créer répertoire pour les modèles
RUN mkdir -p /app/models

# Variables d'environnement pour les modèles
ENV TRANSFORMERS_CACHE=/app/models
ENV HF_HOME=/app/models
ENV TORCH_HOME=/app/models

# Port d'écoute (adapté pour serverless)
EXPOSE 8080

# Commande de démarrage
CMD ["python3", "main.py"]
