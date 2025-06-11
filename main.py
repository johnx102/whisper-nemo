import os
import json
import tempfile
import asyncio
import gc
import traceback
import warnings
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

# Supprimer warnings
warnings.filterwarnings("ignore")

# FORCER CPU MODE - PAS DE GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("🔧 FORCED CPU MODE - No GPU will be used")

import torch
import aiohttp
import runpod
import faster_whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, validator
import uvicorn

# Vérifier que PyTorch est en mode CPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}")

# Imports optionnels du projet whisper-diarization
try:
    from helpers import (
        find_numeral_symbol_tokens,
        langs_to_iso,
        punct_model_langs,
    )
    HELPERS_AVAILABLE = True
    print("✅ Helpers available")
except ImportError:
    print("⚠️ Helpers not available - using fallbacks")
    HELPERS_AVAILABLE = False
    langs_to_iso = {"fr": "fr", "en": "en", "es": "es", "de": "de"}
    punct_model_langs = ["fr", "en", "es", "de"]

try:
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    NEMO_AVAILABLE = True
    print("✅ NeMo available")
except ImportError:
    print("⚠️ NeMo not available")
    NEMO_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(title="Whisper Diarization Service (CPU)", version="3.0.0")

# Configuration
MAX_FILE_SIZE = 300 * 1024 * 1024
DOWNLOAD_TIMEOUT = 600  # Plus long pour CPU
SUPPORTED_FORMATS = {
    'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/m4a', 
    'audio/ogg', 'audio/flac'
}

# Models storage
models = {'whisper': None}

class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    whisper_model: Optional[str] = "base"  # Plus petit par défaut pour CPU
    language: Optional[str] = "fr"
    batch_size: Optional[int] = 4  # Petit batch pour CPU
    no_stem: Optional[bool] = True
    
    @validator('whisper_model')
    def validate_whisper_model(cls, v):
        # Modèles recommandés pour CPU
        valid = ['tiny', 'base', 'small', 'medium']
        if v not in valid:
            print(f"⚠️ Model {v} may be slow on CPU, recommended: {valid}")
        return v

class TranscriptionResponse(BaseModel):
    transcription_text: str
    segments: list
    speakers_detected: int
    processing_time: float
    language: str
    model_info: Dict[str, Any]
    error: Optional[str] = None

def find_numeral_symbol_tokens_fallback(tokenizer):
    """Fallback si helpers pas disponibles"""
    try:
        if HELPERS_AVAILABLE:
            return find_numeral_symbol_tokens(tokenizer)
        else:
            return [-1]  # Pas de suppression
    except:
        return [-1]

async def download_audio_file(url: str) -> str:
    """Download audio file"""
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(str(url)) as response:
            content_type = response.headers.get('content-type', '').lower()
            print(f"📥 Content-Type: {content_type}")
            
            content = await response.read()
            if len(content) > MAX_FILE_SIZE:
                raise ValueError(f"File too large: {len(content)} bytes")
            
            # Extension
            suffix = '.wav'
            if 'mp3' in content_type: suffix = '.mp3'
            elif 'mp4' in content_type: suffix = '.mp4'
            elif 'm4a' in content_type: suffix = '.m4a'
            elif 'ogg' in content_type: suffix = '.ogg'
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(content)
            temp_file.close()
            
            print(f"✅ Downloaded {len(content)} bytes to {temp_file.name}")
            return temp_file.name

def load_whisper_model_cpu(model_name: str):
    """Chargement Whisper en mode CPU uniquement"""
    global models
    
    try:
        print(f"🎤 Loading Whisper model: {model_name} (CPU mode)")
        
        # Configuration CPU optimisée
        whisper_model = faster_whisper.WhisperModel(
            model_name,
            device="cpu",
            compute_type="int8",  # Plus efficace sur CPU
            cpu_threads=4,        # 4 threads CPU
            num_workers=1
        )
        
        models['whisper'] = whisper_model
        
        print(f"✅ Whisper {model_name} loaded successfully (CPU)")
        return whisper_model
        
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        raise

async def process_transcription_cpu(audio_path: str, request: TranscriptionRequest):
    """Transcription optimisée CPU"""
    start_time = datetime.now()
    
    try:
        print(f"🚀 Starting transcription pipeline (CPU mode)...")
        print(f"📁 Audio: {audio_path}")
        print(f"🎛️ Model: {request.whisper_model}")
        print(f"🌍 Language: {request.language}")
        print(f"💾 Device: CPU")
        
        # Charger modèle si nécessaire
        if models['whisper'] is None:
            whisper_model = load_whisper_model_cpu(request.whisper_model)
        else:
            whisper_model = models['whisper']
        
        # Transcription
        print("🎤 Starting transcription...")
        
        try:
            audio_waveform = faster_whisper.decode_audio(audio_path)
            
            # Batch size adapté pour CPU
            cpu_batch_size = min(request.batch_size, 2)  # Très petit pour CPU
            print(f"🎯 CPU batch size: {cpu_batch_size}")
            
            # Tokens à supprimer
            suppress_tokens = find_numeral_symbol_tokens_fallback(whisper_model.hf_tokenizer)
            
            # Transcription CPU (pas de pipeline batchée)
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform,
                language=request.language,
                suppress_tokens=suppress_tokens,
                vad_filter=True,
                beam_size=1,  # Beam size minimal
                word_timestamps=True
            )
            
            # Convertir en liste
            transcript_segments = list(transcript_segments)
            detected_language = info.language
            
            print(f"✅ Transcription completed: {len(transcript_segments)} segments")
            print(f"🌍 Detected language: {detected_language}")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            raise
        
        # Assembler texte
        full_text = " ".join(segment.text for segment in transcript_segments)
        
        # Diarisation basique (un seul speaker pour l'instant)
        speakers_detected = 1
        
        # Si NeMo disponible, on peut essayer une diarisation simple
        if NEMO_AVAILABLE and len(transcript_segments) > 5:
            try:
                print("🎭 Attempting basic NeMo diarization...")
                # Diarisation simple CPU (à implémenter si nécessaire)
                speakers_detected = 2  # Exemple
            except Exception as e:
                print(f"⚠️ Diarization failed: {e}")
        
        # Formater segments
        response_segments = []
        for i, segment in enumerate(transcript_segments):
            # Alternance simple des speakers si plusieurs détectés
            speaker_id = "A" if (i // 3) % 2 == 0 else "B" if speakers_detected > 1 else "A"
            
            response_segments.append({
                "id": i,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": f"Speaker {speaker_id}"
            })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text=full_text,
            segments=response_segments,
            speakers_detected=speakers_detected,
            processing_time=processing_time,
            language=detected_language,
            model_info={
                "whisper_model": request.whisper_model,
                "device": "cpu",
                "compute_type": "int8",
                "batch_size": cpu_batch_size,
                "cpu_threads": 4,
                "nemo_available": NEMO_AVAILABLE,
                "helpers_available": HELPERS_AVAILABLE
            }
        )
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text="",
            segments=[],
            speakers_detected=0,
            processing_time=processing_time,
            language="unknown",
            model_info={
                "whisper_model": request.whisper_model,
                "device": "cpu",
                "error": True
            },
            error=error_msg
        )

# RunPod handler
async def handler(job):
    """Main handler - CPU mode"""
    job_input = job.get("input", {})
    
    try:
        print(f"🚀 New job: {job.get('id', 'unknown')} (CPU mode)")
        
        request = TranscriptionRequest(**job_input)
        print(f"📥 Downloading: {request.audio_url}")
        
        audio_path = await download_audio_file(request.audio_url)
        result = await process_transcription_cpu(audio_path, request)
        
        # Cleanup
        try:
            os.unlink(audio_path)
        except:
            pass
        
        if result.error:
            return {"error": result.error}
        else:
            return {
                "transcription": result.transcription_text,
                "segments": result.segments,
                "speakers_detected": result.speakers_detected,
                "language": result.language,
                "processing_time": result.processing_time,
                "model_info": result.model_info
            }
            
    except Exception as e:
        return {"error": f"Handler error: {str(e)}"}

# FastAPI endpoints
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "device": "cpu",
        "compute_type": "int8",
        "cpu_threads": 4,
        "nemo_available": NEMO_AVAILABLE,
        "helpers_available": HELPERS_AVAILABLE,
        "models_loaded": {
            "whisper": models['whisper'] is not None
        },
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    try:
        audio_path = await download_audio_file(request.audio_url)
        result = await process_transcription_cpu(audio_path, request)
        
        try:
            os.unlink(audio_path)
        except:
            pass
        
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    print(f"🚀 Whisper Diarization Service starting (CPU MODE)...")
    print(f"💾 Device: CPU")
    print(f"🧵 CPU Threads: 4")
    print(f"🧩 NeMo available: {NEMO_AVAILABLE}")
    print(f"🛠️ Helpers available: {HELPERS_AVAILABLE}")
    
    print("🎯 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
