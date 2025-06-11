# Ajout au début du main.py pour gérer cuDNN 8 vs 9

import os
import warnings
warnings.filterwarnings("ignore")

def detect_cudnn_compatibility():
    """Détecte la compatibilité cuDNN et force CPU si nécessaire"""
    try:
        import torch
        
        if not torch.cuda.is_available():
            print("CUDA not available - using CPU")
            return "cpu", "int8"
        
        # Vérifier version cuDNN
        cudnn_version = torch.backends.cudnn.version()
        print(f"Detected cuDNN version: {cudnn_version}")
        
        # Si cuDNN 9+, forcer CPU (incompatible avec notre stack)
        if cudnn_version >= 9000:
            print("WARNING: cuDNN 9+ detected - incompatible with current PyTorch")
            print("Forcing CPU mode for stability")
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            return "cpu", "int8"
        
        # cuDNN 8 OK
        elif cudnn_version >= 8000:
            print(f"cuDNN 8 detected ({cudnn_version}) - GPU mode OK")
            
            # Configuration conservative pour cuDNN 8
            torch.backends.cudnn.benchmark = False
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
            torch.backends.cudnn.deterministic = True
            
            # Test rapide GPU
            try:
                x = torch.randn(10, 10).cuda()
                y = torch.mm(x, x)
                del x, y
                torch.cuda.empty_cache()
                print("GPU test passed")
                return "cuda", "float16"  # float16 OK avec cuDNN 8
            except Exception as e:
                print(f"GPU test failed: {e}")
                os.environ["CUDA_VISIBLE_DEVICES"] = ""
                return "cpu", "int8"
        
        else:
            print(f"Unknown cuDNN version: {cudnn_version}")
            return "cpu", "int8"
            
    except Exception as e:
        print(f"cuDNN detection failed: {e}")
        return "cpu", "int8"

# Détection automatique au démarrage
DEVICE, COMPUTE_TYPE = detect_cudnn_compatibility()
print(f"Selected device: {DEVICE} with compute type: {COMPUTE_TYPE}")

# Le reste du main.py...
import torch
import aiohttp
import runpod
import faster_whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, validator
import uvicorn
import json
import tempfile
import asyncio
import gc
import traceback
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse

# Initialize FastAPI
app = FastAPI(title="Whisper Diarization Service", version="2.1.0")

# Configuration
MAX_FILE_SIZE = 300 * 1024 * 1024
DOWNLOAD_TIMEOUT = 300
SUPPORTED_FORMATS = {
    'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/m4a', 
    'audio/ogg', 'audio/flac'
}

# Models storage
models = {'whisper': None, 'whisper_pipeline': None}

class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    whisper_model: Optional[str] = "large-v2"
    language: Optional[str] = "fr"
    batch_size: Optional[int] = 8
    no_stem: Optional[bool] = True
    
    @validator('whisper_model')
    def validate_whisper_model(cls, v):
        valid = ['tiny', 'base', 'small', 'medium', 'large', 'large-v1', 'large-v2', 'large-v3']
        if v not in valid:
            raise ValueError(f'Model must be one of: {", ".join(valid)}')
        return v

class TranscriptionResponse(BaseModel):
    transcription_text: str
    segments: list
    speakers_detected: int
    processing_time: float
    language: str
    model_info: Dict[str, Any]
    error: Optional[str] = None

def cleanup_gpu_memory():
    """Nettoyage GPU sécurisé"""
    try:
        gc.collect()
        if DEVICE == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            print(f"GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
    except Exception as e:
        print(f"GPU cleanup error: {e}")

async def download_audio_file(url: str) -> str:
    """Download audio file"""
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        async with session.get(str(url)) as response:
            content_type = response.headers.get('content-type', '').lower()
            print(f"Content-Type: {content_type}")
            
            content = await response.read()
            if len(content) > MAX_FILE_SIZE:
                raise ValueError(f"File too large: {len(content)} bytes")
            
            # Déterminer extension
            suffix = '.wav'
            if 'mp3' in content_type: suffix = '.mp3'
            elif 'mp4' in content_type: suffix = '.mp4'
            elif 'm4a' in content_type: suffix = '.m4a'
            elif 'ogg' in content_type: suffix = '.ogg'
            
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
            temp_file.write(content)
            temp_file.close()
            
            print(f"Downloaded {len(content)} bytes to {temp_file.name}")
            return temp_file.name

def load_whisper_model_safe(model_name: str):
    """Chargement Whisper compatible cuDNN 8"""
    global models
    
    try:
        print(f"Loading Whisper model: {model_name} on {DEVICE}")
        cleanup_gpu_memory()
        
        # Configuration adaptée au device détecté
        if DEVICE == "cuda":
            # cuDNN 8 compatible settings
            device_to_use = "cuda"
            compute_to_use = COMPUTE_TYPE
            
            # Test GPU avant chargement
            try:
                x = torch.randn(10, 10).cuda()
                y = torch.mm(x, x)
                del x, y
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"GPU test failed, switching to CPU: {e}")
                device_to_use = "cpu"
                compute_to_use = "int8"
        else:
            device_to_use = "cpu"
            compute_to_use = "int8"
        
        # Chargement modèle
        whisper_model = faster_whisper.WhisperModel(
            model_name,
            device=device_to_use,
            compute_type=compute_to_use,
            cpu_threads=4 if device_to_use == "cpu" else 2,
            num_workers=1
        )
        
        # Pipeline batchée seulement si GPU stable
        whisper_pipeline = None
        if device_to_use == "cuda":
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory > 20:
                    whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
                    print(f"Batched pipeline enabled (GPU: {gpu_memory:.1f}GB)")
            except Exception as e:
                print(f"Batched pipeline failed: {e}")
        
        models['whisper'] = whisper_model
        models['whisper_pipeline'] = whisper_pipeline
        
        print(f"Whisper {model_name} loaded successfully on {device_to_use}")
        cleanup_gpu_memory()
        
        return whisper_model, whisper_pipeline
        
    except Exception as e:
        print(f"Error loading Whisper: {e}")
        cleanup_gpu_memory()
        
        # Fallback CPU
        try:
            print("Attempting CPU fallback...")
            whisper_model = faster_whisper.WhisperModel(
                model_name,
                device="cpu",
                compute_type="int8",
                cpu_threads=4
            )
            models['whisper'] = whisper_model
            models['whisper_pipeline'] = None
            print(f"Whisper {model_name} loaded on CPU fallback")
            return whisper_model, None
        except Exception as e2:
            print(f"CPU fallback failed: {e2}")
            raise

async def process_transcription_safe(audio_path: str, request: TranscriptionRequest):
    """Transcription cuDNN-safe"""
    start_time = datetime.now()
    
    try:
        print(f"Starting transcription pipeline...")
        print(f"Audio: {audio_path}")
        print(f"Model: {request.whisper_model}")
        print(f"Device: {DEVICE} ({COMPUTE_TYPE})")
        
        # Charger modèle si nécessaire
        if models['whisper'] is None:
            whisper_model, whisper_pipeline = load_whisper_model_safe(request.whisper_model)
        else:
            whisper_model = models['whisper']
            whisper_pipeline = models['whisper_pipeline']
        
        # Transcription
        print("Starting transcription...")
        
        try:
            audio_waveform = faster_whisper.decode_audio(audio_path)
            
            # Batch size adaptatif selon device
            if DEVICE == "cuda":
                # Conservateur pour cuDNN 8
                optimal_batch = min(request.batch_size, 8)
            else:
                optimal_batch = min(request.batch_size, 4)  # CPU
            
            print(f"Using batch size: {optimal_batch}")
            
            # Transcription avec gestion d'erreur cuDNN
            try:
                if whisper_pipeline and DEVICE == "cuda":
                    transcript_segments, info = whisper_pipeline.transcribe(
                        audio_waveform,
                        language=request.language,
                        batch_size=optimal_batch
                    )
                    print("Used batched pipeline")
                else:
                    transcript_segments, info = whisper_model.transcribe(
                        audio_waveform,
                        language=request.language,
                        vad_filter=True,
                        beam_size=1
                    )
                    print("Used standard model")
                
                # Convertir en liste
                transcript_segments = list(transcript_segments)
                detected_language = info.language
                
                print(f"Transcription completed: {len(transcript_segments)} segments")
                print(f"Detected language: {detected_language}")
                
                cleanup_gpu_memory()
                
            except Exception as e:
                error_str = str(e)
                if "cudnn" in error_str.lower() or "cudnnCreateTensorDescriptor" in error_str:
                    print(f"cuDNN error detected: {e}")
                    print("Switching to CPU mode and retrying...")
                    
                    # Forcer CPU pour le reste
                    os.environ["CUDA_VISIBLE_DEVICES"] = ""
                    
                    # Recharger modèle en CPU
                    whisper_model_cpu = faster_whisper.WhisperModel(
                        request.whisper_model,
                        device="cpu",
                        compute_type="int8",
                        cpu_threads=4
                    )
                    
                    transcript_segments, info = whisper_model_cpu.transcribe(
                        audio_waveform,
                        language=request.language,
                        vad_filter=True,
                        beam_size=1
                    )
                    
                    transcript_segments = list(transcript_segments)
                    detected_language = info.language
                    
                    print(f"CPU fallback completed: {len(transcript_segments)} segments")
                else:
                    raise
                
        except Exception as e:
            print(f"Audio processing failed: {e}")
            raise
        
        # Assembler texte
        full_text = "".join(segment.text for segment in transcript_segments)
        
        # Formater segments
        response_segments = []
        for i, segment in enumerate(transcript_segments):
            response_segments.append({
                "id": i,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": "Speaker A"
            })
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text=full_text,
            segments=response_segments,
            speakers_detected=1,
            processing_time=processing_time,
            language=detected_language,
            model_info={
                "whisper_model": request.whisper_model,
                "device": DEVICE,
                "compute_type": COMPUTE_TYPE,
                "batch_size": optimal_batch if 'optimal_batch' in locals() else request.batch_size,
                "cudnn_safe_mode": True
            }
        )
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        print(traceback.format_exc())
        
        cleanup_gpu_memory()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text="",
            segments=[],
            speakers_detected=0,
            processing_time=processing_time,
            language="unknown",
            model_info={
                "whisper_model": request.whisper_model,
                "device": DEVICE,
                "error": True
            },
            error=error_msg
        )

# RunPod handler
async def handler(job):
    """Main handler"""
    job_input = job.get("input", {})
    
    try:
        print(f"New job: {job.get('id', 'unknown')}")
        
        request = TranscriptionRequest(**job_input)
        print(f"Downloading: {request.audio_url}")
        
        audio_path = await download_audio_file(request.audio_url)
        result = await process_transcription_safe(audio_path, request)
        
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
    cudnn_info = {}
    if torch.cuda.is_available():
        try:
            cudnn_info = {
                "cudnn_enabled": torch.backends.cudnn.enabled,
                "cudnn_version": torch.backends.cudnn.version(),
                "gpu_name": torch.cuda.get_device_name(),
            }
        except:
            cudnn_info = {"cudnn_error": "Could not get cuDNN info"}
    
    return {
        "status": "healthy",
        "device": DEVICE,
        "compute_type": COMPUTE_TYPE,
        "cudnn_info": cudnn_info,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    try:
        audio_path = await download_audio_file(request.audio_url)
        result = await process_transcription_safe(audio_path, request)
        
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
    print(f"Whisper Diarization Service starting...")
    print(f"Device: {DEVICE}")
    print(f"Compute type: {COMPUTE_TYPE}")
    
    if torch.cuda.is_available():
        try:
            print(f"cuDNN version: {torch.backends.cudnn.version()}")
        except:
            print("cuDNN version: unknown")
    
    print("Starting RunPod handler...")
    runpod.serverless.start({"handler": handler})
