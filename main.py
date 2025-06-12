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

# FORCER GPU MODE
os.environ["TOKENIZERS_PARALLELISM"] = "false"

print("🚀 GPU MODE ENABLED")

import torch
import aiohttp
import runpod
import faster_whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
import uvicorn

# Vérifier que PyTorch fonctionne avec GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

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
    print("🔍 Testing PyAnnote import...")
    from pyannote.audio import Pipeline
    PYANNOTE_AVAILABLE = True
    print("✅ PyAnnote available for advanced diarization")
except ImportError as e:
    print(f"⚠️ PyAnnote not available: {e}")
    PYANNOTE_AVAILABLE = False

try:
    print("🔍 Testing NeMo import...")
    import nemo
    print(f"📦 NeMo version: {nemo.__version__}")
    
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    print("✅ NeuralDiarizer imported successfully")
    
    NEMO_AVAILABLE = True
    print("✅ NeMo available (backup option)")
except ImportError as e:
    print(f"❌ NeMo import failed: {e}")
    print(f"❌ Error type: {type(e).__name__}")
    NEMO_AVAILABLE = False
except Exception as e:
    print(f"❌ NeMo unexpected error: {e}")
    print(f"❌ Error type: {type(e).__name__}")
    NEMO_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(title="Whisper Diarization Service (GPU)", version="4.0.0")

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
DOWNLOAD_TIMEOUT = 300
SUPPORTED_FORMATS = {
    'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/m4a', 
    'audio/ogg', 'audio/flac', 'audio/webm'
}

# Models storage avec metadata
models = {'whisper': None, 'whisper_model_name': None, 'diarizer': None}

class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    whisper_model: Optional[str] = "medium"  # Meilleur modèle par défaut avec GPU
    language: Optional[str] = "fr"
    batch_size: Optional[int] = 16  # Plus gros batch avec GPU
    no_stem: Optional[bool] = True
    enable_diarization: Optional[bool] = True
    min_speakers: Optional[int] = 1
    max_speakers: Optional[int] = 8
    
    @field_validator('whisper_model')
    @classmethod
    def validate_whisper_model(cls, v):
        valid = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        if v not in valid:
            raise ValueError(f"Invalid model. Choose from: {valid}")
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
            # Tokens numériques classiques à supprimer
            return [50362, 50363, 50364, 50365, 50366, 50367, 50368, 50369]
    except Exception as e:
        print(f"Warning: Could not get numeral tokens: {e}")
        return []

async def download_audio_file(url: str) -> str:
    """Download audio file avec retry"""
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(str(url)) as response:
                if response.status != 200:
                    raise ValueError(f"HTTP {response.status}: {response.reason}")
                
                content_type = response.headers.get('content-type', '').lower()
                print(f"📥 Content-Type: {content_type}")
                
                content = await response.read()
                if len(content) > MAX_FILE_SIZE:
                    raise ValueError(f"File too large: {len(content)} bytes (max: {MAX_FILE_SIZE})")
                
                # Déterminer l'extension
                suffix = '.wav'  # défaut
                if 'mp3' in content_type or 'mpeg' in content_type: 
                    suffix = '.mp3'
                elif 'mp4' in content_type: 
                    suffix = '.mp4'
                elif 'm4a' in content_type: 
                    suffix = '.m4a'
                elif 'ogg' in content_type: 
                    suffix = '.ogg'
                elif 'flac' in content_type: 
                    suffix = '.flac'
                elif 'webm' in content_type: 
                    suffix = '.webm'
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(content)
                temp_file.close()
                
                print(f"✅ Downloaded {len(content)} bytes to {temp_file.name}")
                return temp_file.name
                
        except asyncio.TimeoutError:
            raise ValueError("Download timeout")
        except Exception as e:
            raise ValueError(f"Download failed: {str(e)}")

def load_whisper_model_gpu(model_name: str):
    """Chargement Whisper optimisé GPU"""
    global models
    
    try:
        print(f"🎤 Loading Whisper model: {model_name} (GPU mode)")
        
        # Vérifier la mémoire GPU disponible
        if torch.cuda.is_available():
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            print(f"💾 GPU Memory available: {gpu_memory:.1f} GB")
            
            # Choisir le compute_type selon la mémoire
            if gpu_memory >= 8:
                compute_type = "float16"
            else:
                compute_type = "int8"
        else:
            print("⚠️ No GPU available, falling back to CPU")
            compute_type = "int8"
        
        # Configuration GPU optimisée
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        whisper_model = faster_whisper.WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=4 if device == "cuda" else 1
        )
        
        models['whisper'] = whisper_model
        models['whisper_model_name'] = model_name  # Stocker séparément
        
        print(f"✅ Whisper {model_name} loaded successfully ({device}, {compute_type})")
        return whisper_model
        
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        print(traceback.format_exc())
        raise

def basic_speaker_diarization(segments, max_speakers=2):
    """Diarisation basique basée sur les pauses - UTILISE max_speakers"""
    if len(segments) <= 2:
        return ["A"] * len(segments), 1
    
    # Utiliser le paramètre max_speakers
    num_speakers = min(max_speakers, max(2, len(segments) // 5))
    
    # Analyser les pauses entre segments
    pauses = []
    for i in range(1, len(segments)):
        pause = segments[i].start - segments[i-1].end
        pauses.append(pause)
    
    # Seuil adaptatif pour changer de speaker
    avg_pause = sum(pauses) / len(pauses) if pauses else 1.0
    speaker_change_threshold = max(1.0, avg_pause * 1.2)
    
    # Attribution des speakers
    current_speaker = 0
    speaker_labels = []
    
    for i, segment in enumerate(segments):
        if i > 0:
            pause = segment.start - segments[i-1].end
            if pause > speaker_change_threshold:
                current_speaker = (current_speaker + 1) % num_speakers
        
        speaker_labels.append(chr(65 + current_speaker))  # A, B, C...
    
    detected_speakers = min(num_speakers, len(set(speaker_labels)))
    return speaker_labels, detected_speakers

async def process_transcription_gpu(audio_path: str, request: TranscriptionRequest):
    """Transcription avec GPU et diarisation optionnelle"""
    start_time = datetime.now()
    
    try:
        print(f"🚀 Starting transcription pipeline (GPU mode)...")
        print(f"📁 Audio: {audio_path}")
        print(f"🎛️ Model: {request.whisper_model}")
        print(f"🌍 Language: {request.language}")
        print(f"💾 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")
        print(f"🎭 Diarization: {'Enabled' if request.enable_diarization else 'Disabled'}")
        
        # Charger modèle Whisper si nécessaire - CORRECTION ICI
        current_model_name = models.get('whisper_model_name')
        if models['whisper'] is None or current_model_name != request.whisper_model:
            whisper_model = load_whisper_model_gpu(request.whisper_model)
        else:
            whisper_model = models['whisper']
        
        # Nettoyer la mémoire GPU avant transcription
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Transcription
        print("🎤 Starting transcription...")
        
        try:
            # Charger et préparer l'audio
            audio_waveform = faster_whisper.decode_audio(audio_path)
            print(f"🎵 Audio duration: {len(audio_waveform) / 16000:.2f} seconds")
            
            # Tokens à supprimer pour améliorer la qualité
            suppress_tokens = find_numeral_symbol_tokens_fallback(whisper_model.hf_tokenizer)
            
            # Configuration de transcription optimisée
            transcribe_options = {
                "language": request.language,
                "suppress_tokens": suppress_tokens,
                "vad_filter": True,
                "vad_parameters": dict(
                    min_silence_duration_ms=500,
                    max_speech_duration_s=30
                ),
                "beam_size": 5 if torch.cuda.is_available() else 1,
                "word_timestamps": True,
                "condition_on_previous_text": False,
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "log_prob_threshold": -1.0,
                "no_speech_threshold": 0.6,
            }
            
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform, 
                **transcribe_options
            )
            
            # Convertir en liste
            transcript_segments = list(transcript_segments)
            detected_language = info.language
            
            print(f"✅ Transcription completed: {len(transcript_segments)} segments")
            print(f"🌍 Detected language: {detected_language}")
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            print(traceback.format_exc())
            raise
        
        # Assembler le texte complet
        full_text = " ".join(segment.text.strip() for segment in transcript_segments if segment.text.strip())
        
        # Diarisation si demandée
        speakers_detected = 1
        speaker_labels = ["A"] * len(transcript_segments)
        
        if request.enable_diarization and len(transcript_segments) > 2:
            try:
                print("🎭 Starting speaker diarization...")
                
                if PYANNOTE_AVAILABLE:
                    print(f"🎯 Using PyAnnote with {request.min_speakers}-{request.max_speakers} speakers")
                    
                    # Charger le pipeline PyAnnote
                    if models.get('pyannote_pipeline') is None:
                        print("📥 Loading PyAnnote pipeline...")
                        models['pyannote_pipeline'] = Pipeline.from_pretrained(
                            "pyannote/speaker-diarization-3.1",
                            use_auth_token=os.getenv("HF_TOKEN", "hf_...")  # Token HuggingFace nécessaire
                        )
                        print("✅ PyAnnote pipeline loaded")
                    
                    # Diarisation avec PyAnnote
                    pipeline = models['pyannote_pipeline']
                    diarization = pipeline(audio_path, num_speakers=request.max_speakers)
                    
                    # Mapper les résultats PyAnnote aux segments Whisper
                    speaker_labels = []
                    speakers_detected = len(diarization.labels())
                    print(f"🎤 PyAnnote detected {speakers_detected} speakers: {list(diarization.labels())}")
                    
                    for segment in transcript_segments:
                        segment_start = segment.start
                        segment_end = segment.end
                        segment_mid = (segment_start + segment_end) / 2
                        
                        # Trouver le speaker actif au milieu du segment
                        assigned_speaker = "A"  # défaut
                        for turn, _, speaker in diarization.itertracks(yield_label=True):
                            if turn.start <= segment_mid <= turn.end:
                                # Mapper speaker PyAnnote vers lettre
                                speaker_labels_list = sorted(list(diarization.labels()))
                                speaker_index = speaker_labels_list.index(speaker)
                                assigned_speaker = chr(65 + speaker_index)  # A, B, C...
                                break
                        
                        speaker_labels.append(assigned_speaker)
                    
                    print(f"✅ PyAnnote diarization completed: {speakers_detected} speakers")
                    
                elif NEMO_AVAILABLE:
                    print(f"🎯 Using NeMo MSDD with {request.min_speakers}-{request.max_speakers} speakers")
                    
                    try:
                        # Configuration NeMo diarisation simplifiée
                        from omegaconf import OmegaConf
                        
                        # Créer config NeMo complète
                        cfg = OmegaConf.create({
                            'diarizer': {
                                'manifest_filepath': None,
                                'out_dir': '/tmp/nemo_outputs',
                                'oracle_vad': False,
                                'clustering': {
                                    'parameters': {
                                        'oracle_num_speakers': False,
                                        'max_num_speakers': request.max_speakers,
                                        'enhanced_count_thres': 0.8,
                                    }
                                },
                                'msdd_model': {
                                    'model_path': 'diar_msdd_telephonic',
                                    'parameters': {
                                        'use_speaker_model_from_ckpt': True,
                                        'infer_batch_size': 25,
                                        'sigmoid_threshold': [0.7],
                                        'seq_eval_mode': False,
                                        'split_infer': True,
                                        'diar_window_length': 50,
                                        'overlap_infer_spk_limit': 5,
                                    }
                                },
                                'vad': {
                                    'model_path': 'vad_multilingual_marblenet',
                                    'external_vad_manifest': None,
                                    'parameters': {
                                        'onset': 0.8,
                                        'offset': 0.6,
                                        'pad_onset': 0.05,
                                        'pad_offset': -0.05,
                                        'min_duration_on': 0.2,
                                        'min_duration_off': 0.2
                                    }
                                },
                                'speaker_embeddings': {
                                    'model_path': 'titanet_large',
                                    'parameters': {
                                        'window_length_in_sec': 1.5,
                                        'shift_length_in_sec': 0.75,
                                        'multiscale_weights': [1, 1, 1, 1, 1],
                                        'multiscale_args': {"scale_dict": "{1: [1.5], 2: [1.5, 1.0], 3: [1.5, 1.0, 0.5]}"}
                                    }
                                }
                            }
                        })
                        
                        # Créer manifest temporaire
                        temp_dir = '/tmp/nemo_temp'
                        os.makedirs(temp_dir, exist_ok=True)
                        temp_manifest = os.path.join(temp_dir, f'manifest_{hash(audio_path)}.json')
                        
                        manifest_entry = {
                            "audio_filepath": audio_path,
                            "offset": 0,
                            "duration": len(audio_waveform) / 16000,
                            "label": "infer",
                            "text": "-",
                            "num_speakers": None,
                            "rttm_filepath": None,
                            "uem_filepath": None
                        }
                        
                        with open(temp_manifest, 'w') as f:
                            f.write(json.dumps(manifest_entry) + '\n')
                        
                        # Configurer les chemins
                        cfg.diarizer.manifest_filepath = temp_manifest
                        cfg.diarizer.out_dir = temp_dir
                        
                        # Créer le diarizer
                        diarizer = NeuralDiarizer(cfg=cfg)
                        print("✅ NeMo diarizer created, starting diarization...")
                        
                        # Lancer la diarisation
                        diarizer.diarize()
                        
                        # Lire les résultats RTTM
                        pred_rttm = os.path.join(temp_dir, 'pred_rttms', f'{os.path.splitext(os.path.basename(audio_path))[0]}.rttm')
                        
                        speaker_labels = []
                        speakers_detected = 1
                        
                        if os.path.exists(pred_rttm):
                            speaker_ts = []
                            with open(pred_rttm, 'r') as f:
                                for line in f:
                                    parts = line.strip().split()
                                    if len(parts) >= 8:
                                        start_time = float(parts[3])
                                        duration = float(parts[4])
                                        speaker_id = parts[7]
                                        speaker_ts.append({
                                            'start': start_time,
                                            'end': start_time + duration,
                                            'speaker': speaker_id
                                        })
                            
                            if speaker_ts:
                                speakers_detected = len(set(ts['speaker'] for ts in speaker_ts))
                                print(f"🎤 Found {speakers_detected} speakers in RTTM")
                                
                                # Mapper aux segments Whisper
                                for segment in transcript_segments:
                                    segment_mid = (segment.start + segment.end) / 2
                                    assigned_speaker = "A"
                                    
                                    for ts in speaker_ts:
                                        if ts['start'] <= segment_mid <= ts['end']:
                                            speaker_num = hash(ts['speaker']) % 26
                                            assigned_speaker = chr(65 + speaker_num)
                                            break
                                    
                                    speaker_labels.append(assigned_speaker)
                            else:
                                print("⚠️ No speaker data in RTTM, using basic fallback")
                                speaker_labels, speakers_detected = basic_speaker_diarization(transcript_segments, request.max_speakers)
                        else:
                            print(f"⚠️ RTTM file not found: {pred_rttm}")
                            speaker_labels, speakers_detected = basic_speaker_diarization(transcript_segments, request.max_speakers)
                        
                        # Cleanup
                        try:
                            import shutil
                            shutil.rmtree(temp_dir)
                        except:
                            pass
                        
                        print(f"✅ NeMo diarization completed: {speakers_detected} speakers")
                        
                    except Exception as nemo_error:
                        print(f"⚠️ NeMo diarization failed: {nemo_error}")
                        print(traceback.format_exc())
                        speaker_labels, speakers_detected = basic_speaker_diarization(transcript_segments, request.max_speakers)
                        print(f"✅ Fallback to basic diarization: {speakers_detected} speakers")
                    
                else:
                    print("🔤 Using basic diarization")
                    speaker_labels, speakers_detected = basic_speaker_diarization(transcript_segments, request.max_speakers)
                
            except Exception as e:
                print(f"⚠️ Diarization failed: {e}")
                print(traceback.format_exc())
                speakers_detected = 1
                speaker_labels = ["A"] * len(transcript_segments)
        
        # Formater les segments avec speakers
        response_segments = []
        for i, segment in enumerate(transcript_segments):
            speaker_id = speaker_labels[i] if i < len(speaker_labels) else "A"
            
            response_segments.append({
                "id": i,
                "start": round(segment.start, 2),
                "end": round(segment.end, 2),
                "text": segment.text.strip(),
                "speaker": f"Speaker {speaker_id}",
                "confidence": getattr(segment, 'avg_logprob', 0.0)
            })
        
        # Nettoyer la mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text=full_text,
            segments=response_segments,
            speakers_detected=speakers_detected,
            processing_time=processing_time,
            language=detected_language,
            model_info={
                "whisper_model": request.whisper_model,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "compute_type": "float16" if torch.cuda.is_available() else "int8",
                "batch_size": request.batch_size,
                "diarization_enabled": request.enable_diarization,
                "pyannote_available": PYANNOTE_AVAILABLE,
                "nemo_available": NEMO_AVAILABLE,
                "helpers_available": HELPERS_AVAILABLE,
                "gpu_memory_gb": torch.cuda.get_device_properties(0).total_memory / 1024**3 if torch.cuda.is_available() else 0
            }
        )
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        # Nettoyer en cas d'erreur
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text="",
            segments=[],
            speakers_detected=0,
            processing_time=processing_time,
            language="unknown",
            model_info={
                "whisper_model": request.whisper_model,
                "device": "cuda" if torch.cuda.is_available() else "cpu",
                "error": True
            },
            error=error_msg
        )

# RunPod handler
async def handler(job):
    """Main handler optimisé GPU"""
    job_input = job.get("input", {})
    
    try:
        print(f"🚀 New job: {job.get('id', 'unknown')} (GPU mode)")
        
        # Validation des paramètres
        request = TranscriptionRequest(**job_input)
        print(f"📥 Processing: {request.audio_url}")
        print(f"🎛️ Model: {request.whisper_model}")
        print(f"🎭 Diarization: {request.enable_diarization}")
        
        # Télécharger l'audio
        audio_path = await download_audio_file(request.audio_url)
        
        # Traitement
        result = await process_transcription_gpu(audio_path, request)
        
        # Cleanup du fichier temporaire
        try:
            os.unlink(audio_path)
            print(f"🗑️ Cleaned up {audio_path}")
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")
        
        # Retourner le résultat
        if result.error:
            return {"error": result.error, "model_info": result.model_info}
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
        error_msg = f"Handler error: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        return {"error": error_msg}

# FastAPI endpoints
@app.get("/health")
async def health_check():
    """Health check avec infos GPU"""
    gpu_info = {}
    if torch.cuda.is_available():
        gpu_info = {
            "gpu_count": torch.cuda.device_count(),
            "gpu_names": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            "gpu_memory_gb": [torch.cuda.get_device_properties(i).total_memory / 1024**3 for i in range(torch.cuda.device_count())],
            "cuda_version": torch.version.cuda
        }
    
    return {
        "status": "healthy",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "pytorch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "pyannote_available": PYANNOTE_AVAILABLE,
        "nemo_available": NEMO_AVAILABLE,
        "helpers_available": HELPERS_AVAILABLE,
        "models_loaded": {
            "whisper": models['whisper'] is not None,
            "current_model": models.get('whisper_model_name'),
            "pyannote_pipeline": models.get('pyannote_pipeline') is not None,
            "diarizer": models['diarizer'] is not None
        },
        "gpu_info": gpu_info,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    """Endpoint de transcription"""
    try:
        audio_path = await download_audio_file(request.audio_url)
        result = await process_transcription_gpu(audio_path, request)
        
        # Cleanup
        try:
            os.unlink(audio_path)
        except:
            pass
        
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        return result
        
    except Exception as e:
        error_msg = str(e)
        print(f"❌ API Error: {error_msg}")
        raise HTTPException(status_code=500, detail=error_msg)

@app.get("/debug/nemo")
async def debug_nemo():
    """Debug NeMo availability"""
    try:
        import nemo
        nemo_version = nemo.__version__
        
        try:
            from nemo.collections.asr.models.msdd_models import NeuralDiarizer
            msdd_available = True
            msdd_error = None
        except Exception as e:
            msdd_available = False
            msdd_error = str(e)
        
        return {
            "nemo_installed": True,
            "nemo_version": nemo_version,
            "msdd_available": msdd_available,
            "msdd_error": msdd_error,
            "nemo_available": NEMO_AVAILABLE
        }
    except Exception as e:
        return {
            "nemo_installed": False,
            "error": str(e),
            "nemo_available": NEMO_AVAILABLE
        }
    """Liste des modèles disponibles"""
    return {
        "whisper_models": ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        "current_whisper": models.get('whisper_model_name'),
        "diarization_available": NEMO_AVAILABLE,
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }

if __name__ == "__main__":
    print(f"🚀 Whisper Diarization Service starting (GPU MODE)...")
    print(f"💾 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"🧩 NeMo available: {NEMO_AVAILABLE}")
    print(f"🛠️ Helpers available: {HELPERS_AVAILABLE}")
    
    # Précharger le modèle par défaut
    try:
        print("🎤 Preloading default Whisper model...")
        load_whisper_model_gpu("medium")
        print("✅ Default model loaded")
    except Exception as e:
        print(f"⚠️ Could not preload model: {e}")
    
    print("🎯 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
