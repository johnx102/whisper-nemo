import os
import json
import tempfile
import asyncio
import gc
import traceback
import subprocess
import shutil
import warnings
from typing import Optional, Dict, Any
from datetime import datetime
from urllib.parse import urlparse
from pathlib import Path

# Supprimer les warnings Pydantic
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*Pydantic V1 style.*")

# Configuration mémoire CUDA plus aggressive
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,garbage_collection_threshold:0.6"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

import torch
import aiohttp
import aiofiles
import runpod
import faster_whisper
import torchaudio
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, validator
import uvicorn

# Configuration cuDNN plus conservative
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = False  # False pour plus de stabilité
    torch.backends.cuda.matmul.allow_tf32 = False  # False pour éviter les erreurs
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.deterministic = True  # Pour plus de stabilité

# Import des modules du projet whisper-diarization avec fallbacks
try:
    from ctc_forced_aligner import (
        generate_emissions,
        get_alignments,
        get_spans,
        load_alignment_model,
        postprocess_results,
        preprocess_text,
    )
    CTC_AVAILABLE = True
    print("✅ CTC forced aligner available")
except ImportError as e:
    print(f"⚠️ CTC forced aligner not available: {e}")
    CTC_AVAILABLE = False

try:
    from deepmultilingualpunctuation import PunctuationModel
    PUNCT_AVAILABLE = True
    print("✅ Punctuation model available")
except ImportError as e:
    print(f"⚠️ Punctuation model not available: {e}")
    PUNCT_AVAILABLE = False

try:
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    NEMO_AVAILABLE = True
    print("✅ NeMo available")
except ImportError as e:
    print(f"⚠️ NeMo not available: {e}")
    NEMO_AVAILABLE = False

try:
    from helpers import (
        cleanup,
        create_config,
        find_numeral_symbol_tokens,
        get_realigned_ws_mapping_with_punctuation,
        get_sentences_speaker_mapping,
        get_speaker_aware_transcript,
        get_words_speaker_mapping,
        langs_to_iso,
        process_language_arg,
        punct_model_langs,
        whisper_langs,
        write_srt,
    )
    HELPERS_AVAILABLE = True
    print("✅ Helpers available")
except ImportError as e:
    print(f"⚠️ Helpers not available: {e}")
    HELPERS_AVAILABLE = False
    # Définir des fallbacks basiques
    langs_to_iso = {"fr": "fr", "en": "en", "es": "es", "de": "de"}
    punct_model_langs = ["fr", "en", "es", "de"]
    whisper_langs = ["fr", "en", "es", "de"]

# Initialize FastAPI app
app = FastAPI(title="Whisper Diarization Service", version="1.0.0")

# Configuration
MAX_FILE_SIZE = 300 * 1024 * 1024  # 300MB
DOWNLOAD_TIMEOUT = 300  # 5 minutes
PROCESSING_TIMEOUT = 1800  # 30 minutes
SUPPORTED_FORMATS = {
    'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/m4a', 
    'audio/ogg', 'audio/flac', 'video/mp4', 'video/avi'
}

# Global model storage
models = {
    'whisper': None,
    'whisper_pipeline': None,
    'alignment_model': None,
    'punct_model': None,
    'nemo_diarizer': None,
}

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
mtypes = {"cpu": "int8", "cuda": "float16"}
compute_type = mtypes[device]

class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    whisper_model: Optional[str] = "large-v2"
    language: Optional[str] = None  # None pour auto-détection
    device: Optional[str] = None  # Auto-détecté
    batch_size: Optional[int] = 8
    suppress_numerals: Optional[bool] = False
    no_stem: Optional[bool] = True  # Pas de séparation vocale par défaut
    min_speakers: Optional[int] = None
    max_speakers: Optional[int] = None
    num_speakers: Optional[int] = None
    
    @validator('whisper_model')
    def validate_whisper_model(cls, v):
        valid_models = [
            'tiny', 'tiny.en', 'base', 'base.en', 'small', 'small.en',
            'medium', 'medium.en', 'large', 'large-v1', 'large-v2', 'large-v3'
        ]
        if v not in valid_models:
            raise ValueError(f'Model must be one of: {", ".join(valid_models)}')
        return v

class TranscriptionResponse(BaseModel):
    transcription_text: str
    segments: list
    speakers_detected: int
    processing_time: float
    language: str
    model_info: Dict[str, Any]
    srt_content: Optional[str] = None
    error: Optional[str] = None

def cleanup_gpu_memory_aggressive():
    """Nettoyage GPU très agressif"""
    try:
        # Forcer la collecte du garbage collector
        for _ in range(3):
            gc.collect()
        
        if torch.cuda.is_available():
            # Vider tous les caches plusieurs fois
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            try:
                torch.cuda.ipc_collect()  # Nettoie la mémoire IPC
            except:
                pass  # Peut ne pas être supporté sur tous les systèmes
            
            # Stats détaillées
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            max_allocated = torch.cuda.max_memory_allocated() / 1e9
            
            print(f"🧹 GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            print(f"🧹 Max allocated: {max_allocated:.1f}GB")
            
            # Reset des stats mémoire
            torch.cuda.reset_peak_memory_stats()
            
    except Exception as e:
        print(f"⚠️ GPU cleanup error: {str(e)}")

def setup_models():
    """Initialize models with better error handling"""
    global device, compute_type
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    compute_type = mtypes[device]
    
    try:
        print(f"🔧 Setting up models on device: {device}")
        
        if device == "cuda":
            print(f"🎯 GPU Info:")
            print(f"   - Name: {torch.cuda.get_device_name()}")
            print(f"   - Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f}GB")
            print(f"   - CUDA Version: {torch.version.cuda}")
            
            # Vérification cuDNN
            try:
                print(f"   - cuDNN enabled: {torch.backends.cudnn.enabled}")
                print(f"   - cuDNN version: {torch.backends.cudnn.version()}")
            except Exception as e:
                print(f"   - cuDNN error: {e}")
                # Basculer en mode CPU si cuDNN problématique
                print("   - ⚠️ Switching to CPU mode due to cuDNN issues")
                device = "cpu"
                compute_type = "int8"
            
            if device == "cuda":
                # Configuration GPU conservative
                torch.backends.cudnn.benchmark = False
                torch.backends.cuda.matmul.allow_tf32 = False
                torch.backends.cudnn.allow_tf32 = False
                torch.backends.cudnn.deterministic = True
                
                # Nettoyage initial
                cleanup_gpu_memory_aggressive()
        
        # Créer répertoires
        os.makedirs("temp_outputs", exist_ok=True)
        os.makedirs("outputs", exist_ok=True)
        
        return device, compute_type
        
    except Exception as e:
        print(f"❌ Error setting up models: {str(e)}")
        print("🔄 Falling back to CPU mode")
        return "cpu", "int8"

async def validate_url_security(url: str) -> bool:
    """Validate URL for security"""
    try:
        parsed = urlparse(str(url))
        
        if parsed.scheme not in ['http', 'https']:
            raise ValueError("Invalid URL scheme")
        
        if not parsed.hostname:
            raise ValueError("Invalid hostname")
            
        if parsed.hostname.lower() in ['localhost', '127.0.0.1', '0.0.0.0']:
            raise ValueError("Access to localhost forbidden")
            
        return True
    except Exception as e:
        raise ValueError(f"URL validation failed: {str(e)}")

async def download_audio_file(url: str) -> str:
    """Download audio file and return path"""
    await validate_url_security(url)
    
    timeout = aiohttp.ClientTimeout(total=DOWNLOAD_TIMEOUT)
    
    async with aiohttp.ClientSession(timeout=timeout) as session:
        try:
            async with session.get(str(url)) as response:
                # Vérifier le type de contenu
                content_type = response.headers.get('content-type', '').lower()
                print(f"📥 Content-Type: {content_type}")
                
                # Vérifier la taille
                content_length = response.headers.get('content-length')
                if content_length and int(content_length) > MAX_FILE_SIZE:
                    raise ValueError(f"File too large: {content_length} bytes")
                
                # Télécharger
                content = await response.read()
                
                if len(content) > MAX_FILE_SIZE:
                    raise ValueError(f"Downloaded file too large: {len(content)} bytes")
                
                # Sauvegarder dans un fichier temporaire
                suffix = '.wav'  # Défaut
                if 'mp3' in content_type:
                    suffix = '.mp3'
                elif 'mp4' in content_type:
                    suffix = '.mp4'
                elif 'm4a' in content_type:
                    suffix = '.m4a'
                elif 'ogg' in content_type:
                    suffix = '.ogg'
                elif 'flac' in content_type:
                    suffix = '.flac'
                
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
                temp_file.write(content)
                temp_file.close()
                
                print(f"✅ Downloaded {len(content)} bytes to {temp_file.name}")
                return temp_file.name
                
        except asyncio.TimeoutError:
            raise ValueError("Download timeout")
        except aiohttp.ClientError as e:
            raise ValueError(f"Download failed: {str(e)}")

def load_whisper_models(model_name: str, device: str, compute_type: str):
    """Load Whisper models with better memory management"""
    global models
    
    try:
        print(f"🎤 Loading Whisper model: {model_name}")
        
        # Nettoyage préventif
        cleanup_gpu_memory_aggressive()
        
        # Configuration de batch size adaptatif selon device
        if device == "cuda":
            # Plus conservateur pour éviter les crashes
            effective_compute_type = "float16"
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_memory < 20:  # Moins de 20GB
                effective_compute_type = "int8"  # Plus économe
                print(f"   - Using int8 for memory efficiency (GPU: {gpu_memory:.1f}GB)")
            else:
                print(f"   - Using float16 (GPU: {gpu_memory:.1f}GB)")
        else:
            effective_compute_type = "int8"
            print(f"   - Using int8 on CPU")
        
        # Modèle standard avec configuration conservative
        whisper_model = faster_whisper.WhisperModel(
            model_name, 
            device=device, 
            compute_type=effective_compute_type,
            cpu_threads=4 if device == "cpu" else 0,
            num_workers=1  # Limiter le parallélisme
        )
        
        # Pipeline batchée seulement si GPU stable
        whisper_pipeline = None
        if device == "cuda":
            try:
                whisper_pipeline = faster_whisper.BatchedInferencePipeline(
                    whisper_model
                )
                print(f"   - ✅ Batched pipeline enabled")
            except Exception as e:
                print(f"   - ⚠️ Batched pipeline failed: {e}")
                print(f"   - Continuing with standard model")
        
        models['whisper'] = whisper_model
        models['whisper_pipeline'] = whisper_pipeline
        
        print(f"✅ Whisper {model_name} loaded successfully")
        return whisper_model, whisper_pipeline
        
    except Exception as e:
        print(f"❌ Error loading Whisper model: {str(e)}")
        # Nettoyage en cas d'erreur
        cleanup_gpu_memory_aggressive()
        raise

def load_alignment_models(language: str, device: str):
    """Load CTC alignment model for the detected language"""
    global models
    
    if not CTC_AVAILABLE:
        print("⚠️ CTC alignment not available")
        return None
    
    try:
        print(f"🔤 Loading alignment model for language: {language}")
        
        alignment_model = load_alignment_model(
            device=device,
            dtype=torch.float16 if device == "cuda" else torch.float32,
        )
        
        models['alignment_model'] = alignment_model
        print(f"✅ Alignment model loaded")
        return alignment_model
        
    except Exception as e:
        print(f"⚠️ Could not load alignment model: {str(e)}")
        return None

def load_punctuation_model(language: str):
    """Load punctuation restoration model"""
    global models
    
    if not PUNCT_AVAILABLE:
        print("⚠️ Punctuation model not available")
        return None
    
    try:
        if language in punct_model_langs:
            print(f"📝 Loading punctuation model for {language}")
            punct_model = PunctuationModel(model="kredor/punctuate-all")
            models['punct_model'] = punct_model
            print(f"✅ Punctuation model loaded")
            return punct_model
        else:
            print(f"⚠️ No punctuation model available for {language}")
            return None
            
    except Exception as e:
        print(f"⚠️ Could not load punctuation model: {str(e)}")
        return None

def load_nemo_diarizer():
    """Load NeMo diarization model"""
    global models
    
    if not NEMO_AVAILABLE:
        print("⚠️ NeMo not available")
        return None
    
    try:
        print(f"🎭 Loading NeMo diarization model...")
        
        # Configuration pour la diarisation
        if HELPERS_AVAILABLE:
            config_path = create_config("temp_outputs")
        else:
            # Fallback simple
            config_path = "temp_outputs/diar_infer_general.yaml"
            os.makedirs("temp_outputs", exist_ok=True)
            
        nemo_diarizer = NeuralDiarizer(cfg=config_path)
        
        models['nemo_diarizer'] = nemo_diarizer
        print(f"✅ NeMo diarizer loaded")
        return nemo_diarizer
        
    except Exception as e:
        print(f"❌ Error loading NeMo diarizer: {str(e)}")
        return None

def extract_vocals(audio_path: str, no_stem: bool = True) -> str:
    """Extract vocals using Demucs (optionnel)"""
    if no_stem:
        print("🎵 Skipping vocal separation (no_stem=True)")
        return audio_path
    
    try:
        print("🎵 Extracting vocals with Demucs...")
        
        # Utiliser Demucs pour séparer les vocaux
        output_dir = "temp_outputs"
        cmd = [
            "python", "-m", "demucs.separate",
            "--name", "htdemucs",
            "--out", output_dir,
            audio_path
        ]
        
        subprocess.run(cmd, check=True, capture_output=True)
        
        # Chemin vers les vocaux extraits
        audio_name = Path(audio_path).stem
        vocal_path = Path(output_dir) / "htdemucs" / audio_name / "vocals.wav"
        
        if vocal_path.exists():
            print(f"✅ Vocals extracted to {vocal_path}")
            return str(vocal_path)
        else:
            print("⚠️ Vocal extraction failed, using original audio")
            return audio_path
            
    except Exception as e:
        print(f"⚠️ Vocal extraction error: {str(e)}, using original audio")
        return audio_path

def find_numeral_symbol_tokens_fallback(tokenizer):
    """Fallback pour find_numeral_symbol_tokens si helpers non disponibles"""
    try:
        if HELPERS_AVAILABLE:
            return find_numeral_symbol_tokens(tokenizer)
        else:
            # Fallback simple
            return [-1]  # Pas de suppression de tokens
    except:
        return [-1]

async def process_diarization(
    audio_path: str,
    request: TranscriptionRequest,
    device: str,
    compute_type: str
) -> TranscriptionResponse:
    """Main diarization processing pipeline with better error handling"""
    
    start_time = datetime.now()
    
    try:
        print(f"🚀 Starting diarization pipeline...")
        print(f"📁 Audio file: {audio_path}")
        print(f"🎛️ Model: {request.whisper_model}")
        print(f"🌍 Language: {request.language or 'auto-detect'}")
        print(f"💾 Device: {device} ({compute_type})")
        
        # 1. Extraction vocale (optionnelle)
        vocal_target = extract_vocals(audio_path, request.no_stem)
        
        # 2. Charger les modèles Whisper avec gestion d'erreur
        try:
            if models['whisper'] is None:
                whisper_model, whisper_pipeline = load_whisper_models(
                    request.whisper_model, device, compute_type
                )
            else:
                whisper_model = models['whisper']
                whisper_pipeline = models['whisper_pipeline']
        except Exception as e:
            print(f"❌ Whisper loading failed: {e}")
            # Fallback vers CPU
            if device == "cuda":
                print("🔄 Retrying with CPU...")
                device = "cpu"
                compute_type = "int8"
                cleanup_gpu_memory_aggressive()
                whisper_model, whisper_pipeline = load_whisper_models(
                    request.whisper_model, device, compute_type
                )
            else:
                raise
        
        # 3. Transcription avec gestion mémoire
        print("🎤 Starting transcription...")
        
        try:
            audio_waveform = faster_whisper.decode_audio(vocal_target)
            
            # Batch size adaptatif et conservative
            if device == "cuda":
                gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
                if gpu_memory_gb >= 40:  # A40/A100
                    optimal_batch = min(request.batch_size, 16)  # Plus conservateur
                elif gpu_memory_gb >= 20:  # V100
                    optimal_batch = min(request.batch_size, 8)
                else:  # T4 etc
                    optimal_batch = min(request.batch_size, 4)
                print(f"🎯 Conservative batch size: {optimal_batch} (GPU: {gpu_memory_gb:.1f}GB)")
            else:
                optimal_batch = 4  # CPU très conservateur
                print(f"🎯 CPU batch size: {optimal_batch}")
            
            # Gestion des tokens à supprimer
            suppress_tokens = find_numeral_symbol_tokens_fallback(whisper_model.hf_tokenizer) if request.suppress_numerals else [-1]
            
            # Paramètres de transcription
            transcribe_params = {
                'audio': audio_waveform,
                'batch_size': optimal_batch
            }
            
            if request.language and request.language.lower() != "auto":
                transcribe_params['language'] = request.language
            
            # Utiliser pipeline si disponible, sinon modèle standard
            if whisper_pipeline and device == "cuda":
                try:
                    transcript_segments, info = whisper_pipeline.transcribe(
                        audio_waveform,
                        language=request.language,
                        suppress_tokens=suppress_tokens,
                        batch_size=optimal_batch,
                    )
                    print("✅ Used batched pipeline")
                except Exception as e:
                    print(f"⚠️ Batched pipeline failed: {e}, falling back to standard")
                    transcript_segments, info = whisper_model.transcribe(
                        audio_waveform,
                        language=request.language,
                        suppress_tokens=suppress_tokens,
                        vad_filter=True,
                        beam_size=1  # Plus économe
                    )
            else:
                transcript_segments, info = whisper_model.transcribe(
                    audio_waveform,
                    language=request.language,
                    suppress_tokens=suppress_tokens,
                    vad_filter=True,
                    beam_size=1  # Plus économe
                )
            
            # Convertir en liste
            transcript_segments = list(transcript_segments)
            detected_language = info.language
            print(f"🌍 Detected language: {detected_language}")
            
            # Nettoyage mémoire après transcription
            cleanup_gpu_memory_aggressive()
            
        except Exception as e:
            print(f"❌ Transcription failed: {e}")
            cleanup_gpu_memory_aggressive()
            raise
        
        # 4. Assemblage du texte
        full_transcript = "".join(segment.text for segment in transcript_segments)
        print(f"📝 Transcription length: {len(full_transcript)} characters")
        
        # 5. Alignment temporel (optionnel, peut être désactivé si problématique)
        word_segments = []
        try:
            if CTC_AVAILABLE:
                print("🔤 Starting forced alignment...")
                alignment_model = load_alignment_models(detected_language, device)
                
                if alignment_model:
                    emissions, stride = generate_emissions(
                        alignment_model, 
                        vocal_target, 
                        batch_size=4  # Très conservateur
                    )
                    
                    tokens_starred, text_starred = preprocess_text(
                        full_transcript,
                        langs_to_iso[detected_language] if detected_language in langs_to_iso else "en",
                        alignment_model.tokenizer,
                    )
                    
                    segments, scores, blank_token = get_alignments(
                        emissions,
                        tokens_starred,
                        stride,
                        alignment_model.tokenizer,
                    )
                    
                    spans = get_spans(tokens_starred, segments, blank_token)
                    word_segments = postprocess_results(text_starred, spans, stride, scores)
                    
                    print(f"✅ Alignment completed: {len(word_segments)} word segments")
                    cleanup_gpu_memory_aggressive()
            else:
                print("⚠️ CTC alignment skipped (not available)")
                
        except Exception as e:
            print(f"⚠️ Alignment failed: {e} - continuing without alignment")
            cleanup_gpu_memory_aggressive()
        
        # 6. Diarisation avec NeMo (optionnelle si problématique)
        speaker_segments = []
        speakers_detected = 1
        
        try:
            if NEMO_AVAILABLE:
                print("🎭 Starting speaker diarization...")
                nemo_diarizer = load_nemo_diarizer()
                
                if nemo_diarizer:
                    # Configuration NeMo conservative
                    audio_duration = len(audio_waveform) / 16000
                    manifest_entry = {
                        "audio_filepath": vocal_target,
                        "offset": 0,
                        "duration": audio_duration,
                        "label": "infer",
                        "text": "-",
                        "rttm_filepath": None,
                        "uem_filepath": None,
                    }
                    
                    manifest_path = "temp_outputs/manifest.json"
                    with open(manifest_path, "w") as f:
                        json.dump(manifest_entry, f)
                        f.write("\n")
                    
                    # Exécuter diarisation
                    nemo_diarizer.diarize()
                    
                    # Lire résultats
                    rttm_path = f"temp_outputs/pred_rttms/{Path(vocal_target).stem}.rttm"
                    if os.path.exists(rttm_path):
                        with open(rttm_path, "r") as f:
                            for line in f:
                                parts = line.strip().split()
                                if len(parts) >= 8 and parts[0] == "SPEAKER":
                                    start_time = float(parts[3])
                                    duration = float(parts[4])
                                    end_time = start_time + duration
                                    speaker_id = parts[7]
                                    
                                    speaker_segments.append({
                                        "start": start_time,
                                        "end": end_time,
                                        "speaker": f"Speaker {speaker_id}",
                                    })
                        
                        unique_speakers = set(seg["speaker"] for seg in speaker_segments)
                        speakers_detected = len(unique_speakers)
                        print(f"✅ Diarization completed: {speakers_detected} speakers")
                        cleanup_gpu_memory_aggressive()
            else:
                print("⚠️ NeMo diarization skipped (not available)")
                    
        except Exception as e:
            print(f"⚠️ Diarization failed: {e} - continuing with single speaker")
            cleanup_gpu_memory_aggressive()
        
        # 7. Restauration de la ponctuation
        punctuated_transcript = full_transcript
        try:
            if PUNCT_AVAILABLE:
                print("📝 Restoring punctuation...")
                punct_model = load_punctuation_model(detected_language)
                
                if punct_model:
                    punctuated_transcript = punct_model.restore_punctuation(full_transcript)
                    print("✅ Punctuation restored")
            else:
                print("⚠️ Punctuation restoration skipped (not available)")
        except Exception as e:
            print(f"⚠️ Punctuation restoration failed: {e}")
        
        # 8. Mapping speakers aux mots et phrases
        word_speaker_mapping = []
        sentence_speaker_mapping = []
        
        if word_segments and speaker_segments and HELPERS_AVAILABLE:
            try:
                word_speaker_mapping = get_words_speaker_mapping(
                    word_segments, speaker_segments, "Speaker A"
                )
                
                if models.get('punct_model'):
                    word_speaker_mapping = get_realigned_ws_mapping_with_punctuation(
                        word_speaker_mapping, punctuated_transcript, detected_language
                    )
                
                sentence_speaker_mapping = get_sentences_speaker_mapping(
                    word_speaker_mapping, punctuated_transcript
                )
                
                print(f"✅ Speaker mapping completed")
                
            except Exception as e:
                print(f"⚠️ Speaker mapping failed: {str(e)}")
        
        # 9. Générer la transcription finale avec speakers
        final_transcript = punctuated_transcript
        if HELPERS_AVAILABLE and sentence_speaker_mapping:
            try:
                final_transcript = get_speaker_aware_transcript(
                    sentence_speaker_mapping, punctuated_transcript
                )
            except Exception as e:
                print(f"⚠️ Speaker-aware transcript failed: {e}")
        
        # 10. Générer le fichier SRT
        srt_content = None
        try:
            if HELPERS_AVAILABLE and word_speaker_mapping:
                srt_path = "temp_outputs/output.srt"
                write_srt(word_speaker_mapping, srt_path)
                
                if os.path.exists(srt_path):
                    with open(srt_path, "r", encoding="utf-8") as f:
                        srt_content = f.read()
                    print("✅ SRT file generated")
        except Exception as e:
            print(f"⚠️ SRT generation failed: {str(e)}")
        
        # 11. Préparer les segments pour la réponse
        response_segments = []
        for i, segment in enumerate(transcript_segments):
            seg_dict = {
                "id": i,
                "start": segment.start,
                "end": segment.end,
                "text": segment.text,
                "speaker": "Speaker A",  # Défaut si pas de diarisation
            }
            
            # Essayer d'assigner le bon speaker
            if sentence_speaker_mapping:
                for sent_map in sentence_speaker_mapping:
                    if (segment.start >= sent_map.get("start_time", 0) and 
                        segment.end <= sent_map.get("end_time", float('inf'))):
                        seg_dict["speaker"] = sent_map.get("speaker", "Speaker A")
                        break
            
            response_segments.append(seg_dict)
        
        # IMPORTANT: Nettoyage final agressif
        cleanup_gpu_memory_aggressive()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text=final_transcript,
            segments=response_segments,
            speakers_detected=speakers_detected,
            processing_time=processing_time,
            language=detected_language,
            model_info={
                "whisper_model": request.whisper_model,
                "device": device,
                "compute_type": compute_type,
                "batch_size": optimal_batch if 'optimal_batch' in locals() else request.batch_size,
                "vocal_separation": not request.no_stem,
                "punctuation_restored": models.get('punct_model') is not None,
                "alignment_performed": len(word_segments) > 0,
                "diarization_performed": len(speaker_segments) > 0,
                "nemo_available": NEMO_AVAILABLE,
                "ctc_available": CTC_AVAILABLE,
                "punct_available": PUNCT_AVAILABLE,
                "helpers_available": HELPERS_AVAILABLE,
                "safe_mode": True
            },
            srt_content=srt_content
        )
        
    except Exception as e:
        error_msg = f"Processing failed: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        # Nettoyage en cas d'erreur
        cleanup_gpu_memory_aggressive()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TranscriptionResponse(
            transcription_text="",
            segments=[],
            speakers_detected=0,
            processing_time=processing_time,
            language="unknown",
            model_info={
                "whisper_model": request.whisper_model,
                "device": device,
                "error": True,
                "nemo_available": NEMO_AVAILABLE,
                "ctc_available": CTC_AVAILABLE,
                "punct_available": PUNCT_AVAILABLE,
                "helpers_available": HELPERS_AVAILABLE,
                "safe_mode": True
            },
            error=error_msg
        )

def cleanup_temp_files():
    """Clean up temporary files"""
    try:
        # Nettoyer les répertoires temporaires
        if os.path.exists("temp_outputs"):
            shutil.rmtree("temp_outputs")
        
        os.makedirs("temp_outputs", exist_ok=True)
        
        # Nettoyer la mémoire GPU
        cleanup_gpu_memory_aggressive()
        
        print("🧹 Temporary files cleaned")
        
    except Exception as e:
        print(f"⚠️ Cleanup error: {str(e)}")

# RunPod handler
async def handler(job):
    """Main RunPod serverless handler"""
    job_input = job.get("input", {})
    
    try:
        print(f"🚀 New job received: {job.get('id', 'unknown')}")
        
        # Valider et parser la requête
        request = TranscriptionRequest(**job_input)
        
        # Télécharger le fichier audio
        print(f"📥 Downloading audio from: {request.audio_url}")
        audio_path = await download_audio_file(request.audio_url)
        
        # Configurer les modèles
        device, compute_type = setup_models()
        
        # Traiter la diarisation
        result = await process_diarization(audio_path, request, device, compute_type)
        
        # Nettoyer les fichiers temporaires
        cleanup_temp_files()
        
        # Supprimer le fichier audio téléchargé
        try:
            os.unlink(audio_path)
        except:
            pass
        
        # Retourner le résultat
        if result.error:
            return {
                "error": result.error,
                "processing_time": result.processing_time,
                "model_info": result.model_info
            }
        else:
            return {
                "transcription": result.transcription_text,
                "segments": result.segments,
                "speakers_detected": result.speakers_detected,
                "language": result.language,
                "processing_time": result.processing_time,
                "model_info": result.model_info,
                "srt_content": result.srt_content
            }
            
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        
        # Nettoyer même en cas d'erreur
        cleanup_temp_files()
        
        return {
            "error": error_msg,
            "timestamp": datetime.utcnow().isoformat()
        }

# FastAPI endpoints pour tests
@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        device_info = {
            "device": device,
            "compute_type": compute_type,
        }
        
        if torch.cuda.is_available():
            device_info.update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_reserved(),
                "cudnn_enabled": torch.backends.cudnn.enabled,
            })
            
            try:
                device_info["cudnn_version"] = torch.backends.cudnn.version()
            except:
                device_info["cudnn_version"] = "unknown"
        
        return {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "device_info": device_info,
            "modules_available": {
                "nemo": NEMO_AVAILABLE,
                "ctc_aligner": CTC_AVAILABLE,
                "punctuation": PUNCT_AVAILABLE,
                "helpers": HELPERS_AVAILABLE,
            },
            "models_loaded": {
                "whisper": models['whisper'] is not None,
                "whisper_pipeline": models['whisper_pipeline'] is not None,
                "alignment_model": models['alignment_model'] is not None,
                "punct_model": models['punct_model'] is not None,
                "nemo_diarizer": models['nemo_diarizer'] is not None,
            }
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat(),
            "modules_available": {
                "nemo": NEMO_AVAILABLE,
                "ctc_aligner": CTC_AVAILABLE,
                "punctuation": PUNCT_AVAILABLE,
                "helpers": HELPERS_AVAILABLE,
            }
        }

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    """Direct transcription endpoint for testing"""
    try:
        # Télécharger l'audio
        audio_path = await download_audio_file(request.audio_url)
        
        # Configurer les modèles
        device, compute_type = setup_models()
        
        # Traiter
        result = await process_diarization(audio_path, request, device, compute_type)
        
        # Nettoyer
        cleanup_temp_files()
        try:
            os.unlink(audio_path)
        except:
            pass
        
        if result.error:
            raise HTTPException(status_code=500, detail=result.error)
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/debug")
async def debug_info():
    """Debug information endpoint"""
    try:
        debug_data = {
            "timestamp": datetime.utcnow().isoformat(),
            "modules_available": {
                "nemo": NEMO_AVAILABLE,
                "ctc_aligner": CTC_AVAILABLE,
                "punctuation": PUNCT_AVAILABLE,
                "helpers": HELPERS_AVAILABLE,
                "torch": True,
                "faster_whisper": True,
                "runpod": True,
            },
            "device_info": {
                "device": device,
                "compute_type": compute_type,
                "cuda_available": torch.cuda.is_available(),
            },
            "models_loaded": {
                "whisper": models['whisper'] is not None,
                "whisper_pipeline": models['whisper_pipeline'] is not None,
                "alignment_model": models['alignment_model'] is not None,
                "punct_model": models['punct_model'] is not None,
                "nemo_diarizer": models['nemo_diarizer'] is not None,
            }
        }
        
        if torch.cuda.is_available():
            debug_data["device_info"].update({
                "gpu_name": torch.cuda.get_device_name(),
                "gpu_memory_total": torch.cuda.get_device_properties(0).total_memory,
                "gpu_memory_allocated": torch.cuda.memory_allocated(),
                "gpu_memory_cached": torch.cuda.memory_reserved(),
                "cudnn_enabled": torch.backends.cudnn.enabled,
            })
            
            try:
                debug_data["device_info"]["cudnn_version"] = torch.backends.cudnn.version()
            except:
                debug_data["device_info"]["cudnn_version"] = "unknown"
        
        return debug_data
        
    except Exception as e:
        return {
            "error": str(e),
            "timestamp": datetime.utcnow().isoformat()
        }

if __name__ == "__main__":
    # Initialiser l'environnement
    device, compute_type = setup_models()
    print(f"🚀 Whisper Diarization Service initialized on {device}")
    print(f"📊 Compute type: {compute_type}")
    print(f"🧩 Modules available:")
    print(f"   - NeMo: {'✅' if NEMO_AVAILABLE else '❌'}")
    print(f"   - CTC Aligner: {'✅' if CTC_AVAILABLE else '❌'}")
    print(f"   - Punctuation: {'✅' if PUNCT_AVAILABLE else '❌'}")
    print(f"   - Helpers: {'✅' if HELPERS_AVAILABLE else '❌'}")
    
    # Démarrer le handler RunPod serverless
    print("🎯 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
