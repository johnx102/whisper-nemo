import os
import json
import tempfile
import asyncio
import gc
import traceback
import warnings
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime
import re

# Supprimer warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

# FORCER GPU MODE
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["TORCH_CUDNN_V8_API_DISABLED"] = "1"

print("🚀 GPU MODE ENABLED - Whisper + NeMo Diarization")

import torch
import torchaudio
import aiohttp
import runpod
import faster_whisper
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, HttpUrl, field_validator
from pydub import AudioSegment

# Configuration des types de compute
mtypes = {"cpu": "int8", "cuda": "float16"}

# Vérifier que PyTorch fonctionne avec GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# Imports optionnels avec fallbacks
try:
    from helpers import (
        find_numeral_symbol_tokens,
        create_config,
        get_words_speaker_mapping,
        get_sentences_speaker_mapping,
        get_speaker_aware_transcript,
        cleanup
    )
    HELPERS_AVAILABLE = True
    print("✅ Helpers available")
except ImportError:
    print("⚠️ Helpers not available - using fallbacks")
    HELPERS_AVAILABLE = False
    
    def find_numeral_symbol_tokens(tokenizer):
        """Fallback pour find_numeral_symbol_tokens"""
        try:
            # Tokens numériques classiques à supprimer
            return [50362, 50363, 50364, 50365, 50366, 50367, 50368, 50369]
        except:
            return [-1]

try:
    from nemo.collections.asr.models.msdd_models import NeuralDiarizer
    NEMO_AVAILABLE = True
    print("✅ NeMo available")
except ImportError as e:
    print(f"❌ NeMo import failed: {e}")
    NEMO_AVAILABLE = False

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
    print("✅ CTC Forced Aligner available")
except ImportError:
    print("⚠️ CTC Forced Aligner not available")
    CTC_AVAILABLE = False

try:
    from deepmultilingualpunctuation import PunctuationModel
    PUNCT_AVAILABLE = True
    print("✅ Punctuation model available")
except ImportError:
    print("⚠️ Punctuation model not available")
    PUNCT_AVAILABLE = False

# Initialize FastAPI
app = FastAPI(title="Whisper NeMo Diarization Service", version="5.0.0")

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
DOWNLOAD_TIMEOUT = 300
SUPPORTED_FORMATS = {
    'audio/wav', 'audio/mpeg', 'audio/mp4', 'audio/m4a', 
    'audio/ogg', 'audio/flac', 'audio/webm'
}

# Models storage
models = {
    'whisper': None, 
    'whisper_model_name': None, 
    'alignment_model': None,
    'punct_model': None
}

class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    whisper_model: Optional[str] = "medium"
    language: Optional[str] = "fr"
    batch_size: Optional[int] = 16
    no_stem: Optional[bool] = True
    enable_diarization: Optional[bool] = True
    min_speakers: Optional[int] = 1
    max_speakers: Optional[int] = 8
    suppress_numerals: Optional[bool] = True
    
    @field_validator('whisper_model')
    @classmethod
    def validate_whisper_model(cls, v):
        valid = ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3']
        if v not in valid:
            raise ValueError(f"Invalid model. Choose from: {valid}")
        return v

class TranscriptionResponse(BaseModel):
    transcription_text: str
    segments: List[Dict[str, Any]]
    speakers_detected: int
    processing_time: float
    language: str
    model_info: Dict[str, Any]
    error: Optional[str] = None

def create_nemo_config(temp_dir: str, audio_path: str, max_speakers: int = 8):
    """Créer une configuration NeMo optimisée basée sur whisper-diarization"""
    
    # Configuration inspirée du projet original whisper-diarization
    config_content = f"""
device: cuda

diarizer:
  manifest_filepath: {os.path.join(temp_dir, 'manifest.json')}
  out_dir: {temp_dir}
  oracle_vad: false
  clustering:
    parameters:
      oracle_num_speakers: false
      max_num_speakers: {max_speakers}
      enhanced_count_thres: 0.8
  msdd_model:
    model_path: diar_msdd_telephonic
    parameters:
      use_speaker_model_from_ckpt: true
      infer_batch_size: 25
      sigmoid_threshold: [0.7]
      seq_eval_mode: false
      split_infer: true
      diar_window_length: 50
      overlap_infer_spk_limit: 5
  vad:
    model_path: vad_multilingual_marblenet
    external_vad_manifest: null
    parameters:
      onset: 0.8
      offset: 0.6
      pad_onset: 0.05
      pad_offset: -0.05
      min_duration_on: 0.2
      min_duration_off: 0.2
  speaker_embeddings:
    model_path: titanet_large
    parameters:
      window_length_in_sec: 1.5
      shift_length_in_sec: 0.75
      multiscale_weights: [1, 1, 1, 1, 1]
      multiscale_args:
        scale_dict:
          1: [1.5]
          2: [1.5, 1.0] 
          3: [1.5, 1.0, 0.5]
"""
    
    config_path = os.path.join(temp_dir, 'config.yaml')
    with open(config_path, 'w') as f:
        f.write(config_content)
    
    print(f"✅ NeMo config created: {config_path}")
    
    # Debug: afficher la config pour vérifier
    print("🔍 NeMo config content:")
    with open(config_path, 'r') as f:
        print(f.read())
    
    return config_path

def convert_audio_to_mono(audio_path: str, temp_dir: str) -> str:
    """Convertir l'audio en mono pour la compatibilité NeMo"""
    try:
        sound = AudioSegment.from_file(audio_path).set_channels(1)
        mono_path = os.path.join(temp_dir, "mono_audio.wav")
        sound.export(mono_path, format="wav")
        print(f"✅ Audio converted to mono: {mono_path}")
        return mono_path
    except Exception as e:
        print(f"⚠️ Could not convert to mono: {e}")
        return audio_path

def create_manifest(audio_path: str, temp_dir: str) -> str:
    """Créer le manifest pour NeMo"""
    try:
        # Obtenir la durée de l'audio
        waveform, sample_rate = torchaudio.load(audio_path)
        duration = waveform.shape[1] / sample_rate
        
        manifest_entry = {
            "audio_filepath": audio_path,
            "offset": 0,
            "duration": duration,
            "label": "infer",
            "text": "-",
            "num_speakers": None,
            "rttm_filepath": None,
            "uem_filepath": None
        }
        
        manifest_path = os.path.join(temp_dir, 'manifest.json')
        with open(manifest_path, 'w') as f:
            f.write(json.dumps(manifest_entry) + '\n')
        
        print(f"✅ Manifest created: {manifest_path}")
        return manifest_path
        
    except Exception as e:
        print(f"❌ Error creating manifest: {e}")
        raise

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
                suffix = '.wav'
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
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = mtypes[device]
        
        whisper_model = faster_whisper.WhisperModel(
            model_name,
            device=device,
            compute_type=compute_type,
            num_workers=4 if device == "cuda" else 1
        )
        
        models['whisper'] = whisper_model
        models['whisper_model_name'] = model_name
        
        print(f"✅ Whisper {model_name} loaded successfully ({device}, {compute_type})")
        return whisper_model
        
    except Exception as e:
        print(f"❌ Error loading Whisper: {e}")
        print(traceback.format_exc())
        raise

def perform_nemo_diarization(audio_path: str, temp_dir: str, max_speakers: int = 8):
    """Effectuer la diarisation avec NeMo"""
    try:
        print(f"🎭 Starting NeMo diarization with max {max_speakers} speakers")
        
        # Convertir en mono
        mono_audio = convert_audio_to_mono(audio_path, temp_dir)
        
        # Créer le manifest
        manifest_path = create_manifest(mono_audio, temp_dir)
        
        # Créer la config - ESSAYER AVEC helpers.create_config D'ABORD
        config_path = None
        
        if HELPERS_AVAILABLE:
            try:
                print("🔧 Trying helpers.create_config...")
                from helpers import create_config
                config_path = create_config(temp_dir)
                
                # Modifier la config pour nos paramètres
                from omegaconf import OmegaConf
                cfg = OmegaConf.load(config_path)
                cfg.diarizer.manifest_filepath = manifest_path
                cfg.diarizer.out_dir = temp_dir
                cfg.diarizer.clustering.parameters.max_num_speakers = max_speakers
                
                # Sauvegarder la config modifiée
                OmegaConf.save(cfg, config_path)
                print("✅ Using helpers.create_config with modifications")
                
            except Exception as e:
                print(f"⚠️ helpers.create_config failed: {e}")
                config_path = None
        
        # Fallback vers notre config personnalisée
        if config_path is None:
            print("🔧 Using custom config...")
            config_path = create_nemo_config(temp_dir, mono_audio, max_speakers)
        
        # Charger la config
        from omegaconf import OmegaConf
        cfg = OmegaConf.load(config_path)
        
        print(f"🔍 Config loaded. Keys: {list(cfg.keys())}")
        print(f"🔍 Diarizer keys: {list(cfg.diarizer.keys())}")
        
        # Créer et lancer le diarizer
        print("🏗️ Creating NeMo diarizer...")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # IMPORTANT: S'assurer que le device est correctement configuré
        if device == "cuda":
            cfg.device = "cuda"
        
        try:
            msdd_model = NeuralDiarizer(cfg=cfg)
            if device == "cuda":
                msdd_model = msdd_model.to(device)
            
            print("🎯 Starting diarization process...")
            msdd_model.diarize()
            
        except Exception as nemo_error:
            print(f"❌ NeuralDiarizer creation/execution failed: {nemo_error}")
            
            # ESSAYER UNE APPROCHE ALTERNATIVE
            print("🔄 Trying alternative NeMo approach...")
            
            # Créer une config plus simple
            simple_config = f"""
device: {device}

diarizer:
  manifest_filepath: {manifest_path}
  out_dir: {temp_dir}
  oracle_vad: false
  clustering:
    parameters:
      oracle_num_speakers: false
      max_num_speakers: {max_speakers}
  msdd_model:
    model_path: diar_msdd_telephonic
    parameters:
      use_speaker_model_from_ckpt: true
      infer_batch_size: 16
      sigmoid_threshold: [0.7]
  vad:
    model_path: vad_multilingual_marblenet
    parameters:
      onset: 0.8
      offset: 0.6
  speaker_embeddings:
    model_path: titanet_large
    parameters:
      window_length_in_sec: 1.5
      shift_length_in_sec: 0.75
"""
            
            simple_config_path = os.path.join(temp_dir, 'simple_config.yaml')
            with open(simple_config_path, 'w') as f:
                f.write(simple_config)
            
            # Réessayer avec la config simplifiée
            cfg_simple = OmegaConf.load(simple_config_path)
            msdd_model = NeuralDiarizer(cfg=cfg_simple)
            if device == "cuda":
                msdd_model = msdd_model.to(device)
            
            print("🎯 Starting diarization with simple config...")
            msdd_model.diarize()
        
        # Lire les résultats RTTM
        pred_rttm_dir = os.path.join(temp_dir, 'pred_rttms')
        audio_basename = os.path.splitext(os.path.basename(mono_audio))[0]
        pred_rttm = os.path.join(pred_rttm_dir, f'{audio_basename}.rttm')
        
        print(f"🔍 Looking for RTTM: {pred_rttm}")
        
        # Debug: lister les fichiers créés
        if os.path.exists(temp_dir):
            print(f"📁 Files in temp_dir: {os.listdir(temp_dir)}")
        if os.path.exists(pred_rttm_dir):
            print(f"📁 Files in pred_rttms: {os.listdir(pred_rttm_dir)}")
        
        if os.path.exists(pred_rttm):
            speaker_timestamps = []
            with open(pred_rttm, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 8:
                        start_time = float(parts[3])
                        duration = float(parts[4])
                        speaker_id = parts[7]
                        speaker_timestamps.append({
                            'start': start_time,
                            'end': start_time + duration,
                            'speaker': speaker_id
                        })
            
            speakers_detected = len(set(ts['speaker'] for ts in speaker_timestamps))
            print(f"✅ NeMo diarization completed: {speakers_detected} speakers detected")
            return speaker_timestamps, speakers_detected
        else:
            print(f"⚠️ RTTM file not found")
            return [], 1
            
    except Exception as e:
        print(f"❌ NeMo diarization failed: {e}")
        print(traceback.format_exc())
        return [], 1

def assign_speakers_to_segments(transcript_segments, speaker_timestamps):
    """Assigner les speakers aux segments de transcription"""
    speaker_labels = []
    
    if not speaker_timestamps:
        return ["A"] * len(transcript_segments)
    
    # Créer un mapping des IDs speakers vers des lettres
    unique_speakers = sorted(list(set(ts['speaker'] for ts in speaker_timestamps)))
    speaker_id_to_letter = {speaker_id: chr(65 + i) for i, speaker_id in enumerate(unique_speakers)}
    
    for segment in transcript_segments:
        segment_start = segment.start
        segment_end = segment.end
        segment_mid = (segment_start + segment_end) / 2
        
        # Trouver le speaker actif au milieu du segment
        assigned_speaker = "A"  # défaut
        best_overlap = 0
        
        for ts in speaker_timestamps:
            # Calculer l'overlap entre le segment et le timestamp du speaker
            overlap_start = max(segment_start, ts['start'])
            overlap_end = min(segment_end, ts['end'])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                assigned_speaker = speaker_id_to_letter[ts['speaker']]
        
        speaker_labels.append(assigned_speaker)
    
    return speaker_labels

def basic_speaker_diarization(segments, max_speakers=2):
    """Diarisation basique comme fallback"""
    if len(segments) <= 2:
        return ["A"] * len(segments), 1
    
    num_speakers = min(max_speakers, max(2, len(segments) // 5))
    
    # Analyser les pauses
    pauses = []
    for i in range(1, len(segments)):
        pause = segments[i].start - segments[i-1].end
        pauses.append(pause)
    
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
        
        speaker_labels.append(chr(65 + current_speaker))
    
    detected_speakers = min(num_speakers, len(set(speaker_labels)))
    return speaker_labels, detected_speakers

async def process_transcription_with_nemo(audio_path: str, request: TranscriptionRequest):
    """Pipeline principal de transcription + diarisation NeMo"""
    start_time = datetime.now()
    
    try:
        print(f"🚀 Starting Whisper + NeMo pipeline...")
        print(f"📁 Audio: {audio_path}")
        print(f"🎛️ Model: {request.whisper_model}")
        print(f"🌍 Language: {request.language}")
        print(f"🎭 Diarization: {'Enabled' if request.enable_diarization else 'Disabled'}")
        
        # Charger modèle Whisper
        current_model_name = models.get('whisper_model_name')
        if models['whisper'] is None or current_model_name != request.whisper_model:
            whisper_model = load_whisper_model_gpu(request.whisper_model)
        else:
            whisper_model = models['whisper']
        
        # Nettoyer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Transcription Whisper
        print("🎤 Starting Whisper transcription...")
        
        # Créer le pipeline batché si batch_size > 0
        if request.batch_size > 0:
            whisper_pipeline = faster_whisper.BatchedInferencePipeline(whisper_model)
            use_batched = True
        else:
            use_batched = False
        
        # Charger l'audio
        audio_waveform = faster_whisper.decode_audio(audio_path)
        print(f"🎵 Audio duration: {len(audio_waveform) / 16000:.2f} seconds")
        
        # Configuration tokens à supprimer
        suppress_tokens = (
            find_numeral_symbol_tokens(whisper_model.hf_tokenizer)
            if request.suppress_numerals
            else [-1]
        )
        
        # Transcription
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
        
        if use_batched:
            transcribe_options["batch_size"] = request.batch_size
            transcript_segments, info = whisper_pipeline.transcribe(
                audio_waveform, 
                **transcribe_options
            )
        else:
            transcript_segments, info = whisper_model.transcribe(
                audio_waveform, 
                **transcribe_options
            )
        
        # Convertir en liste
        transcript_segments = list(transcript_segments)
        detected_language = info.language
        
        print(f"✅ Whisper transcription completed: {len(transcript_segments)} segments")
        print(f"🌍 Detected language: {detected_language}")
        
        # Assembler le texte complet
        full_text = " ".join(segment.text.strip() for segment in transcript_segments if segment.text.strip())
        
        # Diarisation si demandée
        speakers_detected = 1
        speaker_labels = ["A"] * len(transcript_segments)
        
        if request.enable_diarization and len(transcript_segments) > 2 and NEMO_AVAILABLE:
            try:
                # Créer répertoire temporaire pour NeMo
                temp_dir = f'/tmp/nemo_temp_{hash(audio_path)}'
                os.makedirs(temp_dir, exist_ok=True)
                
                # Lancer NeMo diarization
                speaker_timestamps, speakers_detected = perform_nemo_diarization(
                    audio_path, temp_dir, request.max_speakers
                )
                
                if speaker_timestamps:
                    # Assigner les speakers aux segments
                    speaker_labels = assign_speakers_to_segments(transcript_segments, speaker_timestamps)
                    print(f"✅ Speaker assignment completed: {speakers_detected} speakers")
                else:
                    print("⚠️ No speaker timestamps, using basic fallback")
                    speaker_labels, speakers_detected = basic_speaker_diarization(
                        transcript_segments, request.max_speakers
                    )
                
                # Cleanup
                try:
                    import shutil
                    shutil.rmtree(temp_dir)
                    print("🗑️ Cleaned up temp directory")
                except Exception as e:
                    print(f"⚠️ Cleanup failed: {e}")
                    
            except Exception as e:
                print(f"⚠️ NeMo diarization failed: {e}")
                print(traceback.format_exc())
                speaker_labels, speakers_detected = basic_speaker_diarization(
                    transcript_segments, request.max_speakers
                )
                print(f"✅ Fallback to basic diarization: {speakers_detected} speakers")
        
        elif request.enable_diarization and not NEMO_AVAILABLE:
            print("⚠️ NeMo not available, using basic diarization")
            speaker_labels, speakers_detected = basic_speaker_diarization(
                transcript_segments, request.max_speakers
            )
        
        # Formater les segments de réponse
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
                "compute_type": mtypes["cuda" if torch.cuda.is_available() else "cpu"],
                "batch_size": request.batch_size,
                "diarization_enabled": request.enable_diarization,
                "nemo_available": NEMO_AVAILABLE,
                "helpers_available": HELPERS_AVAILABLE,
                "ctc_available": CTC_AVAILABLE,
                "punct_available": PUNCT_AVAILABLE,
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
    """Main handler pour RunPod serverless"""
    job_input = job.get("input", {})
    
    try:
        print(f"🚀 New job: {job.get('id', 'unknown')} (Whisper + NeMo)")
        
        # Validation des paramètres
        request = TranscriptionRequest(**job_input)
        print(f"📥 Processing: {request.audio_url}")
        print(f"🎛️ Model: {request.whisper_model}")
        print(f"🎭 Diarization: {request.enable_diarization}")
        
        # Télécharger l'audio
        audio_path = await download_audio_file(request.audio_url)
        
        # Traitement
        result = await process_transcription_with_nemo(audio_path, request)
        
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

# FastAPI endpoints pour les tests
@app.get("/health")
async def health_check():
    """Health check avec infos système"""
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
        "nemo_available": NEMO_AVAILABLE,
        "helpers_available": HELPERS_AVAILABLE,
        "ctc_available": CTC_AVAILABLE,
        "punct_available": PUNCT_AVAILABLE,
        "models_loaded": {
            "whisper": models['whisper'] is not None,
            "current_model": models.get('whisper_model_name'),
            "alignment_model": models.get('alignment_model') is not None,
            "punct_model": models.get('punct_model') is not None
        },
        "gpu_info": gpu_info,
        "timestamp": datetime.utcnow().isoformat()
    }

@app.post("/transcribe")
async def transcribe_endpoint(request: TranscriptionRequest):
    """Endpoint de transcription pour tests"""
    try:
        audio_path = await download_audio_file(request.audio_url)
        result = await process_transcription_with_nemo(audio_path, request)
        
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

@app.get("/models")
async def list_models():
    """Liste des modèles disponibles"""
    return {
        "whisper_models": ['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3'],
        "current_whisper": models.get('whisper_model_name'),
        "diarization_available": NEMO_AVAILABLE,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "capabilities": {
            "whisper": True,
            "nemo_diarization": NEMO_AVAILABLE,
            "ctc_alignment": CTC_AVAILABLE,
            "punctuation": PUNCT_AVAILABLE,
            "helpers": HELPERS_AVAILABLE
        }
    }

@app.get("/debug/nemo")
async def debug_nemo():
    """Debug NeMo availability et configuration"""
    debug_info = {
        "nemo_available": NEMO_AVAILABLE,
        "cuda_available": torch.cuda.is_available(),
        "device": "cuda" if torch.cuda.is_available() else "cpu"
    }
    
    if NEMO_AVAILABLE:
        try:
            import nemo
            debug_info["nemo_version"] = nemo.__version__
            
            try:
                from nemo.collections.asr.models.msdd_models import NeuralDiarizer
                debug_info["neural_diarizer_available"] = True
            except Exception as e:
                debug_info["neural_diarizer_available"] = False
                debug_info["neural_diarizer_error"] = str(e)
                
        except Exception as e:
            debug_info["nemo_import_error"] = str(e)
    
    return debug_info

if __name__ == "__main__":
    print(f"🚀 Whisper + NeMo Diarization Service starting...")
    print(f"💾 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        print(f"🎮 GPU: {torch.cuda.get_device_name(0)}")
        print(f"💾 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    print(f"🧩 NeMo available: {NEMO_AVAILABLE}")
    print(f"🛠️ Helpers available: {HELPERS_AVAILABLE}")
    print(f"🔗 CTC Aligner available: {CTC_AVAILABLE}")
    print(f"📝 Punctuation available: {PUNCT_AVAILABLE}")
    
    # Précharger le modèle par défaut
    try:
        print("🎤 Preloading default Whisper model...")
        load_whisper_model_gpu("medium")
        print("✅ Default model loaded")
    except Exception as e:
        print(f"⚠️ Could not preload model: {e}")
    
    print("🎯 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
