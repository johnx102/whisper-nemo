import os
import tempfile
import asyncio
import gc
import traceback
import warnings
import logging
from typing import Optional, Dict, Any, List
from datetime import datetime, timedelta
import numpy as np

# Supprimer warnings
warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.INFO)

print("🚀 Whisper LARGE-V2 + PyAnnote Serverless - Based on Working Setup")

import torch
import aiohttp
import runpod
import whisper
from pyannote.audio import Pipeline
from pydantic import BaseModel, HttpUrl, field_validator

# ============= OPTIMISATIONS GPU IDENTIQUES =============
# Configuration GPU stricte exactement comme votre setup
torch.cuda.set_device(0)
device = torch.device('cuda:0')
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_grad_enabled(False)  # Inference seulement

print("🔥 Warmup GPU...")
x = torch.randn(2000, 2000).cuda()
y = torch.mm(x, x)
torch.cuda.synchronize()
del x, y
print("✅ GPU optimisé et warmed up")
# ============= FIN OPTIMISATIONS =============

# Vérifier que PyTorch fonctionne avec GPU
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU count: {torch.cuda.device_count()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")

# Configuration
MAX_FILE_SIZE = 500 * 1024 * 1024  # 500MB
DOWNLOAD_TIMEOUT = 300

# Variables globales pour les modèles - comme votre setup
whisper_model = None
diarization_pipeline = None

class TranscriptionRequest(BaseModel):
    audio_url: HttpUrl
    language: Optional[str] = "fr"
    num_speakers: Optional[int] = None  # Nombre exact de speakers
    min_speakers: Optional[int] = 1
    max_speakers: Optional[int] = 4  # Comme votre setup
    transcription_only: Optional[bool] = False  # Si True, pas de diarisation
    
    @field_validator('language')
    @classmethod
    def validate_language(cls, v):
        valid = ['fr', 'en', 'es', 'de', 'it', 'auto']
        if v not in valid:
            raise ValueError(f"Invalid language. Choose from: {valid}")
        return v

class TranscriptionResponse(BaseModel):
    success: bool
    model: str
    transcription_brute: str
    transcription_formatee: Optional[str] = None
    segments_detailles: Optional[List[Dict[str, Any]]] = None
    parametres: Dict[str, Any]
    statistiques: Dict[str, Any]
    processing_time: float
    error: Optional[str] = None

def monitor_gpu_usage():
    """Affiche l'utilisation GPU - fonction de votre setup"""
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'], 
                              capture_output=True, text=True)
        lines = result.stdout.strip().split('\n')
        if lines and lines[0]:
            gpu_data = lines[0].split(', ')
            if len(gpu_data) >= 3:
                gpu_util, gpu_mem_used, gpu_mem_total = gpu_data
                print(f"🎮 GPU: {gpu_util.strip()}% util | {gpu_mem_used.strip()}/{gpu_mem_total.strip()}MB")
        else:
            print(f"🎮 GPU info: {result.stdout}")
    except Exception as e:
        print(f"⚠️ Monitoring GPU error: {e}")

def load_models():
    """Chargement des modèles - EXACTEMENT comme votre setup"""
    global whisper_model, diarization_pipeline
    
    try:
        # Chargement Whisper LARGE-V2 - comme votre setup
        print("🔄 Chargement Whisper LARGE-V2...")
        DEVICE_WHISPER = "cuda:0"
        whisper_model = whisper.load_model("large-v2", device=DEVICE_WHISPER)
        print(f"✅ Whisper device: {next(whisper_model.parameters()).device}")
        print("✅ Whisper LARGE-V2 chargé")
        
        # Chargement pyannote - comme votre setup
        print("🔄 Chargement pyannote...")
        diarization_pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1", 
            use_auth_token=os.getenv("HF_TOKEN", True)
        )
        
        # Configuration GPU pour pyannote
        if torch.cuda.device_count() > 1:
            # Si 2 GPU disponibles, mettre pyannote sur GPU 1
            DEVICE_PYANNOTE = "cuda:1"
            diarization_pipeline.to(torch.device(DEVICE_PYANNOTE))
            print("✅ pyannote déplacé sur GPU 1")
        else:
            # Sinon partager GPU 0
            diarization_pipeline.to(torch.device("cuda:0"))
            print("✅ pyannote sur GPU 0 (partagé)")
        
        print("🎉 Modèles chargés et optimisés !")
        monitor_gpu_usage()
        
    except Exception as e:
        print(f"❌ Erreur chargement modèles: {e}")
        print(traceback.format_exc())
        raise

def format_timestamp(seconds):
    """Convertit les secondes en format mm:ss - fonction de votre setup"""
    return str(timedelta(seconds=int(seconds)))[2:]

def optimize_diarization(audio_path, num_speakers=None, min_speakers=1, max_speakers=4):
    """Diarization optimisée - EXACTEMENT votre fonction"""
    
    # Paramètres de diarization
    diarization_params = {}
    
    if num_speakers:
        diarization_params['num_speakers'] = num_speakers
        print(f"🎯 Forçage à {num_speakers} speakers")
    else:
        diarization_params['min_speakers'] = min_speakers
        diarization_params['max_speakers'] = max_speakers
        print(f"🔍 Détection entre {min_speakers} et {max_speakers} speakers")
    
    # Le pipeline est déjà optimisé
    diarization = diarization_pipeline(audio_path, **diarization_params)
    
    return diarization

def merge_transcription_with_speakers_improved(whisper_segments, diarization):
    """Fusion améliorée - EXACTEMENT votre fonction"""
    
    # Convertir diarization en liste
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    print(f"👥 Speakers détectés par pyannote: {len(set(seg['speaker'] for seg in speaker_segments))}")
    
    # Associer chaque segment whisper à un speaker avec logique améliorée
    merged_segments = []
    
    for segment in whisper_segments:
        seg_start = segment["start"]
        seg_end = segment["end"]
        seg_middle = (seg_start + seg_end) / 2
        
        # Trouver le speaker avec le plus de recouvrement
        best_speaker = "INCONNU"
        best_overlap = 0
        
        for spk_seg in speaker_segments:
            # Calculer le recouvrement entre le segment whisper et le segment speaker
            overlap_start = max(seg_start, spk_seg["start"])
            overlap_end = min(seg_end, spk_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = spk_seg["speaker"]
        
        merged_segments.append({
            "start": seg_start,
            "end": seg_end,
            "duration": seg_end - seg_start,
            "speaker": best_speaker,
            "text": segment["text"].strip(),
            "confidence": 1 - segment.get("no_speech_prob", 0),
            "overlap_quality": best_overlap / (seg_end - seg_start)  # Qualité de l'attribution
        })
    
    # Post-traitement : lissage des speakers isolés
    merged_segments = smooth_speaker_transitions(merged_segments)
    
    return merged_segments

def smooth_speaker_transitions(segments):
    """Lisse les transitions - EXACTEMENT votre fonction"""
    
    if len(segments) < 3:
        return segments
    
    smoothed = segments.copy()
    
    # Règle : si un segment court (< 2s) est entouré du même speaker, le reassigner
    for i in range(1, len(smoothed) - 1):
        current = smoothed[i]
        prev_speaker = smoothed[i-1]["speaker"]
        next_speaker = smoothed[i+1]["speaker"]
        
        # Si segment court entre même speaker
        if (current["duration"] < 2.0 and 
            prev_speaker == next_speaker and 
            current["speaker"] != prev_speaker and
            current["overlap_quality"] < 0.8):  # Faible confiance d'attribution
            
            print(f"🔧 Lissage: '{current['text'][:30]}...' de {current['speaker']} vers {prev_speaker}")
            smoothed[i]["speaker"] = prev_speaker
            smoothed[i]["smoothed"] = True
    
    return smoothed

def create_readable_transcript_improved(segments):
    """Transcript amélioré - EXACTEMENT votre fonction"""
    
    if not segments:
        return "Aucune transcription disponible."
    
    # Statistiques par speaker
    speaker_stats = {}
    for segment in segments:
        speaker = segment["speaker"]
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                "total_time": 0,
                "segments_count": 0,
                "texts": [],
                "avg_confidence": 0
            }
        
        speaker_stats[speaker]["total_time"] += segment["duration"]
        speaker_stats[speaker]["segments_count"] += 1
        speaker_stats[speaker]["texts"].append(segment["text"])
        speaker_stats[speaker]["avg_confidence"] += segment["confidence"]
    
    # Calculer moyennes
    for speaker in speaker_stats:
        speaker_stats[speaker]["avg_confidence"] /= speaker_stats[speaker]["segments_count"]
        speaker_stats[speaker]["percentage"] = (speaker_stats[speaker]["total_time"] / 
                                               segments[-1]["end"] * 100)
    
    # Créer le transcript formaté
    transcript_lines = []
    transcript_lines.append("=== TRANSCRIPTION OPTIMISÉE ===\n")
    
    # Statistiques des speakers
    transcript_lines.append("📊 ANALYSE DES PARTICIPANTS:")
    for speaker, stats in speaker_stats.items():
        conf = int(stats["avg_confidence"] * 100)
        time_str = f"{stats['total_time']:.1f}s"
        percentage = f"{stats['percentage']:.1f}%"
        
        transcript_lines.append(f"🗣️ {speaker}: {time_str} ({percentage}) - Confiance: {conf}%")
    
    transcript_lines.append("\n" + "="*60)
    
    # Version chronologique avec qualité
    transcript_lines.append("📝 CONVERSATION CHRONOLOGIQUE:")
    current_speaker = None
    
    for segment in segments:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        confidence = int(segment["confidence"] * 100)
        
        # Marqueur de changement de speaker
        speaker_change = ""
        if segment["speaker"] != current_speaker:
            speaker_change = f"\n👤 {segment['speaker']} prend la parole:"
            current_speaker = segment["speaker"]
        
        quality_icon = "🔧" if segment.get("smoothed") else ""
        
        line = f"{speaker_change}\n[{start_time}-{end_time}] {segment['text']} ({confidence}%) {quality_icon}"
        transcript_lines.append(line)
    
    transcript_lines.append("\n" + "="*60)
    
    # Résumé par speaker
    transcript_lines.append("💬 RÉSUMÉ PAR PARTICIPANT:")
    for speaker, stats in speaker_stats.items():
        transcript_lines.append(f"\n🗣️ {speaker} ({stats['percentage']:.1f}% du temps):")
        full_text = " ".join(stats["texts"])
        # Diviser en phrases pour meilleure lisibilité
        sentences = full_text.replace(". ", ".\n   ").replace("? ", "?\n   ").replace("! ", "!\n   ")
        transcript_lines.append(f"   {sentences}")
    
    return "\n".join(transcript_lines)

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

async def process_transcription_optimized(audio_path: str, request: TranscriptionRequest):
    """Pipeline principal - BASÉ sur votre process_optimized"""
    start_time = datetime.now()
    
    try:
        print(f"📁 Traitement OPTIMISÉ: {audio_path}")
        if request.num_speakers:
            print(f"🎯 Paramètre: {request.num_speakers} speakers forcés")
        else:
            print(f"🔍 Paramètre: {request.min_speakers}-{request.max_speakers} speakers")
        
        # Nettoyer la mémoire GPU
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
        
        # Transcription avec large-v2 - EXACTEMENT comme votre setup
        print("🎯 Transcription LARGE-V2...")
        language = None if request.language == 'auto' else request.language
        
        transcription = whisper_model.transcribe(
            audio_path,
            language=language,
            fp16=True,
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            logprob_threshold=-1.0,
            compression_ratio_threshold=2.4,
            temperature=0.0,
            verbose=False
        )
        
        print("✅ Transcription terminée")
        
        # Si transcription seulement
        if request.transcription_only:
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return TranscriptionResponse(
                success=True,
                model="large-v2",
                transcription_brute=transcription["text"],
                parametres={
                    "transcription_only": True,
                    "language": request.language
                },
                statistiques={
                    "duree_totale": f"{transcription.get('duration', 0):.1f}s",
                    "nombre_segments": len(transcription["segments"]),
                    "language_detected": transcription.get("language", "unknown")
                },
                processing_time=processing_time
            )
        
        # Diarization optimisée - comme votre setup
        print("👥 Diarization optimisée...")
        diarization = optimize_diarization(
            audio_path, 
            num_speakers=request.num_speakers,
            min_speakers=request.min_speakers,
            max_speakers=request.max_speakers
        )
        
        # Fusion améliorée - comme votre setup
        print("🔗 Fusion intelligente...")
        merged_segments = merge_transcription_with_speakers_improved(
            transcription["segments"], 
            diarization
        )
        
        # Transcript lisible - comme votre setup
        readable_transcript = create_readable_transcript_improved(merged_segments)
        
        # Nettoyer la mémoire
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        speakers_detected = len(set(seg["speaker"] for seg in merged_segments if seg["speaker"] != "INCONNU"))
        
        return TranscriptionResponse(
            success=True,
            model="large-v2",
            transcription_brute=transcription["text"],
            transcription_formatee=readable_transcript,
            segments_detailles=merged_segments,
            parametres={
                "num_speakers_force": request.num_speakers,
                "min_speakers": request.min_speakers,
                "max_speakers": request.max_speakers,
                "language": request.language
            },
            statistiques={
                "speakers_detectes": speakers_detected,
                "speakers_inconnus": len([seg for seg in merged_segments if seg["speaker"] == "INCONNU"]),
                "duree_totale": f"{max(seg['end'] for seg in merged_segments) if merged_segments else 0:.1f}s",
                "nombre_segments": len(merged_segments),
                "confiance_moyenne": f"{sum(seg['confidence'] for seg in merged_segments) / len(merged_segments) * 100:.1f}%" if merged_segments else "0%",
                "segments_lisses": len([seg for seg in merged_segments if seg.get("smoothed")]),
                "language_detected": transcription.get("language", "unknown")
            },
            processing_time=processing_time
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
            success=False,
            model="large-v2",
            transcription_brute="",
            parametres={"error": True},
            statistiques={"processing_time": processing_time},
            processing_time=processing_time,
            error=error_msg
        )

# RunPod handler
async def handler(job):
    """Main handler pour RunPod serverless"""
    job_input = job.get("input", {})
    
    try:
        print(f"🚀 New job: {job.get('id', 'unknown')} (Whisper LARGE-V2 + PyAnnote)")
        
        # Validation des paramètres
        request = TranscriptionRequest(**job_input)
        print(f"📥 Processing: {request.audio_url}")
        print(f"🌍 Language: {request.language}")
        print(f"🎭 Speakers: {request.num_speakers or f'{request.min_speakers}-{request.max_speakers}'}")
        
        # Télécharger l'audio
        audio_path = await download_audio_file(request.audio_url)
        
        # Traitement
        result = await process_transcription_optimized(audio_path, request)
        
        # Cleanup du fichier temporaire
        try:
            os.unlink(audio_path)
            print(f"🗑️ Cleaned up {audio_path}")
        except Exception as e:
            print(f"⚠️ Could not delete temp file: {e}")
        
        # Retourner le résultat
        if result.error:
            return {
                "success": False,
                "error": result.error, 
                "processing_time": result.processing_time
            }
        else:
            return {
                "success": True,
                "model": result.model,
                "transcription_brute": result.transcription_brute,
                "transcription_formatee": result.transcription_formatee,
                "segments_detailles": result.segments_detailles,
                "parametres": result.parametres,
                "statistiques": result.statistiques,
                "processing_time": result.processing_time
            }
            
    except Exception as e:
        error_msg = f"Handler error: {str(e)}"
        print(f"❌ {error_msg}")
        print(traceback.format_exc())
        return {"success": False, "error": error_msg}

if __name__ == "__main__":
    print(f"🚀 Whisper LARGE-V2 + PyAnnote Serverless starting...")
    print(f"💾 Device: {'CUDA' if torch.cuda.is_available() else 'CPU'}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"🎮 GPU {i}: {torch.cuda.get_device_name(i)}")
            print(f"💾 GPU {i} Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    
    # Charger les modèles - comme votre setup
    try:
        print("🎤 Chargement des modèles...")
        load_models()
        print("✅ Modèles chargés avec succès")
    except Exception as e:
        print(f"❌ Erreur chargement modèles: {e}")
        raise
    
    print("🎯 Starting RunPod serverless handler...")
    runpod.serverless.start({"handler": handler})
