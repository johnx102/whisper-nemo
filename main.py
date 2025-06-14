import runpod
import whisper
from pyannote.audio import Pipeline
import os
import tempfile
import logging
from datetime import timedelta
import torch
import requests
import json
from urllib.parse import urlparse

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🎮 Device: {device}")

# Optimisations GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)
    
    # GPU warmup
    logger.info("🔥 Warmup GPU...")
    x = torch.randn(1000, 1000, device=device)
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    del x, y
    logger.info("✅ GPU warmed up")

# Variables globales pour les modèles (chargement au démarrage)
whisper_model = None
diarization_pipeline = None

def load_models():
    """Chargement des modèles au démarrage du container"""
    global whisper_model, diarization_pipeline
    
    if whisper_model is None:
        logger.info("🔄 Chargement Whisper large-v2...")
        whisper_model = whisper.load_model("large-v2", device=device)
        logger.info("✅ Whisper chargé")
    
    if diarization_pipeline is None:
        logger.info("🔄 Chargement pyannote...")
        try:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            logger.info(f"🔑 Token HF trouvé: {'Oui' if hf_token else 'Non'}")
            
            if hf_token:
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-2.1",
                    use_auth_token=hf_token
                )
                logger.info("✅ pyannote chargé avec token")
            else:
                logger.warning("⚠️ Pas de token HF - tentative sans token")
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-2.1"
                )
            
            if torch.cuda.is_available():
                diarization_pipeline.to(device)
                logger.info("✅ pyannote déplacé sur GPU")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement pyannote: {e}")
            logger.info("💡 Astuce: Définir HUGGINGFACE_TOKEN dans les variables d'environnement")
            diarization_pipeline = None

def format_timestamp(seconds):
    """Convertit les secondes en format mm:ss"""
    return str(timedelta(seconds=int(seconds)))[2:]

def download_audio(audio_url, max_size_mb=100):
    """Télécharge un fichier audio depuis une URL"""
    try:
        logger.info(f"📥 Téléchargement: {audio_url}")
        
        # Vérification préliminaire
        head_response = requests.head(audio_url, timeout=10)
        if head_response.status_code != 200:
            return None, f"URL non accessible: HTTP {head_response.status_code}"
        
        content_length = head_response.headers.get('content-length')
        if content_length:
            size_mb = int(content_length) / (1024 * 1024)
            if size_mb > max_size_mb:
                return None, f"Fichier trop volumineux: {size_mb:.1f}MB > {max_size_mb}MB"
        
        # Téléchargement
        response = requests.get(audio_url, timeout=120, stream=True)
        response.raise_for_status()
        
        # Déterminer l'extension
        content_type = response.headers.get('content-type', '')
        if 'audio/wav' in content_type:
            ext = '.wav'
        elif 'audio/mpeg' in content_type or 'audio/mp3' in content_type:
            ext = '.mp3'
        elif 'audio/mp4' in content_type or 'audio/m4a' in content_type:
            ext = '.m4a'
        else:
            # Fallback basé sur l'URL
            parsed_url = urlparse(audio_url)
            path_ext = os.path.splitext(parsed_url.path)[1].lower()
            ext = path_ext if path_ext in ['.wav', '.mp3', '.m4a', '.aac', '.flac'] else '.wav'
        
        # Sauvegarder dans un fichier temporaire
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            downloaded_size = 0
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded_size += len(chunk)
                    # Vérification de taille pendant le téléchargement
                    if downloaded_size > max_size_mb * 1024 * 1024:
                        tmp_file.close()
                        os.unlink(tmp_file.name)
                        return None, f"Fichier trop volumineux pendant téléchargement"
            
            temp_path = tmp_file.name
        
        logger.info(f"✅ Téléchargé: {downloaded_size/1024/1024:.1f}MB -> {temp_path}")
        return temp_path, None
        
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement: {e}")
        return None, str(e)

def transcribe_and_diarize(audio_path, num_speakers=None, min_speakers=1, max_speakers=4):
    """Effectue la transcription et diarization"""
    try:
        # Transcription avec Whisper
        logger.info("🎯 Transcription Whisper...")
        transcription_result = whisper_model.transcribe(
            audio_path,
            language='fr',
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=False,
            no_speech_threshold=0.6,
            temperature=0.0,
            verbose=False
        )
        
        if not diarization_pipeline:
            # Retour sans diarization si pyannote non disponible
            logger.warning("⚠️ Diarization indisponible - retour transcription seule")
            segments_with_speakers = []
            for segment in transcription_result["segments"]:
                segments_with_speakers.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"].strip(),
                    "speaker": "SPEAKER_00",
                    "confidence": 1 - segment.get("no_speech_prob", 0)
                })
            
            return {
                'success': True,
                'transcription': transcription_result["text"],
                'segments': segments_with_speakers,
                'speakers_detected': 1,
                'language': transcription_result.get("language", "fr"),
                'diarization_available': False
            }
        
        # Diarization avec pyannote
        logger.info("👥 Diarization pyannote...")
        diarization_params = {}
        if num_speakers:
            diarization_params['num_speakers'] = num_speakers
        else:
            diarization_params['min_speakers'] = min_speakers
            diarization_params['max_speakers'] = max_speakers
        
        diarization = diarization_pipeline(audio_path, **diarization_params)
        
        # Conversion diarization en segments
        speaker_segments = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        # Fusion transcription + diarization
        logger.info("🔗 Fusion transcription + diarization...")
        merged_segments = []
        
        for segment in transcription_result["segments"]:
            seg_start = segment["start"]
            seg_end = segment["end"]
            
            # Trouver le speaker avec le plus de recouvrement
            best_speaker = "SPEAKER_UNKNOWN"
            best_overlap = 0
            
            for spk_seg in speaker_segments:
                overlap_start = max(seg_start, spk_seg["start"])
                overlap_end = min(seg_end, spk_seg["end"])
                overlap = max(0, overlap_end - overlap_start)
                
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_speaker = spk_seg["speaker"]
            
            merged_segments.append({
                "start": seg_start,
                "end": seg_end,
                "text": segment["text"].strip(),
                "speaker": best_speaker,
                "confidence": 1 - segment.get("no_speech_prob", 0)
            })
        
        speakers_detected = len(set(seg["speaker"] for seg in merged_segments if seg["speaker"] != "SPEAKER_UNKNOWN"))
        
        logger.info(f"✅ Transcription + Diarization terminée - {speakers_detected} speakers détectés")
        
        return {
            'success': True,
            'transcription': transcription_result["text"],
            'segments': merged_segments,
            'speakers_detected': speakers_detected,
            'language': transcription_result.get("language", "fr"),
            'diarization_available': True
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur transcription/diarization: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def create_formatted_transcript(segments):
    """Crée un transcript formaté avec speakers"""
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
                "texts": []
            }
        
        duration = segment["end"] - segment["start"]
        speaker_stats[speaker]["total_time"] += duration
        speaker_stats[speaker]["segments_count"] += 1
        speaker_stats[speaker]["texts"].append(segment["text"])
    
    # Créer le transcript
    lines = ["=== TRANSCRIPTION AVEC DIARIZATION ===\n"]
    
    # Statistiques
    lines.append("📊 PARTICIPANTS:")
    total_duration = segments[-1]["end"] if segments else 0
    for speaker, stats in speaker_stats.items():
        percentage = (stats["total_time"] / total_duration * 100) if total_duration > 0 else 0
        lines.append(f"🗣️ {speaker}: {stats['total_time']:.1f}s ({percentage:.1f}%)")
    
    lines.append("\n" + "="*50)
    lines.append("📝 CONVERSATION:")
    
    # Format conversation
    current_speaker = None
    for segment in segments:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        confidence = int(segment["confidence"] * 100)
        
        if segment["speaker"] != current_speaker:
            lines.append(f"\n👤 {segment['speaker']}:")
            current_speaker = segment["speaker"]
        
        lines.append(f"[{start_time}-{end_time}] {segment['text']} ({confidence}%)")
    
    return "\n".join(lines)

def handler(event):
    """
    Handler principal pour RunPod Serverless
    Format d'entrée attendu:
    {
        "input": {
            "audio_url": "https://example.com/audio.wav",
            "num_speakers": 2,  # optionnel
            "min_speakers": 1,  # optionnel
            "max_speakers": 4   # optionnel
        }
    }
    """
    try:
        # Chargement des modèles si pas encore fait
        load_models()
        
        # Extraction des paramètres
        job_input = event.get("input", {})
        audio_url = job_input.get("audio_url")
        
        if not audio_url:
            return {
                "error": "Paramètre 'audio_url' manquant dans input"
            }
        
        num_speakers = job_input.get("num_speakers")
        min_speakers = job_input.get("min_speakers", 1)
        max_speakers = job_input.get("max_speakers", 4)
        
        logger.info(f"🚀 Début traitement: {audio_url}")
        logger.info(f"👥 Paramètres speakers: num={num_speakers}, min={min_speakers}, max={max_speakers}")
        
        # Téléchargement du fichier audio
        audio_path, download_error = download_audio(audio_url)
        if download_error:
            return {
                "error": f"Erreur téléchargement: {download_error}"
            }
        
        try:
            # Transcription + Diarization
            result = transcribe_and_diarize(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            if not result['success']:
                return {
                    "error": f"Erreur traitement: {result.get('error', 'Erreur inconnue')}"
                }
            
            # Création du transcript formaté
            formatted_transcript = create_formatted_transcript(result['segments'])
            
            # Retour au format RunPod
            return {
                "transcription": result['transcription'],
                "transcription_formatee": formatted_transcript,
                "segments": result['segments'],
                "speakers_detected": result['speakers_detected'],
                "language": result['language'],
                "diarization_available": result['diarization_available'],
                "device": str(device),
                "model": "whisper-large-v2",
                "success": True
            }
            
        finally:
            # Nettoyage du fichier temporaire
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info("🗑️ Fichier temporaire supprimé")
        
    except Exception as e:
        logger.error(f"❌ Erreur handler: {e}")
        return {
            "error": f"Erreur interne: {str(e)}"
        }

if __name__ == "__main__":
    # Pré-chargement des modèles au démarrage
    logger.info("🚀 Démarrage RunPod Serverless - Transcription + Diarization")
    load_models()
    logger.info("✅ Modèles chargés - Prêt pour les requêtes")
    
    # Démarrage du serveur RunPod
    runpod.serverless.start({"handler": handler})
