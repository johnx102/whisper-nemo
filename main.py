"""
Handler RunPod qui reproduit EXACTEMENT votre ancien pod qui marchait bien
Même logique simple + Whisper large-v2
"""

import runpod
import whisper
from pyannote.audio import Pipeline
import os
import tempfile
import logging
import requests
import torch

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🎮 Device: {device}")

# Variables globales - EXACTEMENT comme votre ancien pod
whisper_model = None
diarization_pipeline = None

def load_models():
    """Chargement EXACTEMENT comme votre ancien pod (avec large-v2)"""
    global whisper_model, diarization_pipeline
    
    if whisper_model is None:
        # WHISPER LARGE-V2 comme demandé (au lieu de base)
        logger.info("Chargement Whisper large-v2...")
        whisper_model = whisper.load_model("large-v2", device=device)
        logger.info("Whisper chargé ✅")
    
    if diarization_pipeline is None:
        logger.info("Chargement pyannote...")
        try:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            if not hf_token:
                logger.error("❌ HUGGINGFACE_TOKEN manquant")
                return
            
            # PYANNOTE 3.1 EXACTEMENT comme votre ancien pod
            diarization_pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1", 
                use_auth_token=hf_token  # EXACTEMENT comme votre ancien code
            )
            
            if torch.cuda.is_available():
                diarization_pipeline.to(device)
            
            logger.info("pyannote chargé ✅")
            
        except Exception as e:
            logger.error(f"❌ Erreur pyannote: {e}")
            diarization_pipeline = None

def download_audio(audio_url):
    """Télécharge l'audio pour RunPod"""
    try:
        logger.info(f"📥 Téléchargement: {audio_url}")
        
        response = requests.get(audio_url, timeout=120, stream=True)
        response.raise_for_status()
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            
            logger.info(f"✅ Téléchargé vers: {tmp_file.name}")
            return tmp_file.name, None
        
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement: {e}")
        return None, str(e)

def transcribe_only(audio_path):
    """Transcription seule - EXACTEMENT comme votre ancien /transcribe"""
    try:
        logger.info("🎯 Transcription Whisper large-v2...")
        
        # EXACTEMENT comme votre ancien code (juste large-v2 au lieu de base)
        result = whisper_model.transcribe(audio_path)
        
        logger.info("✅ Transcription terminée")
        
        return {
            "success": True,
            "transcription": result["text"],
            "segments": result["segments"]
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur transcription: {e}")
        return {"success": False, "error": str(e)}

def process_complete_like_before(audio_path):
    """Traitement complet - EXACTEMENT comme votre ancien /process"""
    try:
        logger.info("🎯 Traitement complet comme votre ancien pod...")
        
        if not diarization_pipeline:
            return {"success": False, "error": "Pipeline diarization non disponible"}
        
        # 1. Transcription - EXACTEMENT comme votre ancien code
        logger.info("📝 Transcription...")
        transcription = whisper_model.transcribe(audio_path)
        
        # 2. Diarization - EXACTEMENT comme votre ancien code  
        logger.info("👥 Diarization...")
        diarization = diarization_pipeline(audio_path)
        
        # 3. Extraction speakers - EXACTEMENT comme votre ancien code
        speakers = []
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speakers.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker
            })
        
        logger.info(f"✅ Traitement terminé - {len(speakers)} segments speakers")
        
        # 4. Retour EXACTEMENT comme votre ancien code - PAS de fusion
        return {
            "success": True,
            "transcription": transcription["text"],
            "segments": transcription["segments"],
            "speakers": speakers
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement: {e}")
        return {"success": False, "error": str(e)}

def handler(event):
    """
    Handler RunPod qui reproduit votre ancien pod
    
    Input format:
    {
        "input": {
            "audio_url": "https://example.com/audio.wav",
            "mode": "process"  // "transcribe" ou "process"
        }
    }
    """
    try:
        # Chargement des modèles
        load_models()
        
        # Extraction des paramètres
        job_input = event.get("input", {})
        audio_url = job_input.get("audio_url")
        mode = job_input.get("mode", "process")  # Par défaut mode complet
        
        if not audio_url:
            return {"error": "Paramètre 'audio_url' manquant dans input"}
        
        logger.info(f"🚀 Mode: {mode} - URL: {audio_url}")
        
        # Téléchargement
        audio_path, download_error = download_audio(audio_url)
        if download_error:
            return {"error": f"Erreur téléchargement: {download_error}"}
        
        try:
            if mode == "transcribe":
                # MODE TRANSCRIPTION SEULE - comme votre ancien /transcribe
                result = transcribe_only(audio_path)
                
                if result["success"]:
                    return {
                        "success": True,
                        "transcription": result["transcription"],
                        "segments": result["segments"],
                        "mode": "transcribe_only",
                        "model": "whisper-large-v2",
                        "device": str(device)
                    }
                else:
                    return {"error": result["error"]}
            
            else:
                # MODE COMPLET - comme votre ancien /process
                result = process_complete_like_before(audio_path)
                
                if result["success"]:
                    return {
                        "success": True,
                        "transcription": result["transcription"],
                        "segments": result["segments"],
                        "speakers": result["speakers"],
                        "mode": "complete_like_before",
                        "model": "whisper-large-v2 + pyannote-3.1",
                        "device": str(device),
                        "speakers_count": len(set(s["speaker"] for s in result["speakers"]))
                    }
                else:
                    return {"error": result["error"]}
            
        finally:
            # Nettoyage
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info("🗑️ Fichier temporaire supprimé")
        
    except Exception as e:
        logger.error(f"❌ Erreur handler: {e}")
        return {"error": f"Erreur interne: {str(e)}"}

if __name__ == "__main__":
    # Pré-chargement des modèles
    logger.info("🚀 Démarrage RunPod - Logique de votre ancien pod + Whisper large-v2")
    load_models()
    logger.info("✅ Prêt - Même simplicité que votre ancien pod qui marchait")
    
    # Démarrage du serveur RunPod
    runpod.serverless.start({"handler": handler})
