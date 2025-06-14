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
import time
import random
import traceback

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

# Variables globales pour les modèles
whisper_model = None
diarization_pipeline = None

def load_models():
    """Chargement des modèles avec gestion du rate limiting"""
    global whisper_model, diarization_pipeline
    
    if whisper_model is None:
        logger.info("🔄 Chargement Whisper large-v2...")
        try:
            whisper_model = whisper.load_model("large-v2", device=device)
            logger.info("✅ Whisper chargé")
        except Exception as e:
            logger.error(f"❌ Erreur chargement Whisper: {e}")
            try:
                logger.info("🔄 Tentative Whisper base...")
                whisper_model = whisper.load_model("base", device=device)
                logger.info("✅ Whisper base chargé en fallback")
            except Exception as e2:
                logger.error(f"❌ Échec total Whisper: {e2}")
                raise e2
    
    if diarization_pipeline is None:
        logger.info("🔄 Chargement pyannote diarization...")
        try:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            logger.info(f"🔑 Token HF trouvé: {'Oui' if hf_token else 'Non'}")
            
            if not hf_token:
                logger.error("❌ HUGGINGFACE_TOKEN manquant - diarization impossible")
                return
            
            # GESTION RATE LIMITING
            delay = random.uniform(2, 5)
            logger.info(f"⏱️ Délai anti-rate-limit: {delay:.1f}s")
            time.sleep(delay)
            
            model_name = "pyannote/speaker-diarization-3.1"
            logger.info(f"📥 Chargement du modèle: {model_name}")
            
            # Retry avec backoff exponentiel
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    diarization_pipeline = Pipeline.from_pretrained(
                        model_name,
                        use_auth_token=hf_token,
                        cache_dir="/tmp/hf_cache_persistent"
                    )
                    break
                    
                except Exception as e:
                    error_str = str(e)
                    if "429" in error_str or "Too Many Requests" in error_str:
                        wait_time = (2 ** attempt) * 10
                        logger.warning(f"⚠️ Rate limit HF (tentative {attempt+1}/{max_retries}), attente {wait_time}s...")
                        if attempt < max_retries - 1:
                            time.sleep(wait_time)
                            continue
                        else:
                            logger.error("❌ Rate limit HF persistant - abandon pyannote")
                            raise e
                    else:
                        raise e
            
            # Configuration GPU
            if torch.cuda.is_available():
                logger.info("🚀 Déplacement du pipeline vers GPU...")
                diarization_pipeline.to(device)
                
                try:
                    pipeline_device = next(diarization_pipeline.parameters()).device
                    logger.info(f"✅ Pipeline sur device: {pipeline_device}")
                except:
                    logger.warning("⚠️ Impossible de vérifier le device du pipeline")
            
            logger.info("✅ pyannote chargé et configuré")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement pyannote: {e}")
            if "429" in str(e):
                logger.info("💡 SOLUTION Rate Limit HuggingFace:")
                logger.info("   - Attendez quelques minutes avant de relancer")
                logger.info("   - Redémarrez le container RunPod")
                logger.info("   - Le service continuera en mode transcription seule")
            else:
                logger.info("💡 Vérifiez :")
                logger.info("   - HUGGINGFACE_TOKEN est défini")
                logger.info("   - Vous avez accepté les conditions")
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

def transcribe_with_whisper(audio_path):
    """ÉTAPE 1: Transcription seule avec Whisper"""
    try:
        logger.info("🎯 ÉTAPE 1: Transcription Whisper...")
        
        if not os.path.exists(audio_path):
            return {'success': False, 'error': f'Fichier audio introuvable: {audio_path}'}
        
        file_size = os.path.getsize(audio_path)
        logger.info(f"📁 Taille fichier: {file_size} bytes ({file_size/1024/1024:.2f}MB)")
        
        if file_size == 0:
            return {'success': False, 'error': 'Fichier audio vide'}
        
        logger.info("🎯 Lancement transcription...")
        
        try:
            # Méthode principale
            result = whisper_model.transcribe(
                audio_path,
                language='fr',
                fp16=torch.cuda.is_available(),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                temperature=0.0,
                verbose=False
            )
            
        except Exception as whisper_error:
            logger.warning(f"⚠️ Erreur transcription standard: {whisper_error}")
            
            # FALLBACK: Méthode simplifiée
            logger.info("🔄 Tentative transcription simplifiée...")
            try:
                result = whisper_model.transcribe(
                    audio_path,
                    language='fr',
                    verbose=False
                )
            except Exception as whisper_error2:
                logger.error(f"❌ Échec transcription simplifiée: {whisper_error2}")
                
                # FALLBACK 2: Transcription minimale
                logger.info("🔄 Tentative transcription minimale...")
                try:
                    result = whisper_model.transcribe(audio_path)
                except Exception as whisper_error3:
                    return {
                        'success': False,
                        'error': f'Échec total transcription: {whisper_error3}'
                    }
        
        logger.info(f"📊 Transcription réussie:")
        logger.info(f"   📝 Texte: '{result.get('text', '')[:100]}...'")
        logger.info(f"   🌍 Langue: {result.get('language', 'unknown')}")
        logger.info(f"   📈 Segments: {len(result.get('segments', []))}")
        
        # Nettoyage des segments
        segments_raw = result.get("segments", [])
        cleaned_segments = []
        
        for segment in segments_raw:
            text = segment.get("text", "").strip()
            if text:
                cleaned_segments.append({
                    "start": segment.get("start", 0),
                    "end": segment.get("end", 0),
                    "text": text,
                    "confidence": 1 - segment.get("no_speech_prob", 0),
                    "words": segment.get("words", [])
                })
        
        logger.info(f"✅ Transcription terminée: {len(cleaned_segments)} segments utiles")
        
        return {
            'success': True,
            'transcription': result.get("text", ""),
            'segments': cleaned_segments,
            'language': result.get("language", "fr")
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur transcription globale: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }

def diarize_with_pyannote(audio_path, num_speakers=None, min_speakers=2, max_speakers=4):
    """ÉTAPE 2: Diarisation seule avec pyannote"""
    try:
        if not diarization_pipeline:
            return {
                'success': False,
                'error': 'Pipeline de diarisation non disponible'
            }
        
        logger.info("👥 ÉTAPE 2: Diarisation pyannote...")
        
        diarization_params = {}
        
        if num_speakers and num_speakers > 0:
            diarization_params['num_speakers'] = num_speakers
            logger.info(f"🎯 Nombre fixe de speakers: {num_speakers}")
        else:
            diarization_params['min_speakers'] = max(1, min_speakers)
            diarization_params['max_speakers'] = min(6, max_speakers)
            logger.info(f"🔍 Auto-détection: {diarization_params['min_speakers']}-{diarization_params['max_speakers']} speakers")
        
        try:
            diarization = diarization_pipeline(audio_path, **diarization_params)
            logger.info("✅ Diarisation terminée")
        except Exception as e:
            logger.error(f"❌ Erreur diarization avec paramètres: {e}")
            logger.info("🔄 Tentative sans paramètres...")
            diarization = diarization_pipeline(audio_path)
            logger.info("✅ Diarisation fallback réussie")
        
        # Extraction des segments de speakers
        speaker_segments = []
        speakers_found = set()
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            speaker_segments.append({
                "start": turn.start,
                "end": turn.end,
                "speaker": speaker,
                "duration": turn.end - turn.start
            })
            speakers_found.add(speaker)
        
        speaker_segments.sort(key=lambda x: x["start"])
        
        logger.info(f"👥 Speakers trouvés: {sorted(list(speakers_found))}")
        logger.info(f"📊 Segments de diarisation: {len(speaker_segments)}")
        
        return {
            'success': True,
            'speaker_segments': speaker_segments,
            'speakers_found': sorted(list(speakers_found)),
            'diarization_params_used': diarization_params
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur diarisation: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def assign_speakers_to_transcription(transcription_segments, speaker_segments):
    """ÉTAPE 3: Attribution des speakers aux segments de transcription"""
    logger.info("🔗 ÉTAPE 3: Attribution des speakers...")
    
    final_segments = []
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_center = (trans_start + trans_end) / 2
        trans_duration = trans_end - trans_start
        
        best_speaker = None
        best_coverage = 0
        
        for spk_seg in speaker_segments:
            spk_start = spk_seg["start"]
            spk_end = spk_seg["end"]
            
            if spk_start <= trans_center <= spk_end:
                overlap_start = max(trans_start, spk_start)
                overlap_end = min(trans_end, spk_end)
                overlap = max(0, overlap_end - overlap_start)
                coverage = overlap / trans_duration if trans_duration > 0 else 0
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_speaker = spk_seg["speaker"]
        
        if not best_speaker:
            for spk_seg in speaker_segments:
                spk_start = spk_seg["start"]
                spk_end = spk_seg["end"]
                
                overlap_start = max(trans_start, spk_start)
                overlap_end = min(trans_end, spk_end)
                overlap = max(0, overlap_end - overlap_start)
                coverage = overlap / trans_duration if trans_duration > 0 else 0
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_speaker = spk_seg["speaker"]
        
        if not best_speaker:
            best_speaker = "SPEAKER_UNKNOWN"
            best_coverage = 0
        
        final_segments.append({
            "start": trans_start,
            "end": trans_end,
            "text": trans_seg["text"],
            "speaker": best_speaker,
            "confidence": trans_seg["confidence"],
            "speaker_coverage": best_coverage,
            "words": trans_seg.get("words", [])
        })
    
    final_segments = smooth_speaker_transitions(final_segments)
    
    speakers_assigned = len(set(seg["speaker"] for seg in final_segments if seg["speaker"] != "SPEAKER_UNKNOWN"))
    logger.info(f"✅ Attribution terminée: {speakers_assigned} speakers assignés sur {len(final_segments)} segments")
    
    return final_segments

def smooth_speaker_transitions(segments, min_segment_duration=1.0, confidence_threshold=0.5):
    """Lisse les transitions de speakers pour éviter les changements trop fréquents"""
    if len(segments) < 3:
        return segments
    
    smoothed = segments.copy()
    changes_made = 0
    
    for i in range(1, len(smoothed) - 1):
        current = smoothed[i]
        prev_seg = smoothed[i-1]
        next_seg = smoothed[i+1]
        
        current_duration = current["end"] - current["start"]
        
        if (current_duration < min_segment_duration and
            prev_seg["speaker"] == next_seg["speaker"] and
            current["speaker"] != prev_seg["speaker"] and
            current.get("speaker_coverage", 0) < confidence_threshold):
            
            logger.info(f"🔧 Lissage: '{current['text'][:30]}...' {current['speaker']} → {prev_seg['speaker']}")
            smoothed[i]["speaker"] = prev_seg["speaker"]
            smoothed[i]["smoothed"] = True
            changes_made += 1
    
    if changes_made > 0:
        logger.info(f"✅ Lissage appliqué: {changes_made} corrections")
    
    return smoothed

def transcribe_and_diarize_separated(audio_path, num_speakers=None, min_speakers=2, max_speakers=4):
    """Fonction principale avec processus séparés"""
    try:
        # ÉTAPE 1: Transcription
        transcription_result = transcribe_with_whisper(audio_path)
        if not transcription_result['success']:
            return transcription_result
        
        # ÉTAPE 2: Diarisation
        diarization_result = diarize_with_pyannote(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        if not diarization_result['success']:
            logger.warning("⚠️ Diarisation échouée - retour transcription seule")
            segments_without_speakers = []
            for segment in transcription_result["segments"]:
                segments_without_speakers.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": "SPEAKER_00",
                    "confidence": segment["confidence"]
                })
            
            return {
                'success': True,
                'transcription': transcription_result["transcription"],
                'segments': segments_without_speakers,
                'speakers_detected': 1,
                'language': transcription_result["language"],
                'diarization_available': False,
                'warning': f'Diarisation échouée: {diarization_result.get("error", "Erreur inconnue")}'
            }
        
        # ÉTAPE 3: Attribution des speakers
        final_segments = assign_speakers_to_transcription(
            transcription_result["segments"],
            diarization_result["speaker_segments"]
        )
        
        speakers_detected = len(set(seg["speaker"] for seg in final_segments if seg["speaker"] != "SPEAKER_UNKNOWN"))
        
        return {
            'success': True,
            'transcription': transcription_result["transcription"],
            'segments': final_segments,
            'speakers_detected': speakers_detected,
            'language': transcription_result["language"],
            'diarization_available': True,
            'speakers_found_by_diarization': diarization_result["speakers_found"],
            'diarization_params_used': diarization_result["diarization_params_used"]
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur processus séparé: {e}")
        return {
            'success': False,
            'error': str(e)
        }

def create_formatted_transcript(segments):
    """Crée un transcript formaté avec speakers et statistiques"""
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
                "avg_confidence": 0,
                "avg_coverage": 0
            }
        
        duration = segment["end"] - segment["start"]
        speaker_stats[speaker]["total_time"] += duration
        speaker_stats[speaker]["segments_count"] += 1
        speaker_stats[speaker]["texts"].append(segment["text"])
        speaker_stats[speaker]["avg_confidence"] += segment["confidence"]
        speaker_stats[speaker]["avg_coverage"] += segment.get("speaker_coverage", 0)
    
    # Calculer moyennes
    total_duration = segments[-1]["end"] if segments else 0
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        stats["avg_confidence"] /= stats["segments_count"]
        stats["avg_coverage"] /= stats["segments_count"]
        stats["percentage"] = (stats["total_time"] / total_duration * 100) if total_duration > 0 else 0
    
    # Créer le transcript
    lines = ["=== TRANSCRIPTION AVEC DIARISATION SÉPARÉE ===\n"]
    
    # Statistiques détaillées
    lines.append("📊 ANALYSE DES PARTICIPANTS:")
    for speaker, stats in speaker_stats.items():
        conf = int(stats["avg_confidence"] * 100)
        coverage = int(stats["avg_coverage"] * 100)
        time_str = f"{stats['total_time']:.1f}s"
        percentage = f"{stats['percentage']:.1f}%"
        lines.append(f"🗣️ {speaker}: {time_str} ({percentage}) - Confiance: {conf}% - Attribution: {coverage}%")
    
    lines.append("\n" + "="*60)
    lines.append("📝 CONVERSATION CHRONOLOGIQUE:")
    
    # Format conversation
    current_speaker = None
    for segment in segments:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        confidence = int(segment["confidence"] * 100)
        coverage = int(segment.get("speaker_coverage", 0) * 100)
        
        if segment["speaker"] != current_speaker:
            lines.append(f"\n👤 {segment['speaker']} prend la parole:")
            current_speaker = segment["speaker"]
        
        quality_icons = ""
        if segment.get("smoothed"):
            quality_icons += "🔧"
        if segment.get("speaker_coverage", 1) < 0.5:
            quality_icons += "⚠️"
        
        lines.append(f"[{start_time}-{end_time}] {segment['text']} (conf:{confidence}% attr:{coverage}%) {quality_icons}")
    
    return "\n".join(lines)

def handler(event):
    """Handler principal RunPod avec processus séparés"""
    try:
        # Chargement des modèles seulement si nécessaire
        if whisper_model is None or diarization_pipeline is None:
            logger.info("🔄 Chargement modèles manquants...")
            load_models()
        
        # Extraction des paramètres
        job_input = event.get("input", {})
        audio_url = job_input.get("audio_url")
        
        if not audio_url:
            return {"error": "Paramètre 'audio_url' manquant dans input"}
        
        num_speakers = job_input.get("num_speakers")
        min_speakers = job_input.get("min_speakers", 2)
        max_speakers = job_input.get("max_speakers", 3)
        
        logger.info(f"🚀 Début traitement: {audio_url}")
        logger.info(f"👥 Paramètres: num={num_speakers}, min={min_speakers}, max={max_speakers}")
        logger.info(f"🎮 Status modèles: Whisper={'✅' if whisper_model else '❌'} Pyannote={'✅' if diarization_pipeline else '❌'}")
        
        # Téléchargement
        audio_path, download_error = download_audio(audio_url)
        if download_error:
            return {"error": f"Erreur téléchargement: {download_error}"}
        
        try:
            # Transcription + Diarisation avec processus séparés
            result = transcribe_and_diarize_separated(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            if not result['success']:
                return {"error": f"Erreur traitement: {result.get('error', 'Erreur inconnue')}"}
            
            # Création du transcript formaté
            formatted_transcript = create_formatted_transcript(result['segments'])
            
            # Retour optimisé
            response = {
                "transcription": result['transcription'],
                "transcription_formatee": formatted_transcript,
                "segments": result['segments'],
                "speakers_detected": result['speakers_detected'],
                "language": result['language'],
                "diarization_available": result['diarization_available'],
                "device": str(device),
                "model": "whisper-large-v2" if whisper_model else "whisper-unavailable",
                "pyannote_model": "speaker-diarization-3.1" if diarization_pipeline else "unavailable",
                "processing_method": "separated_processes",
                "success": True
            }
            
            # Infos de debug optionnelles
            if 'speakers_found_by_diarization' in result:
                response['speakers_found_by_diarization'] = result['speakers_found_by_diarization']
            if 'diarization_params_used' in result:
                response['diarization_params_used'] = result['diarization_params_used']
            if 'warning' in result:
                response['warning'] = result['warning']
            
            logger.info(f"✅ Traitement réussi: {len(result.get('segments', []))} segments, {result.get('speakers_detected', 0)} speakers")
            return response
            
        finally:
            # Nettoyage
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.info("🗑️ Fichier temporaire supprimé")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Erreur nettoyage: {cleanup_error}")
        
    except Exception as e:
        logger.error(f"❌ Erreur handler: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return {"error": f"Erreur interne: {str(e)}"}

if __name__ == "__main__":
    logger.info("🚀 Démarrage RunPod Serverless - Transcription + Diarisation SÉPARÉE")
    logger.info("⏳ Chargement initial des modèles...")
    
    try:
        load_models()
        if whisper_model:
            logger.info("✅ Whisper prêt")
        else:
            logger.error("❌ Whisper non chargé")
            
        if diarization_pipeline:
            logger.info("✅ Pyannote prêt")
        else:
            logger.warning("⚠️ Pyannote non disponible - mode transcription seule")
            
        logger.info("✅ Modèles chargés - Prêt pour les requêtes")
        
    except Exception as startup_error:
        logger.error(f"❌ Erreur chargement initial: {startup_error}")
        logger.info("⚠️ Démarrage en mode dégradé - les modèles se chargeront à la première requête")
    
    # Démarrage du serveur RunPod
    runpod.serverless.start({"handler": handler})
