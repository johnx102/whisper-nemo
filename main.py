def handler(event):
    """Handler principal RunPod avec processus séparés - LOGIQUE ORIGINALE"""
    try:
        # Chargement des modèles - comme votre code original
        load_models()
        
        # Extraction des paramètres
        job_input = event.get("input", {})
        audio_url = job_input.get("audio_url")
        
        if not audio_url:
            return {"error": "Paramètre 'audio_url' manquant dans input"}
        
        # Paramètres par défaut optimisés comme votre code
        num_speakers = job_input.get("num_speakers")
        min_speakers = job_input.get("min_speakers", 2)
        max_speakers = job_input.get("max_speakers", 3)
        
        logger.info(f"🚀 Début traitement avec processus séparés: {audio_url}")
        logger.info(f"👥 Paramètres: num={num_speakers}, min={min_speakers}, max={max_speakers}")
        
        # Téléchargement
        audio_path, download_error = download_audio(audio_url)
        if download_error:
            return {"error": f"Erreur téléchargement: {download_error}"}
        
        try:
            # Transcription + Diarisation avec processus séparés (SEULE NOUVEAUTÉ)
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
            
            # Retour identique à votre code original
            return {
                "transcription": result['transcription'],
                "transcription_formatee": formatted_transcript,
                "segments": result['segments'],
                "speakers_detected": result['speakers_detected'],
                "language": result['language'],
                "diarization_available": result['diarization_available'],
                "device": str(device),
                "model": "whisper-large-v2",
                "pyannote_model": "speaker-diarization-3.1",
                "processing_method": "separated_processes",  # Seule différence
                "success": True,
                # Infos de debug
                "speakers_found_by_diarization": result.get('speakers_found_by_diarization', []),
                "diarization_params_used": result.get('diarization_params_used', {}),
                "warning": result.get('warning')
            }
            
        finally:
            # Nettoyage
            if os.path.exists(audio_path):
                os.unlink(audio_path)
                logger.info("🗑️ Fichier temporaire supprimé")
        
    except Exception as e:
        logger.error(f"❌ Erreur handler: {e}")
        return {"error": f"Erreur interne: {str(e)}"}"""
Handler RunPod Serverless pour Transcription + Diarization SÉPARÉE
Amélioration: processus séparés pour de meilleurs résultats
"""

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

# Variables globales pour les modèles
whisper_model = None
diarization_pipeline = None

def load_models():
    """Chargement des modèles - RETOUR À LA LOGIQUE ORIGINALE"""
    global whisper_model, diarization_pipeline
    
    if whisper_model is None:
        logger.info("🔄 Chargement Whisper large-v2...")
        whisper_model = whisper.load_model("large-v2", device=device)
        logger.info("✅ Whisper chargé")
    
    if diarization_pipeline is None:
        logger.info("🔄 Chargement pyannote diarization...")
        try:
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            logger.info(f"🔑 Token HF trouvé: {'Oui' if hf_token else 'Non'}")
            
            if not hf_token:
                logger.error("❌ HUGGINGFACE_TOKEN manquant - diarization impossible")
                return
            
            # RETOUR À VOTRE CODE ORIGINAL - simple et efficace
            model_name = "pyannote/speaker-diarization-3.1"
            logger.info(f"📥 Chargement du modèle: {model_name}")
            
            diarization_pipeline = Pipeline.from_pretrained(
                model_name,
                use_auth_token=hf_token
            )
            
            # GPU comme dans votre version
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
            logger.info("💡 Vérifiez :")
            logger.info("   - HUGGINGFACE_TOKEN est défini")
            logger.info("   - Vous avez accepté les conditions: https://huggingface.co/pyannote/speaker-diarization-3.1")
            diarization_pipeline = None

def format_timestamp(seconds):
    """Convertit les secondes en format mm:ss"""
    return str(timedelta(seconds=int(seconds)))[2:]

def download_audio(audio_url, max_size_mb=100):
    """Télécharge un fichier audio depuis une URL avec validation approfondie"""
    try:
        logger.info(f"📥 Téléchargement: {audio_url}")
        
        # Vérification préliminaire avec plus d'infos
        try:
            head_response = requests.head(audio_url, timeout=10, allow_redirects=True)
            logger.info(f"🔍 Status HTTP: {head_response.status_code}")
            logger.info(f"🔍 Headers: {dict(head_response.headers)}")
        except Exception as head_error:
            logger.warning(f"⚠️ HEAD request échoué: {head_error}, tentative GET direct")
            head_response = None
        
        if head_response and head_response.status_code != 200:
            return None, f"URL non accessible: HTTP {head_response.status_code}"
        
        if head_response:
            content_length = head_response.headers.get('content-length')
            if content_length:
                size_mb = int(content_length) / (1024 * 1024)
                logger.info(f"📏 Taille annoncée: {size_mb:.2f}MB")
                if size_mb > max_size_mb:
                    return None, f"Fichier trop volumineux: {size_mb:.1f}MB > {max_size_mb}MB"
        
        # Téléchargement avec validation
        logger.info("⬇️ Début téléchargement...")
        response = requests.get(audio_url, timeout=120, stream=True, allow_redirects=True)
        response.raise_for_status()
        
        # Déterminer l'extension avec plus de debug
        content_type = response.headers.get('content-type', '')
        logger.info(f"🎵 Content-Type: {content_type}")
        
        if 'audio/wav' in content_type or 'audio/x-wav' in content_type:
            ext = '.wav'
        elif 'audio/mpeg' in content_type or 'audio/mp3' in content_type:
            ext = '.mp3'
        elif 'audio/mp4' in content_type or 'audio/m4a' in content_type:
            ext = '.m4a'
        elif 'audio/ogg' in content_type:
            ext = '.ogg'
        elif 'audio/flac' in content_type:
            ext = '.flac'
        else:
            # Fallback basé sur l'URL
            parsed_url = urlparse(audio_url)
            path_ext = os.path.splitext(parsed_url.path)[1].lower()
            ext = path_ext if path_ext in ['.wav', '.mp3', '.m4a', '.aac', '.flac', '.ogg'] else '.wav'
            logger.info(f"🔍 Extension fallback depuis URL: {ext}")
        
        # Sauvegarder dans un fichier temporaire avec validation
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp_file:
            downloaded_size = 0
            chunk_count = 0
            
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
                    downloaded_size += len(chunk)
                    chunk_count += 1
                    
                    # Log progress pour gros fichiers
                    if chunk_count % 1000 == 0:
                        logger.info(f"📦 Téléchargé: {downloaded_size/1024/1024:.1f}MB...")
                    
                    if downloaded_size > max_size_mb * 1024 * 1024:
                        tmp_file.close()
                        os.unlink(tmp_file.name)
                        return None, f"Fichier trop volumineux pendant téléchargement"
            
            temp_path = tmp_file.name
        
        # Validation du fichier téléchargé
        final_size = os.path.getsize(temp_path)
        logger.info(f"✅ Téléchargé: {final_size/1024/1024:.2f}MB -> {temp_path}")
        
        # Vérifications supplémentaires
        if final_size < 44:  # En dessous de la taille d'un header WAV
            os.unlink(temp_path)
            return None, f"Fichier téléchargé trop petit: {final_size} bytes"
        
        # Tentative de validation basique du format audio
        try:
            with open(temp_path, 'rb') as f:
                header = f.read(12)
                logger.info(f"🔍 Header fichier: {header[:4]} / {header[8:12] if len(header) >= 12 else 'N/A'}")
                
                # Vérifications basiques
                if ext == '.wav' and not header.startswith(b'RIFF'):
                    logger.warning("⚠️ Fichier .wav sans header RIFF")
                elif ext == '.mp3' and not (header.startswith(b'ID3') or header[0:2] == b'\xff\xfb'):
                    logger.warning("⚠️ Fichier .mp3 suspect")
                    
        except Exception as validation_error:
            logger.warning(f"⚠️ Validation header échouée: {validation_error}")
        
        return temp_path, None
        
    except Exception as e:
        logger.error(f"❌ Erreur téléchargement: {e}")
        import traceback
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return None, str(e)

def transcribe_with_whisper(audio_path):
    """ÉTAPE 1: Transcription seule avec Whisper"""
    try:
        logger.info("🎯 ÉTAPE 1: Transcription Whisper large-v2...")
        
        # Vérifier le fichier audio avant traitement
        if not os.path.exists(audio_path):
            return {'success': False, 'error': f'Fichier audio introuvable: {audio_path}'}
        
        file_size = os.path.getsize(audio_path)
        logger.info(f"📁 Taille fichier: {file_size/1024/1024:.1f}MB")
        
        if file_size < 1024:  # Moins de 1KB
            return {'success': False, 'error': f'Fichier audio trop petit: {file_size} bytes'}
        
        # Transcription avec paramètres plus permissifs pour debug
        result = whisper_model.transcribe(
            audio_path,
            language='fr',
            fp16=torch.cuda.is_available(),
            condition_on_previous_text=False,
            no_speech_threshold=0.4,  # Plus permissif
            logprob_threshold=-1.5,   # Plus permissif
            compression_ratio_threshold=2.4,
            temperature=0.0,
            verbose=True,  # Plus de debug
            word_timestamps=False  # Désactiver pour debug
        )
        
        logger.info(f"🔍 Résultat brut Whisper:")
        logger.info(f"   - Texte complet: '{result.get('text', '')[:100]}...'")
        logger.info(f"   - Langue détectée: {result.get('language', 'unknown')}")
        logger.info(f"   - Nombre de segments: {len(result.get('segments', []))}")
        
        # Debug des segments
        segments_raw = result.get("segments", [])
        logger.info(f"✅ Transcription terminée: {len(segments_raw)} segments bruts")
        
        if len(segments_raw) == 0:
            logger.warning("⚠️ Aucun segment trouvé - fichier potentiellement silencieux")
            return {
                'success': True,
                'transcription': result.get("text", ""),
                'segments': [],
                'language': result.get("language", "fr"),
                'warning': 'Aucun contenu audio détecté'
            }
        
        # Nettoyage des segments avec plus de debug
        cleaned_segments = []
        for i, segment in enumerate(segments_raw):
            text = segment.get("text", "").strip()
            no_speech_prob = segment.get("no_speech_prob", 0)
            
            logger.info(f"   Segment {i}: '{text}' (no_speech: {no_speech_prob:.2f})")
            
            if text:  # Garder même les segments avec peu de confiance pour debug
                cleaned_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": text,
                    "confidence": 1 - no_speech_prob,
                    "no_speech_prob": no_speech_prob,
                    "words": segment.get("words", [])
                })
        
        logger.info(f"🧹 Segments nettoyés: {len(cleaned_segments)}")
        
        return {
            'success': True,
            'transcription': result.get("text", ""),
            'segments': cleaned_segments,
            'language': result.get("language", "fr")
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur transcription: {e}")
        import traceback
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
        
        # Paramètres optimisés pour de meilleurs résultats
        diarization_params = {}
        
        if num_speakers and num_speakers > 0:
            diarization_params['num_speakers'] = num_speakers
            logger.info(f"🎯 Nombre fixe de speakers: {num_speakers}")
        else:
            diarization_params['min_speakers'] = max(1, min_speakers)
            diarization_params['max_speakers'] = min(6, max_speakers)
            logger.info(f"🔍 Auto-détection: {diarization_params['min_speakers']}-{diarization_params['max_speakers']} speakers")
        
        # Exécution de la diarisation
        try:
            diarization = diarization_pipeline(audio_path, **diarization_params)
            logger.info("✅ Diarisation terminée")
        except Exception as e:
            logger.error(f"❌ Erreur diarization avec paramètres: {e}")
            # Fallback sans paramètres
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
        
        # Tri par temps de début
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
        
        # Méthode 1: Trouver le speaker qui couvre le centre du segment
        best_speaker = None
        best_coverage = 0
        
        for spk_seg in speaker_segments:
            spk_start = spk_seg["start"]
            spk_end = spk_seg["end"]
            
            # Vérifier si le centre du segment de transcription est dans ce segment de speaker
            if spk_start <= trans_center <= spk_end:
                # Calculer le pourcentage de recouvrement
                overlap_start = max(trans_start, spk_start)
                overlap_end = min(trans_end, spk_end)
                overlap = max(0, overlap_end - overlap_start)
                coverage = overlap / trans_duration if trans_duration > 0 else 0
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_speaker = spk_seg["speaker"]
        
        # Si aucun speaker trouvé avec le centre, prendre celui avec le plus de recouvrement
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
        
        # Attribution finale
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
    
    # Post-traitement: lissage des changements de speakers trop fréquents
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
        
        # Conditions pour le lissage
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
    """Fonction principale avec processus séparés et fallbacks robustes"""
    try:
        # ÉTAPE 1: Transcription (obligatoire)
        logger.info("="*50)
        transcription_result = transcribe_with_whisper(audio_path)
        if not transcription_result['success']:
            logger.error(f"❌ Transcription échouée: {transcription_result.get('error')}")
            return transcription_result
        
        # Vérifier qu'on a du contenu
        if not transcription_result.get('segments'):
            logger.warning("⚠️ Aucun segment transcrit - fichier probablement silencieux")
            return {
                'success': True,
                'transcription': transcription_result.get('transcription', ''),
                'segments': [],
                'speakers_detected': 0,
                'language': transcription_result.get('language', 'fr'),
                'diarization_available': False,
                'warning': 'Aucun contenu audio détecté'
            }
        
        # ÉTAPE 2: Diarisation (optionnelle - continue si échoue)
        logger.info("="*50)
        diarization_result = diarize_with_pyannote(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        if not diarization_result['success']:
            # FALLBACK: Retourner transcription seule avec speakers génériques
            logger.warning(f"⚠️ Diarisation échouée: {diarization_result.get('error')} - mode fallback")
            
            # Attribuer un speaker unique ou diviser artificiellement
            segments_with_generic_speakers = []
            
            if num_speakers and num_speakers > 1:
                # Diviser artificiellement en N speakers si demandé
                total_duration = transcription_result["segments"][-1]["end"] if transcription_result["segments"] else 0
                segment_per_speaker = len(transcription_result["segments"]) // num_speakers
                
                for i, segment in enumerate(transcription_result["segments"]):
                    speaker_index = min(i // max(1, segment_per_speaker), num_speakers - 1)
                    speaker_name = f"SPEAKER_{speaker_index:02d}"
                    
                    segments_with_generic_speakers.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speaker": speaker_name,
                        "confidence": segment["confidence"],
                        "speaker_coverage": 0.0,  # Pas de vraie diarisation
                        "artificial": True
                    })
                    
                speakers_detected = num_speakers
                
            else:
                # Un seul speaker générique
                for segment in transcription_result["segments"]:
                    segments_with_generic_speakers.append({
                        "start": segment["start"],
                        "end": segment["end"],
                        "text": segment["text"],
                        "speaker": "SPEAKER_00",
                        "confidence": segment["confidence"],
                        "speaker_coverage": 1.0,
                        "artificial": True
                    })
                    
                speakers_detected = 1
            
            return {
                'success': True,
                'transcription': transcription_result["transcription"],
                'segments': segments_with_generic_speakers,
                'speakers_detected': speakers_detected,
                'language': transcription_result["language"],
                'diarization_available': False,
                'warning': f'Diarisation échouée - speakers artificiels: {diarization_result.get("error", "Erreur inconnue")}',
                'fallback_mode': True
            }
        
        # ÉTAPE 3: Attribution des speakers (si diarisation réussie)
        logger.info("="*50)
        final_segments = assign_speakers_to_transcription(
            transcription_result["segments"],
            diarization_result["speaker_segments"]
        )
        
        speakers_detected = len(set(seg["speaker"] for seg in final_segments if seg["speaker"] != "SPEAKER_UNKNOWN"))
        
        logger.info(f"🎉 Processus complet terminé avec succès!")
        
        return {
            'success': True,
            'transcription': transcription_result["transcription"],
            'segments': final_segments,
            'speakers_detected': speakers_detected,
            'language': transcription_result["language"],
            'diarization_available': True,
            'speakers_found_by_diarization': diarization_result["speakers_found"],
            'diarization_params_used': diarization_result["diarization_params_used"],
            'fallback_mode': False
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur processus séparé: {e}")
        import traceback
        logger.error(f"🔍 Traceback complet: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e),
            'traceback': traceback.format_exc()
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
        
        # Indicateurs de qualité
        quality_icons = ""
        if segment.get("smoothed"):
            quality_icons += "🔧"  # Segment lissé
        if segment.get("speaker_coverage", 1) < 0.5:
            quality_icons += "⚠️"  # Attribution incertaine
        
        lines.append(f"[{start_time}-{end_time}] {segment['text']} (conf:{confidence}% attr:{coverage}%) {quality_icons}")
    
    return "\n".join(lines)

def handler(event):
    """Handler principal RunPod avec processus séparés et debug amélioré"""
    try:
        # Chargement des modèles avec gestion d'erreur
        logger.info("🚀 Initialisation des modèles...")
        try:
            load_models()
            logger.info("✅ Modèles initialisés")
        except Exception as model_error:
            logger.error(f"❌ Erreur chargement modèles: {model_error}")
            return {"error": f"Erreur initialisation: {model_error}"}
        
        # Extraction des paramètres avec validation
        job_input = event.get("input", {})
        audio_url = job_input.get("audio_url")
        
        if not audio_url:
            return {"error": "Paramètre 'audio_url' manquant dans input"}
        
        num_speakers = job_input.get("num_speakers")
        min_speakers = job_input.get("min_speakers", 2)
        max_speakers = job_input.get("max_speakers", 4)
        
        logger.info("="*60)
        logger.info(f"🚀 DÉBUT TRAITEMENT AVEC PROCESSUS SÉPARÉS")
        logger.info(f"🔗 URL: {audio_url}")
        logger.info(f"👥 Paramètres speakers: num={num_speakers}, min={min_speakers}, max={max_speakers}")
        logger.info(f"🎮 Device: {device}")
        logger.info(f"🤖 Whisper disponible: {'✅' if whisper_model else '❌'}")
        logger.info(f"🎭 Pyannote disponible: {'✅' if diarization_pipeline else '❌'}")
        logger.info("="*60)
        
        # Téléchargement avec validation approfondie
        audio_path, download_error = download_audio(audio_url)
        if download_error:
            logger.error(f"❌ Échec téléchargement: {download_error}")
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
                logger.error(f"❌ Échec traitement: {result.get('error')}")
                error_response = {"error": f"Erreur traitement: {result.get('error', 'Erreur inconnue')}"}
                if 'traceback' in result:
                    error_response['debug_traceback'] = result['traceback']
                return error_response
            
            # Création du transcript formaté
            formatted_transcript = create_formatted_transcript(result['segments'])
            
            # Construction de la réponse avec informations de debug
            response = {
                "transcription": result['transcription'],
                "transcription_formatee": formatted_transcript,
                "segments": result['segments'],
                "speakers_detected": result['speakers_detected'],
                "language": result['language'],
                "diarization_available": result['diarization_available'],
                "device": str(device),
                "model": "whisper-large-v2",
                "pyannote_model": "speaker-diarization-3.1" if diarization_pipeline else "unavailable",
                "processing_method": "separated_processes",
                "success": True,
                "fallback_mode": result.get('fallback_mode', False)
            }
            
            # Ajouter infos de debug si disponibles
            if 'speakers_found_by_diarization' in result:
                response['speakers_found_by_diarization'] = result['speakers_found_by_diarization']
            if 'diarization_params_used' in result:
                response['diarization_params_used'] = result['diarization_params_used']
            if 'warning' in result:
                response['warning'] = result['warning']
            
            # Logs de succès
            logger.info("="*60)
            logger.info("🎉 TRAITEMENT TERMINÉ AVEC SUCCÈS!")
            logger.info(f"📝 Transcription: {len(result.get('transcription', ''))} caractères")
            logger.info(f"🗣️ Segments: {len(result.get('segments', []))}")
            logger.info(f"👥 Speakers détectés: {result.get('speakers_detected', 0)}")
            logger.info(f"🎭 Diarisation: {'✅' if result.get('diarization_available') else '❌'}")
            logger.info(f"🔄 Mode fallback: {'✅' if result.get('fallback_mode') else '❌'}")
            if result.get('warning'):
                logger.info(f"⚠️ Avertissement: {result['warning']}")
            logger.info("="*60)
            
            return response
            
        finally:
            # Nettoyage obligatoire
            try:
                if audio_path and os.path.exists(audio_path):
                    os.unlink(audio_path)
                    logger.info("🗑️ Fichier temporaire supprimé")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Erreur nettoyage: {cleanup_error}")
        
    except Exception as e:
        logger.error(f"❌ ERREUR CRITIQUE HANDLER: {e}")
        import traceback
        full_traceback = traceback.format_exc()
        logger.error(f"🔍 Traceback complet: {full_traceback}")
        return {
            "error": f"Erreur interne critique: {str(e)}",
            "debug_traceback": full_traceback
        }

if __name__ == "__main__":
    # Pré-chargement des modèles
    logger.info("🚀 Démarrage RunPod Serverless - Transcription + Diarisation SÉPARÉE")
    load_models()
    logger.info("✅ Modèles chargés - Prêt pour les requêtes")
    
    # Démarrage du serveur RunPod
    runpod.serverless.start({"handler": handler})
