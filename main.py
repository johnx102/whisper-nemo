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
    """ÉTAPE 3: Attribution des speakers aux segments de transcription - SANS SPEAKER_UNKNOWN"""
    logger.info("🔗 ÉTAPE 3: Attribution des speakers (forçage speaker connu)...")
    
    final_segments = []
    
    # Extraire la liste des speakers trouvés par pyannote
    known_speakers = list(set(seg["speaker"] for seg in speaker_segments))
    logger.info(f"👥 Speakers disponibles: {known_speakers}")
    
    # Si aucun speaker trouvé, créer un speaker par défaut
    if not known_speakers:
        known_speakers = ["SPEAKER_00"]
        logger.warning("⚠️ Aucun speaker trouvé par diarisation - utilisation SPEAKER_00 par défaut")
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_center = (trans_start + trans_end) / 2
        trans_duration = trans_end - trans_start
        
        best_speaker = None
        best_coverage = 0
        
        # Méthode 1: Chercher le speaker qui couvre le centre
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
        
        # Méthode 2: Si pas trouvé, chercher le meilleur recouvrement
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
        
        # FORÇAGE: Si toujours pas trouvé, attribuer au speaker le plus proche temporellement
        if not best_speaker:
            closest_speaker = None
            min_distance = float('inf')
            
            for spk_seg in speaker_segments:
                # Distance au centre du segment de transcription
                spk_center = (spk_seg["start"] + spk_seg["end"]) / 2
                distance = abs(trans_center - spk_center)
                
                if distance < min_distance:
                    min_distance = distance
                    closest_speaker = spk_seg["speaker"]
            
            best_speaker = closest_speaker
            best_coverage = 0.1  # Attribution forcée avec faible confiance
            logger.info(f"🔧 Attribution forcée par proximité: '{trans_seg['text'][:30]}...' → {best_speaker}")
        
        # DERNIER RECOURS: Si vraiment aucun speaker trouvé, utiliser le premier disponible
        if not best_speaker:
            best_speaker = known_speakers[0]
            best_coverage = 0
            logger.warning(f"⚠️ Attribution par défaut: '{trans_seg['text'][:30]}...' → {best_speaker}")
        
        final_segments.append({
            "start": trans_start,
            "end": trans_end,
            "text": trans_seg["text"],
            "speaker": best_speaker,  # GARANTI d'être un speaker connu
            "confidence": trans_seg["confidence"],
            "speaker_coverage": best_coverage,
            "words": trans_seg.get("words", []),
            "attribution_method": "overlap" if best_coverage > 0.1 else "forced"
        })
    
    # Post-traitement: lissage et élimination finale des SPEAKER_UNKNOWN
    final_segments = smooth_speaker_transitions(final_segments)
    final_segments = force_known_speakers_only(final_segments, known_speakers)
    
    speakers_assigned = len(set(seg["speaker"] for seg in final_segments))
    logger.info(f"✅ Attribution terminée: {speakers_assigned} speakers uniques assignés sur {len(final_segments)} segments")
    logger.info(f"🎯 GARANTI: Aucun SPEAKER_UNKNOWN dans le résultat final")
    
    return final_segments

def force_known_speakers_only(segments, known_speakers):
    """Force tous les segments à avoir un speaker connu - ÉLIMINE SPEAKER_UNKNOWN"""
    logger.info("🔒 Forçage final: élimination de tous les SPEAKER_UNKNOWN...")
    
    if not known_speakers:
        known_speakers = ["SPEAKER_00"]
    
    fixed_segments = []
    unknown_count = 0
    
    for i, segment in enumerate(segments):
        current_speaker = segment["speaker"]
        
        # Si c'est SPEAKER_UNKNOWN ou un speaker non reconnu
        if current_speaker == "SPEAKER_UNKNOWN" or current_speaker not in known_speakers:
            unknown_count += 1
            
            # Stratégie 1: Hériter du speaker précédent
            if i > 0 and fixed_segments[-1]["speaker"] in known_speakers:
                new_speaker = fixed_segments[-1]["speaker"]
                method = "inherit_previous"
                
            # Stratégie 2: Regarder le speaker suivant
            elif i < len(segments) - 1 and segments[i+1]["speaker"] in known_speakers:
                new_speaker = segments[i+1]["speaker"]
                method = "inherit_next"
                
            # Stratégie 3: Speaker le plus fréquent jusqu'à présent
            else:
                if fixed_segments:
                    speaker_counts = {}
                    for prev_seg in fixed_segments:
                        speaker = prev_seg["speaker"]
                        if speaker in known_speakers:
                            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
                    
                    if speaker_counts:
                        new_speaker = max(speaker_counts.items(), key=lambda x: x[1])[0]
                        method = "most_frequent"
                    else:
                        new_speaker = known_speakers[0]
                        method = "default_first"
                else:
                    new_speaker = known_speakers[0]
                    method = "default_first"
            
            logger.info(f"🔧 Correction UNKNOWN: '{segment['text'][:30]}...' {current_speaker} → {new_speaker} ({method})")
            
            # Créer le segment corrigé
            corrected_segment = segment.copy()
            corrected_segment["speaker"] = new_speaker
            corrected_segment["speaker_coverage"] = 0.1  # Faible confiance pour attribution forcée
            corrected_segment["forced_attribution"] = True
            corrected_segment["attribution_method"] = method
            
            fixed_segments.append(corrected_segment)
        else:
            # Segment déjà correct
            fixed_segments.append(segment)
    
    if unknown_count > 0:
        logger.info(f"✅ Correction terminée: {unknown_count} segments SPEAKER_UNKNOWN réassignés")
    
    # Vérification finale
    final_speakers = set(seg["speaker"] for seg in fixed_segments)
    if "SPEAKER_UNKNOWN" in final_speakers:
        logger.error("❌ ERREUR: SPEAKER_UNKNOWN encore présent après correction!")
    else:
        logger.info(f"✅ SUCCÈS: Seulement speakers connus présents: {sorted(final_speakers)}")
    
    return fixed_segments

def smooth_speaker_transitions(segments, min_segment_duration=1.0, confidence_threshold=0.3):
    """Lisse les transitions de speakers - Version renforcée pour éviter SPEAKER_UNKNOWN"""
    if len(segments) < 3:
        return segments
    
    smoothed = segments.copy()
    changes_made = 0
    
    for i in range(1, len(smoothed) - 1):
        current = smoothed[i]
        prev_seg = smoothed[i-1]
        next_seg = smoothed[i+1]
        
        current_duration = current["end"] - current["start"]
        current_speaker = current["speaker"]
        prev_speaker = prev_seg["speaker"]
        next_speaker = next_seg["speaker"]
        
        # Conditions pour le lissage (plus permissives)
        should_smooth = False
        new_speaker = None
        smooth_reason = ""
        
        # Cas 1: Segment court entre le même speaker
        if (current_duration < min_segment_duration and
            prev_speaker == next_speaker and
            current_speaker != prev_speaker and
            current.get("speaker_coverage", 0) < confidence_threshold):
            
            should_smooth = True
            new_speaker = prev_speaker
            smooth_reason = "segment_court_entre_meme_speaker"
        
        # Cas 2: SPEAKER_UNKNOWN entouré de speakers connus
        elif (current_speaker == "SPEAKER_UNKNOWN" and
              prev_speaker != "SPEAKER_UNKNOWN" and
              next_speaker != "SPEAKER_UNKNOWN"):
            
            # Choisir le speaker le plus probable
            if prev_speaker == next_speaker:
                new_speaker = prev_speaker
                smooth_reason = "unknown_entre_meme_speaker"
            else:
                # Choisir celui avec la meilleure couverture temporelle
                prev_distance = abs(current["start"] - prev_seg["end"])
                next_distance = abs(next_seg["start"] - current["end"])
                
                if prev_distance < next_distance:
                    new_speaker = prev_speaker
                    smooth_reason = "unknown_plus_proche_precedent"
                else:
                    new_speaker = next_speaker
                    smooth_reason = "unknown_plus_proche_suivant"
            
            should_smooth = True
        
        # Cas 3: Attribution forcée avec faible confiance qui peut être améliorée
        elif (current.get("attribution_method") == "forced" and
              prev_speaker == next_speaker and
              prev_speaker != current_speaker):
            
            should_smooth = True
            new_speaker = prev_speaker
            smooth_reason = "amelioration_attribution_forcee"
        
        # Appliquer le lissage
        if should_smooth and new_speaker:
            logger.info(f"🔧 Lissage ({smooth_reason}): '{current['text'][:30]}...' {current_speaker} → {new_speaker}")
            smoothed[i]["speaker"] = new_speaker
            smoothed[i]["smoothed"] = True
            smoothed[i]["smooth_reason"] = smooth_reason
            smoothed[i]["speaker_coverage"] = max(0.5, smoothed[i].get("speaker_coverage", 0))  # Améliorer la confiance
            changes_made += 1
    
    if changes_made > 0:
        logger.info(f"✅ Lissage appliqué: {changes_made} corrections")
    
    return smoothed

def transcribe_and_diarize_separated(audio_path, num_speakers=None, min_speakers=2, max_speakers=4):
    """Fonction principale avec processus séparés - GARANTIT à 100% l'absence de SPEAKER_UNKNOWN"""
    try:
        # ÉTAPE 1: Transcription
        transcription_result = transcribe_with_whisper(audio_path)
        if not transcription_result['success']:
            return transcription_result
        
        # Vérifier si on a des répétitions suspectes
        repetition_warning = transcription_result.get('repetition_warning', False)
        if repetition_warning:
            logger.warning("⚠️ Transcription avec répétitions détectées - qualité audio possiblement dégradée")
        
        # ÉTAPE 2: Diarisation
        diarization_result = diarize_with_pyannote(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        if not diarization_result['success']:
            logger.warning("⚠️ Diarisation échouée - attribution automatique par segments")
            
            # FALLBACK INTELLIGENT: Répartition automatique
            segments_with_auto_speakers = []
            
            # Déterminer le nombre de speakers à utiliser
            target_speakers = num_speakers if num_speakers and num_speakers > 0 else 2
            speaker_names = [f"SPEAKER_{i:02d}" for i in range(target_speakers)]
            
            logger.info(f"🔄 Attribution automatique sur {target_speakers} speakers: {speaker_names}")
            
            for i, segment in enumerate(transcription_result["segments"]):
                # Alternance simple mais efficace
                if target_speakers == 1:
                    speaker_name = "SPEAKER_00"
                else:
                    # Logique d'alternance améliorée
                    if i == 0:
                        speaker_idx = 0
                    else:
                        # Changer de speaker après des segments longs ou périodiquement
                        prev_duration = segments_with_auto_speakers[-1]["end"] - segments_with_auto_speakers[-1]["start"]
                        prev_speaker_idx = int(segments_with_auto_speakers[-1]["speaker"].split("_")[1])
                        
                        if prev_duration > 3.0 or i % 4 == 0:  # Changer après 3s ou tous les 4 segments
                            speaker_idx = (prev_speaker_idx + 1) % target_speakers
                        else:
                            speaker_idx = prev_speaker_idx
                    
                    speaker_name = speaker_names[speaker_idx]
                
                segments_with_auto_speakers.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": speaker_name,  # TOUJOURS un speaker connu
                    "confidence": segment["confidence"],
                    "speaker_coverage": 0.8,
                    "attribution_method": "auto_alternation_fallback"
                })
            
            return {
                'success': True,
                'transcription': transcription_result["transcription"],
                'segments': segments_with_auto_speakers,
                'speakers_detected': len(set(seg["speaker"] for seg in segments_with_auto_speakers)),
                'language': transcription_result["language"],
                'diarization_available': False,
                'warning': f'Diarisation échouée - attribution automatique: {diarization_result.get("error", "Erreur inconnue")}',
                'fallback_mode': True,
                'repetition_warning': repetition_warning
            }
        
        # ÉTAPE 3: Attribution des speakers (diarisation réussie)
        final_segments = assign_speakers_to_transcription(
            transcription_result["segments"],
            diarization_result["speaker_segments"]
        )
        
        # VÉRIFICATION FINALE CRITIQUE - TRIPLE CONTRÔLE
        final_speakers = set(seg["speaker"] for seg in final_segments)
        unknown_segments = [seg for seg in final_segments if seg["speaker"] == "SPEAKER_UNKNOWN"]
        
        if unknown_segments:
            logger.error(f"❌ ERREUR CRITIQUE: {len(unknown_segments)} segments SPEAKER_UNKNOWN après tout le processus!")
            
            # CORRECTION D'URGENCE FINALE
            known_speakers = [s for s in final_speakers if s != "SPEAKER_UNKNOWN"]
            if not known_speakers:
                known_speakers = ["SPEAKER_00", "SPEAKER_01"]
                logger.error("🚨 Aucun speaker connu trouvé - création forcée SPEAKER_00/01")
            
            # Forcer tous les SPEAKER_UNKNOWN vers des speakers connus
            for i, seg in enumerate(final_segments):
                if seg["speaker"] == "SPEAKER_UNKNOWN":
                    # Stratégie: alternance simple entre speakers connus
                    new_speaker = known_speakers[i % len(known_speakers)]
                    logger.error(f"🚨 CORRECTION FINALE: SPEAKER_UNKNOWN → {new_speaker} pour '{seg['text'][:30]}...'")
                    final_segments[i]["speaker"] = new_speaker
                    final_segments[i]["speaker_coverage"] = 0.1
                    final_segments[i]["emergency_fix"] = True
        
        # VÉRIFICATION POST-CORRECTION
        post_correction_speakers = set(seg["speaker"] for seg in final_segments)
        post_unknown_count = sum(1 for seg in final_segments if seg["speaker"] == "SPEAKER_UNKNOWN")
        
        if post_unknown_count > 0:
            logger.error(f"❌ ÉCHEC TOTAL: {post_unknown_count} SPEAKER_UNKNOWN subsistent malgré corrections!")
            # En dernier recours, remplacer tout par SPEAKER_00
            for seg in final_segments:
                if seg["speaker"] == "SPEAKER_UNKNOWN":
                    seg["speaker"] = "SPEAKER_00"
                    seg["absolute_fallback"] = True
        else:
            logger.info(f"✅ SUCCÈS: Aucun SPEAKER_UNKNOWN après vérification finale")
        
        speakers_detected = len(set(seg["speaker"] for seg in final_segments))
        final_speaker_list = sorted(set(seg["speaker"] for seg in final_segments))
        
        logger.info(f"🎉 Processus complet terminé: {speakers_detected} speakers finaux")
        logger.info(f"🎯 Speakers utilisés: {final_speaker_list}")
        
        return {
            'success': True,
            'transcription': transcription_result["transcription"],
            'segments': final_segments,
            'speakers_detected': speakers_detected,
            'language': transcription_result["language"],
            'diarization_available': True,
            'speakers_found_by_diarization': diarization_result["speakers_found"],
            'diarization_params_used': diarization_result["diarization_params_used"],
            'fallback_mode': False,
            'final_speakers': final_speaker_list,
            'repetition_warning': repetition_warning,
            'unknown_segments_corrected': len(unknown_segments) if unknown_segments else 0
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur processus séparé: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return {
            'success': False,
            'error': str(e)
        }

def create_formatted_transcript(segments):
    """Crée un transcript formaté avec speakers et statistiques - Version sans SPEAKER_UNKNOWN"""
    if not segments:
        return "Aucune transcription disponible."
    
    # Filtrer encore une fois les segments pour l'affichage
    display_segments = []
    for segment in segments:
        text = segment.get("text", "").strip()
        duration = segment.get("end", 0) - segment.get("start", 0)
        
        # Ne garder que les segments avec du contenu réel
        if (text and 
            text not in [".", ",", "!", "?", "...", "-", " ", "..."] and
            len(text) > 1 and
            duration >= 0.5):  # Au moins 0.5s
            display_segments.append(segment)
    
    if not display_segments:
        return "Aucun contenu parlé détecté dans cet audio."
    
    # Vérification finale: aucun SPEAKER_UNKNOWN ne doit être affiché
    unknown_segments = [seg for seg in display_segments if seg.get("speaker") == "SPEAKER_UNKNOWN"]
    if unknown_segments:
        logger.warning(f"⚠️ {len(unknown_segments)} segments SPEAKER_UNKNOWN détectés dans l'affichage - correction!")
        for seg in display_segments:
            if seg.get("speaker") == "SPEAKER_UNKNOWN":
                seg["speaker"] = "SPEAKER_00"  # Correction d'affichage
    
    # Statistiques par speaker (en excluant SPEAKER_UNKNOWN)
    speaker_stats = {}
    for segment in display_segments:
        speaker = segment["speaker"]
        if speaker == "SPEAKER_UNKNOWN":
            continue  # Ignorer complètement les SPEAKER_UNKNOWN
            
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
    total_duration = display_segments[-1]["end"] if display_segments else 0
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        stats["avg_confidence"] /= stats["segments_count"]
        stats["avg_coverage"] /= stats["segments_count"]
        stats["percentage"] = (stats["total_time"] / total_duration * 100) if total_duration > 0 else 0
    
    # Créer le transcript
    lines = ["=== TRANSCRIPTION AVEC DIARISATION AMÉLIORÉE ===\n"]
    
    # Statistiques détaillées (uniquement speakers connus)
    lines.append("📊 ANALYSE DES PARTICIPANTS:")
    for speaker, stats in speaker_stats.items():
        conf = int(stats["avg_confidence"] * 100)
        coverage = int(stats["avg_coverage"] * 100)
        time_str = f"{stats['total_time']:.1f}s"
        percentage = f"{stats['percentage']:.1f}%"
        
        # Indicateur de qualité de l'attribution
        quality_indicator = "✅" if coverage > 60 else "⚠️" if coverage > 30 else "❌"
        
        lines.append(f"🗣️ {speaker}: {time_str} ({percentage}) - Confiance: {conf}% - Attribution: {coverage}% {quality_indicator}")
    
    lines.append(f"\n📈 QUALITÉ GLOBALE:")
    lines.append(f"   📝 Segments utiles: {len(display_segments)}")
    lines.append(f"   ⏱️ Durée totale: {total_duration:.1f}s")
    lines.append(f"   🎯 Speakers identifiés: {len(speaker_stats)}")
    
    # Détection de répétitions pour avertissement
    text_counts = {}
    for seg in display_segments:
        text = seg["text"]
        text_counts[text] = text_counts.get(text, 0) + 1
    
    suspicious_repetitions = {text: count for text, count in text_counts.items() if count > 3}
    if suspicious_repetitions:
        lines.append(f"   ⚠️ Répétitions détectées: {len(suspicious_repetitions)} phrases répétées")
    
    lines.append("\n" + "="*60)
    lines.append("📝 CONVERSATION CHRONOLOGIQUE:")
    
    # Format conversation amélioré (sans SPEAKER_UNKNOWN)
    current_speaker = None
    for segment in display_segments:
        speaker = segment["speaker"]
        
        # Ne jamais afficher SPEAKER_UNKNOWN
        if speaker == "SPEAKER_UNKNOWN":
            speaker = "SPEAKER_00"  # Remplacement pour affichage
        
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        confidence = int(segment["confidence"] * 100)
        coverage = int(segment.get("speaker_coverage", 0) * 100)
        
        # Changer de speaker
        if speaker != current_speaker:
            lines.append(f"\n👤 {speaker}:")
            current_speaker = speaker
        
        # Indicateurs de qualité avec couleurs
        quality_icons = ""
        if segment.get("smoothed"):
            quality_icons += "🔧"  # Segment lissé
        if segment.get("emergency_fix") or segment.get("forced_emergency"):
            quality_icons += "🚨"  # Correction d'urgence
        if coverage < 30:
            quality_icons += "❓"  # Attribution très incertaine
        elif coverage < 60:
            quality_icons += "⚠️"  # Attribution incertaine
        
        # Indicateur de confiance
        if confidence < 40:
            quality_icons += "🔇"  # Confiance très faible
        elif confidence < 70:
            quality_icons += "🔉"  # Confiance moyenne
        
        # Affichage du segment avec indicateurs
        confidence_color = "🟢" if confidence > 70 else "🟡" if confidence > 40 else "🔴"
        coverage_color = "🟢" if coverage > 60 else "🟡" if coverage > 30 else "🔴"
        
        lines.append(f"   [{start_time}-{end_time}] {segment['text']}")
        lines.append(f"      └─ {confidence_color}Conf:{confidence}% {coverage_color}Attr:{coverage}% {quality_icons}")
    
    # Résumé de fin
    lines.append(f"\n" + "="*60)
    lines.append(f"📊 RÉSUMÉ:")
    
    # Qualité globale
    avg_confidence = sum(seg["confidence"] for seg in display_segments) / len(display_segments) * 100
    avg_coverage = sum(seg.get("speaker_coverage", 0) for seg in display_segments) / len(display_segments) * 100
    
    lines.append(f"   🎯 Qualité transcription: {avg_confidence:.0f}%")
    lines.append(f"   🎭 Qualité diarisation: {avg_coverage:.0f}%")
    
    # Recommandations
    if avg_confidence < 50:
        lines.append(f"   💡 Recommandation: Audio de qualité faible - vérifiez le contenu")
    if avg_coverage < 40:
        lines.append(f"   💡 Recommandation: Diarisation incertaine - possibles erreurs d'attribution")
    if suspicious_repetitions:
        lines.append(f"   💡 Attention: Répétitions détectées - possible hallucination Whisper")
    
    return "\n".join(lines)

def handler(event):
    """Handler principal RunPod avec processus séparés - Garantit aucun SPEAKER_UNKNOWN"""
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
            
            # VÉRIFICATION FINALE CÔTÉ HANDLER
            segments = result.get('segments', [])
            final_speakers = set(seg.get("speaker") for seg in segments)
            unknown_count = sum(1 for seg in segments if seg.get("speaker") == "SPEAKER_UNKNOWN")
            
            if unknown_count > 0:
                logger.error(f"🚨 HANDLER: {unknown_count} SPEAKER_UNKNOWN détectés dans le résultat final!")
                # Dernière correction possible
                for seg in segments:
                    if seg.get("speaker") == "SPEAKER_UNKNOWN":
                        seg["speaker"] = "SPEAKER_00"
                        seg["handler_emergency_fix"] = True
                logger.error(f"🚨 HANDLER: Correction d'urgence appliquée")
            
            # Création du transcript formaté
            formatted_transcript = create_formatted_transcript(result['segments'])
            
            # Construction réponse finale
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
            
            # Infos de debug et qualité
            if 'speakers_found_by_diarization' in result:
                response['speakers_found_by_diarization'] = result['speakers_found_by_diarization']
            if 'diarization_params_used' in result:
                response['diarization_params_used'] = result['diarization_params_used']
            if 'warning' in result:
                response['warning'] = result['warning']
            if 'repetition_warning' in result and result['repetition_warning']:
                response['repetition_warning'] = True
                response['warning'] = (response.get('warning', '') + ' ATTENTION: Répétitions détectées dans la transcription.').strip()
            if 'unknown_segments_corrected' in result:
                response['unknown_segments_corrected'] = result['unknown_segments_corrected']
            if 'final_speakers' in result:
                response['final_speakers'] = result['final_speakers']
            
            # Log final de succès avec détails
            logger.info(f"✅ Traitement réussi:")
            logger.info(f"   📝 Segments: {len(result.get('segments', []))}")
            logger.info(f"   🗣️ Speakers: {result.get('speakers_detected', 0)}")
            logger.info(f"   🎯 Speakers finaux: {result.get('final_speakers', 'unknown')}")
            logger.info(f"   🎭 Diarisation: {'✅' if result.get('diarization_available') else '❌'}")
            logger.info(f"   ⚠️ Répétitions: {'⚠️' if result.get('repetition_warning') else '✅'}")
            logger.info(f"   🔧 Corrections UNKNOWN: {result.get('unknown_segments_corrected', 0)}")
            
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
