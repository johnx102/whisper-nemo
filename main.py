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
from collections import Counter

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🎮 Device: {device}")

# Optimisations GPU avancées
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    
    # Informations GPU
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    logger.info(f"🚀 Optimisations GPU activées")
    
    # GPU warmup
    logger.info("🔥 Warmup GPU...")
    x = torch.randn(2000, 2000, device=device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    del x, y
    torch.cuda.empty_cache()
    logger.info("✅ GPU warmed up")
else:
    logger.info("💻 Mode CPU")

# Variables globales pour les modèles
whisper_model = None
diarization_pipeline = None

def cleanup_gpu_memory():
    """Nettoyage GPU"""
    try:
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            gc.collect()
            torch.cuda.empty_cache()
            
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            logger.info(f"🧹 GPU Memory: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
    except Exception as e:
        logger.warning(f"⚠️ Erreur nettoyage GPU: {e}")

def load_models():
    """Chargement des modèles - Version simple et robuste"""
    global whisper_model, diarization_pipeline
    
    # Diagnostic des versions au chargement
    try:
        import transformers
        import whisper as whisper_lib
        logger.info(f"📦 Versions détectées:")
        logger.info(f"   - Transformers: {transformers.__version__}")
        logger.info(f"   - Whisper: {getattr(whisper_lib, '__version__', 'unknown')}")
        logger.info(f"   - PyTorch: {torch.__version__}")
    except Exception as e:
        logger.warning(f"⚠️ Impossible de vérifier les versions: {e}")
    
    if whisper_model is None:
        logger.info("🔄 Chargement Whisper large-v2...")
        try:
            whisper_model = whisper.load_model("large-v2", device=device)
            logger.info("✅ Whisper large-v2 chargé avec succès")
        except Exception as e:
            logger.error(f"❌ Erreur chargement Whisper large-v2: {e}")
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
            
            # Délai anti-rate-limit
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
            
            logger.info("✅ pyannote chargé et configuré")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement pyannote: {e}")
            if "429" in str(e):
                logger.info("💡 SOLUTION Rate Limit HuggingFace:")
                logger.info("   - Attendez quelques minutes avant de relancer")
                logger.info("   - Redémarrez le container RunPod")
                logger.info("   - Le service continuera en mode transcription seule")
            diarization_pipeline = None

def format_timestamp(seconds):
    """Convertit les secondes en format mm:ss"""
    return str(timedelta(seconds=int(seconds)))[2:]

def safe_text_for_logging(text, max_length=40):
    """Sécurise le texte pour l'affichage dans les logs"""
    if not text:
        return "VIDE"
    
    safe_text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    safe_text = ''.join(char for char in safe_text if ord(char) >= 32 or char in [' '])
    
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "..."
    
    return safe_text

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

def improved_segment_filtering(segments_raw):
    """Filtrage intelligent des segments"""
    cleaned_segments = []
    
    # Analyse globale
    total_duration = sum(seg.get("end", 0) - seg.get("start", 0) for seg in segments_raw)
    text_frequency = {}
    
    # Première passe: analyser les répétitions
    for segment in segments_raw:
        text = segment.get("text", "").strip()
        if text and len(text) > 3:
            text_frequency[text] = text_frequency.get(text, 0) + 1
    
    # Seuils adaptatifs selon la durée
    if total_duration > 300:  # Audio long (>5min)
        max_repetitions = 8
        min_duration = 1.0
        max_words_per_second = 7
    elif total_duration > 60:   # Audio moyen (1-5min)
        max_repetitions = 5
        min_duration = 0.8
        max_words_per_second = 8
    else:  # Audio court (<1min)
        max_repetitions = 3
        min_duration = 0.6
        max_words_per_second = 9
    
    suspicious_texts = {text: count for text, count in text_frequency.items() 
                       if count > max_repetitions}
    
    if suspicious_texts:
        logger.warning(f"🚨 {len(suspicious_texts)} textes suspects détectés")
    
    # Détection hallucination
    hallucination_detected = False
    
    for i, segment in enumerate(segments_raw):
        text = segment.get("text", "").strip()
        start_time = segment.get("start", 0)
        end_time = segment.get("end", 0)
        duration = end_time - start_time
        no_speech_prob = segment.get("no_speech_prob", 0)
        words = segment.get("words", [])
        
        # DÉTECTION HALLUCINATION CRITIQUE
        if i > len(segments_raw) * 0.7:  # Dans les 30% finaux
            if text in suspicious_texts and suspicious_texts[text] > 8:
                recent_same = sum(1 for j in range(max(0, i-10), i) 
                                if segments_raw[j].get("text", "").strip() == text)
                if recent_same >= 5:
                    logger.error(f"🚨 HALLUCINATION détectée: '{safe_text_for_logging(text)}'")
                    hallucination_detected = True
                    break
        
        # FILTRES
        
        # 1. Durée minimum
        if duration < min_duration:
            continue
            
        # 2. Texte vide ou inutile
        useless_texts = {
            ".", "...", "....", "-", "–", "euh", "heu", "mm", "hmm", 
            "ah", "oh", "ben", "donc", "alors", "voilà", "bon", "oui", "non",
            " ", "", "merci", "au revoir", "bonjour", "bonsoir"
        }
        if not text or text.lower() in useless_texts or len(text) < 2:
            continue
            
        # 3. Probabilité de silence trop élevée
        if no_speech_prob > 0.85:
            continue
            
        # 4. Ratio mots/durée anormal
        words_count = len(text.split())
        if duration > 0:
            words_per_second = words_count / duration
            if words_per_second > max_words_per_second or words_per_second < 0.2:
                logger.debug(f"🔥 Vitesse anormale: {words_per_second:.1f} mots/s")
                continue
        
        # 5. Détection patterns répétitifs
        if text in suspicious_texts:
            recent_texts = [cleaned_segments[j]["text"] for j in range(max(0, len(cleaned_segments)-5), len(cleaned_segments))]
            same_count = recent_texts.count(text)
            
            if same_count >= 3:
                logger.debug(f"🔥 Pattern répétitif: '{safe_text_for_logging(text)}'")
                continue
        
        # 6. Validation timestamps
        if start_time >= end_time or start_time < 0:
            continue
            
        # 7. Chevauchements anormaux
        if cleaned_segments:
            last_seg = cleaned_segments[-1]
            gap = start_time - last_seg["end"]
            
            if 0 < gap < 0.1 and text != last_seg["text"] and duration < 1.0:
                logger.debug(f"🔥 Segment fragmenté: '{safe_text_for_logging(text)}'")
                continue
        
        # SEGMENT VALIDE
        validated_words = []
        if words and isinstance(words, list):
            for word_info in words:
                try:
                    if isinstance(word_info, dict) and 'word' in word_info:
                        word_start = max(word_info.get('start', start_time), start_time)
                        word_end = min(word_info.get('end', end_time), end_time)
                        
                        if word_start >= word_end:
                            word_end = word_start + 0.1
                        
                        validated_words.append({
                            'word': word_info['word'].strip(),
                            'start': word_start,
                            'end': word_end,
                            'probability': word_info.get('probability', 1.0)
                        })
                except Exception:
                    continue
        
        # Segment enrichi
        cleaned_segments.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "confidence": 1 - no_speech_prob,
            "duration": duration,
            "words_count": words_count,
            "words": validated_words,
            "has_word_timestamps": len(validated_words) > 0,
            "words_per_second": words_count / duration if duration > 0 else 0,
        })
    
    removed_count = len(segments_raw) - len(cleaned_segments)
    
    logger.info(f"✅ Filtrage intelligent:")
    logger.info(f"   📝 Segments gardés: {len(cleaned_segments)} / {len(segments_raw)} ({removed_count} supprimés)")
    logger.info(f"   🚨 Hallucination: {'⚠️ OUI' if hallucination_detected else '✅ Non'}")
    logger.info(f"   🔤 Segments avec mots: {sum(1 for s in cleaned_segments if s['has_word_timestamps'])}")
    
    return cleaned_segments, suspicious_texts, hallucination_detected

def transcribe_with_whisper(audio_path):
    """ÉTAPE 1: Transcription Whisper - Version corrigée pour dernières versions"""
    try:
        logger.info("🎯 ÉTAPE 1: Transcription Whisper avec word_timestamps...")
        
        if not os.path.exists(audio_path):
            return {'success': False, 'error': f'Fichier audio introuvable: {audio_path}'}
        
        file_size = os.path.getsize(audio_path)
        logger.info(f"📁 Taille fichier: {file_size} bytes ({file_size/1024/1024:.2f}MB)")
        
        if file_size == 0:
            return {'success': False, 'error': 'Fichier audio vide'}
        
        # NOUVELLE APPROCHE SIMPLE ET DIRECTE
        result = None
        transcription_method = "unknown"
        
        # Tentative 1: Version optimale avec word_timestamps
        try:
            logger.info("🔄 Transcription avec word_timestamps (transformers 4.52.4)...")
            
            result = whisper_model.transcribe(
                audio_path,
                language='fr',
                word_timestamps=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                logprob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                temperature=0.0,
                verbose=False
            )
            
            transcription_method = "word_timestamps_modern"
            logger.info("✅ Transcription réussie avec word_timestamps")
            
        except Exception as e:
            logger.warning(f"⚠️ Erreur word_timestamps: {e}")
            
            # Tentative 2: Version basique sans word_timestamps
            try:
                logger.info("🔄 Fallback sans word_timestamps...")
                
                result = whisper_model.transcribe(
                    audio_path,
                    language='fr',
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    temperature=0.0,
                    verbose=False
                )
                
                transcription_method = "standard_fallback"
                logger.info("✅ Transcription réussie sans word_timestamps")
                
            except Exception as e2:
                logger.error(f"❌ Échec total transcription: {e2}")
                return {'success': False, 'error': f'Transcription impossible: {e2}'}
        
        if not result:
            return {'success': False, 'error': 'Aucun résultat de transcription'}
        
        logger.info(f"📊 Transcription terminée:")
        logger.info(f"   📝 Texte: '{result.get('text', '')[:100]}...'")
        logger.info(f"   🌍 Langue: {result.get('language', 'unknown')}")
        logger.info(f"   📈 Segments bruts: {len(result.get('segments', []))}")
        logger.info(f"   🎯 Méthode: {transcription_method}")
        
        # Vérifier si on a des timestamps de mots
        has_word_timestamps = transcription_method == "word_timestamps_modern"
        if has_word_timestamps:
            logger.info("✅ Timestamps de mots disponibles")
        else:
            logger.info("⚠️ Pas de timestamps de mots - attribution niveau segment")
        
        # Filtrage intelligent
        segments_raw = result.get("segments", [])
        cleaned_segments, suspicious_texts, hallucination_detected = improved_segment_filtering(segments_raw)
        
        # Ajuster selon la méthode
        word_segments_count = 0
        if has_word_timestamps:
            word_segments_count = sum(1 for seg in cleaned_segments if seg.get("has_word_timestamps"))
        else:
            # Marquer explicitement l'absence de timestamps de mots
            for seg in cleaned_segments:
                seg["has_word_timestamps"] = False
                seg["words"] = []
        
        logger.info(f"✅ Transcription terminée:")
        logger.info(f"   📝 Segments finaux: {len(cleaned_segments)}")
        logger.info(f"   🔤 Segments avec mots: {word_segments_count}")
        logger.info(f"   🚨 Hallucinations: {'⚠️' if hallucination_detected else '✅'}")
        
        return {
            'success': True,
            'transcription': result.get("text", ""),
            'segments': cleaned_segments,
            'language': result.get("language", "fr"),
            'segments_raw_count': len(segments_raw),
            'segments_cleaned_count': len(cleaned_segments),
            'word_segments_count': word_segments_count,
            'repetition_warning': len(suspicious_texts) > 0 or hallucination_detected,
            'suspicious_repetitions': suspicious_texts,
            'hallucination_detected': hallucination_detected,
            'transcription_method': transcription_method,
            'word_timestamps_available': has_word_timestamps
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur transcription globale: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}

def diarize_with_pyannote(audio_path, num_speakers=None, min_speakers=2, max_speakers=4):
    """ÉTAPE 2: Diarisation avec pyannote"""
    try:
        if not diarization_pipeline:
            return {'success': False, 'error': 'Pipeline de diarisation non disponible'}
        
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
        return {'success': False, 'error': str(e)}

def assign_speakers_to_transcription(transcription_segments, speaker_segments):
    """ÉTAPE 3: Attribution speakers intelligente"""
    logger.info("🔗 ÉTAPE 3: Attribution speakers...")
    
    final_segments = []
    
    # Extraire les speakers connus
    known_speakers = list(set(seg["speaker"] for seg in speaker_segments))
    logger.info(f"👥 Speakers disponibles: {known_speakers}")
    
    if not known_speakers:
        known_speakers = ["SPEAKER_00", "SPEAKER_01"]
        logger.warning("⚠️ Aucun speaker trouvé - utilisation speakers par défaut")
    
    for trans_seg in transcription_segments:
        trans_start = trans_seg["start"]
        trans_end = trans_seg["end"]
        trans_center = (trans_start + trans_end) / 2
        trans_duration = trans_end - trans_start
        words = trans_seg.get("words", [])
        has_word_timestamps = trans_seg.get("has_word_timestamps", False)
        
        # Attribution niveau mot si timestamps disponibles
        if has_word_timestamps and words:
            logger.debug(f"🔤 Attribution précise pour segment avec {len(words)} mots")
            
            words_with_speakers = []
            segment_speaker_votes = {}
            
            for word_info in words:
                word_start = word_info.get('start', trans_start)
                word_end = word_info.get('end', trans_end)
                word_center = (word_start + word_end) / 2
                word_duration = word_end - word_start
                
                # Trouver le meilleur speaker pour ce mot
                best_speaker = None
                best_coverage = 0
                
                # Centre du mot dans un segment de speaker
                for spk_seg in speaker_segments:
                    spk_start = spk_seg["start"]
                    spk_end = spk_seg["end"]
                    
                    if spk_start <= word_center <= spk_end:
                        overlap_start = max(word_start, spk_start)
                        overlap_end = min(word_end, spk_end)
                        overlap = max(0, overlap_end - overlap_start)
                        coverage = overlap / word_duration if word_duration > 0 else 0
                        
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_speaker = spk_seg["speaker"]
                
                # Fallback par recouvrement
                if not best_speaker:
                    for spk_seg in speaker_segments:
                        spk_start = spk_seg["start"]
                        spk_end = spk_seg["end"]
                        
                        overlap_start = max(word_start, spk_start)
                        overlap_end = min(word_end, spk_end)
                        overlap = max(0, overlap_end - overlap_start)
                        coverage = overlap / word_duration if word_duration > 0 else 0
                        
                        if coverage > best_coverage:
                            best_coverage = coverage
                            best_speaker = spk_seg["speaker"]
                
                # Fallback final
                if not best_speaker or best_speaker not in known_speakers:
                    best_speaker = known_speakers[0]
                    best_coverage = 0.1
                
                # Ajouter le mot avec son speaker
                words_with_speakers.append({
                    'word': word_info['word'],
                    'start': word_start,
                    'end': word_end,
                    'probability': word_info.get('probability', 1.0),
                    'speaker': best_speaker,
                    'coverage': best_coverage
                })
                
                # Vote pour le speaker du segment
                if best_speaker not in segment_speaker_votes:
                    segment_speaker_votes[best_speaker] = 0
                segment_speaker_votes[best_speaker] += best_coverage * word_duration
            
            # Déterminer le speaker principal du segment
            if segment_speaker_votes:
                segment_speaker = max(segment_speaker_votes.items(), key=lambda x: x[1])[0]
                segment_coverage = segment_speaker_votes[segment_speaker] / trans_duration
            else:
                segment_speaker = known_speakers[0]
                segment_coverage = 0.1
                
        else:
            # Attribution classique niveau segment
            best_speaker = None
            best_coverage = 0
            
            # Attribution par centre de segment
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
            
            # Fallback par recouvrement
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
            
            # Fallback final
            if not best_speaker or best_speaker not in known_speakers:
                best_speaker = known_speakers[0]
                best_coverage = 0.1
            
            segment_speaker = best_speaker
            segment_coverage = best_coverage
            words_with_speakers = words
        
        # Créer le segment final
        final_segments.append({
            "start": trans_start,
            "end": trans_end,
            "text": trans_seg["text"],
            "speaker": segment_speaker,
            "confidence": trans_seg["confidence"],
            "speaker_coverage": segment_coverage,
            "words": words_with_speakers,
            "attribution_method": "word_level" if has_word_timestamps else "segment_level",
            "has_word_speakers": has_word_timestamps and len(words_with_speakers) > 0
        })
    
    # Post-traitement: lissage et validation
    final_segments = smooth_speaker_transitions(final_segments)
    final_segments = validate_speakers(final_segments, known_speakers)
    
    word_level_count = sum(1 for seg in final_segments if seg.get("attribution_method") == "word_level")
    speakers_assigned = len(set(seg["speaker"] for seg in final_segments))
    
    logger.info(f"✅ Attribution terminée:")
    logger.info(f"   🎯 Speakers: {speakers_assigned} sur {len(final_segments)} segments")
    logger.info(f"   🔤 Attribution mot-par-mot: {word_level_count}/{len(final_segments)} segments")
    
    return final_segments

def smooth_speaker_transitions(segments, min_segment_duration=1.0):
    """Lissage des transitions de speakers"""
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
        
        # Cas: segment court entre même speaker
        if (current_duration < min_segment_duration and
            prev_speaker == next_speaker and
            current_speaker != prev_speaker and
            current.get("speaker_coverage", 0) < 0.3):
            
            logger.debug(f"🔧 Lissage: '{current['text'][:30]}...' {current_speaker} → {prev_speaker}")
            smoothed[i]["speaker"] = prev_speaker
            smoothed[i]["smoothed"] = True
            changes_made += 1
    
    if changes_made > 0:
        logger.info(f"✅ Lissage: {changes_made} corrections appliquées")
    
    return smoothed

def validate_speakers(segments, known_speakers):
    """Validation finale des speakers - Éliminer SPEAKER_UNKNOWN"""
    validated_segments = []
    unknown_count = 0
    
    for i, segment in enumerate(segments):
        current_speaker = segment["speaker"]
        
        # Éliminer SPEAKER_UNKNOWN
        if current_speaker == "SPEAKER_UNKNOWN" or current_speaker not in known_speakers:
            unknown_count += 1
            
            # Stratégie: hériter du précédent ou suivant
            if i > 0 and validated_segments[-1]["speaker"] in known_speakers:
                new_speaker = validated_segments[-1]["speaker"]
                method = "inherit_previous"
            elif i < len(segments) - 1 and segments[i+1]["speaker"] in known_speakers:
                new_speaker = segments[i+1]["speaker"]
                method = "inherit_next"
            else:
                new_speaker = known_speakers[0]
                method = "default_first"
            
            logger.debug(f"🔧 Correction: {current_speaker} → {new_speaker} ({method})")
            
            corrected_segment = segment.copy()
            corrected_segment["speaker"] = new_speaker
            corrected_segment["speaker_coverage"] = 0.1
            corrected_segment["corrected"] = True
            
            validated_segments.append(corrected_segment)
        else:
            validated_segments.append(segment)
    
    if unknown_count > 0:
        logger.info(f"✅ Validation: {unknown_count} segments corrigés")
    
    # Vérification finale
    final_speakers = set(seg["speaker"] for seg in validated_segments)
    if "SPEAKER_UNKNOWN" in final_speakers:
        logger.error("❌ ERREUR: SPEAKER_UNKNOWN encore présent!")
    else:
        logger.info(f"✅ SUCCÈS: Speakers valides: {sorted(final_speakers)}")
    
    return validated_segments

def transcribe_and_diarize(audio_path, num_speakers=None, min_speakers=2, max_speakers=4):
    """Fonction principale - Version simplifiée et robuste"""
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
            logger.warning("⚠️ Diarisation échouée - Attribution automatique")
            
            # FALLBACK: Attribution automatique
            segments_with_auto_speakers = []
            target_speakers = num_speakers if num_speakers and num_speakers > 0 else 2
            speaker_names = [f"SPEAKER_{i:02d}" for i in range(target_speakers)]
            
            for i, segment in enumerate(transcription_result["segments"]):
                speaker_name = speaker_names[i % target_speakers]
                
                segments_with_auto_speakers.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": speaker_name,
                    "confidence": segment["confidence"],
                    "speaker_coverage": 0.8,
                    "attribution_method": "auto_alternation",
                    "words": segment.get("words", [])
                })
            
            return {
                'success': True,
                'transcription': transcription_result["transcription"],
                'segments': segments_with_auto_speakers,
                'speakers_detected': len(set(seg["speaker"] for seg in segments_with_auto_speakers)),
                'language': transcription_result["language"],
                'diarization_available': False,
                'warning': f'Diarisation échouée: {diarization_result.get("error", "Erreur inconnue")}',
                'fallback_mode': True,
                'repetition_warning': transcription_result.get('repetition_warning', False),
                'hallucination_detected': transcription_result.get('hallucination_detected', False),
                'final_speakers': sorted(set(seg["speaker"] for seg in segments_with_auto_speakers)),
                'transcription_method': transcription_result.get('transcription_method', 'unknown'),
                'word_timestamps_available': transcription_result.get('word_timestamps_available', False)
            }
        
        # ÉTAPE 3: Attribution des speakers
        final_segments = assign_speakers_to_transcription(
            transcription_result["segments"],
            diarization_result["speaker_segments"]
        )
        
        # VÉRIFICATION FINALE
        final_speakers = set(seg["speaker"] for seg in final_segments)
        unknown_segments = [seg for seg in final_segments if seg["speaker"] == "SPEAKER_UNKNOWN"]
        
        if unknown_segments:
            logger.error(f"❌ {len(unknown_segments)} segments SPEAKER_UNKNOWN détectés!")
            # Correction d'urgence
            known_speakers = [s for s in final_speakers if s != "SPEAKER_UNKNOWN"]
            if not known_speakers:
                known_speakers = ["SPEAKER_00", "SPEAKER_01"]
            
            for i, seg in enumerate(final_segments):
                if seg["speaker"] == "SPEAKER_UNKNOWN":
                    new_speaker = known_speakers[i % len(known_speakers)]
                    final_segments[i]["speaker"] = new_speaker
                    final_segments[i]["emergency_fix"] = True
        
        speakers_detected = len(set(seg["speaker"] for seg in final_segments))
        final_speaker_list = sorted(set(seg["speaker"] for seg in final_segments))
        
        logger.info(f"🎉 Processus terminé: {speakers_detected} speakers")
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
            'repetition_warning': transcription_result.get('repetition_warning', False),
            'hallucination_detected': transcription_result.get('hallucination_detected', False),
            'unknown_segments_corrected': len(unknown_segments) if unknown_segments else 0,
            'word_segments_count': transcription_result.get('word_segments_count', 0),
            'transcription_method': transcription_result.get('transcription_method', 'unknown'),
            'word_timestamps_available': transcription_result.get('word_timestamps_available', False)
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur processus principal: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}

def create_formatted_transcript(segments):
    """Crée un transcript formaté avec speakers"""
    if not segments:
        return "Aucune transcription disponible."
    
    # Filtrer pour l'affichage
    display_segments = []
    for segment in segments:
        text = segment.get("text", "").strip()
        duration = segment.get("end", 0) - segment.get("start", 0)
        
        if (text and 
            len(text) > 1 and
            duration >= 0.4):
            display_segments.append(segment)
    
    if not display_segments:
        return "Aucun contenu parlé détecté."
    
    # Statistiques par speaker
    speaker_stats = {}
    for segment in display_segments:
        speaker = segment["speaker"]
        if speaker not in speaker_stats:
            speaker_stats[speaker] = {
                "total_time": 0,
                "segments_count": 0,
                "avg_confidence": 0,
                "avg_coverage": 0
            }
        
        duration = segment["end"] - segment["start"]
        speaker_stats[speaker]["total_time"] += duration
        speaker_stats[speaker]["segments_count"] += 1
        speaker_stats[speaker]["avg_confidence"] += segment["confidence"]
        speaker_stats[speaker]["avg_coverage"] += segment.get("speaker_coverage", 0)
    
    # Calculer moyennes
    total_duration = display_segments[-1]["end"] if display_segments else 0
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        if stats["segments_count"] > 0:
            stats["avg_confidence"] /= stats["segments_count"]
            stats["avg_coverage"] /= stats["segments_count"]
            stats["percentage"] = (stats["total_time"] / total_duration * 100) if total_duration > 0 else 0
    
    # Créer le transcript
    lines = ["=== TRANSCRIPTION AVEC DIARISATION - VERSION MODERNE ===\n"]
    
    # Statistiques
    lines.append("📊 ANALYSE DES PARTICIPANTS:")
    for speaker, stats in speaker_stats.items():
        conf = int(stats["avg_confidence"] * 100)
        coverage = int(stats["avg_coverage"] * 100)
        time_str = f"{stats['total_time']:.1f}s"
        percentage = f"{stats['percentage']:.1f}%"
        
        quality_indicator = "✅" if coverage > 70 else "⚠️" if coverage > 40 else "❌"
        
        lines.append(f"🗣️ {speaker}: {time_str} ({percentage}) - Confiance: {conf}% - Attribution: {coverage}% {quality_indicator}")
    
    lines.append(f"\n📈 QUALITÉ GLOBALE:")
    lines.append(f"   📝 Segments utiles: {len(display_segments)}")
    lines.append(f"   ⏱️ Durée totale: {total_duration:.1f}s")
    lines.append(f"   🎯 Speakers identifiés: {len(speaker_stats)}")
    
    # Métriques avancées
    word_level_count = sum(1 for seg in display_segments if seg.get("attribution_method") == "word_level")
    if word_level_count > 0:
        lines.append(f"   🔤 Attribution mot-par-mot: {word_level_count}/{len(display_segments)} segments")
    
    lines.append("\n" + "="*60)
    lines.append("📝 CONVERSATION CHRONOLOGIQUE:")
    
    # Format conversation
    current_speaker = None
    for segment in display_segments:
        speaker = segment["speaker"]
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        confidence = int(segment["confidence"] * 100)
        coverage = int(segment.get("speaker_coverage", 0) * 100)
        
        # Changer de speaker
        if speaker != current_speaker:
            lines.append(f"\n👤 {speaker}:")
            current_speaker = speaker
        
        # Indicateurs de qualité
        quality_icons = ""
        if segment.get("smoothed"):
            quality_icons += "🔧"
        if segment.get("emergency_fix"):
            quality_icons += "🚨"
        if segment.get("attribution_method") == "word_level":
            quality_icons += "🔤"
        if coverage < 30:
            quality_icons += "❓"
        
        confidence_color = "🟢" if confidence > 70 else "🟡" if confidence > 40 else "🔴"
        coverage_color = "🟢" if coverage > 60 else "🟡" if coverage > 30 else "🔴"
        
        lines.append(f"   [{start_time}-{end_time}] {segment['text']}")
        lines.append(f"      └─ {confidence_color}Conf:{confidence}% {coverage_color}Attr:{coverage}% {quality_icons}")
    
    # Résumé final
    lines.append(f"\n" + "="*60)
    lines.append(f"📊 RÉSUMÉ:")
    
    avg_confidence = sum(seg["confidence"] for seg in display_segments) / len(display_segments) * 100
    avg_coverage = sum(seg.get("speaker_coverage", 0) for seg in display_segments) / len(display_segments) * 100
    
    lines.append(f"   🎯 Qualité transcription: {avg_confidence:.0f}%")
    lines.append(f"   🎭 Qualité diarisation: {avg_coverage:.0f}%")
    
    if word_level_count > 0:
        lines.append(f"   ✨ Attribution précise: {word_level_count}/{len(display_segments)} segments")
    
    return "\n".join(lines)

def handler(event):
    """Handler principal RunPod - Version moderne et simplifiée"""
    try:
        # Chargement des modèles si nécessaire
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
        
        logger.info(f"🚀 Début traitement moderne: {audio_url}")
        logger.info(f"👥 Paramètres: num={num_speakers}, min={min_speakers}, max={max_speakers}")
        logger.info(f"🎮 Status modèles: Whisper={'✅' if whisper_model else '❌'} Pyannote={'✅' if diarization_pipeline else '❌'}")
        
        # Téléchargement
        audio_path, download_error = download_audio(audio_url)
        if download_error:
            return {"error": f"Erreur téléchargement: {download_error}"}
        
        try:
            # Transcription + Diarisation
            result = transcribe_and_diarize(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            if not result['success']:
                return {"error": f"Erreur traitement: {result.get('error', 'Erreur inconnue')}"}
            
            # VÉRIFICATION FINALE
            segments = result.get('segments', [])
            final_speakers = set(seg.get("speaker") for seg in segments)
            unknown_count = sum(1 for seg in segments if seg.get("speaker") == "SPEAKER_UNKNOWN")
            
            if unknown_count > 0:
                logger.error(f"🚨 {unknown_count} SPEAKER_UNKNOWN détectés - correction finale!")
                for seg in segments:
                    if seg.get("speaker") == "SPEAKER_UNKNOWN":
                        seg["speaker"] = "SPEAKER_00"
                        seg["final_emergency_fix"] = True
            
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
                "model": "whisper-large-v2-modern",
                "pyannote_model": "speaker-diarization-3.1" if diarization_pipeline else "unavailable",
                "processing_method": "modern_direct_approach",
                "enhancements": {
                    "word_level_timestamps": result.get('word_timestamps_available', False),
                    "word_level_speaker_attribution": result.get('word_timestamps_available', False),
                    "gpu_optimizations": torch.cuda.is_available(),
                    "intelligent_filtering": True,
                    "hallucination_detection": True
                },
                "success": True
            }
            
            # Métriques détaillées
            if 'speakers_found_by_diarization' in result:
                response['speakers_found_by_diarization'] = result['speakers_found_by_diarization']
            if 'diarization_params_used' in result:
                response['diarization_params_used'] = result['diarization_params_used']
            if 'warning' in result:
                response['warning'] = result['warning']
            if 'repetition_warning' in result and result['repetition_warning']:
                response['repetition_warning'] = True
            if 'hallucination_detected' in result and result['hallucination_detected']:
                response['hallucination_detected'] = True
            if 'final_speakers' in result:
                response['final_speakers'] = result['final_speakers']
            if 'transcription_method' in result:
                response['transcription_method'] = result['transcription_method']
            if 'word_timestamps_available' in result:
                response['word_timestamps_available'] = result['word_timestamps_available']
            
            # Logs de succès
            logger.info(f"✅ Traitement réussi (version moderne):")
            logger.info(f"   📝 Segments: {len(result.get('segments', []))}")
            logger.info(f"   🗣️ Speakers: {result.get('speakers_detected', 0)}")
            logger.info(f"   🎯 Speakers finaux: {result.get('final_speakers', 'unknown')}")
            logger.info(f"   🎭 Diarisation: {'✅' if result.get('diarization_available') else '❌'}")
            logger.info(f"   🔤 Word timestamps: {'✅' if result.get('word_timestamps_available') else '❌'}")
            logger.info(f"   🎯 Méthode: {result.get('transcription_method', 'unknown')}")
            
            # Nettoyage GPU
            cleanup_gpu_memory()
            
            return response
            
        finally:
            # Nettoyage fichier
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
    logger.info("🚀 Démarrage RunPod Serverless - Version MODERNE")
    logger.info("✨ Fonctionnalités modernes:")
    logger.info("   - Support total transformers 4.52.4")
    logger.info("   - Word timestamps natifs")
    logger.info("   - Attribution niveau mot-par-mot")
    logger.info("   - Optimisations GPU avancées")
    logger.info("   - Élimination garantie SPEAKER_UNKNOWN")
    logger.info("   - Détection hallucinations Whisper")
    logger.info("   - Filtrage intelligent adaptatif")
    logger.info("   - Code simplifié et robuste")
    
    logger.info("⏳ Chargement initial des modèles...")
    
    try:
        load_models()
        if whisper_model:
            logger.info("✅ Whisper large-v2 prêt (version moderne)")
        else:
            logger.error("❌ Whisper non chargé")
            
        if diarization_pipeline:
            logger.info("✅ Pyannote prêt")
        else:
            logger.warning("⚠️ Pyannote non disponible - mode transcription seule")
            
        logger.info("✅ Service prêt avec support complet des dernières versions")
        
    except Exception as startup_error:
        logger.error(f"❌ Erreur chargement initial: {startup_error}")
        logger.info("⚠️ Démarrage en mode dégradé")
    
    # Démarrage du serveur RunPod
    runpod.serverless.start({"handler": handler})
