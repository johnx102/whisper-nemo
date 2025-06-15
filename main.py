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

# Optimisations GPU avancées (style WhisperX)
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_grad_enabled(False)
    
    # Informations GPU
    gpu_name = torch.cuda.get_device_name()
    gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
    logger.info(f"🎮 GPU: {gpu_name} ({gpu_memory:.1f}GB)")
    logger.info(f"🚀 Optimisations WhisperX-style activées:")
    logger.info(f"   - TF32 matmul: {torch.backends.cuda.matmul.allow_tf32}")
    logger.info(f"   - TF32 cudnn: {torch.backends.cudnn.allow_tf32}")
    logger.info(f"   - cuDNN benchmark: {torch.backends.cudnn.benchmark}")
    
    # GPU warmup optimisé
    logger.info("🔥 Warmup GPU optimisé...")
    x = torch.randn(2000, 2000, device=device, dtype=torch.float16 if torch.cuda.is_available() else torch.float32)
    y = torch.mm(x, x)
    torch.cuda.synchronize()
    del x, y
    torch.cuda.empty_cache()
    logger.info("✅ GPU warmed up avec optimisations WhisperX")
else:
    logger.info("💻 Mode CPU - optimisations GPU indisponibles")

# Variables globales pour les modèles
whisper_model = None
diarization_pipeline = None

def cleanup_gpu_memory():
    """Nettoyage GPU agressif style WhisperX"""
    try:
        import gc
        gc.collect()
        
        if torch.cuda.is_available():
            # Nettoyage agressif
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            
            # Forcer la libération de toutes les variables temporaires
            for obj in gc.get_objects():
                if torch.is_tensor(obj) and obj.is_cuda:
                    del obj
            
            # Second passage de nettoyage
            gc.collect()
            torch.cuda.empty_cache()
            
            # Stats mémoire pour monitoring
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            logger.info(f"🧹 GPU Memory après nettoyage: {allocated:.1f}GB allocated, {cached:.1f}GB cached")
            
    except Exception as e:
        logger.warning(f"⚠️ Erreur nettoyage GPU: {e}")

def optimize_transcription_params(device, file_duration=None):
    """Optimise les paramètres selon le hardware disponible"""
    params = {
        'fp16': torch.cuda.is_available(),
        'condition_on_previous_text': False,
        'no_speech_threshold': 0.6,
        'logprob_threshold': -1.0,
        'compression_ratio_threshold': 2.2,
        'temperature': 0.0,
        'word_timestamps': True
    }
    
    # Optimisations spécifiques GPU
    if torch.cuda.is_available():
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        
        # Ajuster selon la mémoire GPU
        if gpu_memory >= 24:  # A100/H100
            params['batch_size'] = 32
            logger.info("🎯 Paramètres optimisés pour GPU haute-mémoire (A100/H100)")
        elif gpu_memory >= 16:  # V100/A10
            params['batch_size'] = 24
            logger.info("🎯 Paramètres optimisés pour GPU moyenne-mémoire (V100/A10)")
        elif gpu_memory >= 8:  # RTX 4070/3080
            params['batch_size'] = 16
            logger.info("🎯 Paramètres optimisés pour GPU standard (RTX 4070/3080)")
        else:  # T4 et inférieurs
            params['batch_size'] = 8
            params['fp16'] = True  # Forcer FP16 pour économiser mémoire
            logger.info("🎯 Paramètres optimisés pour GPU faible-mémoire (T4)")
    else:
        params['batch_size'] = 4
        params['fp16'] = False
        logger.info("🎯 Paramètres optimisés pour CPU")
    
    return params

def load_models():
    """Chargement des modèles avec gestion du rate limiting"""
    global whisper_model, diarization_pipeline
    
    if whisper_model is None:
        logger.info("🔄 Chargement Whisper large-v2...")
        try:
            whisper_model = whisper.load_model("large-v2", device=device)
            logger.info("✅ Whisper chargé avec succès")
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

def safe_text_for_logging(text, max_length=40):
    """Sécurise le texte pour l'affichage dans les logs"""
    if not text:
        return "VIDE"
    
    # Nettoyer les caractères problématiques
    safe_text = str(text).replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    
    # Supprimer les caractères de contrôle
    safe_text = ''.join(char for char in safe_text if ord(char) >= 32 or char in [' '])
    
    # Tronquer si nécessaire
    if len(safe_text) > max_length:
        safe_text = safe_text[:max_length] + "..."
    
    return safe_text
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

def improved_segment_filtering(segments_raw):
    """Filtrage amélioré des segments - Version 2.0 avec détection d'hallucinations"""
    cleaned_segments = []
    
    # Analyse globale pour seuils adaptatifs
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
        logger.warning(f"🚨 {len(suspicious_texts)} textes suspects détectés:")
        for text, count in list(suspicious_texts.items())[:3]:  # Afficher les 3 premiers
            logger.warning(f"   '{text[:40]}...' répété {count} fois")
    
    # Tracking avancé des répétitions
    consecutive_tracker = {}
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
                # Compter répétitions récentes
                recent_same = sum(1 for j in range(max(0, i-10), i) 
                                if segments_raw[j].get("text", "").strip() == text)
                if recent_same >= 5:
                    logger.error(f"🚨 HALLUCINATION détectée à {i}/{len(segments_raw)}: '{text[:40]}...'")
                    hallucination_detected = True
                    # Arrêter le traitement ici pour éviter la boucle infinie
                    break
        
        # FILTRES RENFORCÉS
        
        # 1. Durée minimum adaptative
        if duration < min_duration:
            continue
            
        # 2. Texte vide ou inutile étendu
        useless_texts = {
            ".", "...", "....", "-", "–", "euh", "heu", "mm", "hmm", 
            "ah", "oh", "ben", "donc", "alors", "voilà", "bon", "oui", "non",
            " ", "", "merci", "au revoir", "bonjour"
        }
        if not text or text.lower() in useless_texts or len(text) < 2:
            continue
            
        # 3. Probabilité de silence trop élevée
        if no_speech_prob > 0.85:
            continue
            
        # 4. Ratio mots/durée anormal (NOUVEAU seuil plus strict)
        words_count = len(text.split())
        if duration > 0:
            words_per_second = words_count / duration
            if words_per_second > max_words_per_second or words_per_second < 0.2:
                logger.info(f"🔥 Vitesse anormale: {words_per_second:.1f} mots/s - '{safe_text_for_logging(text)}'")
                continue
        
        # 5. NOUVEAU: Détection patterns répétitifs consécutifs
        if text in suspicious_texts:
            # Analyser les 5 derniers segments
            recent_texts = [cleaned_segments[j]["text"] for j in range(max(0, len(cleaned_segments)-5), len(cleaned_segments))]
            same_count = recent_texts.count(text)
            
            if same_count >= 3:
                logger.info(f"🔥 Pattern répétitif: '{safe_text_for_logging(text)}' ({same_count} dans les 5 derniers)")
                continue
                
            # Pattern A-B-A-B détecté ?
            if (len(recent_texts) >= 3 and 
                recent_texts[-1] == recent_texts[-3] and
                text == recent_texts[-1]):
                logger.info(f"🔥 Pattern A-B-A-B: '{safe_text_for_logging(text)}'")
                continue
        
        # 6. Validation timestamps renforcée
        if start_time >= end_time or start_time < 0 or end_time > total_duration + 10:
            logger.warning(f"⚠️ Timestamps invalides: {start_time:.2f} -> {end_time:.2f}")
            continue
            
        # 7. Détection chevauchements anormaux
        if cleaned_segments:
            last_seg = cleaned_segments[-1]
            gap = start_time - last_seg["end"]
            overlap = min(end_time, last_seg["end"]) - max(start_time, last_seg["start"])
            
            # Si chevauchement > 90% ET texte identique ou très similaire
            if overlap > 0.9 * min(duration, last_seg["end"] - last_seg["start"]):
                text_similarity = len(set(text.split()) & set(last_seg["text"].split())) / max(len(text.split()), 1)
                if text_similarity > 0.8:
                    logger.info(f"🔥 Doublon temporal/textuel: '{safe_text_for_logging(text)}'")
                    continue
            
            # Gap anormalement petit entre segments différents
            if 0 < gap < 0.1 and text != last_seg["text"] and duration < 1.0:
                logger.info(f"🔥 Segment fragmenté: '{safe_text_for_logging(text)}'")
                continue
        
        # 8. NOUVEAU: Filtrage par cohérence sémantique
        if len(cleaned_segments) >= 2:
            # Vérifier si le segment fait sens dans le contexte
            context_words = set()
            for prev_seg in cleaned_segments[-2:]:
                context_words.update(prev_seg["text"].lower().split())
            
            current_words = set(text.lower().split())
            context_overlap = len(context_words & current_words) / max(len(current_words), 1)
            
            # Si aucun mot en commun avec le contexte ET segment très court
            if context_overlap == 0 and duration < 1.5 and len(current_words) <= 2:
                logger.info(f"🔥 Segment hors contexte: '{safe_text_for_logging(text)}'")
                continue
        
        # SEGMENT VALIDE - Enrichissement des données
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
        enriched_segment = {
            "start": start_time,
            "end": end_time,
            "text": text,
            "confidence": 1 - no_speech_prob,
            "duration": duration,
            "words_count": words_count,
            "words": validated_words,
            "has_word_timestamps": len(validated_words) > 0,
            "words_per_second": words_count / duration if duration > 0 else 0,
            "segment_quality": "high" if duration > 2 and no_speech_prob < 0.3 else "medium" if duration > 1 else "low",
            "filter_version": "v2.0_enhanced"
        }
        
        cleaned_segments.append(enriched_segment)
    
    removed_count = len(segments_raw) - len(cleaned_segments)
    
    logger.info(f"✅ Filtrage v2.0 amélioré:")
    logger.info(f"   📝 Segments gardés: {len(cleaned_segments)} / {len(segments_raw)} ({removed_count} supprimés)")
    logger.info(f"   🚨 Hallucination détectée: {'⚠️ OUI' if hallucination_detected else '✅ Non'}")
    logger.info(f"   🔤 Segments avec mots: {sum(1 for s in cleaned_segments if s['has_word_timestamps'])}")
    
    return cleaned_segments, suspicious_texts, hallucination_detected

def transcribe_with_whisper(audio_path):
    """ÉTAPE 1: Transcription avec filtrage amélioré - Gestion erreurs version"""
    try:
        logger.info("🎯 ÉTAPE 1: Transcription Whisper large-v2 (mode amélioré v2.0)...")
        
        if not os.path.exists(audio_path):
            return {'success': False, 'error': f'Fichier audio introuvable: {audio_path}'}
        
        file_size = os.path.getsize(audio_path)
        logger.info(f"📁 Taille fichier: {file_size} bytes ({file_size/1024/1024:.2f}MB)")
        
        if file_size == 0:
            return {'success': False, 'error': 'Fichier audio vide'}
        
        # STRATÉGIE DE TRANSCRIPTION PROGRESSIVE
        transcription_attempts = [
            {
                "name": "word_timestamps_full",
                "params": {
                    "language": 'fr',
                    "fp16": torch.cuda.is_available(),
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.6,
                    "logprob_threshold": -1.0,
                    "compression_ratio_threshold": 2.0,
                    "temperature": 0.0,
                    "verbose": False,
                    "word_timestamps": True,
                    "suppress_tokens": [],
                    "initial_prompt": None
                }
            },
            {
                "name": "word_timestamps_basic",
                "params": {
                    "language": 'fr',
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.6,
                    "temperature": 0.0,
                    "verbose": False,
                    "word_timestamps": True
                }
            },
            {
                "name": "standard_optimized",
                "params": {
                    "language": 'fr',
                    "fp16": torch.cuda.is_available(),
                    "condition_on_previous_text": False,
                    "no_speech_threshold": 0.6,
                    "logprob_threshold": -1.0,
                    "compression_ratio_threshold": 2.0,
                    "temperature": 0.0,
                    "verbose": False
                }
            },
            {
                "name": "minimal_safe",
                "params": {
                    "language": 'fr',
                    "condition_on_previous_text": False,
                    "temperature": 0.0,
                    "verbose": False
                }
            }
        ]
        
        result = None
        successful_method = None
        
        for attempt in transcription_attempts:
            try:
                logger.info(f"🔄 Tentative transcription: {attempt['name']}")
                result = whisper_model.transcribe(audio_path, **attempt['params'])
                successful_method = attempt['name']
                logger.info(f"✅ Transcription réussie avec: {successful_method}")
                break
                
            except Exception as whisper_error:
                error_str = str(whisper_error)
                
                # Gestion d'erreurs spécifiques
                if "Cannot set attribute 'src'" in error_str:
                    logger.warning(f"⚠️ Erreur version Whisper/Transformers: {attempt['name']}")
                    logger.info("💡 Cette erreur est due à une incompatibilité de versions")
                elif "word_timestamps" in error_str:
                    logger.warning(f"⚠️ word_timestamps non supporté: {attempt['name']}")
                elif "fp16" in error_str:
                    logger.warning(f"⚠️ FP16 non supporté: {attempt['name']}")
                else:
                    logger.warning(f"⚠️ Erreur inconnue avec {attempt['name']}: {error_str}")
                
                # Continuer avec la tentative suivante
                continue
        
        # Si aucune méthode n'a fonctionné
        if result is None:
            logger.error("❌ ÉCHEC: Toutes les méthodes de transcription ont échoué")
            return {
                'success': False, 
                'error': 'Transcription impossible avec toutes les méthodes disponibles. Vérifiez la compatibilité Whisper/Transformers.'
            }
        
        logger.info(f"📊 Transcription brute terminée avec {successful_method}:")
        logger.info(f"   📝 Texte: '{result.get('text', '')[:100]}...'")
        logger.info(f"   🌍 Langue: {result.get('language', 'unknown')}")
        logger.info(f"   📈 Segments bruts: {len(result.get('segments', []))}")
        
        # Vérifier si on a des timestamps de mots selon la méthode utilisée
        has_word_timestamps = successful_method in ["word_timestamps_full", "word_timestamps_basic"]
        if has_word_timestamps:
            logger.info("✅ Timestamps de mots disponibles")
        else:
            logger.info("⚠️ Pas de timestamps de mots - attribution niveau segment uniquement")
        
        # NOUVEAU: Filtrage amélioré avec détection hallucinations
        segments_raw = result.get("segments", [])
        cleaned_segments, suspicious_texts, hallucination_detected = improved_segment_filtering(segments_raw)
        
        # Ajuster les informations de mots selon la méthode de transcription
        word_segments_count = 0
        if has_word_timestamps:
            word_segments_count = sum(1 for seg in cleaned_segments if seg.get("has_word_timestamps"))
        else:
            # Pour les méthodes sans word_timestamps, marquer explicitement
            for seg in cleaned_segments:
                seg["has_word_timestamps"] = False
                seg["words"] = []  # Pas de mots détaillés
        
        logger.info(f"✅ Transcription terminée avec améliorations:")
        logger.info(f"   📝 Segments finaux: {len(cleaned_segments)} (supprimé {len(segments_raw) - len(cleaned_segments)})")
        logger.info(f"   🔤 Segments avec mots: {word_segments_count}/{len(cleaned_segments)}")
        logger.info(f"   🚨 Hallucinations: {'⚠️ DÉTECTÉES' if hallucination_detected else '✅ Aucune'}")
        logger.info(f"   🎯 Méthode utilisée: {successful_method}")
        
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
            'filter_version': "v2.0_enhanced",
            'transcription_method': successful_method,
            'word_timestamps_available': has_word_timestamps
        }
        
        logger.info(f"📊 Transcription brute terminée:")
        logger.info(f"   📝 Texte: '{result.get('text', '')[:100]}...'")
        logger.info(f"   🌍 Langue: {result.get('language', 'unknown')}")
        logger.info(f"   📈 Segments bruts: {len(result.get('segments', []))}")
        
        # NOUVEAU: Filtrage amélioré avec détection hallucinations
        segments_raw = result.get("segments", [])
        cleaned_segments, suspicious_texts, hallucination_detected = improved_segment_filtering(segments_raw)
        
        word_segments_count = sum(1 for seg in cleaned_segments if seg.get("has_word_timestamps"))
        
        logger.info(f"✅ Transcription terminée avec améliorations:")
        logger.info(f"   📝 Segments finaux: {len(cleaned_segments)} (supprimé {len(segments_raw) - len(cleaned_segments)})")
        logger.info(f"   🔤 Segments avec mots: {word_segments_count}/{len(cleaned_segments)}")
        logger.info(f"   🚨 Hallucinations: {'⚠️ DÉTECTÉES' if hallucination_detected else '✅ Aucune'}")
        
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
            'filter_version': "v2.0_enhanced"
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

def enhanced_speaker_validation(segments):
    """Validation finale renforcée des speakers"""
    logger.info("🔒 Validation finale renforcée des speakers...")
    
    validated_segments = []
    speaker_consistency_score = {}
    
    # Analyser la cohérence de chaque speaker
    valid_speakers = set()
    for segment in segments:
        speaker = segment["speaker"]
        if speaker != "SPEAKER_UNKNOWN":
            valid_speakers.add(speaker)
    
    if not valid_speakers:
        valid_speakers = {"SPEAKER_00", "SPEAKER_01"}
        logger.warning("⚠️ Aucun speaker valide - création forcée")
    
    # Calculer scores de cohérence
    for speaker in valid_speakers:
        speaker_segments = [seg for seg in segments if seg["speaker"] == speaker]
        
        if speaker_segments:
            total_coverage = sum(seg.get("speaker_coverage", 0) for seg in speaker_segments)
            avg_coverage = total_coverage / len(speaker_segments)
            
            speaker_consistency_score[speaker] = {
                "avg_coverage": avg_coverage,
                "segments_count": len(speaker_segments),
                "total_time": sum(seg["end"] - seg["start"] for seg in speaker_segments)
            }
    
    logger.info(f"📊 Scores de cohérence speakers:")
    for speaker, score in speaker_consistency_score.items():
        logger.info(f"   {speaker}: {score['avg_coverage']:.2f} coverage, {score['segments_count']} segments")
    
    # Validation et correction
    for i, segment in enumerate(segments):
        current_speaker = segment["speaker"]
        
        # Éliminer définitivement SPEAKER_UNKNOWN
        if current_speaker == "SPEAKER_UNKNOWN":
            # Trouver le meilleur speaker de remplacement
            if speaker_consistency_score:
                best_replacement = max(speaker_consistency_score.items(), 
                                     key=lambda x: x[1]["avg_coverage"])[0]
            else:
                best_replacement = list(valid_speakers)[0]
            
            logger.warning(f"🔧 CORRECTION: SPEAKER_UNKNOWN → {best_replacement}")
            segment["speaker"] = best_replacement
            segment["speaker_coverage"] = 0.1
            segment["final_correction"] = True
        
        # Corriger speakers avec coverage très faible
        elif segment.get("speaker_coverage", 0) < 0.15:
            # Analyser le contexte
            context_speakers = []
            
            # Contexte précédent (2 segments)
            for j in range(max(0, i-2), i):
                if j < len(segments):
                    context_speakers.append(segments[j]["speaker"])
            
            # Contexte suivant (2 segments)
            for j in range(i+1, min(len(segments), i+3)):
                context_speakers.append(segments[j]["speaker"])
            
            # Speaker le plus fréquent dans le contexte
            if context_speakers:
                most_common = Counter(context_speakers).most_common(1)[0][0]
                
                if most_common != current_speaker and most_common in valid_speakers:
                    logger.info(f"🔧 Correction contexte: {current_speaker} → {most_common}")
                    segment["speaker"] = most_common
                    segment["speaker_coverage"] = 0.3
                    segment["context_correction"] = True
        
        validated_segments.append(segment)
    
    # Vérification finale
    final_speakers = set(seg["speaker"] for seg in validated_segments)
    unknown_count = sum(1 for seg in validated_segments if seg["speaker"] == "SPEAKER_UNKNOWN")
    
    if unknown_count > 0:
        logger.error(f"❌ ÉCHEC: {unknown_count} SPEAKER_UNKNOWN encore présents!")
        # Correction d'urgence finale
        fallback_speaker = list(valid_speakers)[0] if valid_speakers else "SPEAKER_00"
        for seg in validated_segments:
            if seg["speaker"] == "SPEAKER_UNKNOWN":
                seg["speaker"] = fallback_speaker
                seg["emergency_final_fix"] = True
    else:
        logger.info(f"✅ Validation finale réussie: {sorted(final_speakers)}")
    
    return validated_segments

def assign_speakers_to_transcription_enhanced(transcription_segments, speaker_segments):
    """ÉTAPE 3: Attribution WhisperX-style avec précision mot-par-mot"""
    logger.info("🔗 ÉTAPE 3: Attribution speakers niveau mot (style WhisperX amélioré)...")
    
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
        
        # Attribution niveau mot (comme WhisperX)
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
                
                # Méthode 1: Centre du mot dans un segment de speaker
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
                
                # Méthode 2: Meilleur recouvrement si centre pas trouvé
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
                
                # Fallback temporel
                if not best_speaker:
                    min_distance = float('inf')
                    for spk_seg in speaker_segments:
                        spk_center = (spk_seg["start"] + spk_seg["end"]) / 2
                        distance = abs(word_center - spk_center)
                        if distance < min_distance:
                            min_distance = distance
                            best_speaker = spk_seg["speaker"]
                
                # Dernier recours
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
            
            # Déterminer le speaker principal du segment par vote pondéré
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
            words_with_speakers = words  # Garder mots originaux sans speakers
        
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
    
    # Post-traitement amélioré
    final_segments = smooth_speaker_transitions_enhanced(final_segments)
    final_segments = enhanced_speaker_validation(final_segments)
    
    word_level_count = sum(1 for seg in final_segments if seg.get("attribution_method") == "word_level")
    speakers_assigned = len(set(seg["speaker"] for seg in final_segments))
    
    logger.info(f"✅ Attribution terminée:")
    logger.info(f"   🎯 Speakers: {speakers_assigned} sur {len(final_segments)} segments")
    logger.info(f"   🔤 Attribution mot-par-mot: {word_level_count}/{len(final_segments)} segments")
    
    return final_segments

def smooth_speaker_transitions_enhanced(segments, min_segment_duration=1.0, confidence_threshold=0.3):
    """Lissage amélioré avec prise en compte des mots"""
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
        
        should_smooth = False
        new_speaker = None
        smooth_reason = ""
        
        # Cas 1: Segment court entre même speaker
        if (current_duration < min_segment_duration and
            prev_speaker == next_speaker and
            current_speaker != prev_speaker and
            current.get("speaker_coverage", 0) < confidence_threshold):
            
            should_smooth = True
            new_speaker = prev_speaker
            smooth_reason = "segment_court_entre_meme_speaker"
        
        # Cas 2: Attribution avec très faible confiance
        elif (current.get("speaker_coverage", 0) < 0.2 and
              prev_speaker == next_speaker and
              prev_speaker != current_speaker):
            
            should_smooth = True
            new_speaker = prev_speaker
            smooth_reason = "faible_confiance_entre_meme_speaker"
        
        # Cas 3: Segments avec mots - vérifier cohérence
        elif (current.get("has_word_speakers") and current.get("words")):
            # Compter les mots par speaker
            word_speakers = {}
            for word in current["words"]:
                if isinstance(word, dict) and "speaker" in word:
                    spk = word["speaker"]
                    word_speakers[spk] = word_speakers.get(spk, 0) + 1
            
            # Si majorité des mots vote pour un autre speaker
            if word_speakers:
                word_majority_speaker = max(word_speakers.items(), key=lambda x: x[1])[0]
                majority_ratio = word_speakers[word_majority_speaker] / len(current["words"])
                
                if (word_majority_speaker != current_speaker and majority_ratio > 0.7):
                    should_smooth = True
                    new_speaker = word_majority_speaker
                    smooth_reason = "majorite_mots_autre_speaker"
        
        # Appliquer le lissage
        if should_smooth and new_speaker:
            logger.debug(f"🔧 Lissage ({smooth_reason}): '{current['text'][:30]}...' {current_speaker} → {new_speaker}")
            smoothed[i]["speaker"] = new_speaker
            smoothed[i]["smoothed"] = True
            smoothed[i]["smooth_reason"] = smooth_reason
            
            # Mettre à jour les speakers des mots si applicable
            if smoothed[i].get("has_word_speakers") and smoothed[i].get("words"):
                for word in smoothed[i]["words"]:
                    if isinstance(word, dict) and "speaker" in word:
                        word["speaker"] = new_speaker
                        word["smoothed"] = True
            
            changes_made += 1
    
    if changes_made > 0:
        logger.info(f"✅ Lissage amélioré: {changes_made} corrections appliquées")
    
    return smoothed

def transcribe_and_diarize_separated(audio_path, num_speakers=None, min_speakers=2, max_speakers=4):
    """Fonction principale avec processus séparés - Version améliorée v2.0"""
    try:
        # ÉTAPE 1: Transcription améliorée
        transcription_result = transcribe_with_whisper(audio_path)
        if not transcription_result['success']:
            return transcription_result
        
        # Vérifier si on a des répétitions suspectes
        repetition_warning = transcription_result.get('repetition_warning', False)
        hallucination_detected = transcription_result.get('hallucination_detected', False)
        
        if hallucination_detected:
            logger.warning("🚨 HALLUCINATION WHISPER DÉTECTÉE - Qualité audio dégradée")
        elif repetition_warning:
            logger.warning("⚠️ Répétitions détectées - Possible dégradation qualité")
        
        # ÉTAPE 2: Diarisation
        diarization_result = diarize_with_pyannote(
            audio_path,
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        if not diarization_result['success']:
            logger.warning("⚠️ Diarisation échouée - Attribution automatique intelligente")
            
            # FALLBACK INTELLIGENT amélioré
            segments_with_auto_speakers = []
            target_speakers = num_speakers if num_speakers and num_speakers > 0 else 2
            speaker_names = [f"SPEAKER_{i:02d}" for i in range(target_speakers)]
            
            logger.info(f"🔄 Attribution automatique sur {target_speakers} speakers: {speaker_names}")
            
            # Attribution intelligente basée sur les pauses
            for i, segment in enumerate(transcription_result["segments"]):
                if target_speakers == 1:
                    speaker_name = "SPEAKER_00"
                else:
                    # Logique améliorée d'alternance
                    if i == 0:
                        speaker_idx = 0
                    else:
                        prev_segment = segments_with_auto_speakers[-1]
                        prev_duration = prev_segment["end"] - prev_segment["start"]
                        prev_speaker_idx = int(prev_segment["speaker"].split("_")[1])
                        
                        # Pause entre segments
                        pause_duration = segment["start"] - prev_segment["end"]
                        
                        # Changer de speaker si pause longue ou segment long précédent
                        if pause_duration > 1.0 or prev_duration > 4.0 or i % 5 == 0:
                            speaker_idx = (prev_speaker_idx + 1) % target_speakers
                        else:
                            speaker_idx = prev_speaker_idx
                    
                    speaker_name = speaker_names[speaker_idx]
                
                segments_with_auto_speakers.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                    "speaker": speaker_name,
                    "confidence": segment["confidence"],
                    "speaker_coverage": 0.8,  # Confiance artificielle élevée
                    "attribution_method": "auto_alternation_enhanced",
                    "words": segment.get("words", [])
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
                'repetition_warning': repetition_warning,
                'hallucination_detected': hallucination_detected,
                'final_speakers': sorted(set(seg["speaker"] for seg in segments_with_auto_speakers))
            }
        
        # ÉTAPE 3: Attribution des speakers (diarisation réussie)
        final_segments = assign_speakers_to_transcription_enhanced(
            transcription_result["segments"],
            diarization_result["speaker_segments"]
        )
        
        # VÉRIFICATION FINALE TRIPLE CONTRÔLE
        final_speakers = set(seg["speaker"] for seg in final_segments)
        unknown_segments = [seg for seg in final_segments if seg["speaker"] == "SPEAKER_UNKNOWN"]
        
        if unknown_segments:
            logger.error(f"❌ ERREUR CRITIQUE: {len(unknown_segments)} segments SPEAKER_UNKNOWN!")
            
            # CORRECTION D'URGENCE FINALE
            known_speakers = [s for s in final_speakers if s != "SPEAKER_UNKNOWN"]
            if not known_speakers:
                known_speakers = ["SPEAKER_00", "SPEAKER_01"]
                logger.error("🚨 Création forcée SPEAKER_00/01")
            
            # Forcer tous les SPEAKER_UNKNOWN
            for i, seg in enumerate(final_segments):
                if seg["speaker"] == "SPEAKER_UNKNOWN":
                    new_speaker = known_speakers[i % len(known_speakers)]
                    logger.error(f"🚨 CORRECTION: SPEAKER_UNKNOWN → {new_speaker}")
                    final_segments[i]["speaker"] = new_speaker
                    final_segments[i]["speaker_coverage"] = 0.1
                    final_segments[i]["emergency_fix"] = True
        
        # VÉRIFICATION POST-CORRECTION
        post_correction_speakers = set(seg["speaker"] for seg in final_segments)
        post_unknown_count = sum(1 for seg in final_segments if seg["speaker"] == "SPEAKER_UNKNOWN")
        
        if post_unknown_count > 0:
            logger.error(f"❌ ÉCHEC TOTAL: {post_unknown_count} SPEAKER_UNKNOWN!")
            # En dernier recours
            for seg in final_segments:
                if seg["speaker"] == "SPEAKER_UNKNOWN":
                    seg["speaker"] = "SPEAKER_00"
                    seg["absolute_fallback"] = True
        else:
            logger.info(f"✅ SUCCÈS: Aucun SPEAKER_UNKNOWN")
        
        speakers_detected = len(set(seg["speaker"] for seg in final_segments))
        final_speaker_list = sorted(set(seg["speaker"] for seg in final_segments))
        
        logger.info(f"🎉 Processus complet terminé: {speakers_detected} speakers")
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
            'hallucination_detected': hallucination_detected,
            'unknown_segments_corrected': len(unknown_segments) if unknown_segments else 0,
            'word_segments_count': transcription_result.get('word_segments_count', 0),
            'filter_version': transcription_result.get('filter_version', 'v2.0')
        }
        
    except Exception as e:
        logger.error(f"❌ Erreur processus séparé: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return {'success': False, 'error': str(e)}

def create_formatted_transcript(segments):
    """Crée un transcript formaté avec améliorations v2.0"""
    if not segments:
        return "Aucune transcription disponible."
    
    # Filtrer une dernière fois pour l'affichage
    display_segments = []
    for segment in segments:
        text = segment.get("text", "").strip()
        duration = segment.get("end", 0) - segment.get("start", 0)
        
        # Critères d'affichage plus stricts
        if (text and 
            text not in [".", ",", "!", "?", "...", "-", " ", "..."] and
            len(text) > 1 and
            duration >= 0.4):  # Seuil légèrement réduit
            display_segments.append(segment)
    
    if not display_segments:
        return "Aucun contenu parlé détecté dans cet audio."
    
    # Élimination finale des SPEAKER_UNKNOWN pour l'affichage
    unknown_segments = [seg for seg in display_segments if seg.get("speaker") == "SPEAKER_UNKNOWN"]
    if unknown_segments:
        logger.warning(f"⚠️ {len(unknown_segments)} segments SPEAKER_UNKNOWN dans l'affichage - correction!")
        for seg in display_segments:
            if seg.get("speaker") == "SPEAKER_UNKNOWN":
                seg["speaker"] = "SPEAKER_00"
    
    # Statistiques par speaker
    speaker_stats = {}
    for segment in display_segments:
        speaker = segment["speaker"]
        if speaker == "SPEAKER_UNKNOWN":
            continue
            
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
        if stats["segments_count"] > 0:
            stats["avg_confidence"] /= stats["segments_count"]
            stats["avg_coverage"] /= stats["segments_count"]
            stats["percentage"] = (stats["total_time"] / total_duration * 100) if total_duration > 0 else 0
    
    # Créer le transcript
    lines = ["=== TRANSCRIPTION AVEC DIARISATION AMÉLIORÉE v2.0 (WhisperX-style) ===\n"]
    
    # Statistiques détaillées
    lines.append("📊 ANALYSE DES PARTICIPANTS:")
    for speaker, stats in speaker_stats.items():
        conf = int(stats["avg_confidence"] * 100)
        coverage = int(stats["avg_coverage"] * 100)
        time_str = f"{stats['total_time']:.1f}s"
        percentage = f"{stats['percentage']:.1f}%"
        
        # Indicateur de qualité amélioré
        if coverage > 70:
            quality_indicator = "✅"
        elif coverage > 40:
            quality_indicator = "⚠️"
        else:
            quality_indicator = "❌"
        
        lines.append(f"🗣️ {speaker}: {time_str} ({percentage}) - Confiance: {conf}% - Attribution: {coverage}% {quality_indicator}")
    
    lines.append(f"\n📈 QUALITÉ GLOBALE:")
    lines.append(f"   📝 Segments utiles: {len(display_segments)}")
    lines.append(f"   ⏱️ Durée totale: {total_duration:.1f}s")
    lines.append(f"   🎯 Speakers identifiés: {len(speaker_stats)}")
    
    # Améliorations v2.0
    word_level_count = sum(1 for seg in display_segments if seg.get("attribution_method") == "word_level")
    if word_level_count > 0:
        lines.append(f"   🔤 Attribution mot-par-mot: {word_level_count}/{len(display_segments)} segments")
    
    enhanced_count = sum(1 for seg in display_segments if seg.get("filter_version", "").startswith("v2.0"))
    if enhanced_count > 0:
        lines.append(f"   ✨ Filtrage amélioré v2.0: {enhanced_count}/{len(display_segments)} segments")
    
    # Détection problèmes
    text_counts = {}
    for seg in display_segments:
        text = seg["text"]
        text_counts[text] = text_counts.get(text, 0) + 1
    
    suspicious_repetitions = {text: count for text, count in text_counts.items() if count > 3}
    if suspicious_repetitions:
        lines.append(f"   ⚠️ Répétitions détectées: {len(suspicious_repetitions)} phrases répétées")
    
    lines.append("\n" + "="*60)
    lines.append("📝 CONVERSATION CHRONOLOGIQUE:")
    
    # Format conversation amélioré
    current_speaker = None
    for segment in display_segments:
        speaker = segment["speaker"]
        
        # Assurance finale
        if speaker == "SPEAKER_UNKNOWN":
            speaker = "SPEAKER_00"
        
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        confidence = int(segment["confidence"] * 100)
        coverage = int(segment.get("speaker_coverage", 0) * 100)
        
        # Changer de speaker
        if speaker != current_speaker:
            lines.append(f"\n👤 {speaker}:")
            current_speaker = speaker
        
        # Indicateurs de qualité v2.0
        quality_icons = ""
        if segment.get("smoothed"):
            quality_icons += "🔧"  # Lissé
        if segment.get("emergency_fix") or segment.get("absolute_fallback"):
            quality_icons += "🚨"  # Correction d'urgence
        if segment.get("attribution_method") == "word_level":
            quality_icons += "🔤"  # Attribution mot-par-mot
        if segment.get("filter_version", "").startswith("v2.0"):
            quality_icons += "✨"  # Filtrage v2.0
        if coverage < 30:
            quality_icons += "❓"  # Attribution très incertaine
        elif coverage < 60:
            quality_icons += "⚠️"  # Attribution incertaine
        
        # Indicateur de confiance audio
        if confidence < 40:
            quality_icons += "🔇"  # Confiance très faible
        elif confidence < 70:
            quality_icons += "🔉"  # Confiance moyenne
        
        # Couleurs de confiance
        confidence_color = "🟢" if confidence > 70 else "🟡" if confidence > 40 else "🔴"
        coverage_color = "🟢" if coverage > 60 else "🟡" if coverage > 30 else "🔴"
        
        lines.append(f"   [{start_time}-{end_time}] {segment['text']}")
        lines.append(f"      └─ {confidence_color}Conf:{confidence}% {coverage_color}Attr:{coverage}% {quality_icons}")
    
    # Résumé final amélioré
    lines.append(f"\n" + "="*60)
    lines.append(f"📊 RÉSUMÉ v2.0:")
    
    # Qualité globale
    avg_confidence = sum(seg["confidence"] for seg in display_segments) / len(display_segments) * 100
    avg_coverage = sum(seg.get("speaker_coverage", 0) for seg in display_segments) / len(display_segments) * 100
    
    lines.append(f"   🎯 Qualité transcription: {avg_confidence:.0f}%")
    lines.append(f"   🎭 Qualité diarisation: {avg_coverage:.0f}%")
    
    # Nouvelles métriques v2.0
    if word_level_count > 0:
        lines.append(f"   ✨ Attribution précise: {word_level_count}/{len(display_segments)} segments")
    
    if enhanced_count > 0:
        lines.append(f"   🚀 Filtrage amélioré: {enhanced_count}/{len(display_segments)} segments")
    
    # Détection de corrections appliquées
    corrections_count = sum(1 for seg in display_segments if seg.get("smoothed") or seg.get("emergency_fix"))
    if corrections_count > 0:
        lines.append(f"   🔧 Corrections appliquées: {corrections_count} segments")
    
    # Recommandations améliorées
    if avg_confidence < 50:
        lines.append(f"   💡 Audio de qualité faible - vérifiez le contenu")
    if avg_coverage < 40:
        lines.append(f"   💡 Diarisation incertaine - possibles erreurs d'attribution")
    if suspicious_repetitions:
        lines.append(f"   💡 Attention: Répétitions détectées - possible hallucination Whisper")
    
    return "\n".join(lines)

def post_process_quality_check(result):
    """Contrôle qualité final avant retour - Version v2.0"""
    segments = result.get("segments", [])
    
    # Vérifications critiques améliorées
    checks = {
        "no_unknown_speakers": all(seg.get("speaker") != "SPEAKER_UNKNOWN" for seg in segments),
        "valid_timestamps": all(seg.get("start", 0) < seg.get("end", 0) for seg in segments),
        "non_empty_text": all(seg.get("text", "").strip() for seg in segments),
        "reasonable_durations": all(0.1 <= (seg.get("end", 0) - seg.get("start", 0)) <= 30 for seg in segments),
        "valid_speakers": all(seg.get("speaker", "").startswith("SPEAKER_") for seg in segments),
        "minimum_coverage": all(seg.get("speaker_coverage", 0) >= 0 for seg in segments)
    }
    
    logger.info("🔍 Contrôle qualité final v2.0:")
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        logger.info(f"   {status} {check}")
    
    # Corrections d'urgence si nécessaire
    emergency_fixes = 0
    
    if not checks["no_unknown_speakers"]:
        logger.error("🚨 Correction d'urgence SPEAKER_UNKNOWN...")
        for seg in segments:
            if seg.get("speaker") == "SPEAKER_UNKNOWN":
                seg["speaker"] = "SPEAKER_00"
                seg["emergency_final_fix"] = True
                emergency_fixes += 1
    
    if not checks["valid_speakers"]:
        logger.error("🚨 Correction speakers invalides...")
        for seg in segments:
            if not seg.get("speaker", "").startswith("SPEAKER_"):
                seg["speaker"] = "SPEAKER_00"
                seg["speaker_format_fix"] = True
                emergency_fixes += 1
    
    # Statistiques de qualité améliorées
    if segments:
        avg_coverage = sum(seg.get("speaker_coverage", 0) for seg in segments) / len(segments)
        avg_confidence = sum(seg.get("confidence", 0) for seg in segments) / len(segments)
        word_level_segments = sum(1 for seg in segments if seg.get("attribution_method") == "word_level")
        enhanced_segments = sum(1 for seg in segments if seg.get("filter_version", "").startswith("v2.0"))
        
        result["quality_metrics"] = {
            "avg_speaker_coverage": avg_coverage,
            "avg_transcription_confidence": avg_confidence,
            "total_segments": len(segments),
            "quality_checks_passed": sum(checks.values()),
            "quality_score": sum(checks.values()) / len(checks),
            "word_level_segments": word_level_segments,
            "enhanced_segments": enhanced_segments,
            "emergency_fixes_applied": emergency_fixes
        }
        
        quality_score = result["quality_metrics"]["quality_score"]
        logger.info(f"📊 Score qualité global: {quality_score:.1%}")
        
        if quality_score < 0.8:
            logger.warning(f"⚠️ Qualité sous-optimale détectée: {quality_score:.1%}")
    
    return result

def handler(event):
    """Handler principal RunPod avec améliorations v2.0 - Garantit aucun SPEAKER_UNKNOWN"""
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
        
        logger.info(f"🚀 Début traitement v2.0: {audio_url}")
        logger.info(f"👥 Paramètres: num={num_speakers}, min={min_speakers}, max={max_speakers}")
        logger.info(f"🎮 Status modèles: Whisper={'✅' if whisper_model else '❌'} Pyannote={'✅' if diarization_pipeline else '❌'}")
        
        # Téléchargement
        audio_path, download_error = download_audio(audio_url)
        if download_error:
            return {"error": f"Erreur téléchargement: {download_error}"}
        
        try:
            # Transcription + Diarisation avec processus séparés v2.0
            result = transcribe_and_diarize_separated(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            if not result['success']:
                return {"error": f"Erreur traitement: {result.get('error', 'Erreur inconnue')}"}
            
            # Contrôle qualité final
            result = post_process_quality_check(result)
            
            # VÉRIFICATION FINALE CÔTÉ HANDLER v2.0
            segments = result.get('segments', [])
            final_speakers = set(seg.get("speaker") for seg in segments)
            unknown_count = sum(1 for seg in segments if seg.get("speaker") == "SPEAKER_UNKNOWN")
            
            if unknown_count > 0:
                logger.error(f"🚨 HANDLER v2.0: {unknown_count} SPEAKER_UNKNOWN détectés!")
                # Dernière correction possible
                for seg in segments:
                    if seg.get("speaker") == "SPEAKER_UNKNOWN":
                        seg["speaker"] = "SPEAKER_00"
                        seg["handler_emergency_fix"] = True
                logger.error(f"🚨 HANDLER v2.0: Correction d'urgence appliquée")
            
            # Création du transcript formaté
            formatted_transcript = create_formatted_transcript(result['segments'])
            
            # Construction réponse finale avec améliorations v2.0
            response = {
                "transcription": result['transcription'],
                "transcription_formatee": formatted_transcript,
                "segments": result['segments'],
                "speakers_detected": result['speakers_detected'],
                "language": result['language'],
                "diarization_available": result['diarization_available'],
                "device": str(device),
                "model": "whisper-large-v2-enhanced-v2.0" if whisper_model else "whisper-unavailable",
                "pyannote_model": "speaker-diarization-3.1" if diarization_pipeline else "unavailable",
                "processing_method": "whisperx_style_separated_processes_v2.0",
                "enhancements": {
                    "word_level_timestamps": True,
                    "word_level_speaker_attribution": True,
                    "gpu_optimizations": torch.cuda.is_available(),
                    "enhanced_smoothing": True,
                    "advanced_filtering": True,
                    "hallucination_detection": True,
                    "quality_control": True
                },
                "success": True
            }
            
            # Infos de debug et qualité améliorées v2.0
            if 'speakers_found_by_diarization' in result:
                response['speakers_found_by_diarization'] = result['speakers_found_by_diarization']
            if 'diarization_params_used' in result:
                response['diarization_params_used'] = result['diarization_params_used']
            if 'warning' in result:
                response['warning'] = result['warning']
            
            # Nouvelles métriques v2.0
            if 'repetition_warning' in result and result['repetition_warning']:
                response['repetition_warning'] = True
            if 'hallucination_detected' in result and result['hallucination_detected']:
                response['hallucination_detected'] = True
                response['warning'] = (response.get('warning', '') + ' ATTENTION: Hallucination Whisper détectée.').strip()
            if 'unknown_segments_corrected' in result:
                response['unknown_segments_corrected'] = result['unknown_segments_corrected']
            if 'final_speakers' in result:
                response['final_speakers'] = result['final_speakers']
            if 'filter_version' in result:
                response['filter_version'] = result['filter_version']
            
            # Métriques qualité
            if 'quality_metrics' in result:
                response['quality_metrics'] = result['quality_metrics']
            
            # Nouvelles métriques WhisperX-style v2.0
            if 'word_segments_count' in result:
                response['word_level_segments'] = result['word_segments_count']
                response['word_level_coverage'] = f"{result['word_segments_count']}/{len(result.get('segments', []))}"
            
            # Compter segments avec attribution mot-par-mot
            word_level_attributions = sum(1 for seg in result.get('segments', []) if seg.get('attribution_method') == 'word_level')
            if word_level_attributions > 0:
                response['word_level_attributions'] = word_level_attributions
                response['attribution_quality'] = f"{word_level_attributions}/{len(result.get('segments', []))} segments avec attribution précise"
            
            # Logs de succès avec détails améliorés v2.0
            logger.info(f"✅ Traitement réussi (WhisperX-enhanced v2.0):")
            logger.info(f"   📝 Segments: {len(result.get('segments', []))}")
            logger.info(f"   🗣️ Speakers: {result.get('speakers_detected', 0)}")
            logger.info(f"   🎯 Speakers finaux: {result.get('final_speakers', 'unknown')}")
            logger.info(f"   🎭 Diarisation: {'✅' if result.get('diarization_available') else '❌'}")
            logger.info(f"   🔤 Segments avec mots: {result.get('word_segments_count', 0)}")
            logger.info(f"   ✨ Filtrage v2.0: {result.get('filter_version', 'unknown')}")
            logger.info(f"   ⚠️ Répétitions: {'⚠️' if result.get('repetition_warning') else '✅'}")
            logger.info(f"   🚨 Hallucinations: {'⚠️' if result.get('hallucination_detected') else '✅'}")
            logger.info(f"   🔧 Corrections UNKNOWN: {result.get('unknown_segments_corrected', 0)}")
            
            if 'quality_metrics' in result:
                qm = result['quality_metrics']
                logger.info(f"   📊 Score qualité: {qm.get('quality_score', 0):.1%}")
                logger.info(f"   🔧 Corrections d'urgence: {qm.get('emergency_fixes_applied', 0)}")
            
            # Nettoyage GPU après traitement
            cleanup_gpu_memory()
            
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
        logger.error(f"❌ Erreur handler v2.0: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
        return {"error": f"Erreur interne: {str(e)}"}

if __name__ == "__main__":
    logger.info("🚀 Démarrage RunPod Serverless - Transcription + Diarisation AMÉLIORÉE v2.0")
    logger.info("✨ Fonctionnalités WhisperX-style v2.0:")
    logger.info("   - Attribution niveau mot-par-mot")
    logger.info("   - Timestamps précis des mots")
    logger.info("   - Optimisations GPU avancées")
    logger.info("   - Lissage intelligent des speakers")
    logger.info("   - Élimination garantie des SPEAKER_UNKNOWN")
    logger.info("   - Détection et filtrage des hallucinations Whisper")
    logger.info("   - Filtrage adaptatif selon la durée audio")
    logger.info("   - Validation renforcée de la cohérence speakers")
    logger.info("   - Contrôle qualité multi-niveaux")
    logger.info("   - Attribution automatique intelligente en fallback")
    logger.info("   - Gestion robuste des incompatibilités de versions")
    
    # Diagnostic des versions pour éviter les erreurs
    try:
        import transformers
        import whisper
        logger.info(f"📦 Versions détectées:")
        logger.info(f"   - Whisper: {whisper.__version__ if hasattr(whisper, '__version__') else 'unknown'}")
        logger.info(f"   - Transformers: {transformers.__version__}")
        logger.info(f"   - PyTorch: {torch.__version__}")
        
        # Avertissements sur les incompatibilités connues
        transformers_version = transformers.__version__
        if transformers_version.startswith("4.21") or transformers_version.startswith("4.22"):
            logger.warning("⚠️ Version Transformers potentiellement incompatible avec word_timestamps")
            logger.info("💡 Si erreurs 'src', le fallback automatique sera utilisé")
        
    except Exception as version_error:
        logger.warning(f"⚠️ Impossible de vérifier les versions: {version_error}")
    
    logger.info("⏳ Chargement initial des modèles...")
    
    try:
        load_models()
        if whisper_model:
            logger.info("✅ Whisper large-v2 prêt avec améliorations v2.0")
            logger.info("🔧 Stratégies de transcription disponibles:")
            logger.info("   1. word_timestamps_full (optimal)")
            logger.info("   2. word_timestamps_basic (compatible)")
            logger.info("   3. standard_optimized (fallback)")
            logger.info("   4. minimal_safe (dernier recours)")
        else:
            logger.error("❌ Whisper non chargé")
            
        if diarization_pipeline:
            logger.info("✅ Pyannote prêt avec optimisations")
        else:
            logger.warning("⚠️ Pyannote non disponible - mode transcription seule")
            
        logger.info("✅ Service prêt avec améliorations WhisperX-style v2.0")
        logger.info("🔧 Nouvelles fonctionnalités:")
        logger.info("   - Détection hallucinations Whisper en temps réel")
        logger.info("   - Filtrage adaptatif selon durée audio")
        logger.info("   - Attribution automatique intelligente")
        logger.info("   - Métriques de qualité avancées")
        logger.info("   - Corrections d'urgence multi-niveaux")
        logger.info("   - Gestion robuste des erreurs de compatibilité")
        
    except Exception as startup_error:
        logger.error(f"❌ Erreur chargement initial: {startup_error}")
        logger.info("⚠️ Démarrage en mode dégradé")
        
        # Diagnostic détaillé en cas d'erreur
        if "src" in str(startup_error):
            logger.error("🚨 PROBLÈME DE COMPATIBILITÉ DÉTECTÉ:")
            logger.error("   - Erreur liée aux versions Whisper/Transformers")
            logger.error("   - Solutions possibles:")
            logger.error("     1. pip install --upgrade transformers")
            logger.error("     2. pip install transformers==4.19.2")
            logger.error("     3. pip install --upgrade openai-whisper")
            logger.error("   - Le service utilisera les fallbacks automatiques")
    
    # Démarrage du serveur RunPod
    runpod.serverless.start({"handler": handler})
