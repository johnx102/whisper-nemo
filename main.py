import os
import json
import torch
import tempfile
import traceback
import warnings
import requests
import time
import random
from typing import Tuple, List, Optional
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from datetime import timedelta
from urllib.parse import urlparse
from collections import Counter
import logging
import gc

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Suppress warnings
warnings.filterwarnings("ignore", message="Using `TRANSFORMERS_CACHE` is deprecated")
os.environ["HF_HOME"] = os.environ.get("HF_HOME", "/tmp/hf_cache")

# Configuration GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device_str = "cuda" if torch.cuda.is_available() else "cpu"
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

def cleanup_gpu_memory():
    """Nettoyage GPU"""
    try:
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
    """Filtrage intelligent des segments - Version avancée du code original"""
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
        confidence = segment.get("avg_logprob", 0)  # faster-whisper utilise avg_logprob
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
            
        # 3. Confiance trop faible (faster-whisper utilise des valeurs négatives)
        if confidence < -1.0:  # Seuil adapté pour faster-whisper
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
        
        # SEGMENT VALIDE - Adapter pour faster-whisper
        validated_words = []
        if words and isinstance(words, list):
            for word_info in words:
                try:
                    # faster-whisper structure: word_info a des attributs au lieu de dict
                    if hasattr(word_info, 'word') and hasattr(word_info, 'start'):
                        word_start = max(float(word_info.start), start_time)
                        word_end = min(float(word_info.end), end_time)
                        
                        if word_start >= word_end:
                            word_end = word_start + 0.1
                        
                        validated_words.append({
                            'word': str(word_info.word).strip(),
                            'start': word_start,
                            'end': word_end,
                            'probability': float(getattr(word_info, 'probability', 1.0))
                        })
                except Exception:
                    continue
        
        # Segment enrichi
        cleaned_segments.append({
            "start": start_time,
            "end": end_time,
            "text": text,
            "confidence": abs(confidence),  # Convertir en valeur positive
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

class WhisperAudioTranscriber:
    def __init__(self, model_name="large-v2"):
        """Initialize faster-whisper transcriber avec optimisations avancées"""
        self.model = WhisperModel(
            model_name,
            device=device_str,
            compute_type="float16" if torch.cuda.is_available() else "float32"
        )
        logger.info(f"✅ Faster-Whisper {model_name} chargé avec succès")

    def transcribe_with_faster_whisper(self, audio_path):
        """ÉTAPE 1: Transcription avec faster-whisper - Version avancée"""
        try:
            logger.info("🎯 ÉTAPE 1: Transcription Faster-Whisper avec word_timestamps...")
            
            if not os.path.exists(audio_path):
                return {'success': False, 'error': f'Fichier audio introuvable: {audio_path}'}
            
            file_size = os.path.getsize(audio_path)
            logger.info(f"📁 Taille fichier: {file_size} bytes ({file_size/1024/1024:.2f}MB)")
            
            if file_size == 0:
                return {'success': False, 'error': 'Fichier audio vide'}
            
            # TRANSCRIPTION FASTER-WHISPER
            try:
                logger.info("🔄 Transcription avec faster-whisper (word_timestamps natifs)...")
                
                # faster-whisper API - plus simple et direct
                segments_generator, info = self.model.transcribe(
                    audio_path,
                    language='fr',
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    no_speech_threshold=0.6,
                    log_prob_threshold=-1.0,
                    compression_ratio_threshold=2.4,
                    temperature=0.0
                )
                
                # Convertir le générateur en liste et extraire les infos
                segments_list = list(segments_generator)
                transcription_text = " ".join([seg.text for seg in segments_list])
                
                logger.info("✅ Transcription faster-whisper réussie avec word_timestamps")
                transcription_method = "faster_whisper_with_words"
                
            except Exception as e:
                logger.warning(f"⚠️ Erreur faster-whisper avec word_timestamps: {e}")
                
                # Fallback sans word_timestamps
                try:
                    logger.info("🔄 Fallback faster-whisper sans word_timestamps...")
                    
                    segments_generator, info = self.model.transcribe(
                        audio_path,
                        language='fr',
                        word_timestamps=False,
                        condition_on_previous_text=False,
                        no_speech_threshold=0.6,
                        temperature=0.0
                    )
                    
                    segments_list = list(segments_generator)
                    transcription_text = " ".join([seg.text for seg in segments_list])
                    
                    logger.info("✅ Transcription faster-whisper réussie sans word_timestamps")
                    transcription_method = "faster_whisper_segments_only"
                    
                except Exception as e2:
                    logger.error(f"❌ Échec total faster-whisper: {e2}")
                    return {'success': False, 'error': f'Transcription impossible: {e2}'}
            
            if not segments_list:
                return {'success': False, 'error': 'Aucun segment de transcription'}
            
            logger.info(f"📊 Transcription terminée:")
            logger.info(f"   📝 Texte: '{transcription_text[:100]}...'")
            logger.info(f"   🌍 Langue: {info.language}")
            logger.info(f"   📈 Segments bruts: {len(segments_list)}")
            logger.info(f"   🎯 Méthode: {transcription_method}")
            
            # Vérifier si on a des timestamps de mots
            has_word_timestamps = transcription_method == "faster_whisper_with_words"
            if has_word_timestamps:
                logger.info("✅ Timestamps de mots disponibles (faster-whisper)")
            else:
                logger.info("⚠️ Pas de timestamps de mots - attribution niveau segment")
            
            # Convertir les segments faster-whisper au format standard
            converted_segments = []
            for seg in segments_list:
                # Convertir les mots faster-whisper
                words_converted = []
                if has_word_timestamps and hasattr(seg, 'words') and seg.words:
                    for word in seg.words:
                        words_converted.append({
                            'word': word.word,
                            'start': word.start,
                            'end': word.end,
                            'probability': getattr(word, 'probability', 1.0)
                        })
                
                converted_segments.append({
                    "start": seg.start,
                    "end": seg.end,
                    "text": seg.text.strip(),
                    "avg_logprob": seg.avg_logprob,
                    "words": words_converted
                })
            
            # Filtrage intelligent
            cleaned_segments, suspicious_texts, hallucination_detected = improved_segment_filtering(converted_segments)
            
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
                'transcription': transcription_text,
                'segments': cleaned_segments,
                'language': info.language,
                'segments_raw_count': len(converted_segments),
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

class PyannoteDiarizer:
    def __init__(self, hf_token: str):
        """Initialize PyAnnote diarization avec gestion avancée"""
        try:
            logger.info("🔄 Chargement pyannote diarization...")
            
            if not hf_token:
                logger.error("❌ HUGGINGFACE_TOKEN manquant - diarization impossible")
                self.pipeline = None
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
                    self.pipeline = Pipeline.from_pretrained(
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
                self.pipeline.to(device)
            
            logger.info("✅ pyannote chargé et configuré")
            
        except Exception as e:
            logger.error(f"❌ Erreur chargement pyannote: {e}")
            if "429" in str(e):
                logger.info("💡 SOLUTION Rate Limit HuggingFace:")
                logger.info("   - Attendez quelques minutes avant de relancer")
                logger.info("   - Le service continuera en mode transcription seule")
            self.pipeline = None

    def diarize_with_pyannote(self, audio_path, num_speakers=None, min_speakers=2, max_speakers=4):
        """ÉTAPE 2: Diarisation avec pyannote - Version avancée"""
        try:
            if not self.pipeline:
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
                diarization = self.pipeline(audio_path, **diarization_params)
                logger.info("✅ Diarisation terminée")
            except Exception as e:
                logger.error(f"❌ Erreur diarization avec paramètres: {e}")
                logger.info("🔄 Tentative sans paramètres...")
                diarization = self.pipeline(audio_path)
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
    """ÉTAPE 3: Attribution speakers intelligente - Version avancée du code original"""
    logger.info("🔗 ÉTAPE 3: Attribution speakers...")
    
    final_segments = []
    
    # Extraire les speakers connus
    known_speakers = list(set(seg["speaker"] for seg in speaker_segments)) if speaker_segments else []
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
        if has_word_timestamps and words and speaker_segments:
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
                if not best_speaker and speaker_segments:
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
            
            if speaker_segments:
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
                # Alternance automatique si pas de diarisation
                segment_index = len(final_segments)
                best_speaker = known_speakers[segment_index % len(known_speakers)]
                best_coverage = 0.5
            
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
    
    # Post-traitement: validation
    final_segments = validate_speakers(final_segments, known_speakers)
    
    word_level_count = sum(1 for seg in final_segments if seg.get("attribution_method") == "word_level")
    speakers_assigned = len(set(seg["speaker"] for seg in final_segments))
    
    logger.info(f"✅ Attribution terminée:")
    logger.info(f"   🎯 Speakers: {speakers_assigned} sur {len(final_segments)} segments")
    logger.info(f"   🔤 Attribution mot-par-mot: {word_level_count}/{len(final_segments)} segments")
    
    return final_segments

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

def create_formatted_transcript(segments):
    """Crée un transcript formaté avec speakers - Version avancée"""
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
    lines = ["=== TRANSCRIPTION AVEC DIARISATION - VERSION FASTER-WHISPER AVANCÉE ===\n"]
    
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
    
    # Métriques faster-whisper
    word_level_count = sum(1 for seg in display_segments if seg.get("attribution_method") == "word_level")
    if word_level_count > 0:
        lines.append(f"   🔤 Attribution mot-par-mot: {word_level_count}/{len(display_segments)} segments")
    
    lines.append(f"   🚀 Engine: faster-whisper avancé")
    
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
    lines.append(f"   🚀 Engine: faster-whisper avancé (optimisé)")
    
    if word_level_count > 0:
        lines.append(f"   ✨ Attribution précise: {word_level_count}/{len(display_segments)} segments")
    
    return "\n".join(lines)

class AudioProcessor:
    def __init__(self, hf_token: str, whisper_model="large-v2"):
        """Initialize with advanced faster-whisper + pyannote"""
        self.transcriber = WhisperAudioTranscriber(whisper_model)
        self.diarizer = PyannoteDiarizer(hf_token)

    def process_audio(self, audio_path: str, language="fr", num_speakers=None, min_speakers=2, max_speakers=4):
        """Process audio with advanced transcription and diarization"""
        try:
            logger.info(f"🚀 Processing audio: {audio_path}")
            
            # ÉTAPE 1: Transcription avec faster-whisper
            transcription_result = self.transcriber.transcribe_with_faster_whisper(audio_path)
            if not transcription_result['success']:
                return transcription_result
            
            # ÉTAPE 2: Diarisation
            diarization_result = self.diarizer.diarize_with_pyannote(
                audio_path,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            speaker_segments = None
            if diarization_result['success']:
                speaker_segments = diarization_result['speaker_segments']
            else:
                logger.warning("⚠️ Diarisation échouée - Attribution automatique")
            
            # ÉTAPE 3: Attribution des speakers
            final_segments = assign_speakers_to_transcription(
                transcription_result["segments"],
                speaker_segments
            )
            
            speakers_detected = len(set(seg["speaker"] for seg in final_segments))
            final_speaker_list = sorted(set(seg["speaker"] for seg in final_segments))
            
            logger.info(f"🎉 Processus terminé: {speakers_detected} speakers")
            logger.info(f"🎯 Speakers utilisés: {final_speaker_list}")
            
            # Création du transcript formaté
            formatted_transcript = create_formatted_transcript(final_segments)
            
            return {
                'success': True,
                'transcription': transcription_result["transcription"],
                'transcription_formatee': formatted_transcript,
                'segments': final_segments,
                'speakers_detected': speakers_detected,
                'language': transcription_result["language"],
                'diarization_available': diarization_result['success'] if diarization_result else False,
                'final_speakers': final_speaker_list,
                'repetition_warning': transcription_result.get('repetition_warning', False),
                'hallucination_detected': transcription_result.get('hallucination_detected', False),
                'word_segments_count': transcription_result.get('word_segments_count', 0),
                'transcription_method': transcription_result.get('transcription_method', 'unknown'),
                'word_timestamps_available': transcription_result.get('word_timestamps_available', False),
                'device': str(device),
                'model': "faster-whisper-advanced",
                'processing_method': "faster_whisper_optimized_advanced"
            }
            
        except Exception as e:
            logger.error(f"❌ Erreur processus principal: {e}")
            logger.error(f"🔍 Traceback: {traceback.format_exc()}")
            return {'success': False, 'error': str(e)}
        finally:
            cleanup_gpu_memory()

def download_test_audio():
    """Download a test audio file for demonstration"""
    test_urls = [
        {
            "name": "French conversation sample",
            "url": "https://www2.cs.uic.edu/~i101/SoundFiles/French1.wav",
            "filename": "french_test.wav"
        },
        {
            "name": "English conversation sample", 
            "url": "https://www2.cs.uic.edu/~i101/SoundFiles/dialogue.wav",
            "filename": "english_test.wav"
        }
    ]
    
    print("📥 No audio file found. Download a test file?")
    for i, sample in enumerate(test_urls):
        print(f"   {i+1}. {sample['name']}")
    print(f"   0. Skip download")
    
    try:
        choice = input("Choose option (0-2): ").strip()
        if choice == "0":
            return None
            
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(test_urls):
            sample = test_urls[choice_idx]
            print(f"📥 Downloading {sample['name']}...")
            
            response = requests.get(sample["url"], timeout=30)
            response.raise_for_status()
            
            with open(sample["filename"], "wb") as f:
                f.write(response.content)
            
            print(f"✅ Downloaded: {sample['filename']}")
            return sample["filename"]
    except Exception as e:
        print(f"❌ Download failed: {e}")
    
    return None

def main():
    """Main function - Version locale du code RunPod"""
    import sys
    
    logger.info("🚀 Démarrage version locale - FASTER-WHISPER AVANCÉ")
    
    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not HF_TOKEN:
        logger.error("❌ Error: Please set HF_TOKEN or HUGGINGFACE_TOKEN environment variable")
        logger.info("🔑 Get token from: https://huggingface.co/settings/tokens")
        logger.info("📋 Example: export HF_TOKEN='your_token_here'")
        return
    
    # Audio file path - check command line, URL, or interactive
    audio_file = None
    temp_file_to_cleanup = None
    
    if len(sys.argv) > 1:
        audio_input = sys.argv[1]
        
        # Check if it's a URL
        if audio_input.startswith(('http://', 'https://')):
            logger.info(f"🌐 URL détectée: {audio_input}")
            audio_file, error = download_audio(audio_input)
            if error:
                logger.error(f"❌ Erreur téléchargement: {error}")
                return
            temp_file_to_cleanup = audio_file
        else:
            audio_file = audio_input
    else:
        # Interactive mode
        audio_input = input("📁 Enter audio file path or URL (or press Enter for test file): ").strip()
        
        if audio_input:
            # Remove quotes
            if audio_input.startswith('"') and audio_input.endswith('"'):
                audio_input = audio_input[1:-1]
            if audio_input.startswith("'") and audio_input.endswith("'"):
                audio_input = audio_input[1:-1]
            
            # Check if URL
            if audio_input.startswith(('http://', 'https://')):
                logger.info(f"🌐 URL détectée: {audio_input}")
                audio_file, error = download_audio(audio_input)
                if error:
                    logger.error(f"❌ Erreur téléchargement: {error}")
                    return
                temp_file_to_cleanup = audio_file
            else:
                audio_file = audio_input
    
    # If no file specified or file doesn't exist, offer to download test file
    if not audio_file or not os.path.exists(audio_file):
        if audio_file:
            logger.error(f"❌ Error: Audio file '{audio_file}' not found")
        
        audio_file = download_test_audio()
        if not audio_file:
            logger.info("📁 Please provide a valid audio file path or URL")
            logger.info("💡 Supported formats: mp3, wav, m4a, aac, flac")
            logger.info("🚀 Usage: python main.py your_audio_file.wav")
            logger.info("🌐 Usage: python main.py https://example.com/audio.mp3")
            return
        temp_file_to_cleanup = audio_file
    
    try:
        # Initialize processor with advanced faster-whisper
        logger.info("🔄 Initializing advanced faster-whisper + PyAnnote processor...")
        processor = AudioProcessor(HF_TOKEN, whisper_model="large-v2")
        
        # Process audio
        logger.info("🔄 Starting advanced transcription and diarization...")
        logger.info("⏳ This may take a few minutes depending on audio length...")
        
        results = processor.process_audio(
            audio_file,
            language="fr",  # Change to "en" for English or detect automatically
            num_speakers=None,  # Auto-detect, or set fixed number like 2, 3, etc.
            min_speakers=2,
            max_speakers=4
        )
        
        if results.get('success'):
            # Display advanced formatted results
            print("\n" + "="*80)
            print("🎉 TRAITEMENT TERMINÉ AVEC SUCCÈS")
            print("="*80)
            print(results.get('transcription_formatee', 'Erreur formatage'))
            
            # Save detailed results
            output_file = "transcription_results_advanced.json"
            with open(output_file, 'w', encoding='utf-8') as f:
                # Convert non-serializable objects
                serializable_results = {}
                for key, value in results.items():
                    try:
                        json.dumps(value)
                        serializable_results[key] = value
                    except:
                        serializable_results[key] = str(value)
                
                json.dump(serializable_results, f, ensure_ascii=False, indent=2)
            
            logger.info(f"💾 Résultats détaillés sauvés: {output_file}")
            
        else:
            logger.error(f"❌ Erreur traitement: {results.get('error', 'Erreur inconnue')}")
        
    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")
        logger.error(f"🔍 Traceback: {traceback.format_exc()}")
    finally:
        # Cleanup temp file
        if temp_file_to_cleanup and os.path.exists(temp_file_to_cleanup):
            try:
                os.unlink(temp_file_to_cleanup)
                logger.info("🗑️ Fichier temporaire supprimé")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ Erreur nettoyage: {cleanup_error}")

if __name__ == "__main__":
    main()
