from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
from pyannote.audio import Pipeline
import os
import tempfile
import logging
from datetime import timedelta
import numpy as np
import torch

app = Flask(__name__)
CORS(app)

# Configuration logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration GPU pour serverless
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logger.info(f"🎮 Device utilisé: {device}")

# Optimisations GPU
if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_grad_enabled(False)  # Inference seulement
    
    # GPU warmup léger
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
    """Chargement paresseux des modèles pour optimiser le cold start"""
    global whisper_model, diarization_pipeline
    
    if whisper_model is None:
        logger.info("🔄 Chargement Whisper large-v2...")
        whisper_model = whisper.load_model("large-v2", device=device)
        logger.info("✅ Whisper chargé")
    
    if diarization_pipeline is None:
        logger.info("🔄 Chargement pyannote...")
        try:
            # Récupération du token Hugging Face depuis l'environnement
            hf_token = os.getenv('HUGGINGFACE_TOKEN')
            logger.info(f"🔑 Token HF trouvé: {'Oui' if hf_token else 'Non'}")
            
            if hf_token:
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1",
                    use_auth_token=hf_token
                )
                logger.info("✅ pyannote chargé avec token")
            else:
                # Essayer sans token (ne fonctionnera probablement pas pour pyannote)
                logger.warning("⚠️ Pas de token HF - tentative sans token")
                diarization_pipeline = Pipeline.from_pretrained(
                    "pyannote/speaker-diarization-3.1"
                )
            
            # Déplacer sur GPU si disponible
            if torch.cuda.is_available():
                diarization_pipeline.to(device)
            
            logger.info("✅ pyannote chargé")
        except Exception as e:
            logger.error(f"❌ Erreur chargement pyannote: {e}")
            logger.info("💡 Astuce: Définir HUGGINGFACE_TOKEN dans les variables d'environnement")
            diarization_pipeline = None

def format_timestamp(seconds):
    """Convertit les secondes en format mm:ss"""
    return str(timedelta(seconds=int(seconds)))[2:]

def optimize_diarization(audio_path, num_speakers=None, min_speakers=1, max_speakers=4):
    """Diarization optimisée pour serverless"""
    if diarization_pipeline is None:
        raise Exception("Pipeline pyannote non disponible")
    
    diarization_params = {}
    
    if num_speakers:
        diarization_params['num_speakers'] = num_speakers
        logger.info(f"🎯 Forçage à {num_speakers} speakers")
    else:
        diarization_params['min_speakers'] = min_speakers
        diarization_params['max_speakers'] = max_speakers
        logger.info(f"🔍 Détection entre {min_speakers} et {max_speakers} speakers")
    
    diarization = diarization_pipeline(audio_path, **diarization_params)
    return diarization

def merge_transcription_with_speakers_improved(whisper_segments, diarization):
    """Fusion améliorée transcription + diarization"""
    
    # Convertir diarization en liste
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    
    logger.info(f"👥 Speakers détectés: {len(set(seg['speaker'] for seg in speaker_segments))}")
    
    # Associer segments whisper aux speakers
    merged_segments = []
    
    for segment in whisper_segments:
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
            "duration": seg_end - seg_start,
            "speaker": best_speaker,
            "text": segment["text"].strip(),
            "confidence": 1 - segment.get("no_speech_prob", 0),
            "overlap_quality": best_overlap / (seg_end - seg_start) if seg_end > seg_start else 0
        })
    
    # Lissage des transitions
    merged_segments = smooth_speaker_transitions(merged_segments)
    
    return merged_segments

def smooth_speaker_transitions(segments):
    """Lisse les changements de speakers trop fréquents"""
    if len(segments) < 3:
        return segments
    
    smoothed = segments.copy()
    
    for i in range(1, len(smoothed) - 1):
        current = smoothed[i]
        prev_speaker = smoothed[i-1]["speaker"]
        next_speaker = smoothed[i+1]["speaker"]
        
        # Si segment court entre même speaker
        if (current["duration"] < 2.0 and 
            prev_speaker == next_speaker and 
            current["speaker"] != prev_speaker and
            current["overlap_quality"] < 0.8):
            
            logger.info(f"🔧 Lissage: '{current['text'][:30]}...' -> {prev_speaker}")
            smoothed[i]["speaker"] = prev_speaker
            smoothed[i]["smoothed"] = True
    
    return smoothed

def create_readable_transcript(segments):
    """Créer un transcript lisible avec statistiques"""
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
    
    # Calculer moyennes et pourcentages
    total_duration = segments[-1]["end"] if segments else 0
    for speaker in speaker_stats:
        stats = speaker_stats[speaker]
        stats["avg_confidence"] /= stats["segments_count"]
        stats["percentage"] = (stats["total_time"] / total_duration * 100) if total_duration > 0 else 0
    
    # Générer le transcript formaté
    lines = ["=== TRANSCRIPTION AVEC DIARIZATION ===\n"]
    
    # Statistiques
    lines.append("📊 ANALYSE DES PARTICIPANTS:")
    for speaker, stats in speaker_stats.items():
        conf = int(stats["avg_confidence"] * 100)
        time_str = f"{stats['total_time']:.1f}s"
        percentage = f"{stats['percentage']:.1f}%"
        lines.append(f"🗣️ {speaker}: {time_str} ({percentage}) - Confiance: {conf}%")
    
    lines.append("\n" + "="*50)
    lines.append("📝 CONVERSATION:")
    
    current_speaker = None
    for segment in segments:
        start_time = format_timestamp(segment["start"])
        end_time = format_timestamp(segment["end"])
        confidence = int(segment["confidence"] * 100)
        
        if segment["speaker"] != current_speaker:
            lines.append(f"\n👤 {segment['speaker']}:")
            current_speaker = segment["speaker"]
        
        quality_icon = "🔧" if segment.get("smoothed") else ""
        lines.append(f"[{start_time}-{end_time}] {segment['text']} ({confidence}%) {quality_icon}")
    
    return "\n".join(lines)

@app.route('/', methods=['GET'])
def home():
    """Endpoint de santé"""
    return jsonify({
        "status": "API Transcription + Diarization",
        "version": "4.0-serverless",
        "gpu_available": torch.cuda.is_available(),
        "device": str(device),
        "endpoints": {
            "/transcribe": "POST - Transcription simple",
            "/process": "POST - Transcription + Diarization complète"
        }
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check pour serverless"""
    return jsonify({"status": "healthy"}), 200

@app.route('/transcribe', methods=['POST'])
def transcribe():
    """Transcription simple avec Whisper"""
    try:
        # Chargement paresseux
        load_models()
        
        if 'audio' not in request.files:
            return jsonify({"error": "Fichier audio manquant"}), 400
        
        file = request.files['audio']
        logger.info(f"📁 Transcription: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            
            # Paramètres optimisés pour serverless
            result = whisper_model.transcribe(
                tmp.name,
                language='fr',
                fp16=torch.cuda.is_available(),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                temperature=0.0,
                verbose=False
            )
            
            os.unlink(tmp.name)
        
        return jsonify({
            "success": True,
            "transcription": result["text"],
            "segments": result["segments"],
            "model": "large-v2",
            "language": result.get("language", "fr"),
            "device": str(device)
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur transcription: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/process', methods=['POST'])
def process_with_diarization():
    """Transcription + Diarization complète"""
    try:
        # Chargement paresseux
        load_models()
        
        if diarization_pipeline is None:
            return jsonify({
                "error": "Pipeline diarization non disponible. Vérifiez HUGGINGFACE_TOKEN."
            }), 500
        
        if 'audio' not in request.files:
            return jsonify({"error": "Fichier audio manquant"}), 400
        
        file = request.files['audio']
        
        # Paramètres optionnels
        num_speakers = request.form.get('num_speakers', type=int)
        min_speakers = request.form.get('min_speakers', 1, type=int)
        max_speakers = request.form.get('max_speakers', 4, type=int)
        
        logger.info(f"📁 Traitement complet: {file.filename}")
        
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp:
            file.save(tmp.name)
            
            # Transcription
            logger.info("🎯 Transcription...")
            transcription = whisper_model.transcribe(
                tmp.name,
                language='fr',
                fp16=torch.cuda.is_available(),
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                temperature=0.0,
                verbose=False
            )
            
            # Diarization
            logger.info("👥 Diarization...")
            diarization = optimize_diarization(
                tmp.name,
                num_speakers=num_speakers,
                min_speakers=min_speakers,
                max_speakers=max_speakers
            )
            
            # Fusion
            logger.info("🔗 Fusion...")
            merged_segments = merge_transcription_with_speakers_improved(
                transcription["segments"],
                diarization
            )
            
            # Transcript formaté
            readable_transcript = create_readable_transcript(merged_segments)
            
            os.unlink(tmp.name)
        
        speakers_detected = len(set(seg["speaker"] for seg in merged_segments 
                                  if seg["speaker"] != "SPEAKER_UNKNOWN"))
        
        return jsonify({
            "success": True,
            "model": "large-v2",
            "device": str(device),
            "transcription_brute": transcription["text"],
            "transcription_formatee": readable_transcript,
            "segments_detailles": merged_segments,
            "parametres": {
                "num_speakers_force": num_speakers,
                "min_speakers": min_speakers,
                "max_speakers": max_speakers
            },
            "statistiques": {
                "speakers_detectes": speakers_detected,
                "speakers_inconnus": len([seg for seg in merged_segments 
                                        if seg["speaker"] == "SPEAKER_UNKNOWN"]),
                "duree_totale": f"{max(seg['end'] for seg in merged_segments) if merged_segments else 0:.1f}s",
                "nombre_segments": len(merged_segments),
                "confiance_moyenne": f"{sum(seg['confidence'] for seg in merged_segments) / len(merged_segments) * 100:.1f}%" if merged_segments else "0%",
                "segments_lisses": len([seg for seg in merged_segments if seg.get("smoothed")])
            }
        })
        
    except Exception as e:
        logger.error(f"❌ Erreur traitement: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Port adapté pour serverless (généralement 8080)
    port = int(os.environ.get('PORT', 8080))
    app.run(host='0.0.0.0', port=port, debug=False, threaded=True)
