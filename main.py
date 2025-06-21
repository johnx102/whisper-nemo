import os
import json
import torch
import tempfile
import traceback
from typing import Tuple, List, Optional
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from pyannote.audio.pipelines.utils.hook import ProgressHook
from datetime import timedelta


class WhisperAudioTranscriber:
    def __init__(self, model_name="large-v2"):
        """
        Initialize faster-whisper transcriber
        
        Args:
            model_name: Model size (tiny, base, small, medium, large, large-v2, large-v3)
        """
        # Configure device
        if torch.cuda.is_available():
            self.device = "cuda"
            self.compute_type = "float16"
            print("🎮 Using GPU with float16")
        else:
            self.device = "cpu"
            self.compute_type = "float32"
            print("💻 Using CPU with float32")
        
        # Load faster-whisper model
        try:
            self.model = WhisperModel(
                model_name,
                device=self.device,
                compute_type=self.compute_type
            )
            print(f"✅ Faster-Whisper {model_name} loaded successfully")
        except Exception as e:
            print(f"❌ Error loading faster-whisper: {e}")
            raise

    def transcribe(self, audio_path: str, language="fr") -> Tuple[Optional[str], Optional[List]]:
        """
        Transcribe audio with faster-whisper
        
        Args:
            audio_path: Path to audio file
            language: Language code (fr, en, etc.)
            
        Returns:
            Tuple of (full_transcription, segments_with_timestamps)
        """
        try:
            print(f"🎯 Transcribing with faster-whisper: {audio_path}")
            
            # Transcribe with word timestamps
            segments_generator, info = self.model.transcribe(
                audio_path,
                language=language,
                word_timestamps=True,
                condition_on_previous_text=False,
                no_speech_threshold=0.6,
                log_prob_threshold=-1.0,
                compression_ratio_threshold=2.4,
                temperature=0.0
            )
            
            # Convert generator to list
            segments_list = list(segments_generator)
            
            # Build full transcription
            full_transcription = " ".join([seg.text for seg in segments_list])
            
            # Convert to standard format
            converted_segments = []
            for seg in segments_list:
                # Convert words if available
                words_converted = []
                if hasattr(seg, 'words') and seg.words:
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
                    "confidence": abs(seg.avg_logprob),  # Convert to positive
                    "words": words_converted,
                    "has_word_timestamps": len(words_converted) > 0
                })
            
            print(f"✅ Transcription completed: {len(converted_segments)} segments")
            print(f"📄 Text preview: '{full_transcription[:100]}...'")
            print(f"🌍 Detected language: {info.language}")
            
            return full_transcription, converted_segments
            
        except Exception as e:
            print(f"❌ Error during transcription: {e}")
            return None, None


class PyannoteDiarizer:
    def __init__(self, hf_token: str):
        """
        Initialize PyAnnote diarization pipeline
        
        Args:
            hf_token: Hugging Face token for accessing PyAnnote models
        """
        try:
            print("🔄 Loading PyAnnote diarization model...")
            self.pipeline = Pipeline.from_pretrained(
                "pyannote/speaker-diarization-3.1",
                use_auth_token=hf_token
            )
            
            # Move to GPU if available
            self.device = torch.device(
                "cuda" if torch.cuda.is_available() 
                else "mps" if torch.backends.mps.is_available() 
                else "cpu"
            )
            
            self.pipeline.to(self.device)
            print("✅ PyAnnote diarization model loaded successfully")
        except Exception as e:
            print(f"❌ Error loading PyAnnote model: {e}")
            self.pipeline = None

    def diarize(self, audio_path: str, num_speakers=None, min_speakers=2, max_speakers=4):
        """
        Perform speaker diarization
        
        Args:
            audio_path: Path to audio file
            num_speakers: Fixed number of speakers (optional)
            min_speakers: Minimum speakers for auto-detection
            max_speakers: Maximum speakers for auto-detection
            
        Returns:
            List of speaker segments or None if failed
        """
        if self.pipeline is None:
            print("❌ PyAnnote pipeline not available")
            return None
        
        try:
            print(f"👥 Performing diarization on: {audio_path}")
            
            # Configure diarization parameters
            diarization_params = {}
            if num_speakers and num_speakers > 0:
                diarization_params['num_speakers'] = num_speakers
                print(f"🎯 Fixed speakers: {num_speakers}")
            else:
                diarization_params['min_speakers'] = max(1, min_speakers)
                diarization_params['max_speakers'] = min(6, max_speakers)
                print(f"🔍 Auto-detection: {min_speakers}-{max_speakers} speakers")
            
            # Run diarization
            with ProgressHook() as hook:
                diarization = self.pipeline(audio_path, hook=hook, **diarization_params)
            
            # Extract speaker segments
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
            
            print(f"✅ Diarization completed")
            print(f"👥 Speakers found: {sorted(list(speakers_found))}")
            print(f"📊 Speaker segments: {len(speaker_segments)}")
            
            return speaker_segments
            
        except Exception as e:
            print(f"❌ Error during diarization: {e}")
            return None


class SpeakerAligner:
    def align(self, transcription_segments, speaker_segments):
        """
        Align transcription segments with speaker segments
        
        Args:
            transcription_segments: List of transcription segments with timestamps
            speaker_segments: List of speaker segments from diarization
            
        Returns:
            List of aligned segments with speaker labels
        """
        print("🔗 Aligning transcription with speaker segments...")
        
        if not speaker_segments:
            print("⚠️ No speaker segments available - using default speakers")
            # Fallback: alternate between SPEAKER_00 and SPEAKER_01
            aligned_segments = []
            for i, seg in enumerate(transcription_segments):
                speaker = f"SPEAKER_{i % 2:02d}"
                aligned_segments.append({
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"],
                    "speaker": speaker,
                    "confidence": seg["confidence"],
                    "words": seg.get("words", []),
                    "speaker_coverage": 0.5,  # Default coverage
                    "attribution_method": "fallback_alternation"
                })
            return aligned_segments
        
        aligned_segments = []
        known_speakers = list(set(seg["speaker"] for seg in speaker_segments))
        
        for trans_seg in transcription_segments:
            trans_start = trans_seg["start"]
            trans_end = trans_seg["end"]
            trans_duration = trans_end - trans_start
            
            # Find best matching speaker segment
            best_speaker = None
            best_coverage = 0
            
            for spk_seg in speaker_segments:
                spk_start = spk_seg["start"]
                spk_end = spk_seg["end"]
                
                # Calculate overlap
                overlap_start = max(trans_start, spk_start)
                overlap_end = min(trans_end, spk_end)
                overlap = max(0, overlap_end - overlap_start)
                coverage = overlap / trans_duration if trans_duration > 0 else 0
                
                if coverage > best_coverage:
                    best_coverage = coverage
                    best_speaker = spk_seg["speaker"]
            
            # Fallback to first speaker if no good match
            if not best_speaker:
                best_speaker = known_speakers[0] if known_speakers else "SPEAKER_00"
                best_coverage = 0.1
            
            # Create aligned segment
            aligned_segments.append({
                "start": trans_start,
                "end": trans_end,
                "text": trans_seg["text"],
                "speaker": best_speaker,
                "confidence": trans_seg["confidence"],
                "words": trans_seg.get("words", []),
                "speaker_coverage": best_coverage,
                "attribution_method": "overlap_based",
                "has_word_timestamps": trans_seg.get("has_word_timestamps", False)
            })
        
        print(f"✅ Alignment completed: {len(aligned_segments)} segments aligned")
        return aligned_segments

    def merge_consecutive_segments(self, segments):
        """Merge consecutive segments from the same speaker"""
        if len(segments) < 2:
            return segments
        
        merged_segments = []
        current_segment = segments[0]
        
        for next_segment in segments[1:]:
            # If same speaker and close in time, merge
            if (current_segment["speaker"] == next_segment["speaker"] and
                next_segment["start"] - current_segment["end"] < 1.0):
                
                # Merge segments
                current_segment = {
                    "start": current_segment["start"],
                    "end": next_segment["end"],
                    "text": current_segment["text"] + " " + next_segment["text"],
                    "speaker": current_segment["speaker"],
                    "confidence": (current_segment["confidence"] + next_segment["confidence"]) / 2,
                    "words": current_segment.get("words", []) + next_segment.get("words", []),
                    "speaker_coverage": (current_segment["speaker_coverage"] + next_segment["speaker_coverage"]) / 2,
                    "attribution_method": current_segment["attribution_method"],
                    "merged": True
                }
            else:
                merged_segments.append(current_segment)
                current_segment = next_segment
        
        merged_segments.append(current_segment)
        
        print(f"🔗 Merged {len(segments)} → {len(merged_segments)} segments")
        return merged_segments


class AudioProcessor:
    def __init__(self, hf_token: str, whisper_model="large-v2"):
        """
        Initialize the audio processor with faster-whisper and PyAnnote
        
        Args:
            hf_token: Hugging Face token for PyAnnote
            whisper_model: Faster-whisper model size
        """
        self.transcriber = WhisperAudioTranscriber(whisper_model)
        self.diarizer = PyannoteDiarizer(hf_token)
        self.aligner = SpeakerAligner()

    def process_audio(self, audio_path: str, language="fr", num_speakers=None, min_speakers=2, max_speakers=4) -> dict:
        """
        Process audio file with transcription and diarization using faster-whisper
        
        Args:
            audio_path: Path to the audio file
            language: Language for transcription
            num_speakers: Fixed number of speakers (optional)
            min_speakers: Minimum speakers for auto-detection
            max_speakers: Maximum speakers for auto-detection
            
        Returns:
            Dictionary containing results
        """
        print(f"🚀 Processing audio: {audio_path}")
        
        # Step 1: Transcription with faster-whisper
        transcription, segments = self.transcriber.transcribe(audio_path, language)
        if transcription is None:
            return {"error": "Transcription failed"}
        
        # Step 2: Speaker diarization
        speaker_segments = self.diarizer.diarize(
            audio_path, 
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers
        )
        
        # Step 3: Align transcription with speakers
        aligned_segments = self.aligner.align(segments, speaker_segments)
        
        # Step 4: Optional merging of consecutive segments
        final_segments = self.aligner.merge_consecutive_segments(aligned_segments)
        
        # Calculate results
        speakers_detected = len(set(seg["speaker"] for seg in final_segments))
        diarization_available = speaker_segments is not None
        
        result = {
            "success": True,
            "transcription": transcription,
            "segments": final_segments,
            "speakers_detected": speakers_detected,
            "language": language,
            "diarization_available": diarization_available,
            "total_segments": len(final_segments),
            "engine": "faster-whisper",
            "word_timestamps_available": any(seg.get("has_word_timestamps", False) for seg in final_segments)
        }
        
        return result

    def format_timestamp(self, seconds):
        """Convert seconds to mm:ss format"""
        return str(timedelta(seconds=int(seconds)))[2:]

    def print_formatted_results(self, results: dict):
        """Print formatted transcription results"""
        if "error" in results:
            print(f"❌ Error: {results['error']}")
            return
        
        print("\n" + "="*60)
        print("🎯 TRANSCRIPTION WITH SPEAKER DIARIZATION")
        print("🚀 Engine: faster-whisper + PyAnnote")
        print("="*60)
        
        segments = results.get("segments", [])
        speakers = set(seg["speaker"] for seg in segments)
        
        print(f"📊 Summary:")
        print(f"   📝 Total segments: {len(segments)}")
        print(f"   👥 Speakers detected: {len(speakers)}")
        print(f"   🌍 Language: {results.get('language', 'unknown')}")
        print(f"   🎭 Diarization: {'✅' if results.get('diarization_available') else '❌'}")
        print(f"   🔤 Word timestamps: {'✅' if results.get('word_timestamps_available') else '❌'}")
        
        print(f"\n📝 FULL TRANSCRIPTION:")
        print(f"   {results.get('transcription', 'No transcription')}")
        
        print(f"\n👥 CONVERSATION BY SPEAKERS:")
        print("-" * 60)
        
        current_speaker = None
        for segment in segments:
            speaker = segment["speaker"]
            start_time = self.format_timestamp(segment["start"])
            end_time = self.format_timestamp(segment["end"])
            confidence = int(segment["confidence"] * 100)
            coverage = int(segment.get("speaker_coverage", 0) * 100)
            
            # Show speaker change
            if speaker != current_speaker:
                print(f"\n🗣️ {speaker}:")
                current_speaker = speaker
            
            # Quality indicators
            conf_icon = "🟢" if confidence > 70 else "🟡" if confidence > 40 else "🔴"
            cov_icon = "🟢" if coverage > 60 else "🟡" if coverage > 30 else "🔴"
            word_icon = "🔤" if segment.get("has_word_timestamps") else ""
            
            print(f"   [{start_time}-{end_time}] {segment['text']}")
            print(f"      └─ {conf_icon}Conf:{confidence}% {cov_icon}Spk:{coverage}% {word_icon}")

    def save_results(self, results: dict, output_path: str = "transcription_results.json"):
        """Save results to JSON file"""
        # Convert any non-serializable objects
        serializable_results = {}
        for key, value in results.items():
            try:
                json.dumps(value)  # Test if serializable
                serializable_results[key] = value
            except:
                serializable_results[key] = str(value)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, ensure_ascii=False, indent=2)
        print(f"💾 Results saved to: {output_path}")


def main():
    """Main function for testing"""
    # Configuration
    HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
    if not HF_TOKEN:
        print("❌ Error: Please set HF_TOKEN or HUGGINGFACE_TOKEN environment variable")
        print("🔑 Get token from: https://huggingface.co/settings/tokens")
        return
    
    # Audio file path
    audio_file = "example_audio.wav"  # Change this to your audio file path
    
    if not os.path.exists(audio_file):
        print(f"❌ Error: Audio file '{audio_file}' not found")
        print("📁 Please provide a valid audio file path")
        return
    
    try:
        # Initialize processor with faster-whisper
        print("🔄 Initializing faster-whisper + PyAnnote processor...")
        processor = AudioProcessor(HF_TOKEN, whisper_model="large-v2")
        
        # Process audio
        results = processor.process_audio(
            audio_file,
            language="fr",  # Change to your language
            num_speakers=None,  # Auto-detect, or set fixed number
            min_speakers=2,
            max_speakers=4
        )
        
        # Display results
        processor.print_formatted_results(results)
        
        # Save results
        processor.save_results(results)
        
    except Exception as e:
        print(f"❌ An error occurred: {e}")
        print(f"🔍 Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()
