import os
import torch
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
import shutil
import traceback
from datetime import datetime
from nemo.collections.asr.models.msdd_models import NeuralDiarizer
from helpers import create_config
import types
from tqdm import tqdm
from faster_whisper import WhisperModel

app = FastAPI()

models = {
    "whisper": None,
    "diarizer": None,
    "whisper_model_name": None,
    "pyannote_pipeline": None,
}

NEMO_AVAILABLE = True
HELPERS_AVAILABLE = True
PYANNOTE_AVAILABLE = False

def transcribe_whisper(audio_path: str, model_size="medium", device="cuda"):
    print("🔤 Loading Whisper model...")
    model = WhisperModel(model_size, device=device, compute_type="float16" if device == "cuda" else "int8")
    print("🎙️ Starting transcription...")
    segments, info = model.transcribe(audio_path)
    print("✅ Transcription complete.")
    return "".join([seg.text for seg in segments])

def run_nemo_diarization(audio_path: str, temp_dir: str, device: str = "cuda"):
    print("🎛️ Starting diarization with NeMo...")
    os.makedirs(temp_dir, exist_ok=True)
    waveform, sr = torchaudio.load(audio_path)

    if sr != 16000 or waveform.shape[0] != 1:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
        waveform = torch.mean(waveform, dim=0, keepdim=True)
        torchaudio.save(os.path.join(temp_dir, "mono_file.wav"), waveform, 16000)
    else:
        torchaudio.save(os.path.join(temp_dir, "mono_file.wav"), waveform, 16000)

    print("📦 Loading NeMo diarizer model...")
    diarizer = NeuralDiarizer(cfg=create_config(temp_dir)).to(device)

    def patched_run_vad(self, manifest_vad_input):
        import soundfile as sf

        vad_outputs = []
        dataloader = self._vad_model.test_dataloader()

        for batch in tqdm(dataloader, desc="vad", leave=True, disable=True):
            inputs, audio_paths = batch[:2]
            logits = self._vad_model.forward(processed_signal=inputs)
            preds = logits.sigmoid().cpu().numpy()

            for idx, (pred, path) in enumerate(zip(preds, audio_paths)):
                out_path = os.path.join(self.vad_out_dir, os.path.basename(path).replace('.wav', '.rttm'))
                speech_segments = []
                threshold = 0.5
                sr = self._cfg.sample_rate
                frame_shift = 0.01

                start = None
                for i, p in enumerate(pred):
                    if p >= threshold and start is None:
                        start = i * frame_shift
                    elif p < threshold and start is not None:
                        end = i * frame_shift
                        speech_segments.append((start, end - start))
                        start = None
                if start is not None:
                    end = len(pred) * frame_shift
                    speech_segments.append((start, end - start))

                with open(out_path, 'w') as fout:
                    for seg in speech_segments:
                        fout.write(
                            f"SPEAKER {path} 1 {seg[0]:.3f} {seg[1]:.3f} <NA> <NA> speaker0 <NA> <NA>\n"
                        )

    diarizer.clustering_embedding.clus_diar_model._run_vad = types.MethodType(
        patched_run_vad, diarizer.clustering_embedding.clus_diar_model
    )

    print("🧠 Running diarization...")
    diarizer.diarize()
    print("✅ Diarization complete.")
    return os.path.join(temp_dir, "pred_rttms", "mono_file.rttm")


@app.post("/diarize")
async def diarize_endpoint(file: UploadFile = File(...)):
    audio_path = f"/tmp/{file.filename}"
    with open(audio_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    temp_dir = "/tmp/diarization"
    transcript = transcribe_whisper(audio_path)
    rttm_path = run_nemo_diarization(audio_path, temp_dir)

    with open(rttm_path, "r") as f:
        rttm_content = f.read()

    return {
        "transcript": transcript,
        "rttm": rttm_content
    }


# RunPod serverless handler
async def handler(job):
    job_input = job.get("input", {})

    try:
        print(f"🚀 New job: {job.get('id', 'unknown')} (GPU mode)")

        if "url" not in job_input:
            return {"error": "Missing 'url' in input."}

        # Download the audio file to /tmp
        import requests
        import tempfile
        audio_url = job_input["url"]
        print(f"🌐 Downloading from: {audio_url}")
        response = requests.get(audio_url, stream=True)

        if response.status_code != 200:
            return {"error": f"Failed to download file from {audio_url}"}

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False, dir="/tmp") as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    tmp_file.write(chunk)
            audio_path = tmp_file.name

        print(f"📥 Downloaded to {audio_path}")

        temp_dir = "/tmp/diarization"
        transcript = transcribe_whisper(audio_path)
        rttm_path = run_nemo_diarization(audio_path, temp_dir)

        with open(rttm_path, "r") as f:
            rttm_content = f.read()

        os.unlink(audio_path)

        return {
            "transcript": transcript,
            "rttm": rttm_content
        }

    except Exception as e:
        print(f"❌ Handler error: {str(e)}")
        print(traceback.format_exc())
        return {"error": str(e)}
