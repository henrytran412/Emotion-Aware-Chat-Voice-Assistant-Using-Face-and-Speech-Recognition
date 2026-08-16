import json
import os
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def load_audio(audio_path: str, target_sr: int = 16000):
    audio, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
    audio = audio.mean(axis=1)
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    waveform = torch.from_numpy(audio).unsqueeze(0)
    if sample_rate != target_sr:
        waveform = torchaudio.functional.resample(waveform, sample_rate, target_sr)
    audio = waveform.squeeze(0).numpy()
    audio = np.nan_to_num(audio, nan=0.0, posinf=0.0, neginf=0.0)
    return audio, target_sr


def main():
    if len(sys.argv) < 2:
        print("Usage: python predict_pretrained.py <audio_file.wav> [model_dir]")
        sys.exit(1)

    audio_file = sys.argv[1]
    root = Path(__file__).resolve().parent
    model_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(
        os.environ.get("MODEL_DIR", str(root / "pretrained_outputs"))
    )

    if not os.path.isfile(audio_file):
        raise FileNotFoundError(f"Audio file not found: {audio_file}")
    if not model_dir.exists():
        raise FileNotFoundError(f"Model dir not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
    model = AutoModelForAudioClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    audio, sr = load_audio(audio_file, target_sr=16000)
    inputs = feature_extractor(
        audio,
        sampling_rate=sr,
        return_attention_mask=False,
        return_tensors="pt",
        truncation=True,
        max_length=sr * 6,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1).squeeze(0).cpu()

    pred_id = int(torch.argmax(probs).item())
    pred_label = model.config.id2label[pred_id]
    pred_conf = float(probs[pred_id].item())

    ranking = sorted(
        [{"label": model.config.id2label[i], "probability": float(probs[i].item())} for i in range(len(probs))],
        key=lambda x: x["probability"],
        reverse=True,
    )

    print(f"Predicted emotion: {pred_label} ({pred_conf:.2%})")
    print(json.dumps(ranking, indent=2))


if __name__ == "__main__":
    main()
