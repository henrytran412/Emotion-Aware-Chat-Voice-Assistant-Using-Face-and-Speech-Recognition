"""Real-time microphone emotion prediction using a trained pretrained SER model.

This runner is intentionally standalone: it does not call the web frontend or LLM.
It only listens to mic audio and prints emotion predictions in the terminal.
"""

import argparse
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch
import requests
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


def parse_args():
    root = Path(__file__).resolve().parent
    default_model_dir = root / "pretrained_outputs_tuned"
    if not default_model_dir.exists():
        default_model_dir = root / "pretrained_outputs"

    parser = argparse.ArgumentParser(
        description="Run live microphone emotion detection using a trained pretrained model"
    )
    parser.add_argument(
        "--model-dir",
        type=str,
        default=str(default_model_dir),
        help="Path to trained model directory (contains config.json/model.safetensors)",
    )
    parser.add_argument("--sample-rate", type=int, default=16000, help="Microphone sample rate")
    parser.add_argument("--window-sec", type=float, default=3.0, help="Audio window length for each inference")
    parser.add_argument("--step-sec", type=float, default=1.0, help="Seconds between inferences")
    parser.add_argument("--top-k", type=int, default=3, help="How many top predictions to print")
    parser.add_argument(
        "--audio-backend",
        choices=["auto", "pyaudio", "sounddevice"],
        default="auto",
        help="Microphone backend (auto tries pyaudio then sounddevice)",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.6,
        help="EMA smoothing alpha for prediction probabilities (0 to disable)",
    )
    parser.add_argument("--backend-url", type=str, default="", help="Optional backend base URL to post voice emotion")
    parser.add_argument("--user-id", type=str, default="", help="User ID for backend emotion updates")
    parser.add_argument("--send-interval", type=float, default=0.7, help="Minimum seconds between backend posts")
    return parser.parse_args()


def main():
    args = parse_args()
    model_dir = Path(args.model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading model from: {model_dir}")
    print(f"Device: {device}")

    feature_extractor = AutoFeatureExtractor.from_pretrained(str(model_dir))
    model = AutoModelForAudioClassification.from_pretrained(str(model_dir)).to(device)
    model.eval()

    id2label = model.config.id2label
    if isinstance(next(iter(id2label.keys())), str):
        id2label = {int(k): v for k, v in id2label.items()}

    labels = [id2label[i] for i in sorted(id2label.keys())]
    print("Emotions:", ", ".join(labels))

    chunk_size = 1024
    window_samples = int(args.sample_rate * args.window_sec)
    step_sec = max(0.1, float(args.step_sec))

    audio_buffer = deque(maxlen=window_samples)
    last_infer_time = 0.0
    last_send_time = 0.0
    ema_probs = None

    post_enabled = bool(args.backend_url and args.user_id)
    post_url = args.backend_url.rstrip("/") + "/api/emotion" if post_enabled else ""
    http = requests.Session() if post_enabled else None

    stream = None
    pa = None
    sd = None
    audio_backend = None

    backend_order = [args.audio_backend] if args.audio_backend != "auto" else ["pyaudio", "sounddevice"]
    backend_error = None

    for backend in backend_order:
        try:
            if backend == "pyaudio":
                import pyaudio  # type: ignore

                pa = pyaudio.PyAudio()
                stream = pa.open(
                    format=pyaudio.paInt16,
                    channels=1,
                    rate=args.sample_rate,
                    input=True,
                    frames_per_buffer=chunk_size,
                )
                audio_backend = "pyaudio"
                break

            if backend == "sounddevice":
                import sounddevice as sd  # type: ignore

                audio_backend = "sounddevice"
                break
        except Exception as exc:
            backend_error = exc

    if audio_backend is None:
        raise RuntimeError(
            "No usable microphone backend found. Install one of:\n"
            "  pip install pyaudio\n"
            "  pip install sounddevice\n"
            f"Last backend error: {backend_error}"
        )

    print("\n" + "=" * 70)
    print("LIVE SPEECH EMOTION DETECTION (standalone)")
    print("Speak into your microphone. Press Ctrl+C to stop.")
    print(f"Window: {args.window_sec:.1f}s | Step: {step_sec:.1f}s | Top-K: {args.top_k}")
    print(f"Audio backend: {audio_backend}")
    if post_enabled:
        print(f"Posting voice predictions to: {post_url} (user_id={args.user_id})")
    print("=" * 70)

    try:
        while True:
            if audio_backend == "pyaudio":
                data = stream.read(chunk_size, exception_on_overflow=False)
                samples_i16 = np.frombuffer(data, dtype=np.int16)
            else:
                frames_f32 = sd.rec(chunk_size, samplerate=args.sample_rate, channels=1, dtype="float32")
                sd.wait()
                samples_i16 = np.clip(frames_f32.squeeze(-1) * 32768.0, -32768, 32767).astype(np.int16)

            audio_buffer.extend(samples_i16.tolist())

            now = time.time()
            if len(audio_buffer) < window_samples or (now - last_infer_time) < step_sec:
                continue

            last_infer_time = now

            audio_i16 = np.asarray(audio_buffer, dtype=np.int16)
            audio_f32 = np.clip(audio_i16.astype(np.float32) / 32768.0, -1.0, 1.0)

            inputs = feature_extractor(
                audio_f32,
                sampling_rate=args.sample_rate,
                return_attention_mask=False,
                return_tensors="pt",
                truncation=True,
                max_length=args.sample_rate * 6,
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                logits = model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).squeeze(0).cpu().numpy()

            alpha = float(args.ema_alpha)
            if alpha > 0:
                if ema_probs is None:
                    ema_probs = probs
                else:
                    ema_probs = alpha * ema_probs + (1.0 - alpha) * probs
                use_probs = ema_probs
            else:
                use_probs = probs

            top_k = max(1, min(int(args.top_k), len(use_probs)))
            top_ids = np.argsort(use_probs)[::-1][:top_k]

            timestamp = time.strftime("%H:%M:%S")
            top1 = int(top_ids[0])
            top1_label = id2label[top1]
            top1_conf = float(use_probs[top1])

            ranking = " | ".join([f"{id2label[int(i)]}:{float(use_probs[int(i)]):.2%}" for i in top_ids])
            print(f"[{timestamp}] {top1_label:10s} {top1_conf:6.2%} || {ranking}")

            if post_enabled and (now - last_send_time) >= max(0.1, float(args.send_interval)):
                last_send_time = now
                payload = {
                    "user_id": args.user_id,
                    "voice_emotion": top1_label,
                    "voice_confidence": top1_conf,
                    "confidence": top1_conf,
                }
                try:
                    r = http.post(post_url, json=payload, timeout=2.0)
                    if not r.ok:
                        print(f"[warn] backend post failed: {r.status_code}")
                except Exception as exc:
                    print(f"[warn] backend post error: {exc}")

    except KeyboardInterrupt:
        print("\nStopping realtime detector...")
    finally:
        if stream is not None:
            stream.stop_stream()
            stream.close()
        if pa is not None:
            pa.terminate()
        if http is not None:
            http.close()
        print("Done.")


if __name__ == "__main__":
    main()
