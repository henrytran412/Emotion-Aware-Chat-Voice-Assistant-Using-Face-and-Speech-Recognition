import argparse
import base64
import json
import os
import time

import cv2
import numpy as np
import requests
import timm
import torch


ROOT = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")

WEIGHTS = os.path.join(OUT_DIR, "best.pth")
LABELS_TXT = os.path.join(OUT_DIR, "labels.txt")
PREP_JSON = os.path.join(OUT_DIR, "preprocess.json")
EVAL_SUMMARY = os.path.join(OUT_DIR, "eval_summary.json")
PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFE = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")


LABEL_MAP = {
    "anger": "angry",
    "disgust": "disgusted",
    "fear": "fearful",
    "happy": "happy",
    "sad": "sad",
    "surprise": "surprised",
    "neutral": "neutral",
}


def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]
    if len(labels) != 7:
        raise ValueError("labels.txt must contain exactly 7 labels")
    return labels


def load_preprocess(path):
    with open(path, "r", encoding="utf-8") as f:
        prep = json.load(f)
    return int(prep["img_size"]), np.array(prep["mean"], dtype=np.float32), np.array(prep["std"], dtype=np.float32)


def load_model_name(default_name="mobilenetv3_large_100"):
    if os.path.isfile(EVAL_SUMMARY):
        try:
            with open(EVAL_SUMMARY, "r", encoding="utf-8") as f:
                data = json.load(f)
            model_name = data.get("model_name")
            if isinstance(model_name, str) and model_name:
                return model_name
        except Exception:
            pass
    return default_name


def preprocess_face(face_bgr, img_size, mean, std):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = face_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return torch.from_numpy(x).float()


def expand_box(x1, y1, x2, y2, w, h, scale=1.25):
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = (x2 - x1) * scale
    bh = (y2 - y1) * scale
    nx1 = int(max(0, cx - bw / 2.0))
    ny1 = int(max(0, cy - bh / 2.0))
    nx2 = int(min(w - 1, cx + bw / 2.0))
    ny2 = int(min(h - 1, cy + bh / 2.0))
    return nx1, ny1, nx2, ny2


def load_state_dict_compat(path, map_location):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def encode_frame_to_jpeg_base64(frame_bgr, max_width=480, quality=70):
    h, w = frame_bgr.shape[:2]
    if max_width and w > max_width:
        scale = max_width / float(w)
        resized = cv2.resize(frame_bgr, (max_width, int(h * scale)), interpolation=cv2.INTER_AREA)
    else:
        resized = frame_bgr

    ok, enc = cv2.imencode(
        ".jpg",
        resized,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(max(25, min(95, quality)))],
    )
    if not ok:
        return None
    return base64.b64encode(enc.tobytes()).decode("ascii")


def main():
    parser = argparse.ArgumentParser(description="Run local emotion model and push emotion updates to qwen backend.")
    parser.add_argument("--backend-url", default="http://127.0.0.1:8000", help="Base URL for qwen backend")
    parser.add_argument("--user-id", required=True, help="Existing user_id from qwen app")
    parser.add_argument("--camera-index", type=int, default=0, help="Webcam index")
    parser.add_argument("--send-interval", type=float, default=0.6, help="Seconds between /api/emotion updates")
    parser.add_argument("--frame-max-width", type=int, default=480, help="Max width for bridge frame upload")
    parser.add_argument("--frame-quality", type=int, default=70, help="JPEG quality for bridge frame upload")
    args = parser.parse_args()

    for p in [WEIGHTS, LABELS_TXT, PREP_JSON, PROTO, CAFFE]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing file: {p}")

    labels = load_labels(LABELS_TXT)
    img_size, mean, std = load_preprocess(PREP_JSON)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"

    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model_name = load_model_name()
    print("Model backbone:", model_name)
    model = timm.create_model(model_name, pretrained=False, num_classes=7)
    state = load_state_dict_compat(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    net = cv2.dnn.readNetFromCaffe(PROTO, CAFFE)
    cap = cv2.VideoCapture(args.camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open camera index {args.camera_index}")

    session = requests.Session()
    endpoint = args.backend_url.rstrip("/") + "/api/emotion"
    last_sent_at = 0.0

    ema_probs = None
    ema_alpha = 0.6
    frame_id = 0
    label_text = "No face"

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1
        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0, size=(300, 300), mean=(104.0, 177.0, 123.0)
        )
        net.setInput(blob)
        dets = net.forward()

        best = None
        best_conf = 0.0
        for i in range(dets.shape[2]):
            conf = float(dets[0, 0, i, 2])
            if conf > best_conf:
                best_conf = conf
                box = dets[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                best = (x1, y1, x2, y2)

        curr_emotion = "neutral"
        curr_conf = 0.0

        if best is not None and best_conf >= 0.6:
            x1, y1, x2, y2 = best
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w - 1, x2), min(h - 1, y2)
            x1, y1, x2, y2 = expand_box(x1, y1, x2, y2, w, h, scale=1.25)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            face = frame[y1:y2, x1:x2]
            if face.size > 0 and frame_id % 2 == 0:
                x = preprocess_face(face, img_size, mean, std).to(device, non_blocking=True)
                with torch.no_grad():
                    with torch.cuda.amp.autocast(enabled=amp):
                        logits = model(x)
                        probs = torch.softmax(logits, dim=1).float().cpu().numpy()[0]
                ema_probs = probs if ema_probs is None else (ema_alpha * ema_probs + (1 - ema_alpha) * probs)
                idx = int(np.argmax(ema_probs))
                raw_label = labels[idx]
                curr_emotion = LABEL_MAP.get(raw_label, "neutral")
                curr_conf = float(ema_probs[idx])
                label_text = f"{curr_emotion} ({curr_conf:.2f})"
        else:
            ema_probs = None
            label_text = "No face"

        now = time.time()
        if now - last_sent_at >= args.send_interval:
            last_sent_at = now
            frame_b64 = encode_frame_to_jpeg_base64(
                frame,
                max_width=args.frame_max_width,
                quality=args.frame_quality,
            )
            payload = {
                "user_id": args.user_id,
                "face_emotion": curr_emotion,
                "voice_emotion": None,
                "confidence": curr_conf,
                "face_confidence": curr_conf,
                "bridge_frame_jpeg": frame_b64,
            }
            try:
                r = session.post(endpoint, json=payload, timeout=3.0)
                if not r.ok:
                    print("Emotion update failed:", r.status_code, r.text)
            except Exception as e:
                print("Emotion update error:", e)

        cv2.putText(
            frame,
            label_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
        )

        cv2.imshow("PC Emotion Bridge (press q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    session.close()


if __name__ == "__main__":
    main()
