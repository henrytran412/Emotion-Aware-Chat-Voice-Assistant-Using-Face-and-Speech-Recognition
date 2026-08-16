import os
import json
import cv2
import numpy as np
import torch
import timm

# ---------------- PATHS (portable: Windows + Jetson) ----------------
ROOT = os.path.dirname(os.path.abspath(__file__))  # folder containing this script
OUT_DIR = os.path.join(ROOT, "outputs")
MODEL_DIR = os.path.join(ROOT, "models")

WEIGHTS = os.path.join(OUT_DIR, "best.pth")
LABELS_TXT = os.path.join(OUT_DIR, "labels.txt")
PREP_JSON = os.path.join(OUT_DIR, "preprocess.json")

PROTO = os.path.join(MODEL_DIR, "deploy.prototxt")
CAFFE = os.path.join(MODEL_DIR, "res10_300x300_ssd_iter_140000_fp16.caffemodel")
# -------------------------------------------------------------------

def load_labels(path):
    with open(path, "r", encoding="utf-8") as f:
        labels = [l.strip() for l in f if l.strip()]
    if len(labels) != 7:
        raise ValueError("labels.txt must contain exactly 7 labels")
    return labels

def load_preprocess(path):
    with open(path, "r", encoding="utf-8") as f:
        prep = json.load(f)
    return int(prep["img_size"]), np.array(prep["mean"], dtype=np.float32), np.array(prep["std"], dtype=np.float32)

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

def preprocess_face(face_bgr, img_size, mean, std):
    face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
    face_rgb = cv2.resize(face_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = face_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))
    x = np.expand_dims(x, axis=0)
    return torch.from_numpy(x).float()

def load_state_dict_compat(path, map_location):
    # torch.load(weights_only=True) isn't available in all versions; handle both.
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)

def open_camera():
    # Try normal OpenCV capture first
    cap = cv2.VideoCapture(0)
    if cap.isOpened():
        return cap

    # Jetson fallback (often works for CSI/USB cams via GStreamer)
    gst = (
        "v4l2src device=/dev/video0 ! "
        "video/x-raw, width=640, height=480, framerate=30/1 ! "
        "videoconvert ! appsink"
    )
    cap = cv2.VideoCapture(gst, cv2.CAP_GSTREAMER)
    return cap

def main():
    for p in [WEIGHTS, LABELS_TXT, PREP_JSON, PROTO, CAFFE]:
        if not os.path.isfile(p):
            raise FileNotFoundError(f"Missing file: {p}")

    labels = load_labels(LABELS_TXT)
    img_size, mean, std = load_preprocess(PREP_JSON)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    amp = (device.type == "cuda")
    print("Device:", device)
    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))

    model = timm.create_model("mobilenetv3_small_100", pretrained=False, num_classes=7)
    state = load_state_dict_compat(WEIGHTS, map_location="cpu")
    model.load_state_dict(state)
    model.to(device)
    model.eval()

    net = cv2.dnn.readNetFromCaffe(PROTO, CAFFE)

    cap = open_camera()
    if not cap.isOpened():
        raise RuntimeError("Cannot open camera (tried /dev/video0).")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    ema_probs = None
    ema_alpha = 0.6
    frame_id = 0
    last_label, last_conf = "No face", 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_id += 1

        h, w = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(
            frame, scalefactor=1.0, size=(300, 300),
            mean=(104.0, 177.0, 123.0)
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
                last_label = labels[idx]
                last_conf = float(ema_probs[idx])
        else:
            last_label, last_conf = "No face", 0.0
            ema_probs = None

        cv2.putText(
            frame, f"{last_label} ({last_conf:.2f})", (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2
        )

        cv2.imshow("Emotion Recognition (OpenCV DNN Face)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
