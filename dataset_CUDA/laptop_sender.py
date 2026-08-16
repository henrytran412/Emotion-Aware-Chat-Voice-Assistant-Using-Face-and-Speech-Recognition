import time
import cv2
import requests

# ===== CONFIG =====
JETSON_IP = "192.168.55.1"
URL = f"http://{JETSON_IP}:8000/infer"
CAMERA_INDEX = 0
CONNECT_TIMEOUT_SEC = 3.0
READ_TIMEOUT_SEC = 15.0
MAX_FPS = 3
JPEG_QUALITY = 80

session = requests.Session()

cap = cv2.VideoCapture(CAMERA_INDEX)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

last_send = 0.0
min_interval = 1.0 / max(1, MAX_FPS)
label_text = "No face"

while True:
    ok, frame = cap.read()
    if not ok:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(60, 60)
    )

    if len(faces) > 0:
        x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
        face = frame[y:y + h, x:x + w]

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        now = time.time()
        if now - last_send >= min_interval:
            last_send = now

            face_small = cv2.resize(face, (224, 224), interpolation=cv2.INTER_LINEAR)

            ok2, jpg = cv2.imencode(
                ".jpg",
                face_small,
                [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY],
            )

            if ok2:
                try:
                    r = session.post(
                        URL,
                        files={"image": ("face.jpg", jpg.tobytes(), "image/jpeg")},
                        timeout=(CONNECT_TIMEOUT_SEC, READ_TIMEOUT_SEC),
                    )

                    if r.ok:
                        out = r.json()
                        label_text = f"{out['label']} ({out['confidence']:.2f})"
                    else:
                        label_text = f"HTTP {r.status_code}"

                except requests.exceptions.ConnectTimeout:
                    label_text = "ERR ConnectTimeout"
                except requests.exceptions.ReadTimeout:
                    label_text = "ERR ReadTimeout"
                except requests.exceptions.ConnectionError as e:
                    label_text = f"ERR ConnectionError: {e}"
                except Exception as e:
                    label_text = f"ERR {type(e).__name__}: {e}"
    else:
        label_text = "No face"

    cv2.putText(
        frame,
        label_text,
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        (255, 255, 255),
        2,
    )

    cv2.imshow("Laptop (face crop -> Jetson inference)", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
session.close()