import os, json
import numpy as np
import cv2
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

app = Flask(__name__)

ROOT = os.path.dirname(os.path.abspath(__file__))

WEIGHTS_PATH = os.path.join(ROOT, "outputs", "best.pth")
PREPROCESS_PATH = os.path.join(ROOT, "outputs", "preprocess.json")
LABELS_PATH = os.path.join(ROOT, "outputs", "labels.txt")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
amp = (device.type == "cuda")

# ---------- Load preprocess + labels ----------
with open(PREPROCESS_PATH, "r") as f:
    pp = json.load(f)

img_size = int(pp["img_size"])
mean = np.array(pp["mean"], dtype=np.float32)
std  = np.array(pp["std"], dtype=np.float32)

with open(LABELS_PATH, "r") as f:
    labels = [ln.strip() for ln in f if ln.strip()]
num_classes = len(labels)

# ---------- Model definition (same as train.py) ----------
class BasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out = out + identity
        out = self.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=7):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64,  layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, 1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18(num_classes):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes)

model = resnet18(num_classes).to(device)
state = torch.load(WEIGHTS_PATH, map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()

if device.type == "cuda":
    model = model.to(memory_format=torch.channels_last)

def preprocess_bgr(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_rgb = cv2.resize(img_rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    x = img_rgb.astype(np.float32) / 255.0
    x = (x - mean) / std
    x = np.transpose(x, (2, 0, 1))          # CHW
    x = np.expand_dims(x, axis=0)           # NCHW
    t = torch.from_numpy(x).float().to(device, non_blocking=True)
    if device.type == "cuda":
        t = t.contiguous(memory_format=torch.channels_last)
    return t

@app.post("/infer")
def infer():
    if "image" not in request.files:
        return jsonify({"error": "missing file field 'image'"}), 400

    data = request.files["image"].read()
    arr = np.frombuffer(data, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)  # BGR
    if img is None:
        return jsonify({"error": "cannot decode image"}), 400

    x = preprocess_bgr(img)

    with torch.no_grad():
        with torch.cuda.amp.autocast(enabled=amp):
            logits = model(x)
            probs = torch.softmax(logits, dim=1).float().cpu().numpy()[0]

    idx = int(np.argmax(probs))
    return jsonify({
        "label": labels[idx],
        "confidence": float(probs[idx]),
        "probs": probs.tolist()
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000)