# Setup Guide: Distributed Deployment

This guide explains how to run the Emotion-Aware AI Assistant across two laptops.

## Architecture Overview

```
┌─────────────────────┐          ┌─────────────────────┐
│   YOUR LAPTOP       │          │  FRIEND'S LAPTOP    │
│   (Frontend)        │   HTTP   │  (Backend Server)   │
│                     │ ───────▶ │                     │
│   Browser only      │          │  - FastAPI server   │
│                     │ ◀─────── │  - Qwen3-4B LLM     │
│   Opens:            │          │  - VibeVoice TTS    │
│   http://friend:8000│          │  - Serves frontend  │
└─────────────────────┘          └─────────────────────┘
```

## What Runs Where

### Friend's Laptop (Backend Server)

Runs the entire project. Needs:
- **GPU with 8GB+ VRAM** (for fast LLM inference)
- Python 3.10+
- All project files

**Files used:**
```
qwen-voice-assistant/
├── backend/
│   ├── main.py              ← RUN THIS
│   ├── llm_service.py       ← LLM inference
│   ├── tts_service.py       ← Voice synthesis
│   ├── emotion_handler.py   ← Emotion logic
│   └── user_service.py      ← User accounts
├── frontend/                 ← Served to browser
│   ├── index.html
│   ├── styles.css
│   ├── app.js
│   └── emotion_detector.js
├── VibeVoice/               ← Optional: for voice output
└── requirements.txt
```

### Your Laptop (Frontend Only)

Just needs a web browser. No files needed locally.

---

## Setup Instructions

### Step 1: Friend's Laptop Setup

#### 1.1 Copy/Clone the Project
```bash
# Copy the qwen-voice-assistant folder to friend's laptop
# Or clone from GitHub:
git clone https://github.com/JacobS4914/qwen-voice-assistant.git
cd qwen-voice-assistant
```

#### 1.2 Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv .venv

# Activate it
# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

#### 1.3 (Optional) Install VibeVoice for Voice Output
```bash
cd qwen-voice-assistant
git clone https://github.com/microsoft/VibeVoice.git
cd VibeVoice
pip install -r requirements.txt
cd ..
```

#### 1.4 Run the Backend Server
```bash
cd backend
python main.py
```

For faster response on RTX GPUs (recommended), set:
```bash
# PowerShell
$env:USE_4BIT="false"
$env:MAX_NEW_TOKENS="120"
$env:TEMPERATURE="0.6"
$env:TOP_P="0.9"
python main.py
```

First run will download the Qwen3-4B model (~8GB). Wait for:
```
Model loaded successfully!
Services loaded successfully
INFO:     Uvicorn running on http://0.0.0.0:8000
```

#### 1.5 Find Your IP Address
```bash
# Windows
ipconfig

# Linux
ip a

# Look for IPv4 Address like: 192.168.1.50
```

#### 1.6 Configure Firewall
Make sure port 8000 is allowed through the firewall:
- Windows: Allow Python through Windows Firewall when prompted
- Or manually add rule for port 8000

---

### Step 2: Your Laptop Setup

Just open a browser and go to:
```
http://<friend-ip>:8000
```

Example: `http://192.168.1.50:8000`

That's it! The frontend will load from your friend's server.

---

## Troubleshooting

### "Connection refused" or page won't load
- Check friend's server is running (`python main.py`)
- Check you're on the same network (WiFi/LAN)
- Check firewall allows port 8000
- Try pinging friend's IP: `ping 192.168.1.50`

### Slow responses (30+ seconds)
- Friend is running on CPU instead of GPU
- Check CUDA is installed: `python -c "import torch; print(torch.cuda.is_available())"`
- Should print `True` for GPU acceleration

### No voice output
- VibeVoice not installed
- See Step 1.3 above

### CSS not loading / ugly UI
- Clear browser cache and refresh
- Check browser console for errors (F12)

---

## Network Requirements

Both laptops must be on the **same local network** (WiFi or LAN).

If you need to connect over the internet (different networks), use:
- **ngrok**: `ngrok http 8000` (creates public URL)
- **Tailscale**: VPN that connects devices
- **Port forwarding**: On friend's router (advanced)

---

## Quick Reference

| Component | Location | Command |
|-----------|----------|---------|
| Backend Server | Friend's laptop | `cd backend && python main.py` |
| Frontend | Your browser | `http://<friend-ip>:8000` |
| Model Download | Automatic | First run (~8GB) |
| VibeVoice | Optional | Clone into project root |

---

## Team Contacts

- **Thuc Bao Tran** - LLM Integration, Web Development
- **Jacob Seward** - Model Fine-tuning, VibeVoice Integration
- **Mert Hamsioglu** - Edge Computing, Emotion Detection
