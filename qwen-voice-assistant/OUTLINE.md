# Deployment Split Outline

A guide for splitting the Qwen Voice Assistant between a GPU machine (backend) and a non-GPU machine (frontend).

## Overview

This outline helps you deploy the project across two machines:
- **Friend's Machine (RTX 4070)**: Runs LLM + TTS backend
- **Your Machine (No GPU)**: Runs frontend chat interface

## Architecture Split

```
┌─────────────────────┐         ┌──────────────────┐
│   Your Machine      │         │  Friend's GPU    │
│   (No GPU needed)   │◀───────▶│  Machine (4070)  │
│                     │  LAN    │                  │
│  - Frontend UI      │         │  - Qwen3 LLM     │
│  - Browser          │         │  - VibeVoice TTS │
│  - Development      │         │  - FastAPI API   │
└─────────────────────┘         └──────────────────┘
         │                               │
         └── http://192.168.x.x:8000 ────┘
```

## What Goes Where

### Friend's Machine (GPU Required)
**Files to send:**
- `backend/` folder (all files)
- `requirements.txt`
- `SETUP_GUIDE.md`
- `qwen_lora/` folder (optional, if fine-tuned)

**What they run:**
- FastAPI server on port 8000
- Qwen3-4B LLM inference
- VibeVoice TTS generation

**Hardware needs:**
- GPU with 8GB+ VRAM ✅ (RTX 4070 = 12GB)
- 16GB RAM
- ~15GB disk space

### Your Machine (No GPU)
**Files you keep:**
- `frontend/` folder (all files)
- `README.md`
- Project documentation

**What you run:**
- Open `frontend/index.html` in browser
- Configure Settings → Server URL
- Chat interface + emotion detection (browser-based)

**Hardware needs:**
- Any computer with modern browser
- Webcam (optional, for face emotion)
- Microphone (optional, for voice input)

## Setup Steps

### Phase 1: Friend Sets Up Backend

1. **Install Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

2. **Setup VibeVoice**
   ```bash
   git clone https://github.com/microsoft/VibeVoice.git
   ```

3. **Configure environment**
   - Create `.env` file in `backend/`
   - Set paths to models and VibeVoice

4. **Start server**
   ```bash
   cd backend
   python main.py
   ```

5. **Get IP address**
   ```bash
   ipconfig  # Windows
   ```
   - Share IP with you (e.g., `192.168.1.50`)

### Phase 2: You Configure Frontend

1. **Open frontend**
   - Open `frontend/index.html` in browser

2. **Configure server**
   - Click Settings icon (⚙️)
   - Enter: `http://192.168.1.50:8000`
   - Click Save

3. **Test connection**
   - Create account
   - Send message
   - Verify response from friend's LLM

## Network Requirements

- **Same WiFi network** required
- **Firewall rule** on friend's machine:
  ```powershell
  New-NetFirewallRule -DisplayName "Qwen Backend" -Direction Inbound -LocalPort 8000 -Protocol TCP -Action Allow
  ```

## API Endpoints

Your frontend will call these on friend's server:

| Endpoint | Purpose |
|----------|---------|
| `POST /api/users` | Create account |
| `POST /api/chat` | Send message, get AI response |
| `GET /api/audio/{filename}` | Get TTS audio file |

## Configuration Files

### Friend's Backend (.env)
```env
MODEL_PATH=Qwen/Qwen3-4B-Instruct-2507
VIBEVOICE_REPO_PATH=../VibeVoice
TTS_SPEAKER_NAME=Carter
```

### Your Frontend (localStorage)
```javascript
api_base: "http://192.168.1.50:8000"
face_emotion_enabled: true
```

## Verification Checklist

### Friend's Backend
- [ ] Server starts without errors
- [ ] Console shows "Model loaded successfully on cuda"
- [ ] Server binds to `0.0.0.0:8000`
- [ ] Firewall allows port 8000

### Your Frontend
- [ ] Settings modal shows server URL
- [ ] Account creation works
- [ ] Chat messages get responses
- [ ] Audio playback works
- [ ] No CORS errors in browser console

### Network Connection
- [ ] Can ping friend's IP
- [ ] Can access `http://friend-ip:8000` in browser
- [ ] API calls succeed from your frontend

## Troubleshooting

### "Cannot connect to server"
1. Check friend's server is running
2. Verify IP address is correct
3. Check firewall on friend's machine
4. Ensure both on same WiFi

### "CUDA out of memory"
- Friend should close other GPU apps
- Model needs ~4-6GB VRAM

### "Model download slow"
- First run downloads ~8GB model
- Can take 10-30 minutes

## Development Workflow

### Your Work (Frontend)
1. Edit `frontend/` files
2. Refresh browser
3. Test against friend's backend
4. No restart needed

### Friend's Work (Backend)
1. Edit `backend/` files
2. Restart `python main.py`
3. You test immediately from frontend

## Performance

- **First message**: 5-10 seconds (model loading)
- **Later messages**: 1-3 seconds
- **VRAM usage**: ~4-6GB
- **Network latency**: <100ms (local LAN)

## File Checklist

### Send to Friend
- [ ] `backend/main.py`
- [ ] `backend/llm_service.py`
- [ ] `backend/tts_service.py`
- [ ] `backend/emotion_handler.py`
- [ ] `backend/user_service.py`
- [ ] `requirements.txt`
- [ ] `SETUP_GUIDE.md`
- [ ] `qwen_lora/` (optional)

### Keep for Yourself
- [ ] `frontend/index.html`
- [ ] `frontend/app.js`
- [ ] `frontend/styles.css`
- [ ] `frontend/emotion_detector.js`

## Quick Reference

| Setting | Value |
|---------|-------|
| Backend Port | 8000 |
| Frontend Config | Settings → Server URL |
| CORS | Enabled for all origins |
| Model Size | ~8GB download |
| VRAM Usage | ~4-6GB runtime |

## Team

- **Thuc Bao Tran** - Frontend Development (You)
- **Friend** - Backend Operations (GPU)

## Next Steps

1. Friend sets up backend following Phase 1
2. You configure frontend following Phase 2
3. Test end-to-end integration
4. Begin development workflow

---

**Ready to split?** Send the backend files to your friend and get started!
