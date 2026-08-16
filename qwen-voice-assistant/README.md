# Emotion-Aware Conversational AI Assistant

An emotion-aware voice assistant that adapts responses based on real-time facial expression and speech emotion recognition. Built with Qwen 3 LLM (LoRA fine-tuned) and Microsoft VibeVoice for natural speech synthesis.

## Project Overview

This project creates a privacy-focused conversational AI that:
- **Detects emotions** from facial expressions and voice tone
- **Adapts responses** dynamically based on detected emotional state
- **Synthesizes natural speech** using VibeVoice TTS with emotional nuance
- **Runs locally** on NVIDIA Jetson Orin Nano for complete privacy

### Architecture

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Jetson Nano    │     │   Web Server     │     │   Web Browser   │
│  (Edge Device)  │────▶│   (FastAPI)      │◀───▶│   (React UI)    │
│                 │     │                  │     │                 │
│ - Face Emotion  │     │ - Qwen3 LLM      │     │ - Chat UI       │
│ - Voice Emotion │     │ - VibeVoice TTS  │     │ - Camera/Mic    │
│ - Camera/Mic    │     │ - Emotion Logic  │     │ - User Profile  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
        │                        │
        └────── JSON ────────────┘
```

## Features

- **Emotion Detection**: 7 emotions (happy, sad, angry, surprised, disgusted, fearful, neutral)
- **Adaptive Responses**: LLM adjusts tone and content based on user's emotional state
- **Voice Synthesis**: Natural TTS with VibeVoice supporting multiple speakers
- **Privacy-First**: All processing happens on-device, no cloud dependency
- **Web Interface**: Modern chat UI with camera/microphone access
- **User Profiles**: Simple account creation with age-appropriate responses

## Project Structure

```
qwen-voice-assistant/
├── backend/
│   ├── main.py              # FastAPI server
│   ├── llm_service.py       # Qwen3 model inference
│   ├── tts_service.py       # VibeVoice integration
│   ├── emotion_handler.py   # Emotion-aware response logic
│   └── user_service.py      # User account management
├── frontend/
│   ├── index.html           # Main chat interface
│   ├── styles.css           # UI styling
│   └── app.js               # Frontend logic
├── train_model.py           # LoRA fine-tuning script (Unsloth)
├── run_model.py             # CLI text chat with emotion
├── voice_chat.py            # CLI chat + VibeVoice TTS
└── requirements.txt         # Python dependencies
```

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/JacobS4914/qwen-voice-assistant.git
cd qwen-voice-assistant
```

### 2. Install Dependencies
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# or: .venv\Scripts\activate  # Windows

pip install -r requirements.txt
```

### 3. Run the Web Server
```bash
cd backend
python main.py
```

### 4. Open the Web Interface
Navigate to `http://localhost:8000` in your browser.

## Configuration

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `VIBEVOICE_REPO_PATH` | Path to VibeVoice repository | `./VibeVoice` |
| `VIBEVOICE_PYTHON` | Python executable for VibeVoice | `python` |
| `TTS_MODEL_ID` | VibeVoice model ID | `microsoft/VibeVoice-Realtime-0.5B` |
| `TTS_SPEAKER_NAME` | TTS voice speaker | `Carter` |
| `MODEL_PATH` | Qwen3 model path | `Qwen/Qwen3-4B-Instruct-2507` |
| `LORA_ADAPTER_PATH` | LoRA adapter path | `qwen_lora` |

## Model Training

The assistant uses Qwen 3 (4B parameters) fine-tuned with LoRA for conversational tasks.

```bash
python train_model.py
```

Training configuration:
- Base model: `unsloth/Qwen3-4B-Instruct-2507`
- LoRA rank: 16
- Learning rate: 2e-4
- Dataset: FineTome-100k

## Emotion Response System

The assistant detects emotions and adapts its responses:

| Emotion | Response Style |
|---------|---------------|
| Happy | Matches positive energy, enthusiastic |
| Sad | Warm, supportive, empathetic |
| Frustrated | Calm, patient, clear explanations |
| Anxious | Grounding, reassuring, measured |
| Surprised | Acknowledging, steady demeanor |
| Disgusted | Redirecting, respectful |
| Neutral | Standard helpful assistant |

## Integration with Jetson

The web server accepts emotion data via JSON from the Jetson edge device:

```json
{
  "face_emotion": "happy",
  "voice_emotion": "neutral",
  "confidence": 0.85
}
```

This allows the Jetson to handle real-time emotion detection while the web server manages the LLM and TTS processing.

## Hardware Requirements

### Web Server (LLM + TTS)
- GPU with 8GB+ VRAM (for Qwen3-4B)
- 16GB RAM
- SSD storage recommended

### Edge Device (Emotion Detection)
- NVIDIA Jetson Orin Nano
- USB Camera
- USB Microphone

## Technologies

- **LLM**: Qwen 3 (4B) with LoRA fine-tuning via Unsloth
- **TTS**: Microsoft VibeVoice (Realtime 0.5B)
- **Backend**: FastAPI, Python 3.10+
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Training**: Unsloth, PEFT, TRL, Accelerate

## Team

- **Thuc Bao Tran** - LLM Integration, Web Development
- **Jacob Seward** - Model Fine-tuning, VibeVoice Integration
- **Mert Hamsioglu** - Edge Computing, Emotion Detection

## License

MIT License

## Acknowledgments

- San Jose State University Engineering Success Program
- Microsoft VibeVoice Team
- Alibaba Qwen Team
- Unsloth Team
