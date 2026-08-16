# Setup Instructions for AI Emotion Detection Project

## Overview
This is the GitHub-ready version of the AI emotion detection project that combines face and speech recognition to adapt AI response tones. Large datasets and regenerateable files have been removed to reduce repository size.

## Directory Structure
```
eng_success_github/
├── dataset_CUDA/          # Face recognition component
├── speech_recognition/    # Speech emotion recognition component  
├── qwen-voice-assistant/  # Integrated voice assistant (Qwen LLM + emotion detection)
├── poster/                # Project documentation and visuals
├── qwen-voice-assistant.code-workspace  # VS Code workspace
├── README.md              # This overview
├── .gitignore             # Recommended git ignore rules
└── SETUP_INSTRUCTIONS.md  # This file
```

## Prerequisites
- Python 3.8+
- Git (for version control)
- CUDA-compatible GPU (for optimal performance, though CPU fallback available)
- Required dependencies (see individual component requirements.txt files)

## Setup Steps

### 1. Initialize Git Repository (Optional)
```bash
cd eng_success_github
git init
git add .
git commit -m "Initial commit: AI emotion detection project"
```

### 2. Setup Individual Components

#### Dataset_CUDA (Face Recognition)
- Download face emotion dataset (e.g., FER-2013, RAF-DB, or similar)
- Place dataset in appropriate directory structure
- Install requirements: `pip install -r dataset_CUDA/requirements.txt`
- Train or download pretrained models (models/deploy.prototxt and models/res10_300x300_ssd_iter_140000_fp16.caffemodel are included)

#### Speech_Recognition (Speech Emotion Recognition)
- Download RAVDESS speech emotion dataset
- Place in: `speech_recognition/speech_emotion_project1/data/RAVDESS/`
- Install requirements: `pip install -r speech_recognition/speech_emotion_project1/requirements_pretrained.txt`
- The small emotion_model.pkl and emotion_model_improved.pkl files are included for testing

#### Qwen-Voice-Assistant (Integrated System)
- Install requirements: `pip install -r qwen-voice-assistant/requirements.txt`
- Setup VibeVoice dependency (already included, excludes large demo data)
- Configure API keys for Qwen LLM in backend/llm_service.py or similar
- Run: `python qwen-voice-assistant/run_model.py`

### 3. Verify Installation
Each component includes test scripts to verify functionality:
- dataset_CUDA: `python dataset_CUDA/infer_webcam_local.py` (requires webcam)
- speech_recognition: `python speech_recognition/speech_emotion_project1/train_model.py` (trains on RAVDESS)
- qwen-voice-assistant: `python qwen-voice-assistant/voice_chat.py` (starts voice interface)

## Important Notes

### Dataset Locations
- Face images: dataset_CUDA/{train,val,test}/{[emotion]/} (ImageNet-style structure)
- Speech audio: speech_recognition/speech_emotion_project1/data/RAVDESS/
- Outputs/generated files: various `outputs/` directories in each component

### Model Files Included
For immediate testing, the following pretrained models are included:
- dataset_CUDA/models/res10_300x300_ssd_iter_140000_fp16.caffemodel (face detection)
- speech_recognition/speech_emotion_project1/emotion_model_improved.pkl (Speech emotion - ~22MB)
- speech_recognition/speech_emotion_project1/deployment_package/emotion_model_improved.pkl (Deployment copy)

### Excluded Content (for GitHub)
The following large/regenerateable content was removed for GitHub:
- ❌ Large image/audio datasets (hundreds of MBs to GBs)
- ❌ Large model output files (*.pt, *.pth, *.h5 >10MB)
- ❌ Virtual environments (.venv, env/)
- ❌ Build/cache directories (__pycache__, .git)
- ❌ Temporary output files
- ❌ IDE and OS-specific files

To regenerate excluded content, run the training scripts in each component.

## Customization
- Adjust emotion categories in dataset_CUDA/dataset.py and speech_recognition code
- Modify tone adaptation logic in qwen-voice-assistant/backend/emotion_handler.py
- Change LLM parameters in qwen-voice-assistant/backend/llm_service.py
- Update voice synthesis parameters in qwen-voice-assistant/backend/tts_service.py

## Troubleshooting
Common issues and solutions:
1. **CUDA not available**: Install CPU-only versions of torch/tensorflow
2. **Missing datasets**: Follow setup instructions above to download required data
3. **Memory errors**: Reduce batch sizes in training scripts
4. **Audio issues**: Check microphone permissions and audio backend configurations
5. **LLM API errors**: Verify API key and internet connectivity for Qwen access

## Credits
This project combines:
- Face detection: OpenCV DNN module with Caffe model
- Speech emotion: Various audio feature extraction and ML models
- Language model: Qwen series from Alibaba Cloud
- Text-to-speech: VibeVoice or similar TTS systems
- Integration: Custom emotion-aware response generation framework

## License
See individual component directories for license information. 
- VibeVoice component includes its own LICENSE file
- Other components: check respective source files for copyright notices
