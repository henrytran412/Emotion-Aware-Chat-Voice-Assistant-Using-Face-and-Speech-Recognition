# AI Emotion Detection Project

This repository contains the cleaned and GitHub-ready version of the AI model for emotion detection using face and speech recognition. The project combines facial expression analysis and speech emotion recognition to detect user emotions and adapt the tone/style of AI responses accordingly.

## Project Structure

- `dataset_CUDA/` - Face recognition component (face detection, emotion classification from images)
- `speech_recognition/` - Speech emotion recognition component (audio processing, emotion classification from speech)
- `qwen-voice-assistant/` - Integrated voice assistant using Qwen LLM with emotion-aware tone adaptation
- `poster/` - Project documentation and presentation materials
- `qwen-voice-assistant.code-workspace` - VS Code workspace configuration

## Important Notes

This version has been cleaned for GitHub distribution by removing:
- Large datasets (face images, speech audio datasets) - users need to download these separately
- Large model output files that can be regenerated through training
- Virtual environments (.venv)
- Build/cache directories (__pycache__, .git)
- Temporary output files

To run the project, users will need to:
1. Download required datasets (instructions in respective component READMEs)
2. Install dependencies using the provided requirements.txt files
3. Train or download pretrained models as needed
4. Run the individual components or the integrated voice assistant

## Components

### Dataset_CUDA
Face detection and emotion recognition from webcam/images. Contains code for training and inference on facial expression datasets.

### Speech_Recognition
Speech emotion recognition using audio features and deep learning models. Includes preprocessing, training, and inference code for emotion detection from speech.

### Qwen-Voice-Assistant
Integrated system that combines:
- Face emotion recognition (from dataset_CUDA component)
- Speech emotion recognition (from speech_recognition component)
- Qwen LLM for text generation
- Adaptive tone generation based on detected emotions
- Voice output generation

See individual component directories for detailed setup and usage instructions.

## Setup

Each component has its own setup instructions:
- See `dataset_CUDA/README.txt` or similar files
- See `speech_recognition/README*.txt` files
- See `qwen-voice-assistant/README.md` and `SETUP_GUIDE.md`

## Citation

If you use this code in your research, please cite the appropriate papers/models used in each component.
