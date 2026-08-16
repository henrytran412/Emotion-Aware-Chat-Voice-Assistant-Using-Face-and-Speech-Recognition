# Speech Emotion Recognition - Deployment Package

## Files Included
- `emotion_model_improved.pkl` - Trained Random Forest model (61% accuracy)
- `scaler_improved.pkl` - Feature normalization scaler
- `label_encoder_improved.pkl` - Emotion label encoder
- `realtime_emotion_improved.py` - Real-time recognition script
- `test_improved_model.py` - Single file testing script
- `requirements.txt` - Python dependencies

## Installation on Jetson Orin Nano

### 1. Install Dependencies
```bash
pip3 install -r requirements.txt
```

### 2. Test Installation
```bash
python3 test_improved_model.py test_audio.wav
```

### 3. Run Real-Time Recognition
```bash
python3 realtime_emotion_improved.py
```

## Model Details
- **Dataset**: RAVDESS (1,440 samples)
- **Features**: eGeMAPS (88 acoustic features)
- **Model**: Random Forest (300 estimators)
- **Accuracy**: 61% on 8 emotion classes
- **Emotions**: angry, calm, disgust, fearful, happy, neutral, sad, surprised
- **Processing Time**: ~0.5 seconds per 3-second audio chunk

## Integration Notes
For multimodal fusion with facial recognition:
1. Call `predict_emotion(audio_file)` to get emotion + confidence
2. Use confidence score for uncertainty-aware fusion
3. Combine with facial emotion detection
4. Apply weighted fusion based on confidence scores
