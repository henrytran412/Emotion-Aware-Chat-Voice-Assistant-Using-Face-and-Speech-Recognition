import opensmile
import pickle
import numpy as np
import sys

def predict_emotion(audio_file, 
                   model_path='emotion_model_improved.pkl',
                   scaler_path='scaler_improved.pkl',
                   label_encoder_path='label_encoder_improved.pkl'):
    """
    Predict emotion from audio file using improved model
    """
    print("="*60)
    print(f"ANALYZING: {audio_file}")
    print("="*60)
    
    # Load model, scaler, and label encoder
    print("\n1. Loading model components...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    print("   ✓ Model loaded")
    print("   ✓ Scaler loaded")
    print("   ✓ Label encoder loaded")
    
    # Initialize openSMILE
    print("\n2. Extracting features with openSMILE...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    # Extract features
    features = smile.process_file(audio_file)
    features_array = features.values[0].reshape(1, -1)
    print(f"   ✓ Extracted {len(features.columns)} features")
    
    # Normalize
    print("\n3. Normalizing features...")
    features_scaled = scaler.transform(features_array)
    print("   ✓ Features normalized")
    
    # Predict
    print("\n4. Making prediction...")
    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]
    
    # Decode emotion
    emotion = label_encoder.inverse_transform([prediction])[0]
    confidence = probabilities[prediction]
    
    # Get all emotion probabilities
    all_emotions = {}
    for idx, prob in enumerate(probabilities):
        emotion_name = label_encoder.inverse_transform([idx])[0]
        all_emotions[emotion_name] = prob
    
    # Display results
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    print(f"\n🎯 Predicted Emotion: {emotion.upper()}")
    print(f"📊 Confidence: {confidence:.2%}")
    
    print(f"\n📈 All Emotion Probabilities:")
    print("-" * 40)
    for emotion_name, prob in sorted(all_emotions.items(), key=lambda x: x[1], reverse=True):
        bar_length = int(prob * 30)
        bar = "█" * bar_length
        print(f"  {emotion_name:12s} {prob:6.2%} {bar}")
    
    print("\n" + "="*60)
    
    return emotion, confidence, all_emotions

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python test_improved_model.py <audio_file.wav>")
        print("\nExample:")
        print("  python test_improved_model.py data/RAVDESS/Actor_01/03-01-03-01-01-01-01.wav")
        print("\nTry different emotions:")
        print("  Happy:    data/RAVDESS/Actor_01/03-01-03-01-01-01-01.wav")
        print("  Sad:      data/RAVDESS/Actor_01/03-01-04-01-01-01-01.wav")
        print("  Angry:    data/RAVDESS/Actor_01/03-01-05-01-01-01-01.wav")
        print("  Fearful:  data/RAVDESS/Actor_01/03-01-06-01-01-01-01.wav")
        sys.exit(1)
    
    audio_file = sys.argv[1]
    predict_emotion(audio_file)
