import opensmile
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
import os

def batch_predict_emotions(audio_folder, 
                          model_path='emotion_model_improved.pkl',
                          scaler_path='scaler_improved.pkl',
                          label_encoder_path='label_encoder_improved.pkl'):
    """
    Predict emotions for all audio files in a folder
    """
    print("Loading model components...")
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_path, 'rb') as f:
        scaler = pickle.load(f)
    with open(label_encoder_path, 'rb') as f:
        label_encoder = pickle.load(f)
    
    # Initialize openSMILE
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    results = []
    
    # Find all wav files
    audio_files = []
    for root, dirs, files in os.walk(audio_folder):
        for file in files:
            if file.endswith('.wav'):
                audio_files.append(os.path.join(root, file))
    
    print(f"\nFound {len(audio_files)} audio files")
    print("Processing...\n")
    
    for audio_path in tqdm(audio_files):
        try:
            # Extract features
            features = smile.process_file(audio_path)
            features_array = features.values[0].reshape(1, -1)
            
            # Normalize and predict
            features_scaled = scaler.transform(features_array)
            prediction = model.predict(features_scaled)[0]
            probabilities = model.predict_proba(features_scaled)[0]
            
            emotion = label_encoder.inverse_transform([prediction])[0]
            confidence = probabilities[prediction]
            
            # Parse true emotion from RAVDESS filename if possible
            filename = os.path.basename(audio_path)
            true_emotion = None
            
            if filename.startswith('03-01-'):
                parts = filename.split('-')
                emotion_code = int(parts[2])
                emotion_map = {
                    1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
                    5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
                }
                true_emotion = emotion_map.get(emotion_code)
            
            results.append({
                'file': filename,
                'predicted': emotion,
                'confidence': confidence,
                'true_emotion': true_emotion,
                'correct': emotion == true_emotion if true_emotion else None
            })
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
            continue
    
    # Create results DataFrame
    df = pd.DataFrame(results)
    
    # Print summary
    print("\n" + "="*60)
    print("BATCH PREDICTION RESULTS")
    print("="*60)
    print(f"\nTotal files processed: {len(df)}")
    
    if 'correct' in df.columns and df['correct'].notna().any():
        accuracy = df['correct'].sum() / df['correct'].notna().sum()
        print(f"Accuracy: {accuracy:.2%}")
        
        print("\nPer-emotion accuracy:")
        for emotion in sorted(df['true_emotion'].unique()):
            if pd.notna(emotion):
                emotion_df = df[df['true_emotion'] == emotion]
                emotion_acc = emotion_df['correct'].sum() / len(emotion_df)
                print(f"  {emotion:12s}: {emotion_acc:.2%} ({emotion_df['correct'].sum()}/{len(emotion_df)})")
    
    print(f"\nAverage confidence: {df['confidence'].mean():.2%}")
    
    print("\nPrediction distribution:")
    for emotion, count in df['predicted'].value_counts().items():
        percentage = (count / len(df)) * 100
        print(f"  {emotion:12s}: {count:3d} ({percentage:.1f}%)")
    
    # Save results
    output_file = 'batch_predictions.csv'
    df.to_csv(output_file, index=False)
    print(f"\n✓ Results saved to: {output_file}")
    
    return df

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python batch_test_emotions.py <audio_folder>")
        print("\nExample:")
        print("  python batch_test_emotions.py data/RAVDESS/Actor_01/")
        sys.exit(1)
    
    audio_folder = sys.argv[1]
    batch_predict_emotions(audio_folder)
