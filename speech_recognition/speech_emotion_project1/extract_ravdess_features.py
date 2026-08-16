import opensmile
import os
import pandas as pd
from tqdm import tqdm
import numpy as np

def extract_ravdess_features(ravdess_dir='./data/RAVDESS', output_file='ravdess_features.csv'):
    """
    Extract openSMILE features from RAVDESS dataset
    """
    print("Initializing openSMILE...")
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.eGeMAPSv02,
        feature_level=opensmile.FeatureLevel.Functionals,
    )
    
    all_features = []
    all_labels = []
    all_metadata = []
    
    # Emotion mapping
    emotion_map = {
        1: 'neutral', 2: 'calm', 3: 'happy', 4: 'sad',
        5: 'angry', 6: 'fearful', 7: 'disgust', 8: 'surprised'
    }
    
    print(f"Processing RAVDESS dataset from: {ravdess_dir}")
    
    # Get all actor folders
    actor_folders = sorted([f for f in os.listdir(ravdess_dir) if f.startswith('Actor_')])
    
    for actor_folder in tqdm(actor_folders, desc="Processing actors"):
        actor_path = os.path.join(ravdess_dir, actor_folder)
        
        if not os.path.isdir(actor_path):
            continue
        
        # Process each audio file
        audio_files = [f for f in os.listdir(actor_path) if f.endswith('.wav')]
        
        for audio_file in audio_files:
            audio_path = os.path.join(actor_path, audio_file)
            
            try:
                # Extract features
                features = smile.process_file(audio_path)
                
                # Parse filename for metadata
                # Format: 03-01-06-01-02-01-12.wav
                parts = audio_file.split('-')
                emotion_code = int(parts[2])
                intensity = int(parts[3])
                statement = int(parts[4])
                repetition = int(parts[5])
                actor_id = int(parts[6].split('.')[0])
                
                emotion_label = emotion_map[emotion_code]
                
                all_features.append(features.values[0])
                all_labels.append(emotion_label)
                all_metadata.append({
                    'filename': audio_file,
                    'actor_id': actor_id,
                    'emotion_code': emotion_code,
                    'intensity': 'normal' if intensity == 1 else 'strong',
                    'statement': statement,
                    'repetition': repetition,
                    'gender': 'female' if actor_id % 2 == 0 else 'male'
                })
                
            except Exception as e:
                print(f"Error processing {audio_path}: {e}")
                continue
    
    # Create DataFrame
    print("\nCreating feature DataFrame...")
    feature_columns = features.columns.tolist()
    df_features = pd.DataFrame(all_features, columns=feature_columns)
    df_labels = pd.DataFrame(all_labels, columns=['emotion'])
    df_metadata = pd.DataFrame(all_metadata)
    
    # Combine all data
    df_complete = pd.concat([df_metadata, df_labels, df_features], axis=1)
    
    # Save to CSV
    df_complete.to_csv(output_file, index=False)
    print(f"\nSaved {len(df_complete)} samples to {output_file}")
    print(f"Features extracted: {len(feature_columns)}")
    print(f"\nEmotion distribution:")
    print(df_complete['emotion'].value_counts())
    
    return df_complete

if __name__ == "__main__":
    df = extract_ravdess_features()
    print("\nFeature extraction complete!")
