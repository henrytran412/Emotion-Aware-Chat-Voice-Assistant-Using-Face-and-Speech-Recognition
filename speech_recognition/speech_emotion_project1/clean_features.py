import pandas as pd
import numpy as np

print("Cleaning RAVDESS features...")

# Load data
df = pd.read_csv('ravdess_features.csv')
print(f"Original shape: {df.shape}")

# Identify feature columns
metadata_cols = ['filename', 'actor_id', 'emotion_code', 'intensity', 
                 'statement', 'repetition', 'gender', 'emotion']
feature_cols = [col for col in df.columns if col not in metadata_cols]

# Replace inf with NaN
df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan)

# Count NaN values
nan_count = df[feature_cols].isna().sum().sum()
print(f"NaN values found: {nan_count}")

if nan_count > 0:
    # Fill NaN with column median
    df[feature_cols] = df[feature_cols].fillna(df[feature_cols].median())
    print("NaN values filled with median")

# Remove any rows with missing emotions
df = df.dropna(subset=['emotion'])
print(f"Final shape: {df.shape}")

# Save cleaned data
df.to_csv('ravdess_features_clean.csv', index=False)
print("\nCleaned data saved to: ravdess_features_clean.csv")
print("\nEmotion distribution:")
print(df['emotion'].value_counts().sort_index())
