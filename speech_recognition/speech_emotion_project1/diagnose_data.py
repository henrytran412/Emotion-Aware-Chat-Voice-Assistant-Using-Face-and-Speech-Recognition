import pandas as pd
import numpy as np

print("="*60)
print("RAVDESS DATA DIAGNOSIS")
print("="*60)

# Load data
df = pd.read_csv('ravdess_features.csv')

print(f"\n1. Dataset Shape: {df.shape}")
print(f"   - Total samples: {len(df)}")
print(f"   - Total columns: {len(df.columns)}")

# Check emotion distribution
print("\n2. Emotion Distribution:")
print(df['emotion'].value_counts().sort_index())

# Check for missing values
print("\n3. Missing Values Check:")
metadata_cols = ['filename', 'actor_id', 'emotion_code', 'intensity', 
                 'statement', 'repetition', 'gender', 'emotion']
feature_cols = [col for col in df.columns if col not in metadata_cols]

print(f"   - Total features: {len(feature_cols)}")
print(f"   - Missing values in features: {df[feature_cols].isna().sum().sum()}")
print(f"   - Missing values in labels: {df['emotion'].isna().sum()}")

# Check for infinite values
feature_data = df[feature_cols].values
print(f"\n4. Data Quality:")
print(f"   - Infinite values: {np.isinf(feature_data).sum()}")
print(f"   - NaN values: {np.isnan(feature_data).sum()}")
print(f"   - Min value: {np.nanmin(feature_data):.4f}")
print(f"   - Max value: {np.nanmax(feature_data):.4f}")

# Check actor distribution
print("\n5. Actor Distribution:")
print(f"   - Unique actors: {df['actor_id'].nunique()}")
print(f"   - Samples per actor: min={df['actor_id'].value_counts().min()}, max={df['actor_id'].value_counts().max()}")

# Check feature statistics
print("\n6. Feature Statistics (first 5 features):")
for col in feature_cols[:5]:
    print(f"   - {col}: mean={df[col].mean():.4f}, std={df[col].std():.4f}")

print("\n" + "="*60)
print("DIAGNOSIS COMPLETE")
print("="*60)
