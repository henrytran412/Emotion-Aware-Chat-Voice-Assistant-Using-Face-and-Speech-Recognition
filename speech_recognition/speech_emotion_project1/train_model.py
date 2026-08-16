import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import os

def train_emotion_model(features_file='ravdess_features.csv', 
                        model_output='emotion_model.pkl',
                        scaler_output='scaler.pkl',
                        label_encoder_output='label_encoder.pkl'):
    """
    Train emotion recognition model
    """
    print("Loading features...")
    df = pd.read_csv(features_file)
    
    print(f"Total samples: {len(df)}")
    print(f"Emotions: {df['emotion'].unique()}")
    
    # Separate features and labels
    # Drop metadata columns
    metadata_cols = ['filename', 'actor_id', 'emotion_code', 'intensity', 
                     'statement', 'repetition', 'gender', 'emotion']
    
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['emotion'].values
    
    print(f"\nNumber of features: {len(feature_cols)}")
    
    # Encode labels
    print("\nEncoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    
    # Split data - Speaker independent
    # Use actor_id if available for proper splitting
    if 'actor_id' in df.columns:
        unique_actors = df['actor_id'].unique()
        np.random.shuffle(unique_actors)
        
        train_actors = unique_actors[:int(len(unique_actors) * 0.75)]
        test_actors = unique_actors[int(len(unique_actors) * 0.75):]
        
        train_mask = df['actor_id'].isin(train_actors)
        test_mask = df['actor_id'].isin(test_actors)
        
        X_train = X[train_mask]
        X_test = X[test_mask]
        y_train = y_encoded[train_mask]
        y_test = y_encoded[test_mask]
    else:
        # Regular random split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.25, random_state=42, stratify=y_encoded
        )
    
    print(f"\nTraining samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    
    # Normalize features
    print("\nNormalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train SVM model
    print("\nTraining SVM model...")
    model = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        probability=True,  # Enable probability estimates
        random_state=42
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    print("\nEvaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTrain Accuracy: {train_accuracy:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    
    print("\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                                target_names=label_encoder.classes_))
    
    # Save model, scaler, and label encoder
    print("\nSaving model...")
    with open(model_output, 'wb') as f:
        pickle.dump(model, f)
    
    with open(scaler_output, 'wb') as f:
        pickle.dump(scaler, f)
    
    with open(label_encoder_output, 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print(f"\nModel saved to: {model_output}")
    print(f"Scaler saved to: {scaler_output}")
    print(f"Label encoder saved to: {label_encoder_output}")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    model, scaler, label_encoder = train_emotion_model()
    print("\nTraining complete!")
