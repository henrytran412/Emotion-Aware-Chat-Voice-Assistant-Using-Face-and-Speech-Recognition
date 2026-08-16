import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

def train_improved_model(features_file='ravdess_features_clean.csv'):
    """
    Train with optimized settings
    """
    print("="*60)
    print("IMPROVED EMOTION RECOGNITION TRAINING")
    print("="*60)
    
    # Load data
    print("\n1. Loading data...")
    df = pd.read_csv(features_file)
    print(f"   Total samples: {len(df)}")
    
    # Separate features and labels
    metadata_cols = ['filename', 'actor_id', 'emotion_code', 'intensity', 
                     'statement', 'repetition', 'gender', 'emotion']
    feature_cols = [col for col in df.columns if col not in metadata_cols]
    
    X = df[feature_cols].values
    y = df['emotion'].values
    
    print(f"   Features: {len(feature_cols)}")
    print(f"   Samples per emotion:")
    for emotion, count in df['emotion'].value_counts().sort_index().items():
        print(f"      {emotion}: {count}")
    
    # Encode labels
    print("\n2. Encoding labels...")
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)
    print(f"   Classes: {label_encoder.classes_}")
    
    # SIMPLE RANDOM SPLIT (easier, better for small datasets)
    print("\n3. Splitting data (80/20 random split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"   Training samples: {len(X_train)}")
    print(f"   Testing samples: {len(X_test)}")
    
    # Check class balance in splits
    print(f"\n   Train set emotion counts:")
    unique, counts = np.unique(y_train, return_counts=True)
    for u, c in zip(unique, counts):
        print(f"      {label_encoder.inverse_transform([u])[0]}: {c}")
    
    # Normalize features
    print("\n4. Normalizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"   Train mean: {X_train_scaled.mean():.6f}, std: {X_train_scaled.std():.6f}")
    print(f"   Test mean: {X_test_scaled.mean():.6f}, std: {X_test_scaled.std():.6f}")
    
    # Train Random Forest (works better than SVM for this)
    print("\n5. Training Random Forest Classifier...")
    model = RandomForestClassifier(
        n_estimators=300,        # More trees
        max_depth=None,          # No limit on depth
        min_samples_split=2,     # Minimum samples to split
        min_samples_leaf=1,      # Minimum samples per leaf
        max_features='sqrt',     # Features to consider at each split
        random_state=42,
        n_jobs=-1,               # Use all CPU cores
        class_weight='balanced'  # Handle any class imbalance
    )
    
    model.fit(X_train_scaled, y_train)
    print("   Training complete!")
    
    # Cross-validation on training set
    print("\n6. Cross-validation (5-fold)...")
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=5)
    print(f"   CV scores: {cv_scores}")
    print(f"   CV mean: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    # Evaluate
    print("\n7. Evaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Train Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Test Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    print(f"{'='*60}")
    
    # Detailed classification report
    print("\n8. Classification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                                target_names=label_encoder.classes_,
                                digits=4))
    
    # Confusion matrix
    print("\n9. Confusion Matrix:")
    cm = confusion_matrix(y_test, y_pred_test)
    print("\n   Predicted ->")
    print(f"   Classes: {label_encoder.classes_}")
    print(cm)
    
    # Check prediction distribution
    print("\n10. Prediction Distribution (Test Set):")
    unique_preds, pred_counts = np.unique(y_pred_test, return_counts=True)
    for u, c in zip(unique_preds, pred_counts):
        emotion = label_encoder.inverse_transform([u])[0]
        print(f"    {emotion}: {c} predictions")
    
    if len(unique_preds) < len(label_encoder.classes_):
        print("\n    ⚠️  WARNING: Model not predicting all classes!")
        missing = set(range(len(label_encoder.classes_))) - set(unique_preds)
        for m in missing:
            print(f"       Missing: {label_encoder.inverse_transform([m])[0]}")
    
    # Feature importance
    print("\n11. Top 10 Most Important Features:")
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    for idx, row in feature_importance.head(10).iterrows():
        print(f"    {row['feature']}: {row['importance']:.4f}")
    
    # Save model
    print("\n12. Saving model...")
    with open('emotion_model_improved.pkl', 'wb') as f:
        pickle.dump(model, f)
    with open('scaler_improved.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    with open('label_encoder_improved.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    print("    ✓ emotion_model_improved.pkl")
    print("    ✓ scaler_improved.pkl")
    print("    ✓ label_encoder_improved.pkl")
    
    print("\n" + "="*60)
    print("TRAINING COMPLETE")
    print("="*60)
    
    # Give recommendation based on results
    if test_accuracy < 0.5:
        print("\n⚠️  VERY LOW ACCURACY - Possible issues:")
        print("   1. Feature extraction may have failed")
        print("   2. Data corruption")
        print("   3. Re-run: python extract_ravdess_features.py")
    elif test_accuracy < 0.65:
        print("\n⚠️  LOW ACCURACY - Try:")
        print("   1. Add more data (TESS, CREMA-D)")
        print("   2. Use ComParE feature set")
        print("   3. Tune hyperparameters")
    elif test_accuracy < 0.75:
        print("\n✓ GOOD ACCURACY - This is normal for speaker-independent")
        print("   To improve: Add more training data")
    else:
        print("\n✓✓ EXCELLENT ACCURACY!")
    
    return model, scaler, label_encoder

if __name__ == "__main__":
    model, scaler, label_encoder = train_improved_model()
