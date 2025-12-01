# scripts/train_ml_enhanced.py

import argparse
import pickle
import os

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.utils import class_weight

from app.config import settings
from app.feature_extractor import LogFeatureExtractor


def train(csv_path: str, text_col: str, label_col: str, model_type: str = 'logistic'):
    """
    Enhanced training with combined features.
    
    Args:
        csv_path: Path to training CSV
        text_col: Column name for log text
        label_col: Column name for labels
        model_type: 'logistic', 'random_forest', or 'gradient_boosting'
    """
    print(f"\n{'='*60}")
    print(f"🚀 Enhanced ML Training with Feature Engineering")
    print(f"{'='*60}\n")
    
    # Load data
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    
    print(f"📊 Dataset: {len(texts)} samples across {len(set(labels))} classes")
    print(f"Class distribution:\n{pd.Series(labels).value_counts()}\n")
    
    # === Step 1: Generate Sentence Embeddings ===
    print("🔤 Generating sentence embeddings...")
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(texts, show_progress_bar=True)
    print(f"   ✓ Embeddings shape: {embeddings.shape}")
    
    # === Step 2: Extract Handcrafted Features ===
    print("\n🔧 Extracting domain-specific features...")
    feature_extractor = LogFeatureExtractor()
    handcrafted_features = feature_extractor.extract_batch(texts)
    print(f"   ✓ Handcrafted features shape: {handcrafted_features.shape}")
    print(f"   ✓ Feature names: {feature_extractor.get_feature_names()[:10]}... (+{len(feature_extractor.get_feature_names())-10} more)")
    
    # === Step 3: Combine Features ===
    print("\n🔗 Combining embeddings + handcrafted features...")
    X_combined = np.hstack([embeddings, handcrafted_features])
    print(f"   ✓ Combined feature shape: {X_combined.shape}")
    
    # === Step 4: Encode Labels ===
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # === Step 5: Train/Test Split ===
    num_classes = len(set(y))
    min_test_ratio = max(num_classes * 2, 20) / len(y)  # At least 2 samples per class
    test_size = max(0.2, min_test_ratio)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_combined, y, test_size=test_size, random_state=42, stratify=y
    )
    
    print(f"\n📑 Split: {len(X_train)} train, {len(X_val)} validation")
    
    # === Step 6: Handle Class Imbalance ===
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y_train),
        y=y_train
    )
    class_weight_dict = dict(enumerate(class_weights))
    print(f"\n⚖️  Class weights: {class_weight_dict}")
    
    # === Step 7: Train Model ===
    print(f"\n🏋️  Training {model_type} model...")
    
    if model_type == 'logistic':
        model = LogisticRegression(
            max_iter=1000,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42,
            C=1.0  # Regularization strength
        )
    elif model_type == 'random_forest':
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=20,
            min_samples_split=5,
            n_jobs=-1,
            class_weight='balanced',
            random_state=42
        )
    elif model_type == 'gradient_boosting':
        model = GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    model.fit(X_train, y_train)
    print("   ✓ Training completed!")
    
    # === Step 8: Cross-Validation ===
    print("\n📊 Running 5-fold cross-validation...")
    cv_scores = cross_val_score(model, X_train, y_train, cv=min(5, len(set(y_train))), n_jobs=-1)
    print(f"   ✓ CV Accuracy: {cv_scores.mean():.3f} (+/- {cv_scores.std() * 2:.3f})")
    
    # === Step 9: Validation Performance ===
    preds = model.predict(X_val)
    print("\n" + "="*60)
    print("📈 VALIDATION PERFORMANCE")
    print("="*60 + "\n")
    print(classification_report(y_val, preds, target_names=le.classes_))
    
    print("\n" + "="*60)
    print("🔢 CONFUSION MATRIX")
    print("="*60 + "\n")
    print(confusion_matrix(y_val, preds))
    
    # === Step 10: Save Model ===
    os.makedirs(os.path.dirname(settings.LR_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(settings.LABEL_ENCODER_PATH), exist_ok=True)
    
    with open(settings.LR_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print(f"\n💾 Model saved to: {settings.LR_MODEL_PATH}")
    
    with open(settings.LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print(f"💾 Label encoder saved to: {settings.LABEL_ENCODER_PATH}")
    
    # === Step 11: Feature Importance (if available) ===
    if hasattr(model, 'feature_importances_'):
        print("\n" + "="*60)
        print("🎯 TOP 20 MOST IMPORTANT FEATURES")
        print("="*60 + "\n")
        
        # Combine feature names
        embedding_names = [f'embed_{i}' for i in range(embeddings.shape[1])]
        all_feature_names = embedding_names + feature_extractor.get_feature_names()
        
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1][:20]
        
        for i, idx in enumerate(indices):
            print(f"{i+1:2d}. {all_feature_names[idx]:30s} : {importances[idx]:.4f}")
    
    print("\n" + "="*60)
    print("✅ Training Complete!")
    print("="*60 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ML training with feature engineering")
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    parser.add_argument("--text-col", default="text", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing labels")
    parser.add_argument("--model", default="logistic", 
                       choices=['logistic', 'random_forest', 'gradient_boosting'],
                       help="Model type to train")
    
    args = parser.parse_args()
    train(args.csv, args.text_col, args.label_col, args.model)
