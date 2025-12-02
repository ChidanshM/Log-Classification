# scripts/train_ml_enhanced.py

import argparse
import pickle
import os

import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import class_weight
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from app.config import settings
from app.feature_extractor import LogFeatureExtractor
from app.utils import preprocess_log


def train(csv_path: str, text_col: str, label_col: str, test_csv_path: str = None, output_metrics_path: str = "evaluation_metrics.txt"):
    """
    Enhanced training with combined features.
    Trains on the full dataset and saves the model.
    
    Args:
        csv_path: Path to training CSV
        text_col: Column name for log text
        label_col: Column name for labels
        model_type: 'logistic', 'random_forest', or 'gradient_boosting'
    """
    
    # Load data
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()
    
    texts = df[text_col].tolist()
    labels = df[label_col].tolist()
    
    # Filter out classes with fewer than 2 samples
    label_counts = pd.Series(labels).value_counts()
    valid_labels = label_counts[label_counts >= 2].index
    
    df_filtered = df[df[label_col].isin(valid_labels)]
    texts = df_filtered[text_col].tolist()
    labels = df_filtered[label_col].tolist()
    
    print("Dataset: ", len(texts), "samples across", len(set(labels)), "classes")
    
    if len(texts) == 0:
        print("Error: No data left after filtering. Please ensure your dataset has classes with at least 2 samples.")
        return
    
    # Step 1: Generate Sentence Embeddings
    print("Generating sentence embeddings...")
    # Preprocess texts for embeddings (masking IDs)
    preprocessed_texts = [preprocess_log(t) for t in texts]
    
    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embeddings = encoder.encode(preprocessed_texts, show_progress_bar=True)
    
    # Step 2: Extract Handcrafted Features
    print("Extracting domain-specific features...")
    feature_extractor = LogFeatureExtractor()
    handcrafted_features = feature_extractor.extract_batch(texts)
    
    # Step 3: Combine Features
    print("Combining embeddings + handcrafted features...")
    X_combined = np.hstack([embeddings, handcrafted_features])
    
    # Step 4: Encode Labels
    le = LabelEncoder()
    y = le.fit_transform(labels)
    
    # Step 5: Handle Class Imbalance
    # Note: We calculate weights but some models (like sklearn's) take them as a parameter during init
    # rather than fit, or handle it internally with class_weight='balanced'.
    class_weights = class_weight.compute_class_weight(
        'balanced',
        classes=np.unique(y),
        y=y
    )
    print("Class weights calculated.")
    
    # Step 6: Train Model
    print("Training Random Forest model on full dataset...")
    
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        n_jobs=-1,
        class_weight='balanced',
        random_state=42
    )
    
    model.fit(X_combined, y)
    print("Training completed!")
    
    # Step 7: Save Model
    os.makedirs(os.path.dirname(settings.LR_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(settings.LABEL_ENCODER_PATH), exist_ok=True)
    
    with open(settings.LR_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Model saved to: ", settings.LR_MODEL_PATH)
    
    with open(settings.LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)
    print("Label encoder saved to: ", settings.LABEL_ENCODER_PATH)

    # Step 8: Evaluate on Test Data (if provided)
    if test_csv_path:
        if not os.path.exists(test_csv_path):
            print(f"Warning: Test CSV file not found at {test_csv_path}")
            return

        print(f"\nEvaluating on test data: {test_csv_path}")
        test_df = pd.read_csv(test_csv_path)
        
        # Ensure columns exist
        if text_col not in test_df.columns or label_col not in test_df.columns:
            print(f"Error: Test CSV must contain '{text_col}' and '{label_col}' columns.")
            return

        test_df = test_df[[text_col, label_col]].dropna()
        
        test_texts = test_df[text_col].tolist()
        test_labels = test_df[label_col].tolist()
        
        if len(test_texts) == 0:
            print("Error: Test dataset is empty.")
            return

        # Preprocess test data
        print("Generating test embeddings...")
        preprocessed_test_texts = [preprocess_log(t) for t in test_texts]
        test_embeddings = encoder.encode(preprocessed_test_texts, show_progress_bar=True)
        
        print("Extracting test features...")
        test_handcrafted_features = feature_extractor.extract_batch(test_texts)
        
        X_test = np.hstack([test_embeddings, test_handcrafted_features])
        
        # Filter test data to only include known labels
        known_labels = set(le.classes_)
        mask = [l in known_labels for l in test_labels]
        
        if not all(mask):
             dropped_count = len(test_labels) - sum(mask)
             print(f"Warning: {dropped_count} samples dropped due to unseen labels in test set.")
             X_test = X_test[mask]
             test_labels = [l for l, m in zip(test_labels, mask) if m]
        
        if len(test_labels) == 0:
            print("Error: No valid test samples remaining after filtering.")
            return

        y_test = le.transform(test_labels)
        
        print("Predicting on test set...")
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, target_names=le.classes_)
        conf_matrix = confusion_matrix(y_test, y_pred)
        
        print(f"Test Accuracy: {accuracy:.4f}")
        print("Classification Report:\n", report)
        
        with open(output_metrics_path, "w") as f:
            f.write(f"Test Accuracy: {accuracy:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report)
            f.write("\n\nConfusion Matrix:\n")
            f.write(str(conf_matrix))
            
        print(f"Metrics saved to {output_metrics_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ML training with feature engineering")
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    parser.add_argument("--text-col", default="text", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing labels")
    parser.add_argument("--test-csv", help="Path to test CSV for evaluation")
    parser.add_argument("--output-metrics", default="evaluation_metrics.txt", help="Path to save evaluation metrics")
    args = parser.parse_args()
    train(args.csv, args.text_col, args.label_col, args.test_csv, args.output_metrics)
