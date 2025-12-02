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

    log_texts = df[text_col].tolist()
    log_labels = df[label_col].tolist()

    encoder_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding logs...")
    log_vectors = encoder_model.encode(log_texts, show_progress_bar=True)

    # Label encode
    label_encoder = LabelEncoder()
    encoded_labels = label_encoder.fit_transform(log_labels)

    #dynamic test size 
    num_classes = len(set(encoded_labels))
    min_test_ratio = num_classes / len(encoded_labels)
    test_ratio = max(0.2, min_test_ratio)

    # Check for rare classes
    class_counts = pd.Series(encoded_labels).value_counts()
    if class_counts.min() < 2:
        print("Warning: Some classes have fewer than 2 samples. Disabling stratification.")
        stratify_vals = None
    else:
        stratify_vals = encoded_labels

    X_train, X_val, y_train, y_val = train_test_split(
        log_vectors, encoded_labels, test_size=test_ratio,
        random_state=42, stratify=stratify_vals
    )

    # Train model
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    # Validation metrics
    val_preds = model.predict(X_val)
    print("\nValidation Performance:\n")
    print(classification_report(
        y_val,
        val_preds,
        labels=range(len(label_encoder.classes_)),
        target_names=label_encoder.classes_,
        zero_division=0
    ))

    # model dir creation
    os.makedirs(os.path.dirname(settings.LR_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(settings.LABEL_ENCODER_PATH), exist_ok=True)
    
    with open(settings.LR_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    print("Model saved to: ", settings.LR_MODEL_PATH)
    
    with open(settings.LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(label_encoder, f)

    print("\nModel Saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced ML training with feature engineering")
    parser.add_argument("--csv", required=True, help="Path to training CSV")
    parser.add_argument("--text-col", default="text", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing labels")
    parser.add_argument("--test-csv", help="Path to test CSV for evaluation")
    parser.add_argument("--output-metrics", default="evaluation_metrics.txt", help="Path to save evaluation metrics")
    args = parser.parse_args()
    train(args.csv, args.text_col, args.label_col, args.test_csv, args.output_metrics)
