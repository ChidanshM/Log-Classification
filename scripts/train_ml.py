# scripts/train_ml.py

import argparse
import pickle
import os

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report

from app.config import settings


def train(csv_path: str, text_col: str, label_col: str):
    df = pd.read_csv(csv_path)
    df = df[[text_col, label_col]].dropna()

    texts = df[text_col].tolist()
    labels = df[label_col].tolist()

    encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    print("Embedding logs...")
    X = encoder.encode(texts, show_progress_bar=True)

    # Label encode
    le = LabelEncoder()
    y = le.fit_transform(labels)

    # ==== Robust dynamic test split size ====
    num_classes = len(set(y))
    min_test_ratio = num_classes / len(y)
    test_size = max(0.2, min_test_ratio)

    # Check for rare classes
    class_counts = pd.Series(y).value_counts()
    if class_counts.min() < 2:
        print("Warning: Some classes have fewer than 2 samples. Disabling stratification.")
        stratify_param = None
    else:
        stratify_param = y

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=stratify_param
    )

    # Train model
    model = LogisticRegression(max_iter=1000, n_jobs=-1)
    model.fit(X_train, y_train)

    # Validation metrics
    preds = model.predict(X_val)
    print("\nValidation Performance:\n")
    print(classification_report(
        y_val, 
        preds, 
        labels=range(len(le.classes_)), 
        target_names=le.classes_, 
        zero_division=0
    ))

    # ==== Create model directory if missing ====
    os.makedirs(os.path.dirname(settings.LR_MODEL_PATH), exist_ok=True)
    os.makedirs(os.path.dirname(settings.LABEL_ENCODER_PATH), exist_ok=True)

    # Save model + encoder
    with open(settings.LR_MODEL_PATH, "wb") as f:
        pickle.dump(model, f)

    with open(settings.LABEL_ENCODER_PATH, "wb") as f:
        pickle.dump(le, f)

    print("\nModel Saved!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--text-col", default="text")
    parser.add_argument("--label-col", default="label")

    args = parser.parse_args()
    train(args.csv, args.text_col, args.label_col)
