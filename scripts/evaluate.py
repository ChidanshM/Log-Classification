# scripts/evaluate.py

import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix

from app.router import HybridClassifier


def load_csv(csv_path: str, text_col: str, label_col: str):
    """
    Load CSV safely, supporting:
      - CSVs with headers
      - CSVs without headers
      - CSVs with unknown separators
    """

    try:
        # First try normal read
        table = pd.read_csv(csv_path)
    except Exception:
        # Fallback: auto-detect separator
        table = pd.read_csv(csv_path, sep=None, engine="python")

    # If expected columns NOT found → assume headerless file
    if text_col not in table.columns or label_col not in table.columns:
        print(f"[WARN] Columns '{text_col}' and '{label_col}' not found. "
              f"Assuming the CSV has NO HEADER.")

        table = pd.read_csv(
            csv_path,
            header=None,
            names=[text_col, label_col]
        )

    return table[[text_col, label_col]].dropna()


def evaluate(csv_path: str, text_col: str, label_col: str):
    records = load_csv(csv_path, text_col, label_col)

    classifier = HybridClassifier()

    actual_tags = []
    predicted_tags = []
    model_layer_used = []  # Which layer classified the log

    print(f"\nLoaded {len(records)} evaluation samples.")
    print(f"Columns: {records.columns.tolist()}")

    for _, entry in records.iterrows():
        true_tag = entry[label_col]
        log_msg = entry[text_col]

        try:
            outcome = classifier.classify(log_msg)
            guess_tag = outcome["label"]
            layer_name = outcome.get("source", "unknown")
        except Exception:
            guess_tag = "error"
            layer_name = "error"

        actual_tags.append(true_tag)
        predicted_tags.append(guess_tag)
        model_layer_used.append(layer_name)

    print("\n HYBRID SYSTEM EVALUATION \n")
    print(classification_report(actual_tags, predicted_tags))

    print("\n LAYER USAGE BREAKDOWN \n")
    print(pd.Series(model_layer_used).value_counts())

    print("\n CONFUSION MATRIX \n")
    print(confusion_matrix(actual_tags, predicted_tags))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    parser.add_argument("--text-col", default="log", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing true labels")

    args = parser.parse_args()
    evaluate(args.csv, args.text_col, args.label_col)
