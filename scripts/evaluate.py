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
        df = pd.read_csv(csv_path)
    except Exception:
        # Fallback: try auto-detect separator
        df = pd.read_csv(csv_path, sep=None, engine="python")

    # If expected columns NOT found → assume headerless file
    if text_col not in df.columns or label_col not in df.columns:
        print(f"[WARN] Columns '{text_col}' and '{label_col}' not found. "
              f"Assuming the CSV has NO HEADER.")

        df = pd.read_csv(
            csv_path,
            header=None,
            names=[text_col, label_col]
        )

    return df[[text_col, label_col]].dropna()


def evaluate(csv_path: str, text_col: str, label_col: str):
    df = load_csv(csv_path, text_col, label_col)

    clf = HybridClassifier()

    y_true = []
    y_pred = []
    used_layer = []  # Store which model classified the log

    print(f"\nLoaded {len(df)} evaluation samples.")
    print(f"Columns: {df.columns.tolist()}")

    for _, row in df.iterrows():
        true_label = row[label_col]
        log_text = row[text_col]

        try:
            result = clf.classify(log_text)
            pred_label = result["label"]
            layer = result.get("source", "unknown")
        except Exception as e:
            pred_label = "error"
            layer = "error"

        y_true.append(true_label)
        y_pred.append(pred_label)
        used_layer.append(layer)

    print("\n HYBRID SYSTEM EVALUATION n")
    print(classification_report(y_true, y_pred))

    print("\n LAYER USAGE BREAKDOWN ==\n")
    print(pd.Series(used_layer).value_counts())

    print("\n CONFUSION MATRIX n")
    print(confusion_matrix(y_true, y_pred))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    parser.add_argument("--text-col", default="log", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing true labels")

    args = parser.parse_args()
    evaluate(args.csv, args.text_col, args.label_col)
