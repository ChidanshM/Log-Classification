import sys
import os
import io

# Add parent directory to path so we can import 'app'
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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


def evaluate(csv_path: str, text_col: str, label_col: str, output_path: str = None):
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
            outcome = classifier.label_logs(log_msg)
            guess_tag = outcome["label"]
            layer_name = outcome.get("layer", "unknown")
        except Exception as e:
            print(f"Error classifying log: {log_msg[:50]}... -> {e}")
            guess_tag = "error"
            layer_name = "error"

        actual_tags.append(true_tag)
        predicted_tags.append(guess_tag)
        model_layer_used.append(layer_name)

    # Capture metrics
    report = classification_report(actual_tags, predicted_tags)
    layer_breakdown = pd.Series(model_layer_used).value_counts().to_string()
    conf_matrix = confusion_matrix(actual_tags, predicted_tags).tolist() # Convert to list for easier saving

    print("\n HYBRID SYSTEM EVALUATION \n")
    print(report)

    print("\n LAYER USAGE BREAKDOWN \n")
    print(layer_breakdown)

    print("\n CONFUSION MATRIX \n")
    print(conf_matrix)

    if output_path:
        records["predicted_label"] = predicted_tags
        records["layer"] = model_layer_used
        records.to_csv(output_path, index=False)
        print(f"\nResults saved to {output_path}")

        # Save metrics to a separate file
        metrics_output_path = output_path.replace(".csv", "_metrics.txt") if output_path.endswith(".csv") else f"{output_path}_metrics.txt"
        with open(metrics_output_path, "w") as f:
            f.write("HYBRID SYSTEM EVALUATION\n")
            f.write(report)
            f.write("\n\nLAYER USAGE BREAKDOWN\n")
            f.write(layer_breakdown)
            f.write("\n\nCONFUSION MATRIX\n")
            f.write(str(conf_matrix)) # Write as string representation
            f.write("\n")
        print(f"Metrics saved to {metrics_output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    parser.add_argument("--text-col", default="log", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing true labels")
    parser.add_argument("--output", help="Path to save evaluation results CSV")

    args = parser.parse_args()
    evaluate(args.csv, args.text_col, args.label_col, args.output)
