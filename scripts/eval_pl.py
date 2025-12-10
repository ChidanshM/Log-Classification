# scripts/evaluate.py

# --- OPTIMIZATION: Prevent CPU oversubscription ---
import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures

from app.router import HybridClassifier


def load_csv(csv_path: str, text_col: str, label_col: str):
    try:
        table = pd.read_csv(csv_path)
    except Exception:
        table = pd.read_csv(csv_path, sep=None, engine="python")

    if text_col not in table.columns or label_col not in table.columns:
        print(f"[WARN] Columns '{text_col}' and '{label_col}' not found. "
              f"Assuming the CSV has NO HEADER.")

        table = pd.read_csv(
            csv_path,
            header=None,
            names=[text_col, label_col]
        )

    return table[[text_col, label_col]].dropna()

# --- HELPER: Process one log entry safely ---
def process_single_log(entry, classifier, text_col, label_col):
    true_tag = entry[label_col]
    log_msg = entry[text_col]
    
    start = time.perf_counter()
    try:
        outcome = classifier.label_logs(log_msg)
        guess_tag = outcome["label"]
        layer_name = outcome.get("layer", "unknown")
    except Exception:
        guess_tag = "error"
        layer_name = "error"
    
    end = time.perf_counter()
    elapsed_ms = (end - start) * 1000
    
    return true_tag, guess_tag, layer_name, elapsed_ms


def evaluate(csv_path: str, text_col: str, label_col: str):
    records = load_csv(csv_path, text_col, label_col)

    print("Initializing classifier...")
    classifier = HybridClassifier()

    # Lists to store results
    actual_tags = []
    predicted_tags = []
    model_layer_used = []
    latencies = [] 

    print(f"\nLoaded {len(records)} evaluation samples.")

    # --- PARALLEL EXECUTION ---
    print(f"Starting parallel evaluation with 16 workers...")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futures = [
            executor.submit(process_single_log, row, classifier, text_col, label_col) 
            for _, row in records.iterrows()
        ]
        
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(records), desc="Evaluating"):
            try:
                true_tag, guess_tag, layer_name, elapsed_ms = future.result()
                
                actual_tags.append(true_tag)
                predicted_tags.append(guess_tag)
                model_layer_used.append(layer_name)
                latencies.append(elapsed_ms)
            except Exception as e:
                print(f"Error processing row: {e}")

    # --- REPORTING ---
    
    # Create a temporary DataFrame for easy grouping
    results_df = pd.DataFrame({
        "actual": actual_tags,
        "predicted": predicted_tags,
        "layer": model_layer_used
    })

    # 1. Overall Report
    print("\n" + "="*40)
    print(" HYBRID SYSTEM EVALUATION (OVERALL)")
    print("="*40)
    print(classification_report(actual_tags, predicted_tags))

    # 2. Accuracy Per Layer (NEW FEATURE)
    print("\n" + "="*40)
    print(" ACCURACY BY LAYER")
    print("="*40)
    
    # Group by layer and calculate accuracy
    for layer_name, group in results_df.groupby("layer"):
        acc = accuracy_score(group["actual"], group["predicted"])
        count = len(group)
        print(f"Layer: {layer_name.upper():<10} | Samples: {count:<5} | Accuracy: {acc:.2%}")

    # 3. Layer Usage
    print("\n" + "="*40)
    print(" LAYER USAGE BREAKDOWN")
    print("="*40)
    print(pd.Series(model_layer_used).value_counts())

    # 4. Confusion Matrix
    print("\n" + "="*40)
    print(" CONFUSION MATRIX")
    print("="*40)
    
    classes = sorted(list(set(actual_tags) | set(predicted_tags)))
    cm = confusion_matrix(actual_tags, predicted_tags, labels=classes)
    print(cm)

    # Plotting
    plt.figure(figsize=(14, 12))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues",
        xticklabels=classes, yticklabels=classes,
        cbar_kws={"shrink": 0.7}
    )
    plt.title("Confusion Matrix", fontsize=20, pad=20)
    plt.xlabel("Predicted Label", fontsize=16, labelpad=15)
    plt.ylabel("Actual Label", fontsize=16, labelpad=15)
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    plt.tight_layout()
    
    save_filename = "confusion_matrix.png"
    plt.savefig(save_filename, dpi=300)
    print(f"\nSaved confusion matrix image → {save_filename}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    parser.add_argument("--text-col", default="log", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing true labels")

    args = parser.parse_args()
    evaluate(args.csv, args.text_col, args.label_col)