# scripts/evaluate.py

import argparse
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
import time
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns



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

def plot_binary_confusion_matrix(cm, label, save_path=None):
    tn, fp, fn, tp = cm.ravel()

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.set_title(f"Confusion Matrix for {label}", fontsize=18, fontweight="bold")

    
    ax.plot([0, 2], [1, 1], 'k-', linewidth=2)
    ax.plot([1, 1], [0, 2], 'k-', linewidth=2)

    ax.set_xlim(0, 2)
    ax.set_ylim(0, 2)

    ax.set_xticks([0.5, 1.5])
    ax.set_xticklabels(["Negative", "Positive"], fontsize=14)
    ax.set_xlabel("Predicted", fontsize=16)

    ax.set_yticks([0.5, 1.5])
    ax.set_yticklabels(["Negative", "Positive"], fontsize=14)
    ax.set_ylabel("Actual", fontsize=16)


    ax.text(0.5, 1.5, f"{tn}\nTrue Negative", ha='center', va='center', fontsize=15)
    ax.text(1.5, 1.5, f"{fp}\nFalse Positive", ha='center', va='center', fontsize=15)
    ax.text(0.5, 0.5, f"{fn}\nFalse Negative", ha='center', va='center', fontsize=15)
    ax.text(1.5, 0.5, f"{tp}\nTrue Positive", ha='center', va='center', fontsize=15)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f"Saved: {save_path}")

    plt.show()


def evaluate(csv_path: str, text_col: str, label_col: str):
    records = load_csv(csv_path, text_col, label_col)

    classifier = HybridClassifier()

    actual_tags = []
    predicted_tags = []
    model_layer_used = []
    latencies = [] 
    layer_latencies = {"regex": [], "ml": [], "llm": [], "unknown": [], "error": []}

    print(f"\nLoaded {len(records)} evaluation samples.")
    print(f"Columns: {records.columns.tolist()}")

    # evaluate each record
    for _, entry in tqdm(records.iterrows(), total=len(records), desc="Evaluating"):
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

        # store global latency
        latencies.append(elapsed_ms)

        # store latency per layer
        if layer_name not in layer_latencies:
            layer_latencies[layer_name] = []
        layer_latencies[layer_name].append(elapsed_ms)

        # store predictions
        actual_tags.append(true_tag)
        predicted_tags.append(guess_tag)
        model_layer_used.append(layer_name)

    # results
    print("\n HYBRID SYSTEM EVALUATION \n")
    print(classification_report(actual_tags, predicted_tags))

    print("\n LAYER USAGE BREAKDOWN \n")
    print(pd.Series(model_layer_used).value_counts())

    print("\n CONFUSION MATRIX \n")
    cm = confusion_matrix(actual_tags, predicted_tags)
    print(confusion_matrix(actual_tags, predicted_tags))
    #  Confusion Matrix 
    classes = sorted(list(set(actual_tags) | set(predicted_tags)))
    
    plt.figure(figsize=(14, 12))
    
    # Compute confusion matrix
    cm = confusion_matrix(actual_tags, predicted_tags, labels=classes)
    
    # Heatmap
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={"shrink": 0.7}
    )
    
    plt.title("Confusion Matrix", fontsize=20, pad=20)
    
    # Updated axis labels
    plt.xlabel("Predicted Label", fontsize=16, labelpad=15)
    plt.ylabel("Actual Label", fontsize=16, labelpad=15)
    
    # Rotate class labels for readability
    plt.xticks(rotation=45, ha="right", fontsize=12)
    plt.yticks(rotation=0, fontsize=12)
    
    plt.tight_layout()
    
    # Save before showing
    plt.savefig("confusion_matrix.png", dpi=300)
    print("\nSaved confusion matrix image → confusion_matrix.png\n")
    
    plt.show()

    
   
    plt.savefig("confusion_matrix.png", dpi=300)
    print("\nSaved confusion matrix image → confusion_matrix.png\n")


    # ---------------- global latencies ----------------
    #latencies = np.array(latencies)
   # print("\n LATENCY STATISTICS (ms) \n")
    #print(f"Average latency: {latencies.mean():.2f} ms")
    #print(f"Median latency: {np.median(latencies):.2f} ms")
    #print(f"p95 latency: {np.percentile(latencies, 95):.2f} ms")
    #print(f"p99 latency: {np.percentile(latencies, 99):.2f} ms")
    #print(f"Max latency: {latencies.max():.2f} ms")
    #print(f"Min latency: {latencies.min():.2f} ms")

    # ---------------- latency by layer ----------------
    #print("\n LATENCY BY LAYER (ms)\n")

    for layer, values in layer_latencies.items():
        if len(values) == 0:
            continue

        arr = np.array(values)
        """
        print(
            f"{layer:10} count={len(values):4d}  "
            f"avg={arr.mean():8.2f}  "
            f"median={np.median(arr):8.2f}  "
            f"p95={np.percentile(arr, 95):8.2f}  "
            f"p99={np.percentile(arr, 99):8.2f}  "
            f"max={arr.max():8.2f}"
        )
        """


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True, help="Path to evaluation CSV")
    parser.add_argument("--text-col", default="log", help="Column containing log messages")
    parser.add_argument("--label-col", default="label", help="Column containing true labels")

    args = parser.parse_args()
    evaluate(args.csv, args.text_col, args.label_col)