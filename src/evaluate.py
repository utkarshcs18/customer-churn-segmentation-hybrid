# stage -> 3 (evaluate global model vs per-segment models, compare results)

import os
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, recall_score, accuracy_score, precision_score, classification_report


def evaluate_global_model(global_model, X_test_scaled, y_test):

    y_pred = global_model.predict(X_test_scaled)

    metrics = {
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
    }

    print("\n" + "="*50)
    print("  GLOBAL MODEL : Evaluation Report")
    print("="*50)
    print(classification_report(y_test, y_pred, target_names=["No Churn", "Churn"]))

    return metrics


def evaluate_segment_models(segment_models, kmeans, X_test_scaled, y_test):

    y_test_array = np.array(y_test)
    test_segments = kmeans.predict(X_test_scaled)

    y_pred = np.zeros(len(y_test_array), dtype=int)

    print("\n" + "="*50)
    print("  PER-SEGMENT MODEL — Per-Segment Breakdown")
    print("="*50)

    for seg_id, model in segment_models.items():
        mask = (test_segments == seg_id)

        if mask.sum() == 0:
            print(f"\nSegment {seg_id}: no test customers — skipped.")
            continue

        X_seg = X_test_scaled[mask]
        y_seg = y_test_array[mask]

        seg_pred = model.predict(X_seg)
        y_pred[mask] = seg_pred

        seg_f1     = round(f1_score(y_seg, seg_pred, zero_division=0), 4)
        seg_recall = round(recall_score(y_seg, seg_pred, zero_division=0), 4)
        churn_rate = round(y_seg.mean() * 100, 1)

        print(f"\n  Segment {seg_id}: {mask.sum()} customers | "
              f"Churn rate: {churn_rate}% | "
              f"F1: {seg_f1} | Recall: {seg_recall}")

    metrics = {
        "accuracy":  round(accuracy_score(y_test_array, y_pred), 4),
        "precision": round(precision_score(y_test_array, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test_array, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test_array, y_pred, zero_division=0), 4),
    }

    print("\n--- Overall Per-Segment Performance ---")
    print(classification_report(y_test_array, y_pred, target_names=["No Churn", "Churn"]))

    return metrics


def compare_results(global_metrics, segment_metrics, results_dir="results"):

    print("\n" + "="*50)
    print("  HYBRID vs GLOBAL — Final Comparison")
    print("="*50)

    comparison_df = pd.DataFrame({
        "Model":     ["Global Model", "Per-Segment Model"],
        "Accuracy":  [global_metrics["accuracy"],  segment_metrics["accuracy"]],
        "Precision": [global_metrics["precision"], segment_metrics["precision"]],
        "Recall":    [global_metrics["recall"],    segment_metrics["recall"]],
        "F1 Score":  [global_metrics["f1"],        segment_metrics["f1"]],
    }).set_index("Model")

    print(f"\n{comparison_df.to_string()}\n")

    f1_diff     = segment_metrics["f1"]     - global_metrics["f1"]
    recall_diff = segment_metrics["recall"] - global_metrics["recall"]

    print("--- Conclusion ---")
    if f1_diff > 0.01:
        conclusion = (
            f"Per-segment models outperform the global model on F1 "
            f"(+{f1_diff:.4f}) and Recall (+{recall_diff:.4f}). "
            f"Segmenting customers before classification captures behavioural "
            f"patterns that a one-size-fits-all model misses."
        )
    elif f1_diff < -0.01:
        conclusion = (
            f"The global model outperforms the per-segment approach on F1 "
            f"({f1_diff:.4f}). The clusters may not be well-separated enough "
            f"for segment-specific patterns to emerge."
        )
    else:
        conclusion = (
            f"The two approaches perform comparably (F1 difference: {f1_diff:.4f}). "
            f"Possible reasons: cluster overlap, uniform churn patterns across "
            f"segments, or the dataset is too small for per-segment gains to appear."
        )
    print(conclusion)

    os.makedirs(results_dir, exist_ok=True)
    report_path = os.path.join(results_dir, "evaluation_report.txt")

    with open(report_path, "w") as f:
        f.write("HYBRID CHURN MODEL : Evaluation Report\n")
        f.write("="*50 + "\n\n")
        f.write(comparison_df.to_string())
        f.write("\n\nConclusion:\n")
        f.write(conclusion + "\n")

    print(f"\nReport saved -> {report_path}")

    return comparison_df