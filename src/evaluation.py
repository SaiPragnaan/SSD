import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
    roc_auc_score,
    average_precision_score
)


def evaluate_state_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None
) -> Dict[str, float]:
    """
    Level 1 Evaluation: Continuous/window sleep state classification metrics.
    """
    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0))
    }
    
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
            metrics["pr_auc"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = 0.0
            metrics["pr_auc"] = 0.0
    else:
        metrics["roc_auc"] = 0.0
        metrics["pr_auc"] = 0.0
        
    return metrics


def match_events_single_series(
    true_steps: np.ndarray,
    pred_steps: np.ndarray,
    tolerance_steps: float
) -> Tuple[int, int, int]:
    """
    Greedy bipartite event matching within a temporal tolerance window.
    
    Returns:
      (TP, FP, FN)
    """
    if len(true_steps) == 0:
        return 0, len(pred_steps), 0
    if len(pred_steps) == 0:
        return 0, 0, len(true_steps)
        
    matched_true = set()
    tp = 0
    fp = 0
    
    # Sort predictions
    sorted_pred_indices = np.argsort(pred_steps)
    sorted_preds = pred_steps[sorted_pred_indices]
    
    for pred_step in sorted_preds:
        # Find closest unmatched true event within tolerance
        best_dist = float("inf")
        best_true_idx = None
        
        for t_idx, true_step in enumerate(true_steps):
            if t_idx in matched_true:
                continue
            dist = abs(pred_step - true_step)
            if dist <= tolerance_steps and dist < best_dist:
                best_dist = dist
                best_true_idx = t_idx
                
        if best_true_idx is not None:
            matched_true.add(best_true_idx)
            tp += 1
        else:
            fp += 1
            
    fn = len(true_steps) - len(matched_true)
    return tp, fp, fn


def evaluate_event_detection(
    true_events_df: pd.DataFrame,
    pred_events_df: pd.DataFrame,
    tolerances_seconds: List[int] = [30, 60, 120, 300, 900],
    sample_rate_sec: int = 5
) -> pd.DataFrame:
    """
    Level 2 Evaluation: Event-level Precision, Recall, and F1 across various timing tolerances.
    """
    results = []
    
    # Clean true events
    true_clean = true_events_df.dropna(subset=["step"]).copy()
    true_clean["step"] = true_clean["step"].astype(int)
    
    for tol_sec in tolerances_seconds:
        tol_steps = tol_sec / sample_rate_sec
        
        for event_type in ["onset", "wakeup", "overall"]:
            total_tp = 0
            total_fp = 0
            total_fn = 0
            
            # Filter by event type if not overall
            if event_type == "overall":
                sub_types = ["onset", "wakeup"]
            else:
                sub_types = [event_type]
                
            for st in sub_types:
                t_sub = true_clean[true_clean["event"] == st]
                p_sub = pred_events_df[pred_events_df["event"] == st] if not pred_events_df.empty else pd.DataFrame()
                
                # Group by series_id
                all_series = set(t_sub["series_id"].unique()).union(
                    set(p_sub["series_id"].unique()) if not p_sub.empty else set()
                )
                
                for s_id in all_series:
                    t_steps = t_sub[t_sub["series_id"] == s_id]["step"].values
                    p_steps = p_sub[p_sub["series_id"] == s_id]["step"].values if not p_sub.empty else np.array([])
                    
                    tp, fp, fn = match_events_single_series(t_steps, p_steps, tol_steps)
                    total_tp += tp
                    total_fp += fp
                    total_fn += fn
                    
            prec = total_tp / (total_tp + total_fp + 1e-12)
            rec = total_tp / (total_tp + total_fn + 1e-12)
            f1 = 2 * prec * rec / (prec + rec + 1e-12)
            
            results.append({
                "tolerance_sec": tol_sec,
                "tolerance_label": f"±{tol_sec}s" if tol_sec < 60 else f"±{tol_sec//60}m",
                "event_type": event_type,
                "precision": float(prec),
                "recall": float(rec),
                "f1": float(f1),
                "tp": int(total_tp),
                "fp": int(total_fp),
                "fn": int(total_fn)
            })
            
    return pd.DataFrame(results)


def print_event_metrics_table(event_metrics_df: pd.DataFrame, title: str = "Event Detection Performance"):
    """
    Prints a formatted markdown table for event metrics across tolerances.
    """
    print(f"\n### {title}")
    overall = event_metrics_df[event_metrics_df["event_type"] == "overall"]
    print("| Tolerance | Event Precision | Event Recall | Event F1 | TP | FP | FN |")
    print("|:---|:---:|:---:|:---:|:---:|:---:|:---:|")
    for _, row in overall.iterrows():
        print(f"| {row['tolerance_label']:<9} | {row['precision']:.4f} | {row['recall']:.4f} | **{row['f1']:.4f}** | {row['tp']:<2} | {row['fp']:<2} | {row['fn']:<2} |")
    print()

