import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class ActivityThresholdBaseline:
    """
    Baseline 0: Pure rule-based ENMO activity threshold model.
    
    Logic:
      - Smooth raw ENMO using a moving window.
      - Predict Sleep (1) if smoothed ENMO < threshold, else Wake (0).
      - Compute transitions: 0 -> 1 = Onset, 1 -> 0 = Wakeup.
    """
    def __init__(
        self,
        enmo_threshold: float = 0.02,
        smoothing_window: int = 15,
        min_persistence_steps: int = 12
    ):
        self.enmo_threshold = enmo_threshold
        self.smoothing_window = smoothing_window
        self.min_persistence_steps = min_persistence_steps
        
    def fit(self, X_df: pd.DataFrame, y_true: np.ndarray, search_grid: Optional[List[float]] = None) -> "ActivityThresholdBaseline":
        """
        Calibrates the optimal ENMO threshold on training/validation data.
        """
        if search_grid is None:
            search_grid = [0.005, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05, 0.08]
            
        enmo_means = X_df["enmo_mean"].values if "enmo_mean" in X_df else X_df["enmo"].values
        
        best_f1 = -1.0
        best_thresh = self.enmo_threshold
        
        for thresh in search_grid:
            pred = (enmo_means < thresh).astype(int)
            # F1 score on binary state
            tp = np.sum((pred == 1) & (y_true == 1))
            fp = np.sum((pred == 1) & (y_true == 0))
            fn = np.sum((pred == 0) & (y_true == 1))
            precision = tp / (tp + fp + 1e-12)
            recall = tp / (tp + fn + 1e-12)
            f1 = 2 * precision * recall / (precision + recall + 1e-12)
            
            if f1 > best_f1:
                best_f1 = f1
                best_thresh = thresh
                
        self.enmo_threshold = best_thresh
        print(f"[ActivityThresholdBaseline] Fitted optimal threshold: {self.enmo_threshold:.4f} (State F1: {best_f1:.4f})")
        return self
        
    def predict_proba(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Estimates sleep probability as an inverse sigmoid of normalized ENMO.
        """
        enmo_vals = X_df["enmo_mean"].values if "enmo_mean" in X_df else X_df.values
        # Logistic transformation: lower enmo -> higher sleep prob
        scaled = (self.enmo_threshold - enmo_vals) / (self.enmo_threshold + 1e-6)
        probs = 1.0 / (1.0 + np.exp(-scaled * 3.0))
        return probs
        
    def predict(self, X_df: pd.DataFrame) -> np.ndarray:
        """
        Predicts binary state: 1 = asleep, 0 = awake.
        """
        probs = self.predict_proba(X_df)
        return (probs >= 0.5).astype(int)

