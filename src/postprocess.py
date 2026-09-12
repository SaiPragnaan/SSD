import numpy as np
import pandas as pd
from typing import List, Dict, Optional


def smooth_probabilities(probs: np.ndarray, window_size: int = 5) -> np.ndarray:
    """
    Smooths continuous probabilities using a centered moving average.
    """
    if window_size <= 1:
        return probs
    pad = window_size // 2
    padded = np.pad(probs, pad, mode='edge')
    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(padded, kernel, mode='valid')
    return smoothed[:len(probs)]


def extract_events_from_probabilities(
    meta_df: pd.DataFrame,
    probs: np.ndarray,
    sleep_onset_threshold: float = 0.70,
    wake_threshold: float = 0.30,
    min_onset_persistence_steps: int = 10,
    min_wake_persistence_steps: int = 6,
    min_event_separation_steps: int = 30,
    smooth_window: int = 5
) -> pd.DataFrame:
    """
    Applies hysteresis persistence and refractory period filtering to extract
    discrete 'onset' and 'wakeup' events from predicted sleep probabilities.
    
    Parameters:
      - meta_df: DataFrame with columns ['series_id', 'step', 'timestamp'] corresponding to probs.
      - probs: Array of P(sleep) predictions.
      - sleep_onset_threshold: Threshold to consider state entering sleep.
      - wake_threshold: Threshold to consider state entering wake.
      - min_onset_persistence_steps: Minimum consecutive steps above threshold for onset.
      - min_wake_persistence_steps: Minimum consecutive steps below threshold for wake.
      - min_event_separation_steps: Minimum steps between consecutive events of same/opposite type.
      - smooth_window: Size of moving average window.
    """
    detected_events = []
    
    # Process each series independently
    for series_id, group in meta_df.groupby("series_id"):
        indices = group.index.values
        series_probs = probs[indices]
        series_steps = group["step"].values
        series_times = group["timestamp"].values
        
        # Smooth probabilities
        smoothed = smooth_probabilities(series_probs, window_size=smooth_window)
        
        # State machine: 0 = awake, 1 = asleep
        current_state = 0 if smoothed[0] < sleep_onset_threshold else 1
        consecutive_sleep = 0
        consecutive_wake = 0
        
        last_event_step = -100000
        last_event_type = None
        
        for i in range(len(smoothed)):
            p = smoothed[i]
            step = series_steps[i]
            ts = pd.Timestamp(series_times[i])
            
            if p >= sleep_onset_threshold:
                consecutive_sleep += 1
                consecutive_wake = 0
            elif p <= wake_threshold:
                consecutive_wake += 1
                consecutive_sleep = 0
            else:
                # Ambiguous intermediate band - decay counters
                consecutive_sleep = max(0, consecutive_sleep - 1)
                consecutive_wake = max(0, consecutive_wake - 1)
                
            # Check transitions
            if current_state == 0 and consecutive_sleep >= min_onset_persistence_steps:
                # Transition: Awake -> Asleep (Onset)
                event_step = step - (min_onset_persistence_steps // 2)
                if (event_step - last_event_step) >= min_event_separation_steps:
                    detected_events.append({
                        "series_id": series_id,
                        "step": int(event_step),
                        "timestamp": ts.isoformat(),
                        "event": "onset",
                        "confidence": float(np.mean(smoothed[max(0, i - min_onset_persistence_steps):i + 1]))
                    })
                    last_event_step = event_step
                    last_event_type = "onset"
                    current_state = 1
                    
            elif current_state == 1 and consecutive_wake >= min_wake_persistence_steps:
                # Transition: Asleep -> Awake (Wakeup)
                event_step = step - (min_wake_persistence_steps // 2)
                if (event_step - last_event_step) >= min_event_separation_steps:
                    detected_events.append({
                        "series_id": series_id,
                        "step": int(event_step),
                        "timestamp": ts.isoformat(),
                        "event": "wakeup",
                        "confidence": float(1.0 - np.mean(smoothed[max(0, i - min_wake_persistence_steps):i + 1]))
                    })
                    last_event_step = event_step
                    last_event_type = "wakeup"
                    current_state = 0
                    
    df_events = pd.DataFrame(detected_events)
    if df_events.empty:
        df_events = pd.DataFrame(columns=["series_id", "step", "timestamp", "event", "confidence"])
    return df_events

