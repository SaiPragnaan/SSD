import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


def compute_spectral_features(signal: np.ndarray) -> Dict[str, float]:
    """
    Extracts frequency-domain features from a 1D windowed signal using FFT.
    """
    if len(signal) < 4:
        return {"spec_energy": 0.0, "spec_entropy": 0.0, "spec_dominant_freq": 0.0}
        
    fft_vals = np.abs(np.fft.rfft(signal - np.mean(signal)))
    power = fft_vals ** 2
    total_power = np.sum(power) + 1e-12
    
    # Normalized power spectral density
    psd = power / total_power
    # Spectral entropy
    spec_entropy = -np.sum(psd * np.log(psd + 1e-12))
    # Dominant frequency index
    dominant_freq = float(np.argmax(power))
    
    return {
        "spec_energy": float(total_power / len(signal)),
        "spec_entropy": float(spec_entropy),
        "spec_dominant_freq": dominant_freq
    }


def extract_features_from_window(
    enmo_window: np.ndarray,
    anglez_window: np.ndarray,
    step_center: int,
    hour_of_day: float,
    use_enmo_stats: bool = True,
    use_anglez_stats: bool = True,
    use_derivatives: bool = True,
    use_spectral: bool = True,
    use_temporal_context: bool = True
) -> Dict[str, float]:
    """
    Computes a comprehensive feature dictionary from a single time window.
    """
    feat = {}
    
    # --- 1. ENMO Statistical Features ---
    if use_enmo_stats:
        feat["enmo_mean"] = float(np.mean(enmo_window))
        feat["enmo_std"] = float(np.std(enmo_window))
        feat["enmo_min"] = float(np.min(enmo_window))
        feat["enmo_max"] = float(np.max(enmo_window))
        feat["enmo_median"] = float(np.median(enmo_window))
        q25, q75 = np.percentile(enmo_window, [25, 75])
        feat["enmo_q25"] = float(q25)
        feat["enmo_q75"] = float(q75)
        feat["enmo_iqr"] = float(q75 - q25)
        feat["enmo_range"] = float(feat["enmo_max"] - feat["enmo_min"])
        
    if use_anglez_stats:
        feat["anglez_mean"] = float(np.mean(anglez_window))
        feat["anglez_std"] = float(np.std(anglez_window))
        feat["anglez_min"] = float(np.min(anglez_window))
        feat["anglez_max"] = float(np.max(anglez_window))
        feat["anglez_median"] = float(np.median(anglez_window))
        feat["anglez_range"] = float(feat["anglez_max"] - feat["anglez_min"])
        
    # --- 3. Dynamic / Derivative Features ---
    if use_derivatives:
        enmo_diff = np.diff(enmo_window) if len(enmo_window) > 1 else np.array([0.0])
        anglez_diff = np.diff(anglez_window) if len(anglez_window) > 1 else np.array([0.0])
        
        feat["enmo_diff_mean_abs"] = float(np.mean(np.abs(enmo_diff)))
        feat["enmo_diff_std"] = float(np.std(enmo_diff))
        feat["enmo_diff_max"] = float(np.max(np.abs(enmo_diff)))
        
        feat["anglez_diff_mean_abs"] = float(np.mean(np.abs(anglez_diff)))
        feat["anglez_diff_std"] = float(np.std(anglez_diff))
        feat["anglez_diff_max"] = float(np.max(np.abs(anglez_diff)))
        
    # --- 4. Frequency / Spectral Features ---
    if use_spectral:
        spec_enmo = compute_spectral_features(enmo_window)
        spec_angle = compute_spectral_features(anglez_window)
        for k, v in spec_enmo.items():
            feat[f"enmo_{k}"] = v
        for k, v in spec_angle.items():
            feat[f"anglez_{k}"] = v
            
    # --- 5. Temporal / Context Features ---
    if use_temporal_context:
        # Cyclical encoding of hour (24-hour periodicity)
        feat["hour_sin"] = float(np.sin(2 * np.pi * hour_of_day / 24.0))
        feat["hour_cos"] = float(np.cos(2 * np.pi * hour_of_day / 24.0))
        feat["hour_raw"] = float(hour_of_day)
        
    return feat


def extract_features_dataset(
    df_series: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    window_size_steps: int = 60,
    stride_steps: int = 12,
    sample_rate_sec: int = 5,
    use_enmo_stats: bool = True,
    use_anglez_stats: bool = True,
    use_derivatives: bool = True,
    use_spectral: bool = True,
    use_temporal_context: bool = True
) -> Tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    """
    Sliding window feature extractor across all series.
    
    Returns:
      - X_df: DataFrame of computed tabular features
      - y: 1D array of binary ground-truth labels (sleep = 1, wake = 0) at center of window
      - meta_df: DataFrame containing metadata (series_id, center_step, timestamp)
    """
    rows_feat = []
    y_list = []
    meta_list = []
    
    for series_id, group in df_series.groupby("series_id"):
        enmo = group["enmo"].values
        anglez = group["anglez"].values
        steps = group["step"].values
        timestamps = pd.to_datetime(group["timestamp"]).values
        
        # Series labels if provided
        group_labels = labels.loc[group.index].values if labels is not None else None
        
        n_steps = len(enmo)
        half_win = window_size_steps // 2
        
        for start_idx in range(0, n_steps - window_size_steps + 1, stride_steps):
            end_idx = start_idx + window_size_steps
            center_idx = start_idx + half_win
            
            enmo_win = enmo[start_idx:end_idx]
            anglez_win = anglez[start_idx:end_idx]
            center_step = steps[center_idx]
            center_time = pd.Timestamp(timestamps[center_idx])
            hour_of_day = center_time.hour + center_time.minute / 60.0 + center_time.second / 3600.0
            
            feat_dict = extract_features_from_window(
                enmo_window=enmo_win,
                anglez_window=anglez_win,
                step_center=center_step,
                hour_of_day=hour_of_day,
                use_enmo_stats=use_enmo_stats,
                use_anglez_stats=use_anglez_stats,
                use_derivatives=use_derivatives,
                use_spectral=use_spectral,
                use_temporal_context=use_temporal_context
            )
            rows_feat.append(feat_dict)
            
            meta_list.append({
                "series_id": series_id,
                "step": center_step,
                "timestamp": center_time
            })
            
            if group_labels is not None:
                # Window label is the label at the window center (or majority)
                y_list.append(group_labels[center_idx])
                
    X_df = pd.DataFrame(rows_feat)
    meta_df = pd.DataFrame(meta_list)
    y_arr = np.array(y_list, dtype=np.int8) if len(y_list) > 0 else np.zeros(len(X_df), dtype=np.int8)
    
    return X_df, y_arr, meta_df

