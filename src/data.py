import os
import math
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional


def generate_synthetic_dataset(
    num_series: int = 10,
    days_per_series: int = 4,
    sample_rate_sec: int = 5,
    noise_level: float = 0.05,
    random_seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generates synthetic wrist-worn accelerometer data mimicking the CMI Sleep States dataset.
    
    Signals generated:
      - 'enmo': Euclidean Norm Minus One (high & variable during wake, low during sleep)
      - 'anglez': Arm angle in degrees (-90 to +90, dynamic during wake, stable during sleep)
      
    Events generated:
      - 'onset': Sleep onset event
      - 'wakeup': Wake-up event
    """
    rng = np.random.RandomState(random_seed)
    
    # 5-second steps per 24-hour day
    steps_per_day = (24 * 3600) // sample_rate_sec
    total_steps_per_series = days_per_series * steps_per_day
    
    series_list = []
    events_list = []
    
    base_start_time = pd.Timestamp("2023-01-01 12:00:00")
    
    for s_idx in range(num_series):
        series_id = f"series_{s_idx:03d}"
        
        # Series-specific sleep habit offsets
        habit_onset_hour = 22.5 + rng.normal(0, 0.5) # ~10:30 PM
        habit_wake_hour = 6.5 + rng.normal(0, 0.5)   # ~6:30 AM
        
        steps = np.arange(total_steps_per_series)
        time_deltas = pd.to_timedelta(steps * sample_rate_sec, unit='s')
        timestamps = base_start_time + time_deltas
        
        # Calculate time of day in fractional hours [0, 24)
        hours_of_day = (timestamps.hour + timestamps.minute / 60.0 + timestamps.second / 3600.0).values
        
        # Base signals
        enmo = np.zeros(total_steps_per_series, dtype=np.float32)
        anglez = np.zeros(total_steps_per_series, dtype=np.float32)
        
        for day in range(days_per_series):
            day_offset = day * steps_per_day
            night_idx = day + 1
            
            # Actual onset and wake times for this night with random jitter
            actual_onset_hour = (habit_onset_hour + rng.normal(0, 0.4)) % 24
            actual_wake_hour = (habit_wake_hour + rng.normal(0, 0.4)) % 24
            
            # Approximate step indices
            # Day starts at 12:00 (noon). 
            # 22.5 is 10.5 hours after 12:00.
            onset_step_day = int(((actual_onset_hour - 12) % 24) * 3600 / sample_rate_sec)
            wake_step_day = int(((actual_wake_hour - 12) % 24) * 3600 / sample_rate_sec)
            
            onset_step = day_offset + onset_step_day
            wake_step = day_offset + wake_step_day
            
            if onset_step < total_steps_per_series and wake_step < total_steps_per_series and onset_step < wake_step:
                events_list.append({
                    "series_id": series_id,
                    "night": night_idx,
                    "event": "onset",
                    "step": onset_step,
                    "timestamp": timestamps[onset_step].isoformat()
                })
                events_list.append({
                    "series_id": series_id,
                    "night": night_idx,
                    "event": "wakeup",
                    "step": wake_step,
                    "timestamp": timestamps[wake_step].isoformat()
                })
        
        # Synthesize continuous ENMO and ANGLEZ signals
        # 1. Awake state: variable ENMO (mean ~0.08, spikes up to 0.4+), active angle changes
        # 2. Sleep state: near-zero ENMO (<0.01) with occasional micro-movements, fixed angle plateaus
        
        # Dense ground truth mask
        is_sleeping = np.zeros(total_steps_per_series, dtype=bool)
        series_events = [e for e in events_list if e["series_id"] == series_id]
        
        onsets = [e["step"] for e in series_events if e["event"] == "onset"]
        wakes = [e["step"] for e in series_events if e["event"] == "wakeup"]
        
        for on, wk in zip(onsets, wakes):
            is_sleeping[on:wk] = True
            
        # Wake ENMO: Gamma distributed motion bursts + baseline
        wake_mask = ~is_sleeping
        enmo[wake_mask] = rng.gamma(shape=1.5, scale=0.04, size=int(wake_mask.sum())) + 0.015
        
        # Sleep ENMO: low noise + occasional brief body turn/twitch
        sleep_mask = is_sleeping
        sleep_count = int(sleep_mask.sum())
        enmo[sleep_mask] = rng.exponential(scale=0.003, size=sleep_count) + 0.001
        
        # Micro-movements during sleep (1% chance of movement burst)
        micro_moves = rng.rand(sleep_count) < 0.015
        enmo_sleep_indices = np.where(sleep_mask)[0]
        enmo[enmo_sleep_indices[micro_moves]] += rng.uniform(0.04, 0.15, size=micro_moves.sum())
        
        # Wake anglez: random walk with mean reversion
        wake_angles = np.zeros(int(wake_mask.sum()))
        curr_ang = rng.uniform(-40, 40)
        for i in range(len(wake_angles)):
            curr_ang = 0.96 * curr_ang + rng.normal(0, 12)
            curr_ang = np.clip(curr_ang, -85, 85)
            wake_angles[i] = curr_ang
        anglez[wake_mask] = wake_angles
        
        # Sleep anglez: stable postures with abrupt posture shifts
        sleep_indices = np.where(sleep_mask)[0]
        if len(sleep_indices) > 0:
            posture_angle = rng.choice([-60, -30, 0, 35, 65])
            for idx in sleep_indices:
                if rng.rand() < 0.003: # posture shift
                    posture_angle = rng.choice([-60, -30, 0, 35, 65])
                anglez[idx] = posture_angle + rng.normal(0, 1.2)
                
        # Add background measurement sensor noise
        enmo += np.abs(rng.normal(0, noise_level * 0.02, size=total_steps_per_series))
        anglez += rng.normal(0, noise_level * 2.0, size=total_steps_per_series)
        anglez = np.clip(anglez, -90.0, 90.0)
        
        df_series_single = pd.DataFrame({
            "series_id": series_id,
            "step": steps,
            "timestamp": timestamps,
            "anglez": anglez.astype(np.float32),
            "enmo": np.maximum(enmo, 0.0).astype(np.float32)
        })
        series_list.append(df_series_single)
        
    df_series = pd.concat(series_list, ignore_index=True)
    df_events = pd.DataFrame(events_list)
    return df_series, df_events


def load_series_and_events(
    series_path: str,
    events_path: str,
    fallback_synthetic: bool = True,
    synthetic_kwargs: Optional[dict] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads series and events from disk. If files are not found and fallback is enabled,
    generates a high-fidelity synthetic dataset automatically.
    """
    if os.path.exists(series_path) and os.path.exists(events_path):
        print(f"Loading data from {series_path} and {events_path}...")
        if series_path.endswith(".parquet"):
            df_series = pd.read_parquet(series_path)
        else:
            df_series = pd.read_csv(series_path)
            
        if events_path.endswith(".parquet"):
            df_events = pd.read_parquet(events_path)
        else:
            df_events = pd.read_csv(events_path)
        return df_series, df_events
    elif fallback_synthetic:
        print(f"Dataset files not found at '{series_path}'. Generating synthetic sleep dataset...")
        kwargs = synthetic_kwargs or {}
        df_series, df_events = generate_synthetic_dataset(**kwargs)
        return df_series, df_events
    else:
        raise FileNotFoundError(f"Could not find {series_path} or {events_path}")


def split_series_ids(
    series_ids: List[str],
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    random_seed: int = 42
) -> Dict[str, List[str]]:
    """
    Groups splits by series_id to strictly prevent window leakage across subjects.
    """
    rng = np.random.RandomState(random_seed)
    unique_ids = np.array(sorted(list(set(series_ids))))
    rng.shuffle(unique_ids)
    
    n = len(unique_ids)
    n_train = max(1, int(n * train_ratio))
    n_val = max(1, int(n * val_ratio))
    
    train_ids = list(unique_ids[:n_train])
    val_ids = list(unique_ids[n_train:n_train + n_val])
    test_ids = list(unique_ids[n_train + n_val:])
    
    # If test_ids is empty due to small n, adjust
    if len(test_ids) == 0 and len(val_ids) > 1:
        test_ids.append(val_ids.pop())
    elif len(test_ids) == 0:
        test_ids = [train_ids[-1]]
        
    return {
        "train": train_ids,
        "val": val_ids,
        "test": test_ids
    }


def create_step_labels(df_series: pd.DataFrame, df_events: pd.DataFrame) -> pd.Series:
    """
    Constructs dense binary labels (0 = awake, 1 = asleep) for each step in df_series.
    """
    labels = np.zeros(len(df_series), dtype=np.int8)
    
    # Drop rows with null step
    valid_events = df_events.dropna(subset=["step"]).copy()
    valid_events["step"] = valid_events["step"].astype(int)
    
    # Build series_id start index lookup for fast indexing
    for series_id, group in df_series.groupby("series_id"):
        series_indices = group.index.values
        series_start_step = group["step"].iloc[0]
        series_events = valid_events[valid_events["series_id"] == series_id].sort_values("step")
        
        onsets = series_events[series_events["event"] == "onset"]["step"].values
        wakes = series_events[series_events["event"] == "wakeup"]["step"].values
        
        for on, wk in zip(onsets, wakes):
            # Convert step numbers to relative dataframe row positions
            rel_on = max(0, on - series_start_step)
            rel_wk = min(len(series_indices), wk - series_start_step)
            if rel_on < rel_wk:
                labels[series_indices[rel_on:rel_wk]] = 1
                
    return pd.Series(labels, index=df_series.index, name="is_asleep")


def summarize_dataset(df_series: pd.DataFrame, df_events: pd.DataFrame) -> Dict:
    """
    Generates summary diagnostics to understand dataset properties.
    """
    unique_series = df_series["series_id"].nunique()
    total_steps = len(df_series)
    total_events = len(df_events)
    null_events = df_events["step"].isna().sum() if "step" in df_events else 0
    
    summary = {
        "num_series": unique_series,
        "total_steps": total_steps,
        "total_events": total_events,
        "null_step_events": int(null_events),
        "enmo_mean": float(df_series["enmo"].mean()),
        "enmo_std": float(df_series["enmo"].std()),
        "anglez_mean": float(df_series["anglez"].mean()),
        "anglez_std": float(df_series["anglez"].std())
    }
    return summary

