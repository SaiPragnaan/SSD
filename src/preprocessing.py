import numpy as np
import pandas as pd
from typing import Tuple, Dict, Optional


def clean_and_interpolate_series(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans raw wearable accelerometer signals by interpolating missing steps/values
    and clipping physically impossible extreme sensor readings.
    """
    df = df.copy()
    
    # Forward-fill then backward-fill any missing signal values
    df["enmo"] = df["enmo"].ffill().bfill().fillna(0.0)
    df["anglez"] = df["anglez"].ffill().bfill().fillna(0.0)
    
    # Clip to physical wearable ranges
    df["enmo"] = np.clip(df["enmo"], 0.0, 10.0) # ENMO in g-force
    df["anglez"] = np.clip(df["anglez"], -90.0, 90.0) # Wrist angle in degrees
    
    return df


def normalize_signals(
    df: pd.DataFrame,
    stats: Optional[Dict[str, Dict[str, float]]] = None
) -> Tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """
    Normalizes enmo and anglez using robust Z-score statistics.
    If stats is not provided, computes mean and standard deviation.
    """
    df = df.copy()
    if stats is None:
        stats = {
            "enmo": {
                "mean": float(df["enmo"].mean()),
                "std": float(df["enmo"].std() + 1e-6)
            },
            "anglez": {
                "mean": float(df["anglez"].mean()),
                "std": float(df["anglez"].std() + 1e-6)
            }
        }
        
    df["enmo_norm"] = (df["enmo"] - stats["enmo"]["mean"]) / stats["enmo"]["std"]
    df["anglez_norm"] = (df["anglez"] - stats["anglez"]["mean"]) / stats["anglez"]["std"]
    
    return df, stats


# =====================================================================
# Phase 14: Sensor Perturbation / Robustness Injection Utilities
# =====================================================================

def inject_gaussian_noise(
    df: pd.DataFrame,
    enmo_noise_std: float = 0.05,
    anglez_noise_std: float = 5.0,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Adds Gaussian noise to sensor signals to evaluate sensitivity to sensor chatter.
    """
    rng = np.random.RandomState(random_seed)
    df_pert = df.copy()
    
    df_pert["enmo"] = np.maximum(0.0, df_pert["enmo"] + rng.normal(0, enmo_noise_std, size=len(df)))
    df_pert["anglez"] = np.clip(df_pert["anglez"] + rng.normal(0, anglez_noise_std, size=len(df)), -90.0, 90.0)
    
    return df_pert


def inject_missing_chunks(
    df: pd.DataFrame,
    missing_fraction: float = 0.10,
    chunk_size_steps: int = 60,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Simulates transmission drops or device off-wrist periods by zeroing/holding chunks of data.
    """
    rng = np.random.RandomState(random_seed)
    df_pert = df.copy()
    n = len(df)
    
    num_chunks = int((n * missing_fraction) / chunk_size_steps)
    for _ in range(num_chunks):
        start_idx = rng.randint(0, max(1, n - chunk_size_steps))
        end_idx = start_idx + chunk_size_steps
        # Sensor drops typically result in zero ENMO and frozen angle
        df_pert.iloc[start_idx:end_idx, df_pert.columns.get_loc("enmo")] = 0.0
        
    return df_pert


def inject_sensor_spikes(
    df: pd.DataFrame,
    spike_rate: float = 0.01,
    spike_magnitude: float = 0.5,
    random_seed: int = 42
) -> pd.DataFrame:
    """
    Simulates high-amplitude artifact spikes (e.g. wrist impact or mechanical shocks).
    """
    rng = np.random.RandomState(random_seed)
    df_pert = df.copy()
    n = len(df)
    
    mask = rng.rand(n) < spike_rate
    df_pert.loc[mask, "enmo"] += rng.uniform(0.1, spike_magnitude, size=int(mask.sum()))
    
    return df_pert


def inject_constant_offset(
    df: pd.DataFrame,
    enmo_offset: float = 0.03,
    anglez_offset: float = 15.0
) -> pd.DataFrame:
    """
    Simulates sensor calibration drift or bias offset.
    """
    df_pert = df.copy()
    df_pert["enmo"] = np.maximum(0.0, df_pert["enmo"] + enmo_offset)
    df_pert["anglez"] = np.clip(df_pert["anglez"] + anglez_offset, -90.0, 90.0)
    return df_pert

