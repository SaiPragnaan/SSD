import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple


class SleepWindowDataset(Dataset):
    """
    PyTorch Dataset yielding 2-channel 1D raw signal windows for Deep Learning models.
    Channel 0: anglez (normalized)
    Channel 1: enmo (normalized)
    """
    def __init__(
        self,
        windows: np.ndarray, # Shape: (N, 2, L)
        labels: Optional[np.ndarray] = None, # Shape: (N,)
        meta_df: Optional[pd.DataFrame] = None
    ):
        self.windows = torch.tensor(windows, dtype=torch.float32)
        if labels is not None:
            self.labels = torch.tensor(labels, dtype=torch.float32)
        else:
            self.labels = None
        self.meta_df = meta_df
        
    def __len__(self) -> int:
        return len(self.windows)
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        x = self.windows[idx]
        if self.labels is not None:
            y = self.labels[idx]
            return x, y
        return (x,)


def extract_raw_windows_dataset(
    df_series: pd.DataFrame,
    labels: Optional[pd.Series] = None,
    window_size_steps: int = 60,
    stride_steps: int = 12,
    normalize: bool = True
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """
    Extracts raw 2-channel time slices of shape (N, 2, seq_len).
    """
    windows_list = []
    y_list = []
    meta_list = []
    
    # Global/Per-series normalization parameters
    enmo_mean = float(df_series["enmo"].mean())
    enmo_std = float(df_series["enmo"].std() + 1e-6)
    anglez_mean = float(df_series["anglez"].mean())
    anglez_std = float(df_series["anglez"].std() + 1e-6)
    
    for series_id, group in df_series.groupby("series_id"):
        enmo = group["enmo"].values
        anglez = group["anglez"].values
        steps = group["step"].values
        timestamps = pd.to_datetime(group["timestamp"]).values
        
        if normalize:
            enmo = (enmo - enmo_mean) / enmo_std
            anglez = (anglez - anglez_mean) / anglez_std
            
        group_labels = labels.loc[group.index].values if labels is not None else None
        
        n_steps = len(enmo)
        half_win = window_size_steps // 2
        
        for start_idx in range(0, n_steps - window_size_steps + 1, stride_steps):
            end_idx = start_idx + window_size_steps
            center_idx = start_idx + half_win
            
            # Stack 2 channels: (2, seq_len)
            win_2ch = np.stack([anglez[start_idx:end_idx], enmo[start_idx:end_idx]], axis=0)
            windows_list.append(win_2ch)
            
            center_step = steps[center_idx]
            center_time = timestamps[center_idx]
            
            meta_list.append({
                "series_id": series_id,
                "step": center_step,
                "timestamp": pd.Timestamp(center_time)
            })
            
            if group_labels is not None:
                y_list.append(group_labels[center_idx])
                
    windows_arr = np.array(windows_list, dtype=np.float32)
    y_arr = np.array(y_list, dtype=np.float32) if len(y_list) > 0 else np.zeros(len(windows_arr), dtype=np.float32)
    meta_df = pd.DataFrame(meta_list)
    
    return windows_arr, y_arr, meta_df


def build_dataloaders(
    train_data: Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    val_data: Tuple[np.ndarray, np.ndarray, pd.DataFrame],
    test_data: Optional[Tuple[np.ndarray, np.ndarray, pd.DataFrame]] = None,
    batch_size: int = 64,
    num_workers: int = 0
) -> Dict[str, DataLoader]:
    """
    Constructs PyTorch DataLoaders for training, validation, and testing.
    """
    train_ds = SleepWindowDataset(train_data[0], train_data[1], train_data[2])
    val_ds = SleepWindowDataset(val_data[0], val_data[1], val_data[2])
    
    loaders = {
        "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    }
    
    if test_data is not None:
        test_ds = SleepWindowDataset(test_data[0], test_data[1], test_data[2])
        loaders["test"] = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
    return loaders

