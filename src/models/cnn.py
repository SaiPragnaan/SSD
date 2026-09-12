import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from typing import Dict, Optional, Tuple


class Sleep1DCNN(nn.Module):
    """
    1D Convolutional Neural Network for Wearable Accelerometer Sleep State Detection.
    
    Architecture (Phase 8):
      Input (B, 2, L)
        ↓
      Conv1D(32, k=7, p=3) -> BatchNorm -> ReLU
        ↓
      Conv1D(64, k=5, p=2) -> BatchNorm -> ReLU -> MaxPool1D(2)
        ↓
      Conv1D(128, k=3, p=1) -> BatchNorm -> ReLU
        ↓
      AdaptiveAvgPool1D(1)
        ↓
      Dense(64) -> ReLU -> Dropout(0.3)
        ↓
      Dense(1) -> Sigmoid -> P(sleep)
    """
    def __init__(
        self,
        in_channels: int = 2,
        conv_filters: Tuple[int, int, int] = (32, 64, 128),
        kernel_sizes: Tuple[int, int, int] = (7, 5, 3),
        dense_dim: int = 64,
        dropout: float = 0.3
    ):
        super().__init__()
        
        c1, c2, c3 = conv_filters
        k1, k2, k3 = kernel_sizes
        
        self.features = nn.Sequential(
            nn.Conv1d(in_channels, c1, kernel_size=k1, padding=k1 // 2),
            nn.BatchNorm1d(c1),
            nn.ReLU(inplace=True),
            
            nn.Conv1d(c1, c2, kernel_size=k2, padding=k2 // 2),
            nn.BatchNorm1d(c2),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2),
            
            nn.Conv1d(c2, c3, kernel_size=k3, padding=k3 // 2),
            nn.BatchNorm1d(c3),
            nn.ReLU(inplace=True),
            
            nn.AdaptiveAvgPool1d(1)
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c3, dense_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(dense_dim, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, L)
        feat = self.features(x)
        logits = self.classifier(feat).squeeze(-1)
        return torch.sigmoid(logits)


class SleepConvGRU(nn.Module):
    """
    Hybrid 1D-CNN + Bidirectional GRU model for joint spatial and recurrent temporal modeling (Phase 10).
    """
    def __init__(
        self,
        in_channels: int = 2,
        conv_dim: int = 64,
        gru_hidden_dim: int = 64,
        num_gru_layers: int = 1,
        dropout: float = 0.3
    ):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, conv_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(conv_dim),
            nn.ReLU(inplace=True),
            nn.MaxPool1d(2)
        )
        self.gru = nn.GRU(
            input_size=conv_dim,
            hidden_size=gru_hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(gru_hidden_dim * 2, 32),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 2, L)
        feat = self.conv(x) # (B, conv_dim, L/2)
        feat = feat.permute(0, 2, 1) # (B, L/2, conv_dim)
        out, _ = self.gru(feat) # (B, L/2, 2*gru_hidden)
        # Take mean over sequence
        pooled = torch.mean(out, dim=1)
        logits = self.classifier(pooled).squeeze(-1)
        return torch.sigmoid(logits)


def train_torch_model(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    epochs: int = 15,
    lr: float = 0.001,
    weight_decay: float = 1e-4,
    device: Optional[str] = None
) -> Tuple[nn.Module, Dict[str, list]]:
    """
    Standard PyTorch training loop with BCE loss, AdamW optimizer, and validation tracking.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    model = model.to(device)
    criterion = nn.BCELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    
    history = {"train_loss": [], "val_loss": [], "val_f1": []}
    best_val_loss = float("inf")
    best_state = None
    
    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * len(y)
            
        train_loss /= len(train_loader.dataset)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch in val_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                pred = model(x)
                loss = criterion(pred, y)
                val_loss += loss.item() * len(y)
                val_preds.append(pred.cpu().numpy())
                val_targets.append(y.cpu().numpy())
                
        val_loss /= len(val_loader.dataset)
        val_preds = np.concatenate(val_preds)
        val_targets = np.concatenate(val_targets)
        
        bin_pred = (val_preds >= 0.5).astype(int)
        tp = np.sum((bin_pred == 1) & (val_targets == 1))
        fp = np.sum((bin_pred == 1) & (val_targets == 0))
        fn = np.sum((bin_pred == 0) & (val_targets == 1))
        val_f1 = (2 * tp) / (2 * tp + fp + fn + 1e-12)
        
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["val_f1"].append(float(val_f1))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            
        if epoch % 5 == 0 or epoch == epochs:
            print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val F1: {val_f1:.4f}")
            
    if best_state is not None:
        model.load_state_dict(best_state)
        
    return model, history


def predict_torch_model(
    model: nn.Module,
    loader: DataLoader,
    device: Optional[str] = None
) -> np.ndarray:
    """
    Runs model inference over a DataLoader and returns 1D array of probabilities.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    
    model = model.to(device)
    model.eval()
    
    probs = []
    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            p = model(x)
            probs.append(p.cpu().numpy())
            
    return np.concatenate(probs)

