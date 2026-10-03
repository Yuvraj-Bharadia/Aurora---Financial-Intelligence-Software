"""
LSTM / BiLSTM / GRU deep learning forecasters using PyTorch Lightning.

These models capture long-range temporal dependencies in financial sequences.
The architecture supports:
  - Unidirectional LSTM
  - Bidirectional LSTM (BiLSTM)
  - GRU (Gated Recurrent Unit)

Multi-step forecasting via direct strategy (one output head per horizon step).
Uncertainty estimation via MC-Dropout at inference.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aurora.configs.config import settings
from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class SequenceDataset(Dataset):
    """Sliding-window time series dataset for sequence models."""

    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        seq_len: int,
    ) -> None:
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self) -> int:
        return len(self.X) - self.seq_len

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x_seq = self.X[idx : idx + self.seq_len]
        y_target = self.y[idx + self.seq_len]
        return x_seq, y_target


class LSTMNet(nn.Module):
    """
    Flexible LSTM/GRU/BiLSTM network with optional attention pooling.

    Architecture:
        Input → [LSTM layers] → Attention/LastStep → [FC layers] → Output
    """

    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_layers: int,
        output_size: int,
        dropout: float,
        cell_type: Literal["LSTM", "GRU", "BiLSTM"],
    ) -> None:
        super().__init__()
        bidirectional = cell_type == "BiLSTM"
        rnn_class = nn.GRU if cell_type == "GRU" else nn.LSTM

        self.rnn = rnn_class(
            input_size=input_size,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        rnn_out_dim = hidden_dim * (2 if bidirectional else 1)

        # Attention to pool over sequence dimension
        self.attention = nn.Linear(rnn_out_dim, 1)

        self.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(rnn_out_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_size),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        rnn_out, _ = self.rnn(x)  # (batch, seq_len, hidden)
        attn_weights = torch.softmax(self.attention(rnn_out), dim=1)  # (batch, seq_len, 1)
        context = (attn_weights * rnn_out).sum(dim=1)  # (batch, hidden)
        return self.fc(context)  # (batch, output_size)


class LSTMForecaster(BaseForecaster):
    """
    LSTM/BiLSTM/GRU forecasting model with MC-Dropout uncertainty estimation.

    Args:
        cell_type: Architecture variant ('LSTM', 'BiLSTM', 'GRU').
        seq_len: Input sequence length (lookback window).
        horizon: Number of steps ahead to forecast.
    """

    name = "lstm"
    requires_sequential = True
    supports_intervals = True

    def __init__(
        self,
        horizon: int = 1,
        cell_type: Literal["LSTM", "GRU", "BiLSTM"] = "LSTM",
        seq_len: int | None = None,
        hidden_dim: int | None = None,
        num_layers: int | None = None,
        dropout: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        max_epochs: int | None = None,
        patience: int | None = None,
        mc_samples: int = 50,
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        cfg = settings.models
        self.cell_type = cell_type
        self.seq_len = seq_len or cfg.sequence_length
        self.hidden_dim = hidden_dim or cfg.hidden_dim
        self.num_layers = num_layers or cfg.num_layers
        self.dropout = dropout or cfg.dropout
        self.learning_rate = learning_rate or cfg.learning_rate
        self.batch_size = batch_size or cfg.batch_size
        self.max_epochs = max_epochs or cfg.max_epochs
        self.patience = patience or cfg.patience
        self.mc_samples = mc_samples
        self._net: LSTMNet | None = None
        self._device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu) else "cpu")
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "LSTMForecaster":
        """Train the LSTM on a feature matrix X and target series y."""
        X_arr, y_arr = self._scale_and_align(X, y, fit=True)
        dataset = SequenceDataset(X_arr, y_arr, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self._net = LSTMNet(
            input_size=X_arr.shape[1],
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            output_size=self.horizon,
            dropout=self.dropout,
            cell_type=self.cell_type,
        ).to(self._device)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=self.patience // 3, factor=0.5
        )
        criterion = nn.HuberLoss()
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(self.max_epochs):
            self._net.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device).unsqueeze(1) if self.horizon == 1 else y_batch.to(self._device)
                optimizer.zero_grad()
                pred = self._net(x_batch)
                if self.horizon == 1:
                    pred = pred.squeeze(-1)
                loss = criterion(pred, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), settings.models.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                logger.info(f"[{self.cell_type}] Early stopping at epoch {epoch + 1}")
                break

            if (epoch + 1) % 20 == 0:
                logger.info(f"[{self.cell_type}] Epoch {epoch + 1}: loss={avg_loss:.6f}")

        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Point forecasts (no dropout at inference)."""
        if not self._fitted or self._net is None:
            raise RuntimeError("Not fitted.")
        self._net.eval()
        X_arr = self._scale_features(X)
        sequences = self._make_sequences(X_arr)
        with torch.no_grad():
            tensor_in = torch.tensor(sequences, dtype=torch.float32).to(self._device)
            out = self._net(tensor_in).cpu().numpy()
        return out[:, 0] if self.horizon == 1 else out

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """MC-Dropout uncertainty estimation."""
        if not self._fitted or self._net is None:
            raise RuntimeError("Not fitted.")

        # Enable dropout at inference (MC-Dropout)
        self._net.train()
        X_arr = self._scale_features(X)
        sequences = self._make_sequences(X_arr)
        tensor_in = torch.tensor(sequences, dtype=torch.float32).to(self._device)

        mc_preds = []
        with torch.no_grad():
            for _ in range(self.mc_samples):
                out = self._net(tensor_in).cpu().numpy()
                mc_preds.append(out[:, 0] if self.horizon == 1 else out)

        self._net.eval()
        stacked = np.stack(mc_preds)  # (mc_samples, n_pred)
        lo = np.percentile(stacked, alpha / 2 * 100, axis=0)
        hi = np.percentile(stacked, (1 - alpha / 2) * 100, axis=0)
        return lo, hi

    def _scale_and_align(
        self, X: pd.DataFrame, y: pd.Series, fit: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = X.ffill().bfill().fillna(0).values
        if fit:
            self._feature_means = X_arr.mean(axis=0)
            self._feature_stds = X_arr.std(axis=0) + 1e-8
        X_arr = (X_arr - self._feature_means) / self._feature_stds
        y_arr = y.fillna(0).values
        return X_arr, y_arr

    def _scale_features(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.ffill().bfill().fillna(0).values
        return (X_arr - self._feature_means) / self._feature_stds

    def _make_sequences(self, X_arr: np.ndarray) -> np.ndarray:
        """Create overlapping sequences of length seq_len."""
        n = len(X_arr)
        if n <= self.seq_len:
            # Pad with zeros
            pad = np.zeros((self.seq_len - n, X_arr.shape[1]))
            X_arr = np.vstack([pad, X_arr])
        sequences = np.stack([
            X_arr[i : i + self.seq_len]
            for i in range(len(X_arr) - self.seq_len + 1)
        ])
        return sequences

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict() if self._net else None,
            "config": {
                "cell_type": self.cell_type,
                "seq_len": self.seq_len,
                "hidden_dim": self.hidden_dim,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "horizon": self.horizon,
            },
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "LSTMForecaster":
        checkpoint = torch.load(path, map_location="cpu")
        cfg = checkpoint["config"]
        obj = cls(**cfg)
        if checkpoint["state_dict"] is not None:
            obj._net = LSTMNet(
                input_size=checkpoint["feature_means"].shape[0],
                hidden_dim=cfg["hidden_dim"],
                num_layers=cfg["num_layers"],
                output_size=cfg["horizon"],
                dropout=cfg["dropout"],
                cell_type=cfg["cell_type"],
            )
            obj._net.load_state_dict(checkpoint["state_dict"])
            obj._net.eval()
        obj._feature_means = checkpoint["feature_means"]
        obj._feature_stds = checkpoint["feature_stds"]
        obj._fitted = True
        return obj
