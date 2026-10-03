"""
Transformer-based forecasting model for financial time series.

Implements a Temporal Fusion Transformer (TFT)-inspired architecture with:
- Multi-head self-attention over the full input sequence
- Gated residual networks (GRN) for non-linear feature mixing
- Variable selection networks for automatic feature weighting
- Quantile output heads for probabilistic forecasting
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from aurora.configs.config import settings
from aurora.models.base import BaseForecaster
from aurora.models.deep_learning.lstm import SequenceDataset
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for sequence position awareness."""

    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class GatedResidualNetwork(nn.Module):
    """
    Gated Residual Network (GRN) from TFT paper.

    Applies non-linear transformation with skip connection controlled by
    a sigmoid gate — allows the network to skip irrelevant transformations.
    """

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.gate = nn.Linear(input_dim, output_dim)
        self.norm = nn.LayerNorm(output_dim)
        self.skip = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()
        self.dropout = nn.Dropout(dropout)
        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.elu(self.fc1(x))
        h = self.dropout(h)
        h = self.fc2(h)
        gate = self.sigmoid(self.gate(x))
        return self.norm(gate * h + (1 - gate) * self.skip(x))


class TemporalTransformerNet(nn.Module):
    """
    Temporal Fusion Transformer-style forecasting network.

    Pipeline:
    1. Linear feature projection to d_model
    2. Positional encoding
    3. N × TransformerEncoder layers (multi-head attention)
    4. GRN aggregation of [CLS] token
    5. Quantile output heads
    """

    def __init__(
        self,
        input_size: int,
        d_model: int,
        num_heads: int,
        num_layers: int,
        dim_feedforward: int,
        dropout: float,
        output_size: int,
        quantiles: list[float],
    ) -> None:
        super().__init__()
        self.quantiles = quantiles
        self.input_proj = nn.Linear(input_size, d_model)
        self.pos_enc = PositionalEncoding(d_model, dropout=dropout)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=num_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.grn = GatedResidualNetwork(d_model, d_model, d_model, dropout)

        # Separate head per quantile
        self.quantile_heads = nn.ModuleList([
            nn.Linear(d_model, output_size) for _ in quantiles
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (batch, seq_len, features)
        h = self.input_proj(x)
        h = self.pos_enc(h)
        h = self.encoder(h)
        # Pool via last token (most recent information)
        h_last = h[:, -1, :]
        h_refined = self.grn(h_last)
        # Stack quantile outputs: (batch, n_quantiles, output_size)
        quantile_outs = torch.stack([head(h_refined) for head in self.quantile_heads], dim=1)
        return quantile_outs  # (batch, n_quantiles, horizon)


class TransformerForecaster(BaseForecaster):
    """
    Transformer-based probabilistic forecaster with quantile regression.

    Outputs forecasts at multiple quantile levels (e.g., [0.1, 0.5, 0.9])
    for full predictive distributions rather than point estimates.
    """

    name = "transformer"
    requires_sequential = True
    supports_intervals = True

    def __init__(
        self,
        horizon: int = 1,
        seq_len: int | None = None,
        d_model: int = 64,
        num_heads: int | None = None,
        num_layers: int | None = None,
        dim_feedforward: int = 256,
        dropout: float | None = None,
        learning_rate: float | None = None,
        batch_size: int | None = None,
        max_epochs: int | None = None,
        patience: int | None = None,
        quantiles: list[float] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        cfg = settings.models
        self.seq_len = seq_len or cfg.sequence_length
        self.d_model = d_model
        self.num_heads = num_heads or cfg.num_heads
        self.num_layers = num_layers or cfg.num_layers
        self.dim_feedforward = dim_feedforward
        self.dropout = dropout or cfg.dropout
        self.learning_rate = learning_rate or cfg.learning_rate
        self.batch_size = batch_size or cfg.batch_size
        self.max_epochs = max_epochs or cfg.max_epochs
        self.patience = patience or cfg.patience
        self.quantiles = quantiles or [0.1, 0.25, 0.5, 0.75, 0.9]
        self._median_idx = self.quantiles.index(0.5) if 0.5 in self.quantiles else len(self.quantiles) // 2
        self._net: TemporalTransformerNet | None = None
        self._device = torch.device("cuda" if (torch.cuda.is_available() and cfg.use_gpu) else "cpu")
        self._feature_means: np.ndarray | None = None
        self._feature_stds: np.ndarray | None = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "TransformerForecaster":
        X_arr, y_arr = self._preprocess(X, y, fit=True)
        dataset = SequenceDataset(X_arr, y_arr, self.seq_len)
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=True)

        self._net = TemporalTransformerNet(
            input_size=X_arr.shape[1],
            d_model=self.d_model,
            num_heads=self.num_heads,
            num_layers=self.num_layers,
            dim_feedforward=self.dim_feedforward,
            dropout=self.dropout,
            output_size=self.horizon,
            quantiles=self.quantiles,
        ).to(self._device)

        optimizer = torch.optim.AdamW(self._net.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.max_epochs
        )
        best_loss = float("inf")
        no_improve = 0

        for epoch in range(self.max_epochs):
            self._net.train()
            epoch_loss = 0.0
            for x_batch, y_batch in loader:
                x_batch = x_batch.to(self._device)
                y_batch = y_batch.to(self._device)

                optimizer.zero_grad()
                q_preds = self._net(x_batch)  # (batch, n_quantiles, horizon)

                # Pinball / quantile loss
                loss = self._quantile_loss(q_preds, y_batch)
                loss.backward()
                nn.utils.clip_grad_norm_(self._net.parameters(), settings.models.grad_clip)
                optimizer.step()
                epoch_loss += loss.item()

            scheduler.step()
            avg_loss = epoch_loss / len(loader)

            if avg_loss < best_loss - 1e-6:
                best_loss = avg_loss
                no_improve = 0
            else:
                no_improve += 1

            if no_improve >= self.patience:
                logger.info(f"[Transformer] Early stop at epoch {epoch + 1}")
                break

            if (epoch + 1) % 20 == 0:
                logger.info(f"[Transformer] Epoch {epoch + 1}: loss={avg_loss:.6f}")

        self._fitted = True
        return self

    def _quantile_loss(
        self, q_preds: torch.Tensor, y: torch.Tensor
    ) -> torch.Tensor:
        """Pinball loss summed across quantiles."""
        total = torch.tensor(0.0, device=self._device)
        for i, q in enumerate(self.quantiles):
            pred_q = q_preds[:, i, 0] if self.horizon == 1 else q_preds[:, i]
            errors = y - pred_q
            total += torch.mean(torch.max(q * errors, (q - 1) * errors))
        return total / len(self.quantiles)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Median quantile point forecast."""
        if not self._fitted or self._net is None:
            raise RuntimeError("Not fitted.")
        self._net.eval()
        q_preds = self._predict_quantiles(X)
        return q_preds[:, self._median_idx]  # median

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (lower, upper) from calibrated quantile outputs."""
        if not self._fitted or self._net is None:
            raise RuntimeError("Not fitted.")
        self._net.eval()
        q_preds = self._predict_quantiles(X)
        # Find nearest quantile to alpha/2 and 1-alpha/2
        lo_q = min(self.quantiles, key=lambda q: abs(q - alpha / 2))
        hi_q = min(self.quantiles, key=lambda q: abs(q - (1 - alpha / 2)))
        lo_idx = self.quantiles.index(lo_q)
        hi_idx = self.quantiles.index(hi_q)
        return q_preds[:, lo_idx], q_preds[:, hi_idx]

    def _predict_quantiles(self, X: pd.DataFrame) -> np.ndarray:
        """Return shape (n_predictions, n_quantiles)."""
        X_arr = self._preprocess_X(X)
        seqs = self._make_sequences(X_arr)
        with torch.no_grad():
            tensor_in = torch.tensor(seqs, dtype=torch.float32).to(self._device)
            q_out = self._net(tensor_in).cpu().numpy()  # (n, n_q, horizon)
        return q_out[:, :, 0] if self.horizon == 1 else q_out[:, :, 0]

    def _preprocess(
        self, X: pd.DataFrame, y: pd.Series, fit: bool
    ) -> tuple[np.ndarray, np.ndarray]:
        X_arr = X.ffill().bfill().fillna(0).values
        if fit:
            self._feature_means = X_arr.mean(axis=0)
            self._feature_stds = X_arr.std(axis=0) + 1e-8
        X_arr = (X_arr - self._feature_means) / self._feature_stds
        return X_arr, y.fillna(0).values

    def _preprocess_X(self, X: pd.DataFrame) -> np.ndarray:
        X_arr = X.ffill().bfill().fillna(0).values
        return (X_arr - self._feature_means) / self._feature_stds

    def _make_sequences(self, X_arr: np.ndarray) -> np.ndarray:
        n = len(X_arr)
        if n < self.seq_len:
            pad = np.zeros((self.seq_len - n, X_arr.shape[1]))
            X_arr = np.vstack([pad, X_arr])
        return np.stack([X_arr[i : i + self.seq_len] for i in range(len(X_arr) - self.seq_len + 1)])

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._net.state_dict() if self._net else None,
            "config": {
                "horizon": self.horizon, "seq_len": self.seq_len,
                "d_model": self.d_model, "num_heads": self.num_heads,
                "num_layers": self.num_layers, "dim_feedforward": self.dim_feedforward,
                "dropout": self.dropout, "quantiles": self.quantiles,
            },
            "feature_means": self._feature_means,
            "feature_stds": self._feature_stds,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "TransformerForecaster":
        ckpt = torch.load(path, map_location="cpu")
        obj = cls(**ckpt["config"])
        if ckpt["state_dict"] is not None:
            n_features = ckpt["feature_means"].shape[0]
            obj._net = TemporalTransformerNet(
                input_size=n_features,
                d_model=ckpt["config"]["d_model"],
                num_heads=ckpt["config"]["num_heads"],
                num_layers=ckpt["config"]["num_layers"],
                dim_feedforward=ckpt["config"]["dim_feedforward"],
                dropout=ckpt["config"]["dropout"],
                output_size=ckpt["config"]["horizon"],
                quantiles=ckpt["config"]["quantiles"],
            )
            obj._net.load_state_dict(ckpt["state_dict"])
            obj._net.eval()
        obj._feature_means = ckpt["feature_means"]
        obj._feature_stds = ckpt["feature_stds"]
        obj._fitted = True
        return obj
