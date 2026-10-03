"""
Layer 3 — QCPAC Joint End-to-End Forecaster.

Trains the differentiable HMM, HRT feature layer, and Transformer critic
in a single backward pass so that regime detection is optimised for
prediction accuracy rather than just observation likelihood.

Architecture
------------
    Macro features (seq)
        │
    DifferentiableHMM  ─────── filtered posterior γ_t
        │                              │
    Classical features (t)  →   HRTLayer(x, γ) → regime-fused features
                                        │
                                TransformerCritic → y_hat, [y_lo, y_hi]

Loss: PhysicsInformedLoss(RMSE + no-arbitrage + elasticity)

Usage::

    model = QCPACForecaster(n_asset_features=50, n_macro_features=10, n_regimes=4)
    model.fit(X, y, macro_seq, depth_series)
    preds = model.predict(X, macro_seq)
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

from aurora.evaluation.physics_loss import PhysicsInformedLoss, compute_rolling_bounds
from aurora.features.hrt_layer import HRTLayer
from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


# ── Differentiable HMM (PyTorch) ──────────────────────────────────────────────

class _DifferentiableHMM(nn.Module):
    """
    End-to-end differentiable Gaussian HMM implemented in PyTorch.
    Runs the causal forward algorithm (no smoothing — no future data).
    """

    def __init__(self, n_states: int, input_dim: int) -> None:
        super().__init__()
        self.n_states = n_states
        self.input_dim = input_dim

        self.transition_logits = nn.Parameter(torch.zeros(n_states, n_states))
        self.means = nn.Parameter(torch.randn(n_states, input_dim) * 0.1)
        self.log_vars = nn.Parameter(torch.zeros(n_states, input_dim))

    def transition_matrix(self) -> torch.Tensor:
        return torch.softmax(self.transition_logits, dim=-1)

    def _log_emission(self, obs: torch.Tensor) -> torch.Tensor:
        """
        obs: (batch, input_dim)
        returns: (batch, n_states)
        """
        batch = obs.shape[0]
        log_probs = torch.zeros(batch, self.n_states, device=obs.device)
        for s in range(self.n_states):
            var = torch.exp(self.log_vars[s]) + 1e-6
            diff = obs - self.means[s]
            log_p = -0.5 * (
                torch.sum(torch.log(2 * math.pi * var))
                + torch.sum(diff ** 2 / var, dim=-1)
            )
            log_probs[:, s] = log_p
        return log_probs

    def forward(self, macro_seq: torch.Tensor) -> torch.Tensor:
        """
        macro_seq: (batch, seq_len, input_dim)
        returns:   (batch, n_states) — filtered posterior at final step
        """
        batch, seq_len, _ = macro_seq.shape
        A = self.transition_matrix()
        log_A = torch.log(A + 1e-12)

        log_alpha = torch.log(
            torch.ones(batch, self.n_states, device=macro_seq.device) / self.n_states
        )

        for t in range(seq_len):
            log_emit = self._log_emission(macro_seq[:, t, :])
            if t == 0:
                log_alpha = log_alpha + log_emit
            else:
                # (batch, n_states, 1) + (1, n_states, n_states) → (batch, n_states, n_states)
                log_alpha_mat = log_alpha.unsqueeze(-1) + log_A.unsqueeze(0)
                log_alpha = torch.logsumexp(log_alpha_mat, dim=1) + log_emit

            # Normalise to prevent underflow
            log_alpha = log_alpha - torch.logsumexp(log_alpha, dim=-1, keepdim=True)

        return torch.softmax(log_alpha, dim=-1)  # (batch, n_states)


# ── Transformer Critic ─────────────────────────────────────────────────────────

class _TransformerCritic(nn.Module):
    """
    Single-token Transformer encoder with quantile output heads.
    Operates on the HRT-fused feature vector.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout, batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.norm = nn.LayerNorm(hidden_dim)

        # Three heads: point, lower quantile (0.05), upper quantile (0.95)
        self.head_point = nn.Linear(hidden_dim, 1)
        self.head_lower = nn.Linear(hidden_dim, 1)
        self.head_upper = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # x: (batch, input_dim)
        h = self.input_proj(x).unsqueeze(1)   # (batch, 1, hidden)
        h = self.encoder(h)                    # (batch, 1, hidden)
        h = self.norm(h.squeeze(1))            # (batch, hidden)
        return self.head_point(h), self.head_lower(h), self.head_upper(h)


# ── Unified QCPAC Module ───────────────────────────────────────────────────────

class _QCPACModule(nn.Module):
    def __init__(
        self,
        n_asset_features: int,
        n_macro_features: int,
        n_regimes: int,
        n_qubits: int,
        hidden_dim: int,
        n_heads: int,
        n_layers: int,
        hrt_dropout: float,
    ) -> None:
        super().__init__()
        self.hmm = _DifferentiableHMM(n_regimes, n_macro_features)
        self.hrt = HRTLayer(n_asset_features, n_regimes, n_qubits, dropout=hrt_dropout)
        self.critic = _TransformerCritic(
            n_qubits, hidden_dim, n_heads, n_layers
        )

    def forward(
        self,
        x_asset: torch.Tensor,     # (batch, n_asset_features)
        macro_seq: torch.Tensor,   # (batch, seq_len, n_macro_features)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        gamma = self.hmm(macro_seq)                         # (batch, n_regimes)
        hrt_feat = self.hrt(x_asset, gamma)                 # (batch, n_qubits)
        y_hat, y_lo, y_hi = self.critic(hrt_feat)
        return y_hat, y_lo, y_hi, gamma


# ── BaseForecaster Wrapper ─────────────────────────────────────────────────────

class QCPACForecaster(BaseForecaster):
    """
    End-to-end regime-adaptive forecaster combining:
    - Differentiable HMM (jointly trained)
    - HRT quantum feature layer
    - Transformer critic with quantile heads
    - Physics-Informed Loss

    Implements BaseForecaster so it slots into Aurora's ensemble.
    """

    name = "qcpac"
    supports_intervals = True
    requires_sequential = True

    def __init__(
        self,
        n_asset_features: int = 50,
        n_macro_features: int = 10,
        n_regimes: int = 4,
        n_qubits: int = 32,
        hidden_dim: int = 64,
        n_heads: int = 4,
        n_layers: int = 2,
        seq_len: int = 5,            # macro sequence length
        hrt_dropout: float = 0.1,
        lr: float = 1e-3,
        max_epochs: int = 50,
        batch_size: int = 64,
        patience: int = 10,
        lambda_arb: float = 0.3,
        lambda_elas: float = 0.2,
        device: str | None = None,
        horizon: int = 1,
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        self.n_asset_features = n_asset_features
        self.n_macro_features = n_macro_features
        self.n_regimes = n_regimes
        self.n_qubits = n_qubits
        self.seq_len = seq_len
        self.lr = lr
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.patience = patience
        self.lambda_arb = lambda_arb
        self.lambda_elas = lambda_elas

        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )

        self._module: _QCPACModule | None = None
        self._loss_fn = PhysicsInformedLoss(lambda_arb=lambda_arb, lambda_elas=lambda_elas)
        self._feature_cols: list[str] = []
        self._macro_cols: list[str] = []

    # ── BaseForecaster interface ───────────────────────────────────────────

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        macro_df: pd.DataFrame | None = None,
        depth_series: pd.Series | None = None,
        **kwargs: Any,
    ) -> "QCPACForecaster":
        """
        Joint end-to-end training.

        Parameters
        ----------
        X         : classical asset features (rows=dates, cols=features)
        y         : log-return target series
        macro_df  : macro features for the HMM (same index as X)
        depth_series : order book depth proxy; defaults to ones if absent
        """
        self._feature_cols = list(X.columns)

        # Build macro sequences from macro_df or use X if macro_df absent
        if macro_df is not None:
            self._macro_cols = list(macro_df.columns)
            macro_aligned = macro_df.reindex(X.index).ffill().bfill().fillna(0.0)
        else:
            self._macro_cols = self._feature_cols[:self.n_macro_features]
            macro_aligned = X[self._macro_cols]

        n_asset = len(self._feature_cols)
        n_macro = len(self._macro_cols)

        # Re-initialise module with correct dims
        self._module = _QCPACModule(
            n_asset_features=n_asset,
            n_macro_features=n_macro,
            n_regimes=self.n_regimes,
            n_qubits=self.n_qubits,
            hidden_dim=kwargs.get("hidden_dim", 64),
            n_heads=kwargs.get("n_heads", 4),
            n_layers=kwargs.get("n_layers", 2),
            hrt_dropout=0.1,
        ).to(self.device)

        # Align and drop NaN
        aligned = pd.concat([X, macro_aligned, y.rename("__target__")], axis=1).dropna()
        feat_vals = aligned[self._feature_cols].values.astype(np.float32)
        macro_vals = aligned[self._macro_cols].values.astype(np.float32)
        target_vals = aligned["__target__"].values.astype(np.float32)

        # Compute causal rolling bounds on target (no look-ahead)
        bounds = compute_rolling_bounds(target_vals, window=63, quantile=0.95).astype(np.float32)

        # Build macro sequences of length seq_len
        seq = self.seq_len
        N = len(feat_vals) - seq
        if N <= 0:
            raise ValueError(f"Too little data for seq_len={seq}")

        x_arr = feat_vals[seq:]
        macro_seq_arr = np.stack([macro_vals[i: i + seq] for i in range(N)])
        y_arr = target_vals[seq:]
        bound_arr = bounds[seq:]

        if depth_series is not None:
            depth_aligned = depth_series.reindex(aligned.index).ffill().fillna(1.0).values[seq:]
        else:
            depth_aligned = np.ones(N, dtype=np.float32)

        dataset = TensorDataset(
            torch.tensor(x_arr),
            torch.tensor(macro_seq_arr),
            torch.tensor(y_arr).unsqueeze(-1),
            torch.tensor(bound_arr).unsqueeze(-1),
            torch.tensor(depth_aligned.astype(np.float32)).unsqueeze(-1),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)

        optimizer = optim.AdamW(self._module.parameters(), lr=self.lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_loss = float("inf")
        no_improve = 0
        self._module.train()

        for epoch in range(self.max_epochs):
            epoch_loss = 0.0
            for x_b, mac_b, y_b, bound_b, depth_b in loader:
                x_b = x_b.to(self.device)
                mac_b = mac_b.to(self.device)
                y_b = y_b.to(self.device)
                bound_b = bound_b.to(self.device)
                depth_b = depth_b.to(self.device)

                optimizer.zero_grad()
                y_hat, _, _, gamma = self._module(x_b, mac_b)
                loss = self._loss_fn(y_hat, y_b, gamma, depth_b, bound_b)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._module.parameters(), 1.0)
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / max(len(loader), 1)
            scheduler.step(avg_loss)

            if avg_loss < best_loss - 1e-5:
                best_loss = avg_loss
                no_improve = 0
                self._best_state = {k: v.clone() for k, v in self._module.state_dict().items()}
            else:
                no_improve += 1

            if (epoch + 1) % 10 == 0:
                logger.info(f"[QCPAC] Epoch {epoch+1}/{self.max_epochs} loss={avg_loss:.6f}")

            if no_improve >= self.patience:
                logger.info(f"[QCPAC] Early stopping at epoch {epoch+1}")
                break

        if hasattr(self, "_best_state"):
            self._module.load_state_dict(self._best_state)

        self._module.eval()
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame, macro_df: pd.DataFrame | None = None, **kwargs: Any) -> np.ndarray:
        if not self._fitted or self._module is None:
            raise RuntimeError("Not fitted.")
        self._module.eval()

        feat, macro_seq = self._prepare_inference(X, macro_df)
        with torch.no_grad():
            y_hat, _, _, _ = self._module(
                feat.to(self.device), macro_seq.to(self.device)
            )
        return y_hat.cpu().numpy().flatten()

    def predict_interval(
        self, X: pd.DataFrame, macro_df: pd.DataFrame | None = None, alpha: float = 0.05, **kwargs: Any
    ) -> tuple[np.ndarray, np.ndarray]:
        if not self._fitted or self._module is None:
            raise RuntimeError("Not fitted.")
        self._module.eval()

        feat, macro_seq = self._prepare_inference(X, macro_df)
        with torch.no_grad():
            _, y_lo, y_hi = self._module(
                feat.to(self.device), macro_seq.to(self.device)
            )[:3]
        return y_lo.cpu().numpy().flatten(), y_hi.cpu().numpy().flatten()

    def get_regime_posterior(
        self, X: pd.DataFrame, macro_df: pd.DataFrame | None = None
    ) -> np.ndarray:
        """Return the filtered regime posterior for each row in X."""
        if not self._fitted or self._module is None:
            raise RuntimeError("Not fitted.")
        self._module.eval()
        feat, macro_seq = self._prepare_inference(X, macro_df)
        with torch.no_grad():
            _, _, _, gamma = self._module(feat.to(self.device), macro_seq.to(self.device))
        return gamma.cpu().numpy()

    # ── Persistence ────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self._module.state_dict() if self._module else None,
            "feature_cols": self._feature_cols,
            "macro_cols": self._macro_cols,
            "config": {
                "n_asset_features": self.n_asset_features,
                "n_macro_features": self.n_macro_features,
                "n_regimes": self.n_regimes,
                "n_qubits": self.n_qubits,
                "seq_len": self.seq_len,
            },
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "QCPACForecaster":
        ckpt = torch.load(path, map_location="cpu")
        cfg = ckpt["config"]
        model = cls(**cfg)
        model._feature_cols = ckpt["feature_cols"]
        model._macro_cols = ckpt["macro_cols"]
        if ckpt["state_dict"] is not None:
            n_asset = len(model._feature_cols)
            n_macro = len(model._macro_cols)
            model._module = _QCPACModule(
                n_asset, n_macro, cfg["n_regimes"], cfg["n_qubits"],
                hidden_dim=64, n_heads=4, n_layers=2, hrt_dropout=0.1,
            )
            model._module.load_state_dict(ckpt["state_dict"])
            model._module.eval()
            model._fitted = True
        return model

    # ── Helpers ────────────────────────────────────────────────────────────

    def _prepare_inference(
        self, X: pd.DataFrame, macro_df: pd.DataFrame | None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feat_cols = [c for c in self._feature_cols if c in X.columns]
        feat = X[feat_cols].ffill().bfill().fillna(0.0).values.astype(np.float32)

        if macro_df is not None:
            macro_cols = [c for c in self._macro_cols if c in macro_df.columns]
            macro = macro_df[macro_cols].reindex(X.index).ffill().bfill().fillna(0.0).values.astype(np.float32)
        else:
            macro_cols = [c for c in self._macro_cols if c in X.columns]
            macro = X[macro_cols].ffill().bfill().fillna(0.0).values.astype(np.float32)

        seq = self.seq_len
        N = len(feat)
        # Build padded macro sequences (repeat first row for initial steps)
        padded_macro = np.vstack([np.tile(macro[0], (seq, 1)), macro])
        macro_seq = np.stack([padded_macro[i: i + seq] for i in range(N)])

        return torch.tensor(feat), torch.tensor(macro_seq)
