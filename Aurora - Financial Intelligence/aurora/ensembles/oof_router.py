"""
Layer 4 — Out-of-Fold Adaptive Router.

Replaces the post-hoc RMSE softmax weighting in RegimeAdaptiveEnsemble with
a properly trained gating network that:

1. Trains every specialist model in walk-forward folds.
2. Stores ONLY out-of-fold (OOF) forecasts — the model never scores itself.
3. Trains a gating MLP on [OOF forecasts, regime posterior, confidence,
   freshness, transition risk] to predict which model will be best.
4. Updates weights online with bounded turnover (max Δw per step) and
   concentration constraints (max weight per model).

The gating network inputs per row:
  - OOF predictions from each specialist (n_models scalars)
  - Filtered posterior (n_regimes scalars)
  - State confidence (1 scalar)
  - Transition risk (1 scalar)
  - Data freshness (1 scalar)
  Total: n_models + n_regimes + 3

Output: softmax over model weights.

Usage::

    store = OOFStore(model_names=["arima", "lstm", "xgboost", "transformer", "qcpac"])
    store.add_fold(fold_idx=0, model_preds={"arima": arr, ...}, actuals=arr, regime_states=list)
    router = OOFRouter(store)
    router.fit()
    weights = router.predict_weights(current_features, current_regime_state)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from aurora.regimes.online_controller import RegimeStateVector
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OOFFold:
    """Stores one walk-forward fold's OOF data."""
    fold_idx: int
    model_preds: dict[str, np.ndarray]   # model_name → predictions (n_test,)
    actuals: np.ndarray                   # (n_test,)
    state_vectors: list[RegimeStateVector] | None = None
    regime_labels: list[str] = field(default_factory=list)


class OOFStore:
    """
    Accumulates out-of-fold forecasts across walk-forward folds.
    Each specialist model must store only predictions it made on data
    it was never trained on.
    """

    def __init__(self, model_names: list[str]) -> None:
        self.model_names = model_names
        self._folds: list[OOFFold] = []

    def add_fold(
        self,
        fold_idx: int,
        model_preds: dict[str, np.ndarray],
        actuals: np.ndarray,
        state_vectors: list[RegimeStateVector] | None = None,
        regime_labels: list[str] | None = None,
    ) -> None:
        self._folds.append(OOFFold(
            fold_idx=fold_idx,
            model_preds=model_preds,
            actuals=actuals,
            state_vectors=state_vectors,
            regime_labels=regime_labels or [],
        ))

    def to_training_arrays(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Assemble all folds into (X_gate, y_best_model, actuals) arrays.

        X_gate : (N, n_models + n_regime_features) — gating network inputs
        y_best  : (N,) int — index of the model with lowest |error| at each step
        actuals : (N,) float — true returns
        """
        X_rows, y_rows, actual_rows = [], [], []

        for fold in self._folds:
            n = len(fold.actuals)
            if n == 0:
                continue

            # Stack model predictions (N, n_models)
            pred_mat = np.stack(
                [fold.model_preds.get(m, np.zeros(n)) for m in self.model_names],
                axis=1,
            )

            # Best model at each step (lowest absolute error)
            errors = np.abs(pred_mat - fold.actuals[:, None])
            best_model = np.argmin(errors, axis=1)

            # Regime features from state vectors
            if fold.state_vectors and len(fold.state_vectors) == n:
                regime_feat = np.stack([
                    np.concatenate([
                        sv.filtered_posterior,
                        [sv.confidence, sv.transition_risk, sv.data_freshness],
                    ])
                    for sv in fold.state_vectors
                ])
            else:
                n_reg = 4
                regime_feat = np.zeros((n, n_reg + 3))

            X_gate = np.concatenate([pred_mat, regime_feat], axis=1)
            X_rows.append(X_gate)
            y_rows.append(best_model)
            actual_rows.append(fold.actuals)

        if not X_rows:
            raise RuntimeError("OOFStore is empty — add fold data before calling to_training_arrays()")

        return (
            np.vstack(X_rows).astype(np.float32),
            np.concatenate(y_rows).astype(np.int64),
            np.concatenate(actual_rows).astype(np.float32),
        )

    @property
    def n_folds(self) -> int:
        return len(self._folds)


class _GatingMLP(nn.Module):
    """Small classification network predicting the best model index."""

    def __init__(self, input_dim: int, n_models: int, hidden: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.LayerNorm(hidden),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden, hidden // 2),
            nn.GELU(),
            nn.Linear(hidden // 2, n_models),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)   # logits; softmax applied externally


class OOFRouter:
    """
    Out-of-fold gating network that learns which model to trust per regime.

    Parameters
    ----------
    oof_store        : filled OOFStore
    max_weight       : maximum weight any single model can receive (concentration limit)
    max_weight_delta : maximum change in any single model's weight per step (turnover bound)
    hidden_dim       : gating MLP hidden size
    lr, epochs       : training hyperparameters
    """

    def __init__(
        self,
        oof_store: OOFStore,
        max_weight: float = 0.60,
        max_weight_delta: float = 0.20,
        hidden_dim: int = 64,
        lr: float = 1e-3,
        epochs: int = 100,
        device: str = "cpu",
    ) -> None:
        self._store = oof_store
        self.model_names = oof_store.model_names
        self.n_models = len(self.model_names)
        self.max_weight = max_weight
        self.max_weight_delta = max_weight_delta
        self._device = torch.device(device)
        self._lr = lr
        self._epochs = epochs
        self._hidden = hidden_dim
        self._net: _GatingMLP | None = None
        self._prev_weights: np.ndarray = np.ones(self.n_models) / self.n_models
        self._fitted = False

    def fit(self) -> "OOFRouter":
        """Train the gating network on OOF data."""
        X, y_best, _ = self._store.to_training_arrays()
        input_dim = X.shape[1]

        self._net = _GatingMLP(input_dim, self.n_models, self._hidden).to(self._device)
        optimizer = optim.Adam(self._net.parameters(), lr=self._lr)
        loss_fn = nn.CrossEntropyLoss()

        X_t = torch.tensor(X).to(self._device)
        y_t = torch.tensor(y_best).to(self._device)

        self._net.train()
        for epoch in range(self._epochs):
            optimizer.zero_grad()
            logits = self._net(X_t)
            loss = loss_fn(logits, y_t)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % 25 == 0:
                acc = (logits.argmax(1) == y_t).float().mean().item()
                logger.info(f"[OOFRouter] Epoch {epoch+1}/{self._epochs} loss={loss.item():.4f} acc={acc:.3f}")

        self._net.eval()
        self._fitted = True
        return self

    def predict_weights(
        self,
        model_preds: dict[str, float],
        state_vector: RegimeStateVector | None = None,
        n_regimes: int = 4,
    ) -> dict[str, float]:
        """
        Predict ensemble weights for the current time step.

        Parameters
        ----------
        model_preds  : {model_name: current_point_forecast}
        state_vector : current RegimeStateVector from OnlineHMMFilter
        n_regimes    : number of regimes (for dimensionality)

        Returns
        -------
        Dict mapping model_name → weight (sums to 1, respects constraints)
        """
        if not self._fitted or self._net is None:
            logger.warning("[OOFRouter] Not fitted — returning equal weights")
            return {m: 1.0 / self.n_models for m in self.model_names}

        pred_vec = np.array([model_preds.get(m, 0.0) for m in self.model_names], dtype=np.float32)

        if state_vector is not None:
            posterior = state_vector.filtered_posterior
            if len(posterior) < n_regimes:
                posterior = np.pad(posterior, (0, n_regimes - len(posterior)))
            regime_feat = np.concatenate([
                posterior[:n_regimes],
                [state_vector.confidence, state_vector.transition_risk, state_vector.data_freshness],
            ]).astype(np.float32)
        else:
            regime_feat = np.zeros(n_regimes + 3, dtype=np.float32)

        x = torch.tensor(np.concatenate([pred_vec, regime_feat])).unsqueeze(0).to(self._device)
        with torch.no_grad():
            logits = self._net(x)
            raw_weights = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        # Apply concentration constraint
        clipped = np.clip(raw_weights, 0, self.max_weight)
        clipped /= clipped.sum() + 1e-9

        # Apply turnover constraint: limit change per step
        delta = clipped - self._prev_weights
        clipped = self._prev_weights + np.clip(delta, -self.max_weight_delta, self.max_weight_delta)
        clipped = np.maximum(clipped, 0)
        clipped /= clipped.sum() + 1e-9

        self._prev_weights = clipped.copy()
        return dict(zip(self.model_names, clipped.tolist()))

    def weight_history(self) -> pd.DataFrame:
        """Return weight statistics from the OOF evaluation."""
        if not self._fitted:
            return pd.DataFrame()
        X, y_best, _ = self._store.to_training_arrays()
        x_t = torch.tensor(X).to(self._device)
        with torch.no_grad():
            weights = torch.softmax(self._net(x_t), dim=-1).cpu().numpy()
        df = pd.DataFrame(weights, columns=self.model_names)
        df["best_model"] = [self.model_names[i] for i in y_best]
        return df

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "OOFRouter":
        return joblib.load(path)
