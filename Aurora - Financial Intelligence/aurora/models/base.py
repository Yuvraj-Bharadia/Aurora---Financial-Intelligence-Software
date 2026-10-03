"""
Abstract base class for all Aurora forecasting models.

Every model—econometric, ML, or deep learning—implements this interface.
This ensures the ensemble layer can treat all models uniformly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


class BaseForecaster(ABC):
    """
    Abstract forecasting model interface.

    Subclasses implement:
    - fit(X, y): train on feature matrix X and target y.
    - predict(X): return point forecasts.
    - predict_interval(X, alpha): return (lower, upper) prediction intervals.
    - save / load: persist model state.
    """

    name: str = "base"
    supports_intervals: bool = False  # whether predict_interval is implemented
    requires_sequential: bool = False  # True for LSTM/Transformer that need ordered data

    def __init__(self, horizon: int = 1, **kwargs: Any) -> None:
        self.horizon = horizon
        self._fitted = False
        self._config: dict[str, Any] = kwargs

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "BaseForecaster":
        """Train the model."""
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate point forecasts for rows in X."""
        ...

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Return (lower, upper) prediction interval bounds at confidence (1-alpha).

        Override in subclasses that support uncertainty estimation.
        """
        preds = self.predict(X)
        # Naive: symmetric ±1.96 × residual std (placeholder)
        width = np.zeros_like(preds)
        return preds - width, preds + width

    def score(self, X: pd.DataFrame, y: pd.Series) -> dict[str, float]:
        """Compute forecast accuracy metrics."""
        from aurora.evaluation.metrics import forecast_metrics
        preds = self.predict(X)
        return forecast_metrics(y.values[: len(preds)], preds)

    @abstractmethod
    def save(self, path: str | Path) -> None:
        """Persist the fitted model."""
        ...

    @classmethod
    @abstractmethod
    def load(cls, path: str | Path) -> "BaseForecaster":
        """Load a previously saved model."""
        ...

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(horizon={self.horizon}, fitted={self._fitted})"
