"""
High-level forecasting orchestrator for Aurora.

Wraps the RegimeAdaptiveEnsemble and individual models behind a clean
interface used by the API, notebooks, and batch pipelines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.configs.config import settings
from aurora.ensembles.regime_adaptive import RegimeAdaptiveEnsemble
from aurora.models.base import BaseForecaster
from aurora.regimes.detector import RegimeDetector
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class AuroraForecaster:
    """
    Top-level forecasting entry point.

    Orchestrates:
    1. Regime detection on the latest features
    2. Ensemble prediction conditioned on the current regime
    3. Multi-horizon forecast generation
    4. Uncertainty interval estimation

    Usage::

        forecaster = AuroraForecaster.load("outputs/models/aurora_full")
        result = forecaster.forecast(features_df, horizons=[1, 5, 21])
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        ensemble: RegimeAdaptiveEnsemble,
        horizons: list[int] | None = None,
    ) -> None:
        self.regime_detector = regime_detector
        self.ensemble = ensemble
        self.horizons = horizons or settings.models.forecast_horizons

    def forecast(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        horizon: int | None = None,
        alpha: float = 0.10,
    ) -> pd.DataFrame:
        """
        Generate regime-adaptive forecasts.

        Args:
            features: Feature matrix (most recent rows).
            returns: Aligned return series for regime detection.
            horizon: Steps-ahead (uses default if None).
            alpha: Significance level for prediction intervals.

        Returns:
            DataFrame with forecast, lower/upper bounds, regime, confidence.
        """
        horizon = horizon or settings.models.default_horizon
        regimes = self.regime_detector.predict(features, returns)
        forecasts = self.ensemble.predict(features, regimes)
        return forecasts

    def forecast_multi_horizon(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
    ) -> dict[int, pd.DataFrame]:
        """Generate forecasts at every horizon in self.horizons."""
        results: dict[int, pd.DataFrame] = {}
        regimes = self.regime_detector.predict(features, returns)

        for h in self.horizons:
            fc = self.ensemble.predict(features, regimes)
            results[h] = fc
            logger.debug(f"Multi-horizon: h={h} forecasts generated")

        return results

    def save(self, directory: str | Path) -> None:
        """Save all components to a directory."""
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        self.regime_detector.save(directory / "regime_detector.pkl")
        self.ensemble.save(directory / "ensemble.pkl")
        logger.info(f"AuroraForecaster saved to {directory}")

    @classmethod
    def load(cls, directory: str | Path) -> "AuroraForecaster":
        """Load a saved AuroraForecaster."""
        directory = Path(directory)
        regime_detector = RegimeDetector.load(directory / "regime_detector.pkl")
        ensemble = RegimeAdaptiveEnsemble.load(directory / "ensemble.pkl")
        return cls(regime_detector=regime_detector, ensemble=ensemble)
