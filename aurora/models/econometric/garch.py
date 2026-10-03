"""
GARCH / EGARCH volatility forecasting models.

Uses the `arch` library for GARCH(p,q), EGARCH, and GJR-GARCH.
These models forecast conditional variance — critical for volatility
regime detection and risk-weighted position sizing.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd

from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class GARCHForecaster(BaseForecaster):
    """
    GARCH family volatility forecasting model.

    Supports GARCH, EGARCH, GJR-GARCH, and FIGARCH specifications.
    The model is fitted on log returns and forecasts conditional volatility
    (annualized), which feeds into regime detection and position sizing.
    """

    name = "garch"
    supports_intervals = True

    def __init__(
        self,
        horizon: int = 1,
        p: int = 1,
        q: int = 1,
        vol_model: Literal["Garch", "EGARCH", "GJR-GARCH", "FIGARCH"] = "Garch",
        mean_model: Literal["Constant", "Zero", "AR"] = "Constant",
        dist: Literal["normal", "t", "skewt", "ged"] = "t",
        annualize: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        self.p = p
        self.q = q
        self.vol_model = vol_model
        self.mean_model = mean_model
        self.dist = dist
        self.annualize = annualize
        self._result: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "GARCHForecaster":
        """
        Fit GARCH on the return series y.

        The return series should be scaled to percentage returns (×100)
        or log returns — the arch library expects mean-zero-ish input.
        """
        try:
            from arch import arch_model
        except ImportError as e:
            raise ImportError("arch library required: pip install arch") from e

        rescaled = y * 100 if y.abs().max() < 1 else y  # arch library prefers 0–5% scale

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = arch_model(
                rescaled.dropna(),
                p=self.p,
                q=self.q,
                vol=self.vol_model,
                mean=self.mean_model,
                dist=self.dist,
            )
            self._result = model.fit(disp="off", show_warning=False, options={"maxiter": 500})

        aic = self._result.aic
        bic = self._result.bic
        logger.info(f"[GARCH] {self.vol_model}({self.p},{self.q}) | AIC={aic:.2f} BIC={bic:.2f}")
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Forecast conditional volatility for the next `horizon` periods.

        Returns annualized volatility (as a fraction, not percentage).
        """
        if not self._fitted:
            raise RuntimeError("Not fitted.")

        forecasts = self._result.forecast(horizon=self.horizon, reindex=False)
        variance_forecast = forecasts.variance.values[-1]  # shape (horizon,)
        # Convert to std dev in original return units (undo ×100 rescaling)
        vol = np.sqrt(variance_forecast) / 100

        if self.annualize:
            vol = vol * np.sqrt(252)

        return vol  # shape (horizon,)

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Bootstrap-based volatility forecast intervals."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        forecasts = self._result.forecast(
            horizon=self.horizon,
            reindex=False,
            method="simulation",
            simulations=1000,
        )
        sim_var = forecasts.simulations.variance[-1]  # shape (simulations, horizon)
        q_lo = np.percentile(sim_var, alpha / 2 * 100, axis=0)
        q_hi = np.percentile(sim_var, (1 - alpha / 2) * 100, axis=0)
        scale = np.sqrt(252) / 100 if self.annualize else 1 / 100
        return np.sqrt(q_lo) * scale, np.sqrt(q_hi) * scale

    def conditional_volatility(self) -> pd.Series:
        """Return the in-sample conditional volatility series (annualized)."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        cond_var = self._result.conditional_volatility
        vol = cond_var / 100
        if self.annualize:
            vol = vol * np.sqrt(252)
        return vol

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "GARCHForecaster":
        return joblib.load(path)
