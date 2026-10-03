"""
ARIMA / SARIMA forecasting model wrapper.

Uses pmdarima (auto_arima) for automatic order selection via AIC/BIC,
with fallback to statsmodels ARIMA for reproducibility.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class ARIMAForecaster(BaseForecaster):
    """
    Auto-ARIMA forecasting model.

    Automatically selects (p, d, q) order parameters via AIC minimisation
    using pmdarima's stepwise search. Supports seasonal ARIMA (SARIMA)
    with configurable periodicity.
    """

    name = "arima"
    supports_intervals = True

    def __init__(
        self,
        horizon: int = 1,
        seasonal: bool = False,
        m: int = 1,
        max_p: int = 5,
        max_q: int = 5,
        max_d: int = 2,
        information_criterion: str = "aic",
        stepwise: bool = True,
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        self.seasonal = seasonal
        self.m = m
        self.max_p = max_p
        self.max_q = max_q
        self.max_d = max_d
        self.information_criterion = information_criterion
        self.stepwise = stepwise
        self._model: Any = None

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "ARIMAForecaster":
        """
        Fit auto-ARIMA on the target series y.

        Note: X (exogenous features) are included as regressors when provided.
        """
        try:
            import pmdarima as pm
        except ImportError:
            # Fallback to statsmodels ARIMA(2,1,2) if pmdarima not installed
            return self._fit_statsmodels(y)

        exog = X.values if not X.empty else None

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model = pm.auto_arima(
                y.values,
                exogenous=exog,
                seasonal=self.seasonal,
                m=self.m,
                max_p=self.max_p,
                max_q=self.max_q,
                max_d=self.max_d,
                information_criterion=self.information_criterion,
                stepwise=self.stepwise,
                error_action="ignore",
                suppress_warnings=True,
            )

        order = self._model.order
        logger.info(f"[ARIMA] Selected order={order} | AIC={self._model.aic():.2f}")
        self._fitted = True
        return self

    def _fit_statsmodels(self, y: pd.Series) -> "ARIMAForecaster":
        """Fallback ARIMA(2,1,2) via statsmodels."""
        from statsmodels.tsa.arima.model import ARIMA as smARIMA
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            result = smARIMA(y.values, order=(2, 1, 2)).fit()
        self._model = result
        self._use_statsmodels = True
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Produce n_periods-ahead point forecasts."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")

        n = len(X) if not X.empty else self.horizon
        exog = X.values[:n] if not X.empty else None

        if getattr(self, "_use_statsmodels", False):
            fc = self._model.forecast(steps=n)
        else:
            fc, _ = self._model.predict(n_periods=n, exogenous=exog, return_conf_int=True)

        return np.array(fc).flatten()

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return confidence intervals from ARIMA in-distribution uncertainty."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        n = max(len(X), self.horizon)
        exog = X.values[:n] if not X.empty else None

        if getattr(self, "_use_statsmodels", False):
            fc_result = self._model.get_forecast(steps=n)
            ci = fc_result.conf_int(alpha=alpha)
            return ci.iloc[:, 0].values, ci.iloc[:, 1].values

        _, conf_int = self._model.predict(
            n_periods=n, exogenous=exog, return_conf_int=True, alpha=alpha
        )
        return conf_int[:, 0], conf_int[:, 1]

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "ARIMAForecaster":
        return joblib.load(path)
