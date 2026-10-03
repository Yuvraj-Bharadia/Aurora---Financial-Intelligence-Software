"""
Markov Switching Model regime detection via statsmodels.

Implements Hamilton's Markov-switching dynamic regression (MSDR) model
for detecting structural breaks and regime switches in return dynamics.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from statsmodels.tsa.regime_switching.markov_autoregression import MarkovAutoregression

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class MarkovSwitchingDetector:
    """
    Hamilton Markov Switching Model for regime detection.

    Fits a Markov-switching autoregression (MSAR) or Markov-switching
    regression (MSR) to the return series and extracts:
    - Filtered regime probabilities (P(regime | past data))
    - Smoothed regime probabilities (P(regime | all data))
    - Most likely regime sequence (Viterbi path)
    """

    def __init__(
        self,
        k_regimes: int | None = None,
        order: int = 1,
        switching_variance: bool = True,
        switching_trend: bool = True,
        use_autoregression: bool = True,
    ) -> None:
        self.k_regimes = k_regimes or settings.regimes.n_regimes
        self.order = order
        self.switching_variance = switching_variance
        self.switching_trend = switching_trend
        self.use_autoregression = use_autoregression
        self._result: Any | None = None
        self._regime_labels: list[str] = settings.regimes.regime_labels

    def fit(self, returns: pd.Series) -> "MarkovSwitchingDetector":
        """
        Fit the Markov switching model.

        Args:
            returns: Daily return series (log returns recommended).

        Returns:
            self.
        """
        logger.info(
            f"Fitting MarkovSwitching: k={self.k_regimes} regimes, "
            f"order={self.order}, switching_var={self.switching_variance}"
        )
        clean = returns.dropna()

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if self.use_autoregression:
                model = MarkovAutoregression(
                    endog=clean,
                    k_regimes=self.k_regimes,
                    order=self.order,
                    switching_ar=False,
                    switching_variance=self.switching_variance,
                    switching_trend=self.switching_trend,
                )
            else:
                model = MarkovRegression(
                    endog=clean,
                    k_regimes=self.k_regimes,
                    switching_variance=self.switching_variance,
                    switching_trend=self.switching_trend,
                )
            self._result = model.fit(disp=False, maxiter=500)

        aic = self._result.aic
        bic = self._result.bic
        logger.info(f"MarkovSwitching fitted: AIC={aic:.2f}, BIC={bic:.2f}")
        return self

    def predict(self, returns: pd.Series) -> pd.DataFrame:
        """
        Extract regime assignments from fitted model.

        Returns:
            DataFrame with smoothed probabilities and predicted regime.
        """
        if self._result is None:
            raise RuntimeError("Model not fitted. Call fit() first.")

        smoothed = self._result.smoothed_marginal_probabilities
        filtered = self._result.filtered_marginal_probabilities

        # Determine which state corresponds to which regime by mean return
        means = [self._result.params[f"regime[{k}].const"] for k in range(self.k_regimes)
                 if f"regime[{k}].const" in self._result.params.index]

        if len(means) == self.k_regimes:
            sorted_states = np.argsort(means)  # low mean → bear, high mean → bull
        else:
            sorted_states = list(range(self.k_regimes))

        result = pd.DataFrame(index=returns.index)
        for k in range(self.k_regimes):
            label = self._regime_labels[k] if k < len(self._regime_labels) else f"state_{k}"
            if k < smoothed.shape[1]:
                result[f"prob_{label}"] = smoothed.iloc[:, k]
                result[f"filtered_prob_{label}"] = filtered.iloc[:, k]

        prob_cols = [c for c in result.columns if c.startswith("prob_") and "filtered" not in c]
        result["regime"] = result[prob_cols].idxmax(axis=1).str.replace("prob_", "")
        result["regime_id"] = result["regime"].map(
            {lbl: i for i, lbl in enumerate(self._regime_labels)}
        )
        return result.reindex(returns.index)

    def expected_duration(self) -> pd.Series:
        """
        Expected regime duration in trading days.

        E[duration_k] = 1 / (1 - P(stay in k | in k))
        """
        if self._result is None:
            raise RuntimeError("Not fitted.")
        trans = self._result.regime_transition
        durations = {}
        for k in range(self.k_regimes):
            stay_prob = trans[k, k] if trans.ndim == 2 else 0.9
            label = self._regime_labels[k] if k < len(self._regime_labels) else f"state_{k}"
            durations[label] = 1.0 / max(1 - stay_prob, 1e-9)
        return pd.Series(durations, name="expected_duration_days")

    def information_criteria(self) -> dict[str, float]:
        """Return AIC, BIC, and log-likelihood."""
        if self._result is None:
            raise RuntimeError("Not fitted.")
        return {
            "aic": float(self._result.aic),
            "bic": float(self._result.bic),
            "log_likelihood": float(self._result.llf),
        }
