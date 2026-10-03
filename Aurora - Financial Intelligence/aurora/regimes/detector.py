"""
Unified regime detection interface.

Combines HMM and Markov Switching outputs into a single regime signal.

Aurora v2 additions:
  - OnlineHMMFilter: filtered (causal) posterior, never smoothed (Layer 1)
  - NoTradeGate: converts state vector → risk mode (Layer 1)
  - Both are attached automatically after fitting when use_online_filter=True
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import joblib
import numpy as np
import pandas as pd

from aurora.configs.config import settings
from aurora.regimes.hmm import GaussianHMMDetector
from aurora.regimes.markov_switching import MarkovSwitchingDetector
from aurora.regimes.online_controller import NoTradeGate, OnlineHMMFilter
from aurora.utils.logging import get_logger

logger = get_logger(__name__)

# Features selected for regime detection — interpretable and regime-relevant
_DEFAULT_REGIME_FEATURES = [
    "log_ret_1d",
    "realized_vol_20d",
    "realized_vol_60d",
    "skew_60d",
    "kurt_60d",
    "momentum_20d",
    "hurst_60d",
    "vix",
    "yield_curve_10y2y",
    "ig_spread",
]


class RegimeDetector:
    """
    Ensemble regime detector: combines HMM and Markov Switching signals.

    Aurora v2: use_online_filter=True (default) runs the causal forward-pass
    HMM filter instead of the smoothed posterior, and attaches a NoTradeGate
    that enriches the output DataFrame with transition_risk, data_freshness,
    liquidity_score, and risk_mode columns.

    The detector exposes:
    - fit(features, returns): trains both sub-models.
    - predict(features, returns): returns a unified regime DataFrame.
    - predict_online(features): returns list[RegimeStateVector] with full state.
    - save / load: persist fitted state to disk.
    """

    def __init__(
        self,
        method: Literal["hmm", "markov_switching", "ensemble"] = "ensemble",
        n_regimes: int | None = None,
        regime_features: list[str] | None = None,
        min_confidence: float | None = None,
        use_online_filter: bool = True,
    ) -> None:
        self.method = method
        self.n_regimes = n_regimes or settings.regimes.n_regimes
        self._regime_features = regime_features or _DEFAULT_REGIME_FEATURES
        self._min_confidence = min_confidence or settings.regimes.confidence_threshold
        self.use_online_filter = use_online_filter

        self._hmm = GaussianHMMDetector(n_regimes=self.n_regimes)
        self._ms = MarkovSwitchingDetector(k_regimes=self.n_regimes)
        self._online_filter: OnlineHMMFilter | None = None
        self._no_trade_gate = NoTradeGate()
        self._fitted = False

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
    ) -> "RegimeDetector":
        """
        Fit the regime detector on historical features and returns.

        Args:
            features: Feature matrix from FeaturePipeline.
            returns: Log return series aligned to features.

        Returns:
            self.
        """
        # Select the subset of features used for regime detection
        feat_cols = [c for c in self._regime_features if c in features.columns]
        if len(feat_cols) < 2:
            # Fallback to returns + volatility if feature names don't match
            feat_cols = list(features.columns[:min(10, len(features.columns))])
            logger.warning(f"Using fallback regime features: {feat_cols[:5]}...")

        regime_feats = features[feat_cols].ffill().bfill().dropna()
        ret_aligned = returns.reindex(regime_feats.index).dropna()

        if self.method in ("hmm", "ensemble"):
            logger.info("Fitting Gaussian HMM regime detector")
            self._hmm.fit(regime_feats)

        if self.method in ("markov_switching", "ensemble"):
            logger.info("Fitting Markov Switching regime detector")
            self._ms.fit(ret_aligned)

        self._feat_cols = feat_cols
        self._fitted = True

        # Layer 1: attach OnlineHMMFilter after HMM is fitted
        if self.use_online_filter and self.method in ("hmm", "ensemble"):
            self._online_filter = OnlineHMMFilter(
                self._hmm, regime_labels=self._hmm.regime_labels
            )
            logger.info("OnlineHMMFilter attached (causal filtered posterior)")

        logger.info("RegimeDetector fitted successfully")
        return self

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
    ) -> pd.DataFrame:
        """
        Assign regime labels to the full time series.

        Returns:
            DataFrame with columns:
            - regime: Primary regime label (string)
            - regime_id: Integer regime index
            - regime_confidence: Maximum posterior probability
            - regime_source: Which model drove the final call
        """
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        feat_cols = [c for c in self._feat_cols if c in features.columns]
        regime_feats = features[feat_cols].ffill().bfill()

        if self.method == "hmm":
            return self._hmm_predict(regime_feats)
        elif self.method == "markov_switching":
            return self._ms_predict(returns)
        else:
            return self._ensemble_predict(regime_feats, returns)

    def predict_online(
        self,
        features: pd.DataFrame,
        freshness_series: pd.Series | None = None,
        liquidity_series: pd.Series | None = None,
    ) -> pd.DataFrame:
        """
        Layer 1: causal filtered posterior with full RegimeStateVector richness.

        Returns a DataFrame with columns: regime, regime_id, regime_confidence,
        transition_risk, state_uncertainty, data_freshness, liquidity_score,
        risk_mode, prob_<label> for each regime, size_multiplier.

        Falls back to standard predict() if OnlineHMMFilter is not available.
        """
        if not self._fitted:
            raise RuntimeError("Not fitted. Call fit() first.")

        if self._online_filter is None:
            logger.warning("[RegimeDetector] OnlineHMMFilter not available — using standard predict()")
            return self.predict(features, pd.Series(dtype=float, name="ret"))

        feat_cols = [c for c in self._feat_cols if c in features.columns]
        regime_feats = features[feat_cols].ffill().bfill()

        state_vectors = self._online_filter.filter_sequence(
            regime_feats, freshness_series, liquidity_series
        )
        state_vectors = self._no_trade_gate.evaluate_series(state_vectors)

        df = self._online_filter.to_dataframe(state_vectors)
        df["size_multiplier"] = [
            self._no_trade_gate.size_multiplier(sv.risk_mode) for sv in state_vectors
        ]
        return df

    def _hmm_predict(self, feats: pd.DataFrame) -> pd.DataFrame:
        df = self._hmm.predict(feats)
        df["regime_source"] = "hmm"
        df["regime_confidence"] = self._hmm.regime_confidence(df)
        return df

    def _ms_predict(self, returns: pd.Series) -> pd.DataFrame:
        df = self._ms.predict(returns)
        df["regime_source"] = "markov_switching"
        # Confidence = max smoothed probability
        prob_cols = [c for c in df.columns if c.startswith("prob_") and "filtered" not in c]
        df["regime_confidence"] = df[prob_cols].max(axis=1)
        return df

    def _ensemble_predict(
        self, feats: pd.DataFrame, returns: pd.Series
    ) -> pd.DataFrame:
        """
        Ensemble: use HMM as primary, fall back to Markov Switching when
        HMM confidence is below threshold.
        """
        hmm_df = self._hmm_predict(feats)
        ms_df = self._ms_predict(returns)

        result = hmm_df.copy()
        low_conf = hmm_df["regime_confidence"] < self._min_confidence
        if low_conf.any():
            # Overwrite low-confidence HMM predictions with MS predictions
            ms_aligned = ms_df.reindex(hmm_df.index)
            result.loc[low_conf, "regime"] = ms_aligned.loc[low_conf, "regime"]
            result.loc[low_conf, "regime_source"] = "markov_switching"
            result.loc[low_conf, "regime_confidence"] = ms_aligned.loc[
                low_conf, "regime_confidence"
            ]

        return result

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        """Serialize detector state to disk via joblib."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"RegimeDetector saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RegimeDetector":
        """Load a previously saved detector."""
        path = Path(path)
        detector: "RegimeDetector" = joblib.load(path)
        logger.info(f"RegimeDetector loaded from {path}")
        return detector

    # ── Analytics ──────────────────────────────────────────────────────────────

    def regime_summary(self, regimes: pd.DataFrame) -> pd.DataFrame:
        """
        Summarise regime frequencies and average durations.

        Args:
            regimes: Output of predict().

        Returns:
            Summary DataFrame indexed by regime name.
        """
        df = regimes[["regime"]].copy()
        df["run"] = (df["regime"] != df["regime"].shift()).cumsum()
        grp = df.groupby(["regime", "run"]).size().reset_index(name="duration")
        summary = grp.groupby("regime")["duration"].agg(
            count="count", mean_duration="mean", total_days="sum"
        )
        summary["pct_time"] = summary["total_days"] / len(df)
        return summary.round(2)

    def transition_matrix(self) -> pd.DataFrame:
        """Empirical transition matrix from the HMM."""
        return self._hmm.transition_matrix()
