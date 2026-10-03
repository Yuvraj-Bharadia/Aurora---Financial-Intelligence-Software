"""
Layer 0 — Causal Data Availability Graph.

Every feature carries four timestamps describing when it was truly available
to a live trader. The CausalGuard enforces these at every walk-forward
boundary, automatically preventing any form of look-ahead bias.

Usage::

    guard = CausalGuard.default()
    clean = guard.filter(features_df, prediction_time=pd.Timestamp("2020-03-15 16:00:00", tz="UTC"))
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
import pandas as pd
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureMetadata:
    """
    Availability envelope for one feature column.

    Attributes
    ----------
    name : feature column name
    event_time_offset : timedelta from bar close to the moment being described
        (e.g. 0 for end-of-day price, -45d for a GDP reading that describes
         the quarter ending 45 days ago)
    release_lag : how many **calendar** days after event_time the value
        is first published by the provider (e.g. 1 for price data, 14 for
        initial jobless claims, 30 for PCE, 90 for GDP advance)
    revision_lag : days until the *final* revised value is available
        (0 = never revised; 90 = GDP is revised three months later)
    intraday_release_hour : UTC hour when the release drops on release day
        (e.g. 8 for most US economic releases at 08:30 ET = 13:30 UTC)
    source : data provider tag for audit trail
    """

    name: str
    release_lag: int = 1            # calendar days after event
    revision_lag: int = 0           # 0 = never revised
    intraday_release_hour: int = 16 # 16:00 UTC ≈ US market close
    event_time_offset: int = 0      # days: negative means event was N days before bar
    source: str = "unknown"

    def first_available(self, bar_date: pd.Timestamp) -> pd.Timestamp:
        """
        Return the earliest UTC timestamp at which this feature value
        for *bar_date* could be used in a live prediction.
        """
        event_date = bar_date + pd.Timedelta(days=self.event_time_offset)
        release_date = event_date + pd.Timedelta(days=self.release_lag)
        return release_date.normalize() + pd.Timedelta(hours=self.intraday_release_hour)

    def is_available_at(self, bar_date: pd.Timestamp, prediction_time: pd.Timestamp) -> bool:
        """True if this feature from bar_date is usable at prediction_time."""
        return self.first_available(bar_date) <= prediction_time


class DataAvailabilityGraph:
    """
    Registry mapping feature column names to their FeatureMetadata.

    Columns not in the registry are assumed to follow the default
    (T+1 open-of-day availability, i.e. usable the next morning).
    """

    def __init__(self) -> None:
        self._registry: dict[str, FeatureMetadata] = {}

    def register(self, meta: FeatureMetadata) -> "DataAvailabilityGraph":
        self._registry[meta.name] = meta
        return self

    def register_many(self, metas: list[FeatureMetadata]) -> "DataAvailabilityGraph":
        for m in metas:
            self._registry[m.name] = m
        return self

    def get(self, name: str) -> FeatureMetadata:
        return self._registry.get(name, FeatureMetadata(name=name, release_lag=1))

    def available_at(self, name: str, bar_date: pd.Timestamp, prediction_time: pd.Timestamp) -> bool:
        return self.get(name).is_available_at(bar_date, prediction_time)

    # ── Pre-built registries ──────────────────────────────────────────────────

    @classmethod
    def default(cls) -> "DataAvailabilityGraph":
        """
        Pre-built registry covering all feature groups produced by Aurora's
        FeaturePipeline: price/technical (T+1), statistical (T+1),
        and FRED macro (variable lags).
        """
        dag = cls()

        # ── Price / technical / statistical features ───────────────────────
        # All derived from end-of-day OHLCV; available at next-day open (T+1 9:30 ET)
        price_tech_features = [
            "open", "high", "low", "close", "volume", "adj_close",
            "log_ret_1d", "log_ret_5d", "log_ret_20d", "log_ret_60d",
            "realized_vol_5d", "realized_vol_20d", "realized_vol_60d", "realized_vol_252d",
            "parkinson_vol_20d", "garman_klass_vol_20d",
            "roll_skew_20d", "roll_skew_60d", "roll_kurt_20d", "roll_kurt_60d",
            "max_drawdown_20d", "max_drawdown_60d",
            "momentum_5d", "momentum_10d", "momentum_20d", "momentum_60d",
            "sma_5", "sma_10", "sma_20", "sma_50", "sma_200",
            "ema_12", "ema_26",
            "macd", "macd_signal", "macd_hist",
            "bb_upper", "bb_lower", "bb_mid", "bb_pct_b", "bb_width",
            "rsi_14", "stoch_k", "stoch_d", "williams_r", "cci_20",
            "obv", "vwap", "atr_14", "adx_14", "di_plus", "di_minus",
            "hurst_60d", "hurst_100d",
            "sample_entropy_20d",
            "autocorr_1", "autocorr_5", "autocorr_10",
            "adf_pval_20d", "adf_pval_60d",
            "beta_60d", "rolling_corr_60d",
        ]
        for feat in price_tech_features:
            dag.register(FeatureMetadata(
                name=feat, release_lag=1, revision_lag=0,
                intraday_release_hour=14,  # 09:30 ET = 14:30 UTC
                source="exchange"
            ))

        # ── FRED macro series ──────────────────────────────────────────────
        # VIX: published same-day by CBOE at close; available T+1
        dag.register(FeatureMetadata("vix", release_lag=1, intraday_release_hour=14, source="cboe"))
        dag.register(FeatureMetadata("vix_regime", release_lag=1, intraday_release_hour=14, source="cboe"))
        dag.register(FeatureMetadata("vix_zscore", release_lag=1, intraday_release_hour=14, source="cboe"))

        # Treasury yields: published next business day by Federal Reserve
        for tenor in ["gs2", "gs10", "yield_curve_10y2y", "yield_curve_slope",
                       "real_rate_10y", "real_rate_2y"]:
            dag.register(FeatureMetadata(tenor, release_lag=1, intraday_release_hour=16, source="fred"))

        # Fed funds rate: updated daily, available T+1
        dag.register(FeatureMetadata("fedfunds", release_lag=1, intraday_release_hour=16, source="fred"))

        # WTI crude oil: daily, available T+1
        dag.register(FeatureMetadata("dcoilwtico", release_lag=1, intraday_release_hour=16, source="fred"))

        # USD broad index: weekly, typically available ~4 days after reference week
        dag.register(FeatureMetadata("dtwexbgs", release_lag=4, intraday_release_hour=16, source="fred"))

        # Unemployment rate: monthly, published ~4 weeks after reference month
        # event_time_offset=-30 means the observation refers to 30 days prior
        dag.register(FeatureMetadata(
            "unrate", release_lag=30, revision_lag=30,
            event_time_offset=-30, intraday_release_hour=13, source="fred"
        ))

        # CPI: monthly, published ~15 days after reference month ends
        dag.register(FeatureMetadata(
            "cpiaucsl", release_lag=15, revision_lag=45,
            event_time_offset=-30, intraday_release_hour=13, source="fred"
        ))

        # Credit / IG spread: daily, available T+1
        for feat in ["ig_spread", "credit_conditions", "commodity_features"]:
            dag.register(FeatureMetadata(feat, release_lag=1, intraday_release_hour=16, source="fred"))

        return dag


class CausalGuard:
    """
    Enforces causal data availability on a feature DataFrame.

    At a given prediction_time, it removes feature *values* (sets to NaN)
    that would not yet be available to a live trader, then forward-fills
    from the most recent available value.

    Parameters
    ----------
    dag : DataAvailabilityGraph
    strict : if True, drops rows where >50% of features are NaN after filtering;
             if False, only logs a warning.
    """

    def __init__(
        self,
        dag: DataAvailabilityGraph | None = None,
        strict: bool = False,
    ) -> None:
        self._dag = dag or DataAvailabilityGraph.default()
        self.strict = strict

    @classmethod
    def default(cls) -> "CausalGuard":
        return cls(DataAvailabilityGraph.default())

    def filter(
        self,
        features: pd.DataFrame,
        prediction_time: pd.Timestamp,
    ) -> pd.DataFrame:
        """
        Return a copy of *features* with future-unavailable values set to NaN.

        Parameters
        ----------
        features : DataFrame with DatetimeIndex (UTC).
        prediction_time : The timestamp at which we are making a prediction.
                          Any feature value from a bar date whose first_available
                          is after prediction_time is masked.
        """
        if features.empty:
            return features

        df = features.copy()
        # Ensure index is UTC
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")

        # For each column, determine the latest bar date whose value is available
        n_masked = 0
        for col in df.columns:
            meta = self._dag.get(col)
            for bar_date in df.index:
                if not meta.is_available_at(bar_date, prediction_time):
                    df.loc[bar_date, col] = float("nan")
                    n_masked += 1

        if n_masked > 0:
            logger.debug(
                f"[CausalGuard] Masked {n_masked} future values at {prediction_time.date()}"
            )

        # Forward-fill from most recent available value
        df = df.ffill()

        if self.strict:
            nan_frac = df.isnull().mean(axis=1)
            bad_rows = nan_frac[nan_frac > 0.5].index
            if len(bad_rows) > 0:
                logger.warning(
                    f"[CausalGuard] {len(bad_rows)} rows have >50% NaN after causal filtering"
                )
                df = df.drop(index=bad_rows)

        return df

    def filter_fold_boundary(
        self,
        train_features: pd.DataFrame,
        test_features: pd.DataFrame,
        embargo_days: int = 5,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Remove the embargo_days rows immediately before the test fold from
        training data (purging) to prevent feature-overlap leakage.

        Returns (purged_train, test).
        """
        if train_features.empty or test_features.empty:
            return train_features, test_features

        test_start = test_features.index[0]
        cutoff = test_start - pd.Timedelta(days=embargo_days)
        purged_train = train_features[train_features.index < cutoff]

        dropped = len(train_features) - len(purged_train)
        if dropped > 0:
            logger.debug(f"[CausalGuard] Purged {dropped} rows within {embargo_days}-day embargo")

        return purged_train, test_features

    def validate_no_lookahead(
        self,
        features: pd.DataFrame,
        target: pd.Series,
        max_lag: int = 1,
    ) -> bool:
        """
        Heuristic check: flag if any feature is highly correlated with
        future target values (potential look-ahead indicator).
        Returns True if clean, False if suspicious.
        """
        suspicious = []
        for col in features.columns:
            for lead in range(1, max_lag + 2):
                corr = features[col].corr(target.shift(-lead))
                if abs(corr) > 0.9:
                    suspicious.append((col, lead, corr))

        if suspicious:
            logger.warning(
                f"[CausalGuard] Potential look-ahead in {len(suspicious)} feature-lead pairs: "
                + str(suspicious[:3])
            )
            return False
        return True
