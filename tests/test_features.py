"""
Unit tests for the Aurora feature engineering pipeline.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.features.price_features import (
    log_returns, simple_returns, realized_volatility,
    parkinson_volatility, build_price_features,
)
from aurora.features.technical_indicators import (
    rsi, macd, bollinger_bands, atr, adx, obv,
    build_technical_features,
)
from aurora.features.statistical_features import (
    hurst_exponent, rolling_autocorrelation, build_statistical_features,
)
from aurora.features.macro_features import build_macro_features


class TestPriceFeatures:
    def test_log_returns_shape(self, synthetic_prices):
        ret = log_returns(synthetic_prices)
        assert len(ret) == len(synthetic_prices)
        assert ret.iloc[0] != ret.iloc[0]  # first value is NaN

    def test_log_returns_finite(self, synthetic_prices):
        ret = log_returns(synthetic_prices).dropna()
        assert np.isfinite(ret.values).all()

    def test_simple_returns_non_negative_prices(self, synthetic_prices):
        ret = simple_returns(synthetic_prices).dropna()
        # Prices are all positive so no -100% returns expected
        assert (ret > -1).all()

    def test_realized_vol_positive(self, synthetic_returns):
        rv = realized_volatility(synthetic_returns, window=20).dropna()
        assert (rv >= 0).all()

    def test_parkinson_vol_positive(self, synthetic_ohlcv):
        pv = parkinson_volatility(
            synthetic_ohlcv["high"], synthetic_ohlcv["low"], window=20
        ).dropna()
        assert (pv >= 0).all()

    def test_build_price_features_columns(self, synthetic_ohlcv):
        df = build_price_features(synthetic_ohlcv)
        assert "log_ret_1d" in df.columns
        assert df.shape[1] > 10
        assert df.shape[0] == len(synthetic_ohlcv)


class TestTechnicalIndicators:
    def test_rsi_bounded(self, synthetic_ohlcv):
        r = rsi(synthetic_ohlcv["close"]).dropna()
        assert (r >= 0).all() and (r <= 100).all()

    def test_macd_has_three_cols(self, synthetic_ohlcv):
        m = macd(synthetic_ohlcv["close"])
        assert set(m.columns) == {"macd_line", "macd_signal", "macd_hist"}

    def test_bollinger_bands_ordering(self, synthetic_ohlcv):
        bb = bollinger_bands(synthetic_ohlcv["close"]).dropna()
        assert (bb["bb_upper"] >= bb["bb_mid"]).all()
        assert (bb["bb_mid"] >= bb["bb_lower"]).all()

    def test_atr_non_negative(self, synthetic_ohlcv):
        a = atr(
            synthetic_ohlcv["high"],
            synthetic_ohlcv["low"],
            synthetic_ohlcv["close"],
        ).dropna()
        assert (a >= 0).all()

    def test_adx_bounded(self, synthetic_ohlcv):
        d = adx(
            synthetic_ohlcv["high"],
            synthetic_ohlcv["low"],
            synthetic_ohlcv["close"],
        ).dropna()
        assert (d["adx_14"] >= 0).all()

    def test_obv_monotonic_check(self, synthetic_ohlcv):
        o = obv(synthetic_ohlcv["close"], synthetic_ohlcv["volume"])
        assert len(o) == len(synthetic_ohlcv)

    def test_build_technical_features_no_inf(self, synthetic_ohlcv):
        df = build_technical_features(synthetic_ohlcv)
        assert not np.isinf(df.select_dtypes(include=np.number).values).any()


class TestStatisticalFeatures:
    def test_hurst_range(self, synthetic_returns):
        h = hurst_exponent(synthetic_returns.values[:300])
        assert 0 <= h <= 2  # theoretically bounded; GBM ≈ 0.5

    def test_rolling_autocorr_shape(self, synthetic_returns):
        ac = rolling_autocorrelation(synthetic_returns, lag=1, window=20)
        assert len(ac) == len(synthetic_returns)

    def test_build_statistical_features_shape(self, synthetic_returns):
        df = build_statistical_features(synthetic_returns)
        assert df.shape[0] == len(synthetic_returns)
        assert df.shape[1] > 0


class TestMacroFeatures:
    def test_build_macro_features(self, synthetic_macro):
        df = build_macro_features(synthetic_macro)
        assert not df.empty
        assert "vix" in df.columns or "yield_curve_10y2y" in df.columns

    def test_macro_no_inf(self, synthetic_macro):
        df = build_macro_features(synthetic_macro)
        num = df.select_dtypes(include=np.number)
        assert not np.isinf(num.values).any()
