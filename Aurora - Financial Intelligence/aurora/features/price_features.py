"""
Price-based feature engineering.

Generates log returns, realized volatility, cumulative return metrics,
and rolling window statistics used across all forecasting models.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


def log_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Log return: ln(P_t / P_{t-k})."""
    return np.log(prices / prices.shift(periods)).rename(f"log_ret_{periods}d")


def simple_returns(prices: pd.Series, periods: int = 1) -> pd.Series:
    """Arithmetic return: (P_t - P_{t-k}) / P_{t-k}."""
    return prices.pct_change(periods).rename(f"ret_{periods}d")


def cumulative_returns(returns: pd.Series, window: int) -> pd.Series:
    """Rolling compound return over *window* periods."""
    return (1 + returns).rolling(window).apply(np.prod, raw=True).sub(1).rename(
        f"cum_ret_{window}d"
    )


def realized_volatility(returns: pd.Series, window: int, annualize: bool = True) -> pd.Series:
    """
    Realized volatility (rolling standard deviation of log returns).

    Annualized by sqrt(252) when annualize=True.
    """
    rv = returns.rolling(window).std()
    if annualize:
        rv = rv * np.sqrt(252)
    return rv.rename(f"realized_vol_{window}d")


def parkinson_volatility(high: pd.Series, low: pd.Series, window: int = 20) -> pd.Series:
    """
    Parkinson range-based volatility estimator.

    More efficient than close-to-close vol; exploits intraday range.
    Formula: sqrt(1/(4*ln(2)) * E[ln(H/L)^2]) * sqrt(252)
    """
    log_hl = np.log(high / low)
    raw = (log_hl**2) / (4 * np.log(2))
    return (raw.rolling(window).mean() * 252).pow(0.5).rename(f"parkinson_vol_{window}d")


def garman_klass_volatility(
    open_: pd.Series, high: pd.Series, low: pd.Series, close: pd.Series, window: int = 20
) -> pd.Series:
    """
    Garman-Klass volatility estimator.

    Uses OHLC; more efficient than Parkinson.
    """
    log_hl = np.log(high / low)
    log_co = np.log(close / open_)
    raw = 0.5 * log_hl**2 - (2 * np.log(2) - 1) * log_co**2
    return (raw.rolling(window).mean() * 252).pow(0.5).rename(f"gk_vol_{window}d")


def rolling_skew(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling skewness of returns (negative → left tail risk)."""
    return returns.rolling(window).skew().rename(f"skew_{window}d")


def rolling_kurt(returns: pd.Series, window: int = 60) -> pd.Series:
    """Rolling excess kurtosis (fat tails when positive)."""
    return returns.rolling(window).kurt().rename(f"kurt_{window}d")


def price_momentum(returns: pd.Series, window: int) -> pd.Series:
    """Cumulative return over past *window* days (cross-sectional momentum signal)."""
    return returns.rolling(window).sum().rename(f"momentum_{window}d")


def max_drawdown_rolling(prices: pd.Series, window: int = 252) -> pd.Series:
    """Rolling maximum drawdown from peak within the window."""
    roll_max = prices.rolling(window, min_periods=1).max()
    dd = (prices - roll_max) / roll_max
    return dd.rolling(window).min().rename(f"max_dd_{window}d")


def build_price_features(
    ohlcv: pd.DataFrame,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Compute the full suite of price features for a single ticker.

    Args:
        ohlcv: DataFrame with columns [open, high, low, close, volume].
        windows: Rolling window sizes; defaults to config value.

    Returns:
        Feature DataFrame aligned to ohlcv.index.
    """
    windows = windows or settings.features.lookback_windows
    close = ohlcv["close"]
    ret_1d = log_returns(close, 1)

    frames: list[pd.Series] = [ret_1d]

    for w in windows:
        frames += [
            log_returns(close, w),
            simple_returns(close, w),
            realized_volatility(ret_1d, w),
            rolling_skew(ret_1d, w),
            rolling_kurt(ret_1d, w),
            price_momentum(ret_1d, w),
        ]

    if {"high", "low"}.issubset(ohlcv.columns):
        frames += [
            parkinson_volatility(ohlcv["high"], ohlcv["low"], w)
            for w in [5, 10, 20, 60]
        ]
        if "open" in ohlcv.columns:
            frames += [
                garman_klass_volatility(ohlcv["open"], ohlcv["high"], ohlcv["low"], close, w)
                for w in [5, 10, 20, 60]
            ]

    frames.append(max_drawdown_rolling(close, 252))

    return pd.concat(frames, axis=1)
