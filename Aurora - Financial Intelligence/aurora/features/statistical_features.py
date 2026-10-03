"""
Statistical and econometric features.

Computes entropy, autocorrelation, Hurst exponent, and stationarity
metrics that capture non-linear market dynamics for regime models.
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, adfuller, kpss

from aurora.utils.logging import get_logger

logger = get_logger(__name__)


def hurst_exponent(ts: np.ndarray, lags: int = 100) -> float:
    """
    Estimate the Hurst exponent using R/S analysis.

    H ≈ 0.5 → random walk (no memory)
    H > 0.5 → persistent (trending)
    H < 0.5 → anti-persistent (mean-reverting)
    """
    lags_range = range(2, min(lags, len(ts) // 2))
    tau = []
    rs_vals = []
    for lag in lags_range:
        rs = []
        for start in range(0, len(ts) - lag, lag):
            sub = ts[start : start + lag]
            mean = sub.mean()
            dev = np.cumsum(sub - mean)
            r_range = dev.max() - dev.min()
            s_std = sub.std(ddof=1)
            if s_std > 0:
                rs.append(r_range / s_std)
        if rs:
            tau.append(lag)
            rs_vals.append(np.mean(rs))

    if len(tau) < 2:
        return 0.5  # degenerate case

    log_tau = np.log(tau)
    log_rs = np.log(rs_vals)
    slope, *_ = np.polyfit(log_tau, log_rs, 1)
    return float(slope)


def rolling_hurst(returns: pd.Series, window: int = 252, lags: int = 50) -> pd.Series:
    """Rolling Hurst exponent over a sliding window."""
    values = returns.values

    def _hurst(arr: np.ndarray) -> float:
        try:
            return hurst_exponent(arr, lags=lags)
        except Exception:
            return np.nan

    result = [
        _hurst(values[i - window : i]) if i >= window else np.nan
        for i in range(len(values))
    ]
    return pd.Series(result, index=returns.index).rename(f"hurst_{window}d")


def sample_entropy(ts: np.ndarray, m: int = 2, r_tol: float = 0.2) -> float:
    """
    Sample entropy: measures irregularity/complexity of a time series.

    Higher entropy → less predictable market conditions.
    """
    n = len(ts)
    r = r_tol * ts.std()

    def _count_templates(m_: int) -> int:
        count = 0
        for i in range(n - m_):
            template = ts[i : i + m_]
            for j in range(i + 1, n - m_):
                if np.max(np.abs(template - ts[j : j + m_])) < r:
                    count += 1
        return count

    b = _count_templates(m)
    a = _count_templates(m + 1)
    if b == 0:
        return 0.0
    return float(-np.log(a / b)) if a > 0 else np.inf


def rolling_autocorrelation(returns: pd.Series, lag: int = 1, window: int = 60) -> pd.Series:
    """
    Rolling autocorrelation at a given lag.

    Positive autocorrelation → momentum; negative → mean-reversion.
    """
    return (
        returns.rolling(window)
        .apply(lambda x: x.autocorr(lag=lag), raw=False)
        .rename(f"autocorr_lag{lag}_{window}d")
    )


def adf_p_value_rolling(returns: pd.Series, window: int = 252) -> pd.Series:
    """
    Rolling ADF test p-value.

    Low p-value (< 0.05) → stationary (mean-reverting signal).
    """
    values = returns.values

    def _adf(arr: np.ndarray) -> float:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = adfuller(arr, autolag="AIC")
            return float(result[1])  # p-value
        except Exception:
            return np.nan

    result = [
        _adf(values[i - window : i]) if i >= window else np.nan
        for i in range(len(values))
    ]
    return pd.Series(result, index=returns.index).rename(f"adf_pval_{window}d")


def rolling_correlation_with_market(
    asset_returns: pd.Series, market_returns: pd.Series, window: int = 60
) -> pd.Series:
    """Rolling Pearson correlation between asset and market."""
    return asset_returns.rolling(window).corr(market_returns).rename(
        f"corr_market_{window}d"
    )


def rolling_beta(
    asset_returns: pd.Series, market_returns: pd.Series, window: int = 60
) -> pd.Series:
    """
    Rolling beta relative to market.

    β = Cov(asset, market) / Var(market)
    """
    cov = asset_returns.rolling(window).cov(market_returns)
    var = market_returns.rolling(window).var()
    return (cov / var.replace(0, np.nan)).rename(f"beta_{window}d")


def drawdown_series(prices: pd.Series) -> pd.Series:
    """Instantaneous drawdown from the running maximum."""
    peak = prices.cummax()
    return ((prices - peak) / peak).rename("drawdown")


def build_statistical_features(
    returns: pd.Series,
    prices: pd.Series | None = None,
    market_returns: pd.Series | None = None,
    windows: list[int] | None = None,
) -> pd.DataFrame:
    """
    Generate statistical and econometric features.

    Args:
        returns: Log or simple return series.
        prices: Price level series (needed for drawdown).
        market_returns: Benchmark series for beta/correlation.
        windows: Rolling windows to use.

    Returns:
        Feature DataFrame aligned to returns.index.
    """
    windows = windows or [20, 60, 120, 252]
    frames: list[pd.Series] = []

    # Autocorrelation at multiple lags
    for lag in [1, 2, 5]:
        for w in [20, 60]:
            frames.append(rolling_autocorrelation(returns, lag=lag, window=w))

    # Hurst exponent (expensive — use larger windows only)
    for w in [60, 252]:
        frames.append(rolling_hurst(returns, window=w))

    # ADF stationarity
    frames.append(adf_p_value_rolling(returns, window=252))

    # Drawdown
    if prices is not None:
        frames.append(drawdown_series(prices))

    # Market-relative features
    if market_returns is not None:
        for w in [20, 60, 252]:
            frames.append(rolling_correlation_with_market(returns, market_returns, w))
            frames.append(rolling_beta(returns, market_returns, w))

    return pd.concat(frames, axis=1)
