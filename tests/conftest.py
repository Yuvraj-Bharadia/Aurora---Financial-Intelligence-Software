"""
Shared pytest fixtures for the Aurora test suite.

All fixtures use synthetic data so that tests run offline
without any API keys or network access.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


# ── Synthetic market data ──────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def date_range() -> pd.DatetimeIndex:
    """5 years of business dates, UTC-aware."""
    return pd.bdate_range("2018-01-01", "2022-12-31", tz="UTC")


@pytest.fixture(scope="session")
def synthetic_prices(date_range: pd.DatetimeIndex) -> pd.Series:
    """Geometric Brownian Motion price path for SPY-like asset."""
    rng = np.random.default_rng(42)
    n = len(date_range)
    log_ret = rng.normal(0.0004, 0.01, n)
    prices = 300 * np.exp(np.cumsum(log_ret))
    return pd.Series(prices, index=date_range, name="close")


@pytest.fixture(scope="session")
def synthetic_ohlcv(synthetic_prices: pd.Series, date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic OHLCV DataFrame."""
    rng = np.random.default_rng(99)
    n = len(date_range)
    close = synthetic_prices.values
    spread = np.abs(rng.normal(0, 0.003, n)) * close
    return pd.DataFrame(
        {
            "open":   close * (1 + rng.normal(0, 0.002, n)),
            "high":   close + spread,
            "low":    close - spread,
            "close":  close,
            "volume": np.abs(rng.normal(1e8, 2e7, n)),
            "adj_close": close,
        },
        index=date_range,
    )


@pytest.fixture(scope="session")
def synthetic_returns(synthetic_prices: pd.Series) -> pd.Series:
    """Log returns from the synthetic price path."""
    return np.log(synthetic_prices / synthetic_prices.shift(1)).dropna().rename("returns")


@pytest.fixture(scope="session")
def synthetic_features(synthetic_returns: pd.Series) -> pd.DataFrame:
    """Simple feature matrix (rolling stats of returns)."""
    r = synthetic_returns
    df = pd.DataFrame(index=r.index)
    for w in [5, 10, 20, 60]:
        df[f"ret_{w}d"] = r.rolling(w).sum()
        df[f"vol_{w}d"] = r.rolling(w).std() * np.sqrt(252)
        df[f"skew_{w}d"] = r.rolling(w).skew()
    df["momentum_20d"] = r.rolling(20).sum()
    df["vix"] = np.abs(np.random.default_rng(7).normal(20, 5, len(r)))
    return df.ffill().bfill().dropna()


@pytest.fixture(scope="session")
def synthetic_macro(date_range: pd.DatetimeIndex) -> pd.DataFrame:
    """Synthetic macro panel."""
    rng = np.random.default_rng(11)
    n = len(date_range)
    return pd.DataFrame(
        {
            "VIXCLS": np.abs(rng.normal(20, 5, n)),
            "GS10":   rng.normal(2.5, 0.5, n),
            "GS2":    rng.normal(1.5, 0.4, n),
            "FEDFUNDS": rng.normal(1.0, 0.8, n),
        },
        index=date_range,
    )
