"""
Technical analysis indicators for Aurora feature engineering.

All indicators are implemented from scratch using NumPy/Pandas
for full reproducibility and auditability without TA-Lib dependency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


# ── Trend Indicators ──────────────────────────────────────────────────────────

def sma(series: pd.Series, period: int) -> pd.Series:
    """Simple moving average."""
    return series.rolling(period).mean().rename(f"sma_{period}")


def ema(series: pd.Series, period: int, adjust: bool = False) -> pd.Series:
    """Exponential moving average."""
    return series.ewm(span=period, adjust=adjust).mean().rename(f"ema_{period}")


def macd(
    close: pd.Series,
    fast: int | None = None,
    slow: int | None = None,
    signal: int | None = None,
) -> pd.DataFrame:
    """
    MACD: Moving Average Convergence Divergence.

    Returns DataFrame with columns: macd_line, signal_line, histogram.
    """
    fast = fast or settings.features.macd_fast
    slow = slow or settings.features.macd_slow
    signal = signal or settings.features.macd_signal

    ema_fast = ema(close, fast)
    ema_slow = ema(close, slow)
    macd_line = (ema_fast - ema_slow).rename("macd_line")
    signal_line = ema(macd_line, signal).rename("macd_signal")
    histogram = (macd_line - signal_line).rename("macd_hist")
    return pd.concat([macd_line, signal_line, histogram], axis=1)


def bollinger_bands(
    close: pd.Series,
    period: int | None = None,
    n_std: float | None = None,
) -> pd.DataFrame:
    """
    Bollinger Bands: middle (SMA), upper, lower band, and %B position.

    %B = (close - lower) / (upper - lower); values near 1 → overbought.
    """
    period = period or settings.features.bb_period
    n_std = n_std or settings.features.bb_std

    mid = close.rolling(period).mean()
    std = close.rolling(period).std()
    upper = mid + n_std * std
    lower = mid - n_std * std
    pct_b = (close - lower) / (upper - lower)
    bb_width = (upper - lower) / mid  # normalized width; spikes → breakouts

    return pd.DataFrame({
        "bb_mid": mid,
        "bb_upper": upper,
        "bb_lower": lower,
        "bb_pct_b": pct_b,
        "bb_width": bb_width,
    })


# ── Momentum Indicators ───────────────────────────────────────────────────────

def rsi(close: pd.Series, period: int | None = None) -> pd.Series:
    """
    Relative Strength Index (Wilder's smoothing).

    RSI < 30 → oversold; RSI > 70 → overbought.
    """
    period = period or settings.features.rsi_period
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - 100 / (1 + rs)).rename(f"rsi_{period}")


def stochastic_oscillator(
    high: pd.Series, low: pd.Series, close: pd.Series,
    k_period: int = 14, d_period: int = 3,
) -> pd.DataFrame:
    """
    Stochastic Oscillator (%K and %D).

    %K: (close - lowest_low) / (highest_high - lowest_low) × 100
    %D: SMA of %K over d_period.
    """
    lowest_low = low.rolling(k_period).min()
    highest_high = high.rolling(k_period).max()
    pct_k = 100 * (close - lowest_low) / (highest_high - lowest_low).replace(0, np.nan)
    pct_d = pct_k.rolling(d_period).mean()
    return pd.DataFrame({"stoch_k": pct_k, "stoch_d": pct_d})


def williams_r(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """Williams %R: measures overbought/oversold on a -100 to 0 scale."""
    hh = high.rolling(period).max()
    ll = low.rolling(period).min()
    return (-(hh - close) / (hh - ll).replace(0, np.nan) * 100).rename(f"willr_{period}")


def cci(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 20
) -> pd.Series:
    """Commodity Channel Index."""
    typical = (high + low + close) / 3
    sma_t = typical.rolling(period).mean()
    mad = typical.rolling(period).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return ((typical - sma_t) / (0.015 * mad)).rename(f"cci_{period}")


# ── Volume Indicators ─────────────────────────────────────────────────────────

def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """
    On-Balance Volume: cumulative volume sign-weighted by price direction.

    Rising OBV confirms trend; falling OBV diverges.
    """
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum().rename("obv")


def vwap(
    high: pd.Series, low: pd.Series, close: pd.Series, volume: pd.Series, period: int = 20
) -> pd.Series:
    """
    Volume-Weighted Average Price (rolling).

    Price above VWAP → bullish intraday momentum.
    """
    typical = (high + low + close) / 3
    tp_vol = typical * volume
    return (tp_vol.rolling(period).sum() / volume.rolling(period).sum()).rename(f"vwap_{period}")


def volume_ratio(volume: pd.Series, period: int = 20) -> pd.Series:
    """Current volume relative to rolling average (volume surprise)."""
    return (volume / volume.rolling(period).mean()).rename(f"vol_ratio_{period}")


# ── Volatility Indicators ─────────────────────────────────────────────────────

def atr(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int | None = None
) -> pd.Series:
    """
    Average True Range.

    Measures market volatility without directional bias.
    """
    period = period or settings.features.atr_period
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean().rename(f"atr_{period}")


def atr_pct(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14
) -> pd.Series:
    """ATR as a percentage of close (normalized for cross-sectional comparisons)."""
    return (atr(high, low, close, period) / close).rename(f"atr_pct_{period}")


# ── Trend Strength ────────────────────────────────────────────────────────────

def adx(
    high: pd.Series, low: pd.Series, close: pd.Series, period: int | None = None
) -> pd.DataFrame:
    """
    Average Directional Index (ADX) with +DI and -DI components.

    ADX > 25 → strong trend; ADX < 20 → sideways market.
    """
    period = period or settings.features.adx_period
    tr_val = atr(high, low, close, period)

    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=close.index).ewm(span=period, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm, index=close.index).ewm(span=period, adjust=False).mean()

    plus_di = 100 * plus_dm_s / tr_val.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / tr_val.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx_val = dx.ewm(span=period, adjust=False).mean()

    return pd.DataFrame({
        f"adx_{period}": adx_val,
        f"plus_di_{period}": plus_di,
        f"minus_di_{period}": minus_di,
    })


def build_technical_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the full technical indicator suite from an OHLCV DataFrame.

    Returns:
        DataFrame of technical features aligned to ohlcv.index.
    """
    close = ohlcv["close"]
    high = ohlcv["high"]
    low = ohlcv["low"]
    volume = ohlcv["volume"]

    frames: list[pd.DataFrame | pd.Series] = []

    # Moving averages at multiple periods
    for p in [5, 10, 20, 50, 100, 200]:
        frames.append(sma(close, p))
        frames.append(ema(close, p))
        frames.append((close / sma(close, p) - 1).rename(f"close_vs_sma_{p}"))

    # MACD
    frames.append(macd(close))

    # Bollinger Bands
    frames.append(bollinger_bands(close))

    # RSI at multiple periods
    for p in [7, 14, 21]:
        frames.append(rsi(close, p))

    # Stochastic
    frames.append(stochastic_oscillator(high, low, close))

    # Williams %R
    frames.append(williams_r(high, low, close))

    # CCI
    frames.append(cci(high, low, close))

    # Volume indicators
    frames.append(obv(close, volume))
    frames.append(vwap(high, low, close, volume))
    for p in [5, 10, 20]:
        frames.append(volume_ratio(volume, p))

    # ATR
    frames.append(atr(high, low, close))
    frames.append(atr_pct(high, low, close))

    # ADX
    frames.append(adx(high, low, close))

    return pd.concat(frames, axis=1)
