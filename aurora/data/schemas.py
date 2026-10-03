"""
Canonical data schemas for Aurora.

All ingestion adapters must produce DataFrames conforming to these schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import pandas as pd


@dataclass
class OHLCVBar:
    """Single OHLCV bar for one ticker at one timestamp."""

    ticker: str
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float
    adj_close: Optional[float] = None
    vwap: Optional[float] = None


@dataclass
class MacroObservation:
    """Single macro indicator observation."""

    series_id: str
    date: datetime
    value: float
    frequency: str = "D"  # D, W, M, Q, A


@dataclass
class NewsArticle:
    """Raw news article with metadata."""

    article_id: str
    source: str
    published_at: datetime
    title: str
    content: str
    url: str
    tickers: list[str] = field(default_factory=list)
    sentiment_score: Optional[float] = None
    sentiment_label: Optional[str] = None
    embedding: Optional[list[float]] = None


@dataclass
class MarketDataBundle:
    """Container for a full market data pull."""

    prices: pd.DataFrame          # DatetimeIndex × tickers (adj_close)
    ohlcv: dict[str, pd.DataFrame]  # ticker → OHLCV DataFrame
    macro: pd.DataFrame           # DatetimeIndex × macro series
    volume: pd.DataFrame          # DatetimeIndex × tickers
    metadata: dict[str, object] = field(default_factory=dict)


# Column name constants — use these everywhere to avoid typos
class Cols:
    OPEN = "open"
    HIGH = "high"
    LOW = "low"
    CLOSE = "close"
    ADJ_CLOSE = "adj_close"
    VOLUME = "volume"
    VWAP = "vwap"
    RETURNS = "returns"
    LOG_RETURNS = "log_returns"
    REALIZED_VOL = "realized_vol"
    REGIME = "regime"
    REGIME_PROB = "regime_prob"
    FORECAST = "forecast"
    SENTIMENT = "sentiment_score"
