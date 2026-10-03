"""
Yahoo Finance ingestion adapter using yfinance.

Provides adjusted OHLCV data, corporate-action-adjusted close prices,
and dividend/split metadata for a broad equity universe.
"""

from __future__ import annotations

import concurrent.futures
from typing import Any

import pandas as pd
import yfinance as yf

from aurora.ingestion.base import BaseIngestionAdapter
from aurora.utils.logging import get_logger
from aurora.utils.validation import validate_ohlcv

logger = get_logger(__name__)


class YahooFinanceAdapter(BaseIngestionAdapter):
    """Fetches OHLCV data from Yahoo Finance via yfinance."""

    name = "yahoo_finance"

    def fetch(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
        auto_adjust: bool = True,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch OHLCV for a single ticker.

        Args:
            ticker: Yahoo Finance ticker symbol.
            start: Start date as "YYYY-MM-DD".
            end: End date as "YYYY-MM-DD".
            interval: Data frequency ('1d', '1h', '5m', etc.).
            auto_adjust: Apply corporate action adjustments.

        Returns:
            Normalized OHLCV DataFrame with UTC DatetimeIndex.
        """
        logger.info(f"[Yahoo] Fetching {ticker} | {start} → {end} | interval={interval}")
        raw = self._retry_fetch(
            yf.download,
            ticker,
            start=start,
            end=end,
            interval=interval,
            auto_adjust=auto_adjust,
            progress=False,
            **kwargs,
        )
        if raw.empty:
            logger.warning(f"[Yahoo] No data returned for {ticker}")
            return pd.DataFrame()

        return self._normalize(raw, ticker)

    def fetch_many(
        self,
        tickers: list[str],
        start: str,
        end: str,
        interval: str = "1d",
        auto_adjust: bool = True,
        max_workers: int = 8,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple tickers in parallel.

        Returns:
            Dict mapping ticker → OHLCV DataFrame.
        """
        results: dict[str, pd.DataFrame] = {}

        def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
            try:
                df = self.fetch(ticker, start, end, interval=interval, auto_adjust=auto_adjust)
                return ticker, df
            except Exception as exc:
                logger.error(f"[Yahoo] Failed {ticker}: {exc}")
                return ticker, pd.DataFrame()

        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
            futures = {pool.submit(_fetch_one, t): t for t in tickers}
            for fut in concurrent.futures.as_completed(futures):
                ticker, df = fut.result()
                results[ticker] = df

        logger.info(
            f"[Yahoo] Fetched {sum(1 for v in results.values() if not v.empty)}/{len(tickers)} tickers"
        )
        return results

    def fetch_bulk(
        self,
        tickers: list[str],
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Bulk download into a multi-index DataFrame (yfinance native batch).

        Returns a wide adj_close DataFrame (DatetimeIndex × tickers).
        """
        logger.info(f"[Yahoo] Bulk download {len(tickers)} tickers")
        raw = yf.download(
            tickers,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            group_by="ticker",
        )
        if raw.empty:
            return pd.DataFrame()

        # Extract adj close (already adjusted when auto_adjust=True → "Close")
        if isinstance(raw.columns, pd.MultiIndex):
            close = raw.xs("Close", axis=1, level=1)
        else:
            close = raw[["Close"]].rename(columns={"Close": tickers[0]})

        close = self._to_datetime_index(close)
        return close

    @staticmethod
    def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Map yfinance column names to Aurora canonical schema."""
        col_map = {
            "Open": "open",
            "High": "high",
            "Low": "low",
            "Close": "close",
            "Volume": "volume",
        }
        df = raw.rename(columns=col_map)[list(col_map.values())]
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df = df.sort_index()
        df["adj_close"] = df["close"]  # auto_adjust=True means Close IS adjusted
        return validate_ohlcv(df, ticker)
