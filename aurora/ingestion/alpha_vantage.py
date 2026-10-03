"""
Alpha Vantage ingestion adapter.

Fetches adjusted OHLCV data via the TIME_SERIES_DAILY_ADJUSTED endpoint.
Requires ALPHA_VANTAGE_API_KEY to be set in the environment or .env file.

Rate limits by plan:
  Free        : 25 requests / day
  Premium 75  : 75 requests / min
  Premium 150 : 150 requests / min
  Premium 300 : 300 requests / min

Set ALPHA_VANTAGE_PREMIUM=true in .env to disable the inter-request sleep.
"""

from __future__ import annotations

import os
import time
import concurrent.futures
from typing import Any

import pandas as pd

from aurora.ingestion.base import BaseIngestionAdapter
from aurora.utils.logging import get_logger
from aurora.utils.validation import validate_ohlcv

logger = get_logger(__name__)

# Alpha Vantage column names for TIME_SERIES_DAILY_ADJUSTED
_AV_COL_MAP = {
    "1. open":           "open",
    "2. high":           "high",
    "3. low":            "low",
    "4. close":          "close",
    "5. adjusted close": "adj_close",
    "6. volume":         "volume",
}


class AlphaVantageAdapter(BaseIngestionAdapter):
    """
    Fetches OHLCV data from Alpha Vantage's TIME_SERIES_DAILY_ADJUSTED endpoint.

    Parameters
    ----------
    api_key:
        Alpha Vantage API key.  Falls back to the ``ALPHA_VANTAGE_API_KEY``
        environment variable if not supplied.
    premium:
        Set True for paid plans to skip the inter-request rate-limit sleep.
        Defaults to the ``ALPHA_VANTAGE_PREMIUM`` env var (false if absent).
    request_delay:
        Seconds to wait between successive single-ticker requests when
        ``premium=False``.  Default 12 s keeps free-tier users under the
        5-req/min cap that Alpha Vantage enforces in practice.
    """

    name = "alpha_vantage"

    def __init__(
        self,
        api_key: str | None = None,
        premium: bool | None = None,
        request_delay: float = 12.0,
    ) -> None:
        super().__init__()
        self._api_key = api_key or os.environ.get("ALPHA_VANTAGE_API_KEY", "")
        if not self._api_key:
            raise ValueError(
                "No Alpha Vantage API key found. "
                "Set ALPHA_VANTAGE_API_KEY in your .env file or pass api_key= directly."
            )

        _premium_env = os.environ.get("ALPHA_VANTAGE_PREMIUM", "false").lower() == "true"
        self._premium = premium if premium is not None else _premium_env
        self._request_delay = 0.0 if self._premium else request_delay
        self._last_request_at: float = 0.0

        # Lazy import — alpha_vantage may not be installed in test environments
        try:
            from alpha_vantage.timeseries import TimeSeries  # type: ignore[import]
            self._ts = TimeSeries(key=self._api_key, output_format="pandas", indexing_type="date")
        except ImportError as exc:
            raise ImportError(
                "alpha-vantage package is required: pip install alpha-vantage"
            ) from exc

    # ── Public API ─────────────────────────────────────────────────────────────

    def fetch(
        self,
        ticker: str,
        start: str,
        end: str,
        outputsize: str = "full",
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch daily adjusted OHLCV for a single ticker.

        Parameters
        ----------
        ticker:
            Equity symbol recognised by Alpha Vantage (e.g. "AAPL", "SPY").
        start:
            Inclusive start date "YYYY-MM-DD".
        end:
            Inclusive end date "YYYY-MM-DD".
        outputsize:
            ``"full"`` returns the complete 20-year history;
            ``"compact"`` returns the last 100 trading days.
        """
        logger.info(f"[AlphaVantage] Fetching {ticker} | {start} → {end}")
        self._rate_limit()

        try:
            raw, _ = self._retry_fetch(
                self._ts.get_daily_adjusted,
                ticker,
                outputsize=outputsize,
            )
        except Exception as exc:
            logger.error(f"[AlphaVantage] API error for {ticker}: {exc}")
            return pd.DataFrame()

        if raw is None or raw.empty:
            logger.warning(f"[AlphaVantage] No data returned for {ticker}")
            return pd.DataFrame()

        df = self._normalize(raw, ticker)
        # Slice to the requested date window
        df = df.loc[start:end]  # type: ignore[misc]
        return df

    def fetch_many(
        self,
        tickers: list[str],
        start: str,
        end: str,
        outputsize: str = "full",
        max_workers: int = 1,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """
        Fetch OHLCV for multiple tickers.

        On free-tier accounts (``premium=False``) requests are serialised and
        throttled automatically.  On premium accounts you can raise
        ``max_workers`` to parallelise calls.
        """
        results: dict[str, pd.DataFrame] = {}

        if not self._premium:
            # Free tier: serialize to stay within 5-req/min hard cap
            for ticker in tickers:
                try:
                    df = self.fetch(ticker, start, end, outputsize=outputsize)
                    results[ticker] = df
                except Exception as exc:
                    logger.error(f"[AlphaVantage] Failed {ticker}: {exc}")
                    results[ticker] = pd.DataFrame()
        else:
            def _fetch_one(ticker: str) -> tuple[str, pd.DataFrame]:
                try:
                    return ticker, self.fetch(ticker, start, end, outputsize=outputsize)
                except Exception as exc:
                    logger.error(f"[AlphaVantage] Failed {ticker}: {exc}")
                    return ticker, pd.DataFrame()

            with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
                futures = {pool.submit(_fetch_one, t): t for t in tickers}
                for fut in concurrent.futures.as_completed(futures):
                    ticker_sym, df = fut.result()
                    results[ticker_sym] = df

        ok = sum(1 for v in results.values() if not v.empty)
        logger.info(f"[AlphaVantage] Fetched {ok}/{len(tickers)} tickers")
        return results

    def fetch_bulk(
        self,
        tickers: list[str],
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Return a wide adj_close DataFrame (DatetimeIndex × tickers).

        Alpha Vantage has no true batch endpoint, so this is a sequential
        call per ticker that extracts only the ``adj_close`` column.
        """
        logger.info(f"[AlphaVantage] Bulk adj_close for {len(tickers)} tickers")
        ohlcv_map = self.fetch_many(tickers, start, end)
        close_frames = {
            t: df["adj_close"]
            for t, df in ohlcv_map.items()
            if not df.empty and "adj_close" in df.columns
        }
        if not close_frames:
            return pd.DataFrame()

        wide = pd.concat(close_frames, axis=1)
        wide = self._to_datetime_index(wide)
        return wide

    # ── Internals ─────────────────────────────────────────────────────────────

    def _rate_limit(self) -> None:
        """Sleep if needed to honour Alpha Vantage's rate limits."""
        if self._request_delay <= 0:
            return
        elapsed = time.monotonic() - self._last_request_at
        wait = self._request_delay - elapsed
        if wait > 0:
            logger.debug(f"[AlphaVantage] Rate-limit sleep {wait:.1f}s")
            time.sleep(wait)
        self._last_request_at = time.monotonic()

    @staticmethod
    def _normalize(raw: pd.DataFrame, ticker: str) -> pd.DataFrame:
        """Map Alpha Vantage column names to Aurora canonical schema."""
        # Drop any columns not in our map (e.g. dividend, split coefficient)
        present = {k: v for k, v in _AV_COL_MAP.items() if k in raw.columns}
        df = raw.rename(columns=present)[list(present.values())]

        # Ensure numeric types (AV occasionally returns object columns)
        for col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Index normalisation to UTC DatetimeIndex
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")

        df = df.sort_index()

        # If adj_close not present, use close as fallback
        if "adj_close" not in df.columns:
            df["adj_close"] = df["close"]

        return validate_ohlcv(df, ticker)
