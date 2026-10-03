"""
Orchestration pipeline for Aurora data ingestion.

Coordinates Alpha Vantage, FRED, and news adapters into a single
MarketDataBundle and persists results to Parquet on disk.

Primary equity data source: Alpha Vantage (TIME_SERIES_DAILY_ADJUSTED).
Requires ALPHA_VANTAGE_API_KEY in the environment / .env file.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aurora.configs.config import settings
from aurora.data.schemas import MarketDataBundle
from aurora.ingestion.alpha_vantage import AlphaVantageAdapter
from aurora.ingestion.fred import FREDAdapter
from aurora.ingestion.news import NewsAPIAdapter, RedditAdapter, articles_to_dataframe
from aurora.utils.caching import AuroraCache
from aurora.utils.logging import get_logger
from aurora.utils.validation import validate_date_range

logger = get_logger(__name__)

_PARQUET_COMPRESSION = "snappy"


class IngestionPipeline:
    """
    End-to-end data ingestion pipeline.

    Usage::

        pipeline = IngestionPipeline()
        bundle = pipeline.run(
            tickers=["SPY", "QQQ"],
            start="2018-01-01",
            end="2023-12-31",
        )
    """

    def __init__(
        self,
        equity_adapter: AlphaVantageAdapter | None = None,
        fred_adapter: FREDAdapter | None = None,
        news_adapter: NewsAPIAdapter | None = None,
        reddit_adapter: RedditAdapter | None = None,
        cache: AuroraCache | None = None,
        output_dir: Path | None = None,
    ) -> None:
        self.equity = equity_adapter or AlphaVantageAdapter()
        self.fred = fred_adapter or FREDAdapter()
        self.news = news_adapter or NewsAPIAdapter()
        self.reddit = reddit_adapter or RedditAdapter()
        self._cache = cache or AuroraCache(namespace="ingestion")
        self._output_dir = output_dir or settings.data.data_dir
        self._output_dir.mkdir(parents=True, exist_ok=True)

    # ── Public API ─────────────────────────────────────────────────────────────

    def run(
        self,
        tickers: list[str] | None = None,
        macro_series: list[str] | None = None,
        start: str | None = None,
        end: str | None = None,
        include_news: bool = False,
        force_refresh: bool = False,
    ) -> MarketDataBundle:
        """
        Run the full ingestion pipeline.

        Args:
            tickers: Equity tickers to fetch.
            macro_series: FRED series IDs to include.
            start: Start date "YYYY-MM-DD".
            end: End date "YYYY-MM-DD".
            include_news: Also ingest news articles.
            force_refresh: Bypass disk cache.

        Returns:
            MarketDataBundle with prices, OHLCV, and macro data.
        """
        tickers = tickers or settings.data.equity_tickers
        macro_series = macro_series or settings.data.macro_series
        start, end = validate_date_range(
            start or settings.data.default_start,
            end or settings.data.default_end,
        )

        cache_key = f"bundle_{','.join(sorted(tickers))}_{start}_{end}"
        if not force_refresh:
            cached = self._load_cached_bundle(cache_key)
            if cached is not None:
                logger.info("Loaded MarketDataBundle from disk cache")
                return cached

        logger.info(f"Running ingestion: {len(tickers)} tickers | {start} → {end}")

        # ── Equities (Alpha Vantage) ───────────────────────────────────────
        ohlcv_map = self.equity.fetch_many(tickers, start, end)
        prices = self.equity.fetch_bulk(tickers, start, end)

        # ── Macro ─────────────────────────────────────────────────────────
        macro = self.fred.fetch_macro_panel(macro_series, start, end)

        # ── Volume panel ──────────────────────────────────────────────────
        volume = pd.concat(
            {t: df["volume"] for t, df in ohlcv_map.items() if not df.empty},
            axis=1,
        )

        bundle = MarketDataBundle(
            prices=prices,
            ohlcv=ohlcv_map,
            macro=macro,
            volume=volume,
            metadata={"start": start, "end": end, "tickers": tickers},
        )

        if include_news:
            self._ingest_news(tickers, start, end)

        self._persist_bundle(bundle, cache_key)
        return bundle

    # ── Persistence ────────────────────────────────────────────────────────────

    def _persist_bundle(self, bundle: MarketDataBundle, cache_key: str) -> None:
        """Save bundle DataFrames to Parquet files."""
        key_dir = self._output_dir / cache_key
        key_dir.mkdir(parents=True, exist_ok=True)

        bundle.prices.to_parquet(key_dir / "prices.parquet", compression=_PARQUET_COMPRESSION)
        bundle.macro.to_parquet(key_dir / "macro.parquet", compression=_PARQUET_COMPRESSION)
        bundle.volume.to_parquet(key_dir / "volume.parquet", compression=_PARQUET_COMPRESSION)

        for ticker, df in bundle.ohlcv.items():
            if not df.empty:
                path = key_dir / "ohlcv" / f"{ticker}.parquet"
                path.parent.mkdir(exist_ok=True)
                df.to_parquet(path, compression=_PARQUET_COMPRESSION)

        logger.info(f"Bundle persisted to {key_dir}")

    def _load_cached_bundle(self, cache_key: str) -> MarketDataBundle | None:
        """Attempt to load a previously persisted bundle."""
        key_dir = self._output_dir / cache_key
        prices_path = key_dir / "prices.parquet"
        if not prices_path.exists():
            return None

        try:
            prices = pd.read_parquet(prices_path)
            macro = pd.read_parquet(key_dir / "macro.parquet")
            volume = pd.read_parquet(key_dir / "volume.parquet")
            ohlcv: dict[str, pd.DataFrame] = {}
            ohlcv_dir = key_dir / "ohlcv"
            if ohlcv_dir.exists():
                for pq in ohlcv_dir.glob("*.parquet"):
                    ohlcv[pq.stem] = pd.read_parquet(pq)
            return MarketDataBundle(prices=prices, ohlcv=ohlcv, macro=macro, volume=volume)
        except Exception as exc:
            logger.warning(f"Failed to load cached bundle: {exc}")
            return None

    # ── News ───────────────────────────────────────────────────────────────────

    def _ingest_news(self, tickers: list[str], start: str, end: str) -> None:
        """Fetch and persist news articles for the given tickers."""
        query = " OR ".join(tickers[:10])  # NewsAPI query length limit
        articles = self.news.fetch_articles(query, start, end)
        reddit_posts = self.reddit.fetch_posts()
        all_articles = articles + reddit_posts

        df = articles_to_dataframe(all_articles)
        out_path = self._output_dir / "news" / f"articles_{start}_{end}.parquet"
        out_path.parent.mkdir(exist_ok=True)
        df.to_parquet(out_path, compression=_PARQUET_COMPRESSION)
        logger.info(f"Persisted {len(df)} articles to {out_path}")
