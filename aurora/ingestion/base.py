"""
Abstract base class for all Aurora data ingestion adapters.

Each data source implements this interface so that the pipeline
can treat all providers uniformly.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

import pandas as pd
from tenacity import retry, stop_after_attempt, wait_exponential

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class BaseIngestionAdapter(ABC):
    """
    Abstract base for data ingestion adapters.

    Concrete subclasses implement `_fetch_raw()` and
    `normalize()` for their specific data source.
    """

    name: str = "base"

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key
        self._max_retries = settings.data.max_retries
        self._backoff = settings.data.retry_backoff
        self._timeout = settings.data.request_timeout

    @abstractmethod
    def fetch(
        self,
        ticker: str,
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """Fetch and return normalized OHLCV data."""
        ...

    @abstractmethod
    def fetch_many(
        self,
        tickers: list[str],
        start: str,
        end: str,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        """Fetch data for multiple tickers."""
        ...

    def _retry_fetch(self, fn: Any, *args: Any, **kwargs: Any) -> Any:
        """Wrap a fetch call with exponential back-off retries."""
        for attempt in range(self._max_retries):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                wait = self._backoff ** attempt
                logger.warning(
                    f"[{self.name}] Attempt {attempt + 1} failed: {exc}. "
                    f"Retrying in {wait:.1f}s"
                )
                if attempt == self._max_retries - 1:
                    raise
                time.sleep(wait)

    @staticmethod
    def _to_datetime_index(df: pd.DataFrame, col: str | None = None) -> pd.DataFrame:
        """Ensure the DataFrame has a tz-aware UTC DatetimeIndex."""
        if col is not None:
            df = df.set_index(col)
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        return df.sort_index()
