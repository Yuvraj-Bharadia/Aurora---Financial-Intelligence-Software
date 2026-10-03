"""
FRED (Federal Reserve Economic Data) ingestion adapter.

Provides macroeconomic indicators: VIX, interest rates, inflation,
unemployment, Treasury spreads, USD index, and more.
"""

from __future__ import annotations

import os
from typing import Any

import pandas as pd

from aurora.ingestion.base import BaseIngestionAdapter
from aurora.utils.logging import get_logger

logger = get_logger(__name__)

# Mapping of friendly names to FRED series IDs
FRED_SERIES_MAP: dict[str, str] = {
    "vix": "VIXCLS",
    "treasury_10y": "GS10",
    "treasury_2y": "GS2",
    "treasury_3m": "TB3MS",
    "fed_funds": "FEDFUNDS",
    "yield_spread_10y_2y": "T10Y2Y",
    "yield_spread_10y_3m": "T10Y3M",
    "inflation_cpi": "CPIAUCSL",
    "core_inflation": "CPILFESL",
    "unemployment": "UNRATE",
    "initial_claims": "ICSA",
    "industrial_production": "INDPRO",
    "retail_sales": "RSXFS",
    "consumer_confidence": "UMCSENT",
    "pce": "PCE",
    "gdp_growth": "A191RL1Q225SBEA",
    "wti_crude": "DCOILWTICO",
    "natural_gas": "DHHNGSP",
    "dollar_index": "DTWEXBGS",
    "m2_money": "M2SL",
    "credit_spread_baa": "BAA10Y",
    "high_yield_spread": "BAMLH0A0HYM2",
}


class FREDAdapter(BaseIngestionAdapter):
    """Downloads macroeconomic series from FRED via the fredapi library."""

    name = "fred"

    def __init__(self, api_key: str | None = None) -> None:
        super().__init__(api_key)
        self.api_key = api_key or os.environ.get("FRED_API_KEY", "")

    def _get_client(self) -> Any:
        try:
            from fredapi import Fred
            return Fred(api_key=self.api_key)
        except ImportError as e:
            raise ImportError("fredapi not installed: pip install fredapi") from e

    def fetch(
        self,
        ticker: str,  # treated as series_id
        start: str,
        end: str,
        **kwargs: Any,
    ) -> pd.DataFrame:
        """
        Fetch a single FRED series.

        Args:
            ticker: FRED series ID (e.g. "VIXCLS") or friendly name.
            start: Start date "YYYY-MM-DD".
            end: End date "YYYY-MM-DD".

        Returns:
            DataFrame with DatetimeIndex and one column named after the series.
        """
        series_id = FRED_SERIES_MAP.get(ticker.lower(), ticker)
        logger.info(f"[FRED] Fetching {series_id}")
        fred = self._get_client()
        raw: pd.Series = self._retry_fetch(
            fred.get_series, series_id, observation_start=start, observation_end=end
        )
        df = raw.to_frame(name=series_id)
        df.index = pd.to_datetime(df.index)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df.sort_index()

    def fetch_many(
        self,
        tickers: list[str],
        start: str,
        end: str,
        **kwargs: Any,
    ) -> dict[str, pd.DataFrame]:
        results: dict[str, pd.DataFrame] = {}
        for t in tickers:
            try:
                results[t] = self.fetch(t, start, end)
            except Exception as exc:
                logger.error(f"[FRED] Failed {t}: {exc}")
                results[t] = pd.DataFrame()
        return results

    def fetch_macro_panel(
        self,
        series: list[str] | None = None,
        start: str = "2010-01-01",
        end: str = "2024-12-31",
    ) -> pd.DataFrame:
        """
        Fetch multiple macro series and join them into a wide panel.

        Missing values forward-filled then backward-filled to handle
        different reporting frequencies (monthly vs daily).
        """
        series = series or list(FRED_SERIES_MAP.values())
        frames: list[pd.DataFrame] = []
        for sid in series:
            try:
                df = self.fetch(sid, start, end)
                frames.append(df)
            except Exception as exc:
                logger.warning(f"[FRED] Skipping {sid}: {exc}")

        if not frames:
            return pd.DataFrame()

        panel = pd.concat(frames, axis=1)
        # Reindex to daily and fill across different frequencies
        daily_idx = pd.bdate_range(start=start, end=end, tz="UTC")
        panel = panel.reindex(daily_idx).ffill().bfill()
        return panel
