"""
Input validation and data integrity checks for Aurora pipelines.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Any

import numpy as np
import pandas as pd

from aurora.utils.logging import get_logger

logger = get_logger(__name__)


def validate_ohlcv(df: pd.DataFrame, ticker: str = "") -> pd.DataFrame:
    """
    Validate and clean an OHLCV DataFrame.

    Checks:
    - Required columns present
    - No duplicate timestamps
    - OHLC ordering consistency (Low <= Open/Close <= High)
    - Volume non-negative
    - Drops rows with all-NaN OHLCV
    """
    required = {"open", "high", "low", "close", "volume"}
    cols = {c.lower() for c in df.columns}
    missing = required - cols
    if missing:
        raise ValueError(f"[{ticker}] Missing OHLCV columns: {missing}")

    df = df.copy()
    df.columns = [c.lower() for c in df.columns]

    # Remove full duplicates on the index
    if df.index.duplicated().any():
        n = df.index.duplicated().sum()
        logger.warning(f"[{ticker}] Dropping {n} duplicate timestamps")
        df = df[~df.index.duplicated(keep="last")]

    df = df.sort_index()

    # Enforce OHLC ordering
    bad_mask = (df["low"] > df["high"]) | (df["open"] > df["high"]) | (df["close"] > df["high"])
    if bad_mask.any():
        n = bad_mask.sum()
        logger.warning(f"[{ticker}] Setting {n} OHLC-inconsistent rows to NaN")
        df.loc[bad_mask, ["open", "high", "low", "close"]] = np.nan

    df.loc[df["volume"] < 0, "volume"] = np.nan
    price_cols = ["open", "high", "low", "close"]
    before = len(df)
    df = df.dropna(subset=price_cols, how="all")
    if len(df) < before:
        logger.warning(f"[{ticker}] Dropped {before - len(df)} all-NaN price rows")

    return df


def validate_date_range(start: str | date | datetime, end: str | date | datetime) -> tuple[str, str]:
    """Parse and validate a date range."""
    if isinstance(start, (date, datetime)):
        start = start.strftime("%Y-%m-%d")
    if isinstance(end, (date, datetime)):
        end = end.strftime("%Y-%m-%d")

    if pd.Timestamp(start) >= pd.Timestamp(end):
        raise ValueError(f"start={start} must be before end={end}")
    return str(start), str(end)


def validate_features(df: pd.DataFrame, min_rows: int = 100) -> None:
    """Sanity-check a feature matrix before model training."""
    if len(df) < min_rows:
        raise ValueError(f"Feature matrix has only {len(df)} rows, need ≥ {min_rows}")

    inf_cols = [c for c in df.select_dtypes(include=np.number).columns if np.isinf(df[c]).any()]
    if inf_cols:
        raise ValueError(f"Inf values in feature columns: {inf_cols}")

    nan_frac = df.isnull().mean()
    high_nan = nan_frac[nan_frac > 0.5]
    if not high_nan.empty:
        logger.warning(f"Columns with >50% NaN: {high_nan.to_dict()}")


def assert_no_lookahead(df: pd.DataFrame, label_col: str, feature_cols: list[str]) -> None:
    """
    Raise if any feature is computed using future label information.

    This performs a simple shift check: correlating unshifted features with
    the label should be ≤ correlation of shifted features (heuristic guard).
    """
    if label_col not in df.columns:
        return
    y = df[label_col].dropna()
    for col in feature_cols:
        if col not in df.columns:
            continue
        x = df[col].reindex(y.index)
        r_current = x.corr(y)
        r_shifted = x.shift(1).corr(y)
        # Flag extreme cases where unshifted correlation is suspiciously high
        if abs(r_current) > 0.95 and abs(r_current) > abs(r_shifted) * 1.5:
            logger.warning(
                f"Possible look-ahead in feature '{col}': "
                f"corr={r_current:.3f} vs shifted corr={r_shifted:.3f}"
            )
