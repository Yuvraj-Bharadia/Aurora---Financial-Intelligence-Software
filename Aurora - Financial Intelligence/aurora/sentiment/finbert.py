"""
FinBERT-based financial sentiment scoring.

Uses ProsusAI/finbert (pre-trained on financial text) to classify
news headlines and article content into positive, negative, neutral.

Integrates with the Aurora pipeline via:
1. Batch scoring of news article DataFrames
2. Daily sentiment aggregation with ticker alignment
3. Temporal smoothing and momentum computation
4. Volatility-adjusted sentiment signals
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
import torch

from aurora.configs.config import settings
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class FinBERTScorer:
    """
    Wrapper around HuggingFace FinBERT for financial sentiment analysis.

    Scores each piece of text as:
    - positive  → +1 label
    - neutral   →  0 label
    - negative  → -1 label

    The continuous sentiment score is computed as:
        score = P(positive) - P(negative)
    ranging from -1 (strongly bearish) to +1 (strongly bullish).
    """

    def __init__(
        self,
        model_name: str | None = None,
        batch_size: int | None = None,
        max_length: int | None = None,
        device: str | None = None,
    ) -> None:
        self.model_name = model_name or settings.sentiment.finbert_model
        self.batch_size = batch_size or settings.sentiment.batch_size
        self.max_length = max_length or settings.sentiment.max_length
        self._device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._pipeline: Any = None

    def _load_model(self) -> None:
        """Lazy-load the FinBERT pipeline."""
        if self._pipeline is not None:
            return
        try:
            from transformers import pipeline as hf_pipeline
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self._pipeline = hf_pipeline(
                    "text-classification",
                    model=self.model_name,
                    tokenizer=self.model_name,
                    device=0 if self._device == "cuda" else -1,
                    top_k=None,  # return all label scores
                    truncation=True,
                    max_length=self.max_length,
                )
            logger.info(f"FinBERT loaded: {self.model_name} on {self._device}")
        except ImportError as e:
            raise ImportError("transformers not installed: pip install transformers") from e

    def score_texts(self, texts: list[str]) -> list[dict[str, float]]:
        """
        Score a list of texts.

        Returns:
            List of dicts with keys: positive, negative, neutral, score.
            `score` = P(positive) - P(negative).
        """
        self._load_model()
        results: list[dict[str, float]] = []

        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            cleaned = [t[:1024] if t else " " for t in batch]  # truncate long texts
            try:
                raw = self._pipeline(cleaned)
            except Exception as exc:
                logger.warning(f"FinBERT batch {i//self.batch_size} failed: {exc}")
                raw = [[]] * len(batch)

            for label_scores in raw:
                probs = {item["label"].lower(): item["score"] for item in label_scores}
                score = probs.get("positive", 0.33) - probs.get("negative", 0.33)
                results.append({
                    "positive": probs.get("positive", 0.33),
                    "negative": probs.get("negative", 0.33),
                    "neutral": probs.get("neutral", 0.33),
                    "score": float(score),
                })

        return results

    def score_dataframe(
        self,
        df: pd.DataFrame,
        text_col: str = "content",
        title_col: str = "title",
    ) -> pd.DataFrame:
        """
        Add sentiment columns to a news articles DataFrame.

        Args:
            df: DataFrame with text content.
            text_col: Column name containing article body.
            title_col: Column name for title (combined with content).

        Returns:
            df with new columns: sentiment_score, sentiment_positive,
            sentiment_negative, sentiment_neutral, sentiment_label.
        """
        df = df.copy()
        texts = []
        for _, row in df.iterrows():
            title = str(row.get(title_col, "") or "")
            content = str(row.get(text_col, "") or "")
            # Prioritize headline (most signal-dense part)
            combined = (title + ". " + content[:512]).strip()
            texts.append(combined)

        scores = self.score_texts(texts)
        df["sentiment_score"] = [s["score"] for s in scores]
        df["sentiment_positive"] = [s["positive"] for s in scores]
        df["sentiment_negative"] = [s["negative"] for s in scores]
        df["sentiment_neutral"] = [s["neutral"] for s in scores]
        df["sentiment_label"] = [
            "positive" if s["score"] > 0.1 else
            ("negative" if s["score"] < -0.1 else "neutral")
            for s in scores
        ]
        return df


class SentimentAggregator:
    """
    Aggregates article-level sentiment into daily trading signals.

    Computes:
    - Equal-weighted daily sentiment
    - Volume-weighted (article-count-weighted) daily sentiment
    - Exponentially smoothed sentiment momentum
    - Volatility-adjusted sentiment (sentiment / recent vol)
    - Sentiment surprise (deviation from rolling mean)
    """

    def aggregate_daily(
        self,
        articles_df: pd.DataFrame,
        ticker: str | None = None,
        lookback: int | None = None,
    ) -> pd.DataFrame:
        """
        Collapse article-level sentiment to daily aggregates.

        Args:
            articles_df: Scored articles with 'published_at' and 'sentiment_score'.
            ticker: If provided, filter to articles mentioning this ticker.
            lookback: Days for rolling aggregation window.

        Returns:
            Daily sentiment DataFrame.
        """
        lookback = lookback or settings.sentiment.sentiment_lookback
        df = articles_df.copy()

        if ticker and "tickers" in df.columns:
            mask = df["tickers"].fillna("").str.contains(ticker, case=False)
            df = df[mask]

        if df.empty:
            return pd.DataFrame()

        df["date"] = pd.to_datetime(df["published_at"]).dt.normalize()
        daily = df.groupby("date").agg(
            sentiment_mean=("sentiment_score", "mean"),
            sentiment_std=("sentiment_score", "std"),
            article_count=("sentiment_score", "count"),
            positive_frac=("sentiment_label", lambda x: (x == "positive").mean()),
            negative_frac=("sentiment_label", lambda x: (x == "negative").mean()),
        ).fillna(0)

        daily.index = pd.to_datetime(daily.index, utc=True)

        # Exponentially smoothed sentiment momentum
        daily["sentiment_ema"] = (
            daily["sentiment_mean"].ewm(span=lookback).mean()
        )

        # Sentiment surprise: deviation from rolling mean (normalized)
        rolling_mean = daily["sentiment_mean"].rolling(21).mean()
        rolling_std = daily["sentiment_mean"].rolling(21).std()
        daily["sentiment_surprise"] = (
            (daily["sentiment_mean"] - rolling_mean) / rolling_std.replace(0, 1)
        )

        # Signed log count (captures article volume as an amplifier)
        daily["sentiment_volume_signal"] = (
            np.sign(daily["sentiment_mean"]) * np.log1p(daily["article_count"])
        )

        return daily
