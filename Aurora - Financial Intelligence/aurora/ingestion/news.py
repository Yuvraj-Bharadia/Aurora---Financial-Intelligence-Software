"""
News ingestion adapter.

Pulls financial news from NewsAPI and Reddit (r/investing, r/wallstreetbets)
and normalises articles into the Aurora NewsArticle schema.
"""

from __future__ import annotations

import hashlib
import os
import re
import time
from datetime import datetime, timedelta, timezone
from typing import Any

import pandas as pd

from aurora.data.schemas import NewsArticle
from aurora.utils.logging import get_logger

logger = get_logger(__name__)

# Rough ticker extraction regex (US equities: 1-5 uppercase letters)
_TICKER_RE = re.compile(r"\b([A-Z]{1,5})\b")

# Common English stop-words we don't want treated as tickers
_NON_TICKERS = {
    "I", "A", "IN", "TO", "OF", "OR", "IS", "IT", "AT", "BE", "BY", "DO",
    "NO", "SO", "UP", "IF", "ON", "AN", "AS", "HE", "HIS", "WE", "US",
    "THE", "AND", "FOR", "ARE", "NOT", "BUT", "WITH", "FROM", "THIS",
    "THAT", "HAVE", "WILL", "BEEN", "SAID", "THEY", "WERE", "WHEN",
    "ETF", "IPO", "CEO", "CFO", "USD", "GDP", "CPI", "FED", "SEC",
}


class NewsAPIAdapter:
    """Fetches news from NewsAPI.org."""

    def __init__(self, api_key: str | None = None) -> None:
        self.api_key = api_key or os.environ.get("NEWS_API_KEY", "")
        if not self.api_key:
            logger.warning("NEWS_API_KEY not set — news ingestion will be limited")

    def fetch_articles(
        self,
        query: str,
        start: str,
        end: str,
        language: str = "en",
        page_size: int = 100,
        max_pages: int = 5,
    ) -> list[NewsArticle]:
        """
        Pull articles matching *query* from NewsAPI.

        NewsAPI free tier is limited to articles within 30 days.
        """
        try:
            from newsapi import NewsApiClient
        except ImportError:
            logger.error("newsapi-python not installed")
            return []

        client = NewsApiClient(api_key=self.api_key)
        articles: list[NewsArticle] = []

        for page in range(1, max_pages + 1):
            try:
                resp = client.get_everything(
                    q=query,
                    from_param=start,
                    to=end,
                    language=language,
                    sort_by="publishedAt",
                    page=page,
                    page_size=page_size,
                )
            except Exception as exc:
                logger.error(f"[NewsAPI] Page {page} failed: {exc}")
                break

            raw_articles = resp.get("articles", [])
            if not raw_articles:
                break

            for art in raw_articles:
                articles.append(self._normalize(art))

            if len(raw_articles) < page_size:
                break
            time.sleep(0.5)  # be polite

        logger.info(f"[NewsAPI] Retrieved {len(articles)} articles for '{query}'")
        return articles

    @staticmethod
    def _normalize(raw: dict[str, Any]) -> NewsArticle:
        content = (raw.get("content") or raw.get("description") or "")
        title = raw.get("title", "")
        published = raw.get("publishedAt", "")
        try:
            pub_dt = datetime.fromisoformat(published.replace("Z", "+00:00"))
        except Exception:
            pub_dt = datetime.now(tz=timezone.utc)

        tickers = _extract_tickers(title + " " + content)
        article_id = hashlib.md5((title + str(pub_dt)).encode()).hexdigest()
        source = raw.get("source", {}).get("name", "unknown")

        return NewsArticle(
            article_id=article_id,
            source=source,
            published_at=pub_dt,
            title=title,
            content=content,
            url=raw.get("url", ""),
            tickers=tickers,
        )


class RedditAdapter:
    """Fetches posts from financial subreddits via PRAW."""

    def __init__(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        user_agent: str | None = None,
    ) -> None:
        self.client_id = client_id or os.environ.get("REDDIT_CLIENT_ID", "")
        self.client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET", "")
        self.user_agent = user_agent or os.environ.get(
            "REDDIT_USER_AGENT", "aurora_sentiment_bot/1.0"
        )

    def _get_client(self) -> Any:
        try:
            import praw
            return praw.Reddit(
                client_id=self.client_id,
                client_secret=self.client_secret,
                user_agent=self.user_agent,
            )
        except ImportError as e:
            raise ImportError("praw not installed: pip install praw") from e

    def fetch_posts(
        self,
        subreddits: list[str] | None = None,
        limit: int = 100,
        time_filter: str = "week",
    ) -> list[NewsArticle]:
        """
        Fetch hot/top posts from financial subreddits.

        Args:
            subreddits: List of subreddit names (without r/).
            limit: Max posts per subreddit.
            time_filter: 'hour', 'day', 'week', 'month', 'year', 'all'.
        """
        subreddits = subreddits or ["investing", "wallstreetbets", "stocks", "finance"]
        reddit = self._get_client()
        articles: list[NewsArticle] = []

        for sub_name in subreddits:
            try:
                sub = reddit.subreddit(sub_name)
                for post in sub.top(time_filter=time_filter, limit=limit):
                    content = post.selftext or post.title
                    pub_dt = datetime.fromtimestamp(post.created_utc, tz=timezone.utc)
                    tickers = _extract_tickers(post.title + " " + content)
                    article_id = hashlib.md5(post.id.encode()).hexdigest()

                    articles.append(
                        NewsArticle(
                            article_id=article_id,
                            source=f"reddit/{sub_name}",
                            published_at=pub_dt,
                            title=post.title,
                            content=content[:2000],  # cap length
                            url=f"https://reddit.com{post.permalink}",
                            tickers=tickers,
                        )
                    )
            except Exception as exc:
                logger.error(f"[Reddit] r/{sub_name} failed: {exc}")

        logger.info(f"[Reddit] Retrieved {len(articles)} posts")
        return articles


def articles_to_dataframe(articles: list[NewsArticle]) -> pd.DataFrame:
    """Convert a list of NewsArticles to a tidy DataFrame."""
    rows = [
        {
            "article_id": a.article_id,
            "source": a.source,
            "published_at": a.published_at,
            "title": a.title,
            "content": a.content,
            "url": a.url,
            "tickers": ",".join(a.tickers),
            "sentiment_score": a.sentiment_score,
            "sentiment_label": a.sentiment_label,
        }
        for a in articles
    ]
    df = pd.DataFrame(rows)
    if not df.empty:
        df["published_at"] = pd.to_datetime(df["published_at"], utc=True)
        df = df.sort_values("published_at")
    return df


def _extract_tickers(text: str) -> list[str]:
    """Heuristic extraction of stock ticker symbols from text."""
    candidates = _TICKER_RE.findall(text)
    return [t for t in dict.fromkeys(candidates) if t not in _NON_TICKERS]
