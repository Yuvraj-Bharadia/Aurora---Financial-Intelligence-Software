from aurora.ingestion.alpha_vantage import AlphaVantageAdapter
from aurora.ingestion.fred import FREDAdapter
from aurora.ingestion.news import NewsAPIAdapter, RedditAdapter
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.ingestion.yahoo_finance import YahooFinanceAdapter  # kept for backwards compat

__all__ = [
    "AlphaVantageAdapter",
    "FREDAdapter",
    "NewsAPIAdapter",
    "RedditAdapter",
    "IngestionPipeline",
    "YahooFinanceAdapter",
]
