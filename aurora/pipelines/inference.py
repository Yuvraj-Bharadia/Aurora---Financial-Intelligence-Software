"""
Aurora inference pipeline.

Loads a trained AuroraForecaster and generates live forecasts
from the most recent market data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from aurora.configs.config import settings
from aurora.features.pipeline import FeaturePipeline
from aurora.forecasting.forecaster import AuroraForecaster
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class InferencePipeline:
    """
    Online inference pipeline for Aurora.

    Fetches the latest data, computes features, and runs the
    trained ensemble to generate current-day forecasts.
    """

    def __init__(
        self,
        model_dir: str | Path,
        ticker: str = "SPY",
        lookback_days: int = 504,   # 2 years of history for features
    ) -> None:
        self.model_dir = Path(model_dir)
        self.ticker = ticker
        self.lookback_days = lookback_days
        self._forecaster: AuroraForecaster | None = None

    def _load_model(self) -> None:
        if self._forecaster is None:
            logger.info(f"Loading AuroraForecaster from {self.model_dir}")
            self._forecaster = AuroraForecaster.load(self.model_dir)

    def run(
        self,
        end_date: str | None = None,
        horizons: list[int] | None = None,
    ) -> dict[str, Any]:
        """
        Run the inference pipeline for the latest available data.

        Args:
            end_date: Use today if None.
            horizons: Forecast horizons (steps ahead).

        Returns:
            Dict with forecasts, current regime, and sentiment.
        """
        from datetime import date, timedelta
        end_date = end_date or date.today().isoformat()
        start_date = (
            pd.Timestamp(end_date) - pd.Timedelta(days=self.lookback_days)
        ).strftime("%Y-%m-%d")

        self._load_model()

        # Ingest latest data
        ingestion = IngestionPipeline()
        bundle = ingestion.run(tickers=[self.ticker], start=start_date, end=end_date)

        if self.ticker not in bundle.ohlcv or bundle.ohlcv[self.ticker].empty:
            logger.error(f"No data for {self.ticker}")
            return {}

        # Build features
        feat_pipeline = FeaturePipeline()
        feature_map = feat_pipeline.run(bundle)
        features = feature_map.get(self.ticker, pd.DataFrame())
        if features.empty:
            return {}

        ohlcv = bundle.ohlcv[self.ticker]
        close = ohlcv["close"].reindex(features.index)
        import numpy as np
        returns = np.log(close / close.shift(1)).dropna()
        features = features.reindex(returns.index)

        # Generate forecasts
        forecast_df = self._forecaster.forecast(features, returns)
        multi_horizon = self._forecaster.forecast_multi_horizon(features, returns)

        latest = forecast_df.iloc[-1] if not forecast_df.empty else pd.Series()
        return {
            "ticker": self.ticker,
            "as_of": end_date,
            "current_regime": str(latest.get("regime", "unknown")),
            "regime_confidence": float(latest.get("regime_confidence", 0.0)),
            "forecast_1d": float(latest.get("forecast", 0.0)),
            "forecast_lower": float(latest.get("forecast_lower", 0.0)),
            "forecast_upper": float(latest.get("forecast_upper", 0.0)),
            "dominant_model": str(latest.get("dominant_model", "unknown")),
            "multi_horizon": {
                h: float(df["forecast"].iloc[-1]) if not df.empty else 0.0
                for h, df in multi_horizon.items()
            },
        }


def main() -> None:
    """CLI entrypoint for inference."""
    import typer
    app = typer.Typer()

    @app.command()
    def predict(
        ticker: str = typer.Option("SPY"),
        model_dir: str = typer.Option(str(settings.models.model_dir / "SPY")),
    ) -> None:
        pipeline = InferencePipeline(model_dir=model_dir, ticker=ticker)
        result = pipeline.run()
        typer.echo(result)

    app()


if __name__ == "__main__":
    main()
