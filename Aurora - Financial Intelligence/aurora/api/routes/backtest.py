"""
/backtest endpoint — on-demand strategy backtesting.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from aurora.backtesting.engine import BacktestEngine
from aurora.configs.config import settings
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/backtest", tags=["Backtest"])


class BacktestRequest(BaseModel):
    ticker: str = Field("SPY")
    start: str = Field("2018-01-01")
    end: str = Field("2023-12-31")
    strategy: str = Field("long_short", description="long_only|long_short|market_neutral|volatility_targeting")
    commission_bps: float = Field(5.0)
    slippage_bps: float = Field(2.0)
    initial_capital: float = Field(1_000_000.0)


class BacktestResponse(BaseModel):
    ticker: str
    strategy: str
    start: str
    end: str
    metrics: dict[str, float]


@router.post("/", response_model=BacktestResponse)
async def run_backtest(req: BacktestRequest) -> BacktestResponse:
    """Run a backtest for a pre-trained model on a historical period."""
    model_dir = settings.models.model_dir / req.ticker
    if not model_dir.exists():
        raise HTTPException(404, f"No model for {req.ticker}")

    try:
        from aurora.forecasting.forecaster import AuroraForecaster
        from aurora.features.pipeline import FeaturePipeline
        import numpy as np

        forecaster = AuroraForecaster.load(model_dir)
        ingestion = IngestionPipeline()
        bundle = ingestion.run(tickers=[req.ticker], start=req.start, end=req.end)
        feat_pipeline = FeaturePipeline()
        features = feat_pipeline.run(bundle).get(req.ticker)
        if features is None:
            raise HTTPException(500, "Feature generation failed")

        ohlcv = bundle.ohlcv[req.ticker]
        close = ohlcv["close"].reindex(features.index)
        returns = np.log(close / close.shift(1)).dropna()
        features = features.reindex(returns.index)
        forecast_df = forecaster.forecast(features, returns)
        regimes = forecaster.regime_detector.predict(features, returns)

        engine = BacktestEngine(
            initial_capital=req.initial_capital,
            commission_bps=req.commission_bps,
            slippage_bps=req.slippage_bps,
        )
        result = engine.run(
            forecast_df["forecast"].dropna(),
            close.reindex(forecast_df.index),
            strategy=req.strategy,  # type: ignore
            regimes=regimes,
        )
        return BacktestResponse(
            ticker=req.ticker, strategy=req.strategy,
            start=req.start, end=req.end, metrics=result.metrics,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))
