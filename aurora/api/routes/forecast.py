"""
/forecast endpoint — regime-adaptive point and interval forecasts.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from aurora.configs.config import settings
from aurora.pipelines.inference import InferencePipeline
from aurora.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/forecast", tags=["Forecast"])


class ForecastRequest(BaseModel):
    ticker: str = Field("SPY", description="Equity ticker")
    horizon: int = Field(1, ge=1, le=252, description="Days ahead")
    end_date: str | None = Field(None, description="As-of date (YYYY-MM-DD); defaults to today")


class ForecastResponse(BaseModel):
    ticker: str
    as_of: str
    horizon: int
    current_regime: str
    regime_confidence: float
    forecast: float
    forecast_lower: float
    forecast_upper: float
    dominant_model: str
    multi_horizon: dict[str, float]


@router.post("/", response_model=ForecastResponse)
async def get_forecast(req: ForecastRequest) -> ForecastResponse:
    """
    Generate a regime-adaptive forecast for a single ticker.

    Returns a point forecast and 90% prediction interval, the current
    detected regime and model confidence.
    """
    model_dir = settings.models.model_dir / req.ticker
    if not model_dir.exists():
        raise HTTPException(
            status_code=404,
            detail=f"No trained model found for {req.ticker}. Train first via /train.",
        )
    try:
        pipeline = InferencePipeline(model_dir=model_dir, ticker=req.ticker)
        result = pipeline.run(end_date=req.end_date)
        if not result:
            raise HTTPException(status_code=500, detail="Inference failed")
        return ForecastResponse(
            ticker=result["ticker"],
            as_of=result["as_of"],
            horizon=req.horizon,
            current_regime=result["current_regime"],
            regime_confidence=result["regime_confidence"],
            forecast=result["forecast_1d"],
            forecast_lower=result["forecast_lower"],
            forecast_upper=result["forecast_upper"],
            dominant_model=result["dominant_model"],
            multi_horizon=result["multi_horizon"],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error(f"Forecast error for {req.ticker}: {exc}")
        raise HTTPException(status_code=500, detail=str(exc))


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}
