"""
/regime endpoint — current and historical regime detection.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from aurora.configs.config import settings
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.features.pipeline import FeaturePipeline
from aurora.regimes.detector import RegimeDetector
from aurora.utils.logging import get_logger

logger = get_logger(__name__)
router = APIRouter(prefix="/regime", tags=["Regime"])


class RegimeResponse(BaseModel):
    ticker: str
    as_of: str
    current_regime: str
    regime_confidence: float
    regime_probabilities: dict[str, float]
    transition_matrix: dict[str, dict[str, float]]


@router.get("/{ticker}", response_model=RegimeResponse)
async def get_regime(
    ticker: str,
    end_date: str | None = None,
) -> RegimeResponse:
    """Detect the current market regime for a ticker."""
    model_dir = settings.models.model_dir / ticker
    detector_path = model_dir / "regime_detector.pkl"
    if not detector_path.exists():
        raise HTTPException(404, f"No regime model for {ticker}")

    try:
        detector = RegimeDetector.load(detector_path)
        ingestion = IngestionPipeline()
        bundle = ingestion.run(tickers=[ticker], end=end_date)
        feat_pipeline = FeaturePipeline()
        features = feat_pipeline.run(bundle).get(ticker, None)
        if features is None or features.empty:
            raise HTTPException(500, "Feature generation failed")

        import numpy as np
        ohlcv = bundle.ohlcv[ticker]
        close = ohlcv["close"].reindex(features.index)
        returns = np.log(close / close.shift(1)).dropna()
        features = features.reindex(returns.index)
        regimes = detector.predict(features, returns)
        latest = regimes.iloc[-1]

        prob_cols = {c: float(latest[c]) for c in regimes.columns if c.startswith("prob_")}
        trans = detector.transition_matrix()
        trans_dict = {r: trans.loc[r].to_dict() for r in trans.index}

        return RegimeResponse(
            ticker=ticker,
            as_of=str(regimes.index[-1].date()),
            current_regime=str(latest["regime"]),
            regime_confidence=float(latest.get("regime_confidence", 1.0)),
            regime_probabilities=prob_cols,
            transition_matrix=trans_dict,
        )
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(500, str(exc))
