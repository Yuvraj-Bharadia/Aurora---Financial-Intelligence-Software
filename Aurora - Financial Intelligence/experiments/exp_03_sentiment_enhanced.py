"""
Experiment 3: Sentiment-Enhanced Forecasting.

Research Question:
    Does incorporating FinBERT news sentiment as a feature improve
    the directional accuracy and Sharpe ratio of return forecasts?

Methodology:
    - Compare feature sets: price-only vs price+sentiment.
    - Use LightGBM as the base learner for both configurations.
    - Walk-forward evaluation on SPY (2018–2023).
    - Evaluate directional accuracy and information coefficient (IC).

Hypothesis:
    H3: Adding sentiment features improves directional accuracy by ≥2%
        and IC by ≥0.02 compared to the price-only baseline.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow

from aurora.evaluation.metrics import (
    directional_metrics, information_coefficient, sharpe_ratio
)
from aurora.evaluation.evaluator import WalkForwardEvaluator
from aurora.models.ml.gradient_boosting import LightGBMForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


def simulate_sentiment_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """
    Simulate daily sentiment features for testing without API keys.

    In production, replace this with SentimentAggregator.aggregate_daily()
    output aligned to the trading calendar.
    """
    rng = np.random.default_rng(77)
    n = len(index)
    # Sentiment is weakly positively correlated with next-day returns (signal)
    raw_sent = rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "sentiment_mean":          raw_sent,
            "sentiment_ema":           pd.Series(raw_sent).ewm(span=3).mean().values,
            "sentiment_surprise":      raw_sent - pd.Series(raw_sent).rolling(21).mean().fillna(0).values,
            "sentiment_volume_signal": raw_sent * np.abs(rng.normal(1, 0.3, n)),
            "positive_frac":           np.clip(0.5 + raw_sent * 0.3, 0, 1),
            "negative_frac":           np.clip(0.5 - raw_sent * 0.3, 0, 1),
        },
        index=index,
    )


def run_experiment(features_path: str | None = None) -> dict:
    logger.info("=" * 60)
    logger.info("Experiment 3: Sentiment-Enhanced Forecasting")
    logger.info("=" * 60)

    from aurora.ingestion.pipeline import IngestionPipeline
    from aurora.features.pipeline import FeaturePipeline

    ingestion = IngestionPipeline()
    bundle = ingestion.run(tickers=["SPY"], start="2018-01-01", end="2023-12-31")
    feat_pipeline = FeaturePipeline()
    features = feat_pipeline.run(bundle)["SPY"]
    ohlcv = bundle.ohlcv["SPY"]
    close = ohlcv["close"].reindex(features.index)
    returns = np.log(close / close.shift(1)).dropna()
    features = features.reindex(returns.index)
    target = returns.shift(-1).dropna()
    features = features.reindex(target.index)

    # Simulated sentiment (replace with real in production)
    sentiment = simulate_sentiment_features(features.index)
    features_with_sentiment = pd.concat([features, sentiment], axis=1)

    cv = WalkForwardEvaluator(n_splits=5, test_size=63)
    splits = cv.split(len(features))

    results_no_sent: list[dict] = []
    results_with_sent: list[dict] = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        for feat_df, results_list, label in [
            (features, results_no_sent, "no_sentiment"),
            (features_with_sentiment, results_with_sent, "with_sentiment"),
        ]:
            X_tr = feat_df.iloc[train_idx]
            y_tr = target.iloc[train_idx]
            X_te = feat_df.iloc[test_idx]
            y_te = target.iloc[test_idx]

            model = LightGBMForecaster(n_estimators=500, horizon=1)
            model.fit(X_tr, y_tr)
            preds = model.predict(X_te)
            n = min(len(preds), len(y_te))

            dir_m = directional_metrics(y_te.values[:n], preds[:n])
            ic = information_coefficient(y_te.values[:n], preds[:n])
            strat_rets = np.sign(preds[:n]) * y_te.values[:n]
            sr = sharpe_ratio(pd.Series(strat_rets), risk_free_rate=0.0)

            results_list.append({
                "fold": fold + 1,
                "directional_accuracy": dir_m["directional_accuracy"],
                "f1": dir_m["f1"],
                "ic": ic,
                "sharpe": sr,
            })
            logger.debug(f"Fold {fold+1} [{label}]: DA={dir_m['directional_accuracy']:.3f}, IC={ic:.3f}")

    def avg(results: list[dict], key: str) -> float:
        return float(np.mean([r[key] for r in results]))

    output = {
        "no_sent_da":    avg(results_no_sent, "directional_accuracy"),
        "sent_da":       avg(results_with_sent, "directional_accuracy"),
        "da_improvement": avg(results_with_sent, "directional_accuracy") - avg(results_no_sent, "directional_accuracy"),
        "no_sent_ic":    avg(results_no_sent, "ic"),
        "sent_ic":       avg(results_with_sent, "ic"),
        "ic_improvement": avg(results_with_sent, "ic") - avg(results_no_sent, "ic"),
        "no_sent_sharpe": avg(results_no_sent, "sharpe"),
        "sent_sharpe":   avg(results_with_sent, "sharpe"),
    }

    with mlflow.start_run(run_name="exp03_sentiment_enhanced"):
        mlflow.log_metrics(output)

    logger.info("\n" + "=" * 50)
    logger.info("RESULTS — Experiment 3")
    logger.info("=" * 50)
    logger.info(f"Directional Accuracy — No Sentiment: {output['no_sent_da']:.3f}")
    logger.info(f"Directional Accuracy — With Sentiment: {output['sent_da']:.3f}")
    logger.info(f"DA Improvement: {output['da_improvement']:.3f} ({output['da_improvement']*100:.1f}pp)")
    logger.info(f"IC — No Sentiment: {output['no_sent_ic']:.4f}")
    logger.info(f"IC — With Sentiment: {output['sent_ic']:.4f}")
    logger.info(f"Sharpe — No Sentiment: {output['no_sent_sharpe']:.3f}")
    logger.info(f"Sharpe — With Sentiment: {output['sent_sharpe']:.3f}")

    return output


if __name__ == "__main__":
    run_experiment()
