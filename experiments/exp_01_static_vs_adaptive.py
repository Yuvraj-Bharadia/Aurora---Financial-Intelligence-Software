"""
Experiment 1: Static vs Regime-Adaptive Forecasting.

Research Question:
    Does conditioning forecasts on hidden market regimes yield statistically
    significant improvements over a static (regime-agnostic) model?

Methodology:
    - Walk-forward evaluation on SPY daily returns (2010–2023).
    - Baseline: XGBoost trained on all data uniformly (static).
    - Aurora: RegimeAdaptiveEnsemble (HMM regimes + per-regime model weights).
    - Compare with Diebold-Mariano test for equal predictive accuracy.

Hypothesis:
    H1: The regime-adaptive ensemble produces lower RMSE and higher Sharpe
        ratio than the static model across the full out-of-sample period.
"""

from __future__ import annotations

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import mlflow

from aurora.configs.config import settings
from aurora.evaluation.metrics import (
    forecast_metrics, trading_metrics, diebold_mariano_test, sharpe_ratio
)
from aurora.evaluation.evaluator import WalkForwardEvaluator
from aurora.features.pipeline import FeaturePipeline
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.models.ml.gradient_boosting import XGBoostForecaster, LightGBMForecaster
from aurora.models.econometric.arima import ARIMAForecaster
from aurora.regimes.detector import RegimeDetector
from aurora.ensembles.regime_adaptive import RegimeAdaptiveEnsemble
from aurora.utils.logging import get_logger

logger = get_logger(__name__)

TICKER = "SPY"
START = "2010-01-01"
END = "2023-12-31"
N_SPLITS = 8
TEST_SIZE = 63  # one quarter


def run_experiment() -> dict:
    logger.info("=" * 60)
    logger.info("Experiment 1: Static vs Adaptive Forecasting")
    logger.info("=" * 60)

    # ── Data ──────────────────────────────────────────────────────────────────
    logger.info("Loading data…")
    ingestion = IngestionPipeline()
    bundle = ingestion.run(tickers=[TICKER], start=START, end=END)
    feat_pipeline = FeaturePipeline()
    features = feat_pipeline.run(bundle)[TICKER]
    ohlcv = bundle.ohlcv[TICKER]
    close = ohlcv["close"].reindex(features.index)
    returns = np.log(close / close.shift(1)).dropna().rename("returns")
    features = features.reindex(returns.index)
    feat_cols = list(features.columns)
    target = returns.shift(-1).dropna()
    X = features.reindex(target.index)

    logger.info(f"Dataset: {len(X)} rows × {len(feat_cols)} features")

    # ── Walk-forward evaluation ────────────────────────────────────────────────
    cv = WalkForwardEvaluator(n_splits=N_SPLITS, test_size=TEST_SIZE, expanding=True)
    splits = cv.split(len(X))

    static_errors: list[np.ndarray] = []
    adaptive_errors: list[np.ndarray] = []
    static_returns: list[np.ndarray] = []
    adaptive_returns: list[np.ndarray] = []

    for fold, (train_idx, test_idx) in enumerate(splits):
        X_tr, y_tr = X.iloc[train_idx], target.iloc[train_idx]
        X_te, y_te = X.iloc[test_idx], target.iloc[test_idx]
        ret_tr = returns.iloc[train_idx]

        logger.info(f"Fold {fold+1}/{N_SPLITS}: train={len(train_idx)}, test={len(test_idx)}")

        # ── Static model ──────────────────────────────────────────────────────
        static = XGBoostForecaster(n_estimators=300, horizon=1)
        static.fit(X_tr, y_tr)
        static_preds = static.predict(X_te)
        n = min(len(static_preds), len(y_te))
        static_errors.append(y_te.values[:n] - static_preds[:n])
        static_ret = np.sign(static_preds[:n]) * y_te.values[:n]
        static_returns.append(static_ret)

        # ── Regime-adaptive model ─────────────────────────────────────────────
        detector = RegimeDetector(method="hmm", n_regimes=4)
        detector.fit(X_tr, ret_tr)
        regimes_tr = detector.predict(X_tr, ret_tr)

        ensemble = RegimeAdaptiveEnsemble(detector)
        xgb2 = XGBoostForecaster(n_estimators=300, horizon=1)
        lgbm = LightGBMForecaster(n_estimators=300, horizon=1)
        arima = ARIMAForecaster(horizon=1)

        xgb2.fit(X_tr, y_tr)
        lgbm.fit(X_tr, y_tr)
        arima.fit(pd.DataFrame(index=y_tr.index), y_tr)

        ensemble.register("xgboost", xgb2)
        ensemble.register("lightgbm", lgbm, regimes=["bull", "sideways"])
        ensemble.register("arima", arima, regimes=["sideways"])
        ensemble.fit_weights(X_tr, y_tr, regimes_tr)

        regimes_te = detector.predict(X_te, returns.iloc[test_idx])
        adaptive_fc = ensemble.predict(X_te, regimes_te)
        adaptive_preds = adaptive_fc["forecast"].reindex(X_te.index).ffill().values
        n = min(len(adaptive_preds), len(y_te))
        adaptive_errors.append(y_te.values[:n] - adaptive_preds[:n])
        adaptive_ret = np.sign(adaptive_preds[:n]) * y_te.values[:n]
        adaptive_returns.append(adaptive_ret)

    # ── Aggregate results ─────────────────────────────────────────────────────
    all_static_e = np.concatenate(static_errors)
    all_adaptive_e = np.concatenate(adaptive_errors)
    all_actual = np.concatenate([e + p for e, p in zip(
        static_errors, [np.sign(r) * r for r in static_returns]
    )])

    static_metrics = forecast_metrics(
        all_actual, all_actual - all_static_e
    )
    adaptive_metrics = forecast_metrics(
        all_actual, all_actual - all_adaptive_e
    )

    static_sharpe = sharpe_ratio(
        pd.Series(np.concatenate(static_returns))
    )
    adaptive_sharpe = sharpe_ratio(
        pd.Series(np.concatenate(adaptive_returns))
    )

    dm = diebold_mariano_test(all_static_e**2, all_adaptive_e**2)

    results = {
        "static_rmse":      static_metrics["rmse"],
        "adaptive_rmse":    adaptive_metrics["rmse"],
        "rmse_improvement": (static_metrics["rmse"] - adaptive_metrics["rmse"]) / static_metrics["rmse"],
        "static_sharpe":    static_sharpe,
        "adaptive_sharpe":  adaptive_sharpe,
        "dm_stat":          dm["dm_stat"],
        "dm_p_value":       dm["p_value"],
        "statistically_significant": dm["p_value"] < 0.05,
    }

    # ── Log to MLflow ─────────────────────────────────────────────────────────
    with mlflow.start_run(run_name="exp01_static_vs_adaptive"):
        mlflow.log_params({"ticker": TICKER, "start": START, "end": END, "n_splits": N_SPLITS})
        mlflow.log_metrics(results)

    # ── Print summary ─────────────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("RESULTS — Experiment 1")
    logger.info("=" * 50)
    logger.info(f"Static   RMSE:   {results['static_rmse']:.6f}")
    logger.info(f"Adaptive RMSE:   {results['adaptive_rmse']:.6f}")
    logger.info(f"RMSE Improvement: {results['rmse_improvement']:.1%}")
    logger.info(f"Static   Sharpe: {results['static_sharpe']:.3f}")
    logger.info(f"Adaptive Sharpe: {results['adaptive_sharpe']:.3f}")
    logger.info(f"DM Test: stat={results['dm_stat']:.3f}, p={results['dm_p_value']:.4f}")
    conclusion = (
        "✅ Regime-adaptive model is STATISTICALLY SIGNIFICANTLY better (p<0.05)"
        if results["statistically_significant"]
        else "❌ No statistically significant improvement detected"
    )
    logger.info(conclusion)

    return results


if __name__ == "__main__":
    results = run_experiment()
