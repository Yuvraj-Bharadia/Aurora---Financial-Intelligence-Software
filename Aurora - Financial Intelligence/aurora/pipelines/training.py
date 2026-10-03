"""
End-to-end Aurora training pipeline.

Orchestrates the full model training workflow:
1. Data ingestion
2. Feature engineering
3. Regime detection model training
4. Individual forecaster training (all models)
5. Ensemble weight learning
6. Walk-forward evaluation
7. Experiment logging to MLflow
8. Model persistence

Designed to be run from the CLI or as part of a scheduled job.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd

from aurora.backtesting.engine import BacktestEngine
from aurora.configs.config import settings
from aurora.data.schemas import MarketDataBundle
from aurora.ensembles.regime_adaptive import RegimeAdaptiveEnsemble
from aurora.evaluation.evaluator import ModelEvaluator
from aurora.features.pipeline import FeaturePipeline
from aurora.ingestion.pipeline import IngestionPipeline
from aurora.models.deep_learning.lstm import LSTMForecaster
from aurora.models.deep_learning.transformer import TransformerForecaster

try:
    from aurora.models.deep_learning.qcpac import QCPACForecaster
    _QCPAC_AVAILABLE = True
except ImportError:
    _QCPAC_AVAILABLE = False
from aurora.models.econometric.arima import ARIMAForecaster
from aurora.models.econometric.garch import GARCHForecaster
from aurora.models.ml.gradient_boosting import (
    LightGBMForecaster,
    RandomForestForecaster,
    XGBoostForecaster,
)
from aurora.regimes.detector import RegimeDetector
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Full Aurora training pipeline with MLflow experiment tracking.

    Args:
        ticker: Primary ticker to train on.
        start: Training data start date.
        end: Training data end date.
        model_dir: Directory to save fitted models.
        experiment_name: MLflow experiment name.
    """

    def __init__(
        self,
        ticker: str = "SPY",
        start: str | None = None,
        end: str | None = None,
        model_dir: Path | None = None,
        experiment_name: str | None = None,
        run_name: str | None = None,
    ) -> None:
        self.ticker = ticker
        self.start = start or settings.data.default_start
        self.end = end or settings.data.default_end
        self.model_dir = model_dir or (settings.models.model_dir / ticker)
        self.experiment_name = experiment_name or settings.env
        self.run_name = run_name or f"{ticker}_{int(time.time())}"
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Artefact storage
        self._bundle: MarketDataBundle | None = None
        self._features: pd.DataFrame | None = None
        self._returns: pd.Series | None = None
        self._regimes: pd.DataFrame | None = None
        self._models: dict[str, Any] = {}
        self._ensemble: RegimeAdaptiveEnsemble | None = None

    # ── Main entrypoint ────────────────────────────────────────────────────────

    def run(
        self,
        force_refresh: bool = False,
        train_deep_learning: bool = True,
        n_eval_splits: int = 5,
    ) -> dict[str, Any]:
        """
        Execute the complete training pipeline.

        Args:
            force_refresh: Re-ingest data even if cached.
            train_deep_learning: Whether to train LSTM / Transformer models.
            n_eval_splits: Walk-forward folds for evaluation.

        Returns:
            Dict with evaluation metrics and artifact paths.
        """
        mlflow.set_tracking_uri(settings.env)
        mlflow.set_experiment(self.experiment_name)

        with mlflow.start_run(run_name=self.run_name) as run:
            run_id = run.info.run_id
            logger.info(f"MLflow run started: {run_id}")

            mlflow.log_params({
                "ticker": self.ticker,
                "start": self.start,
                "end": self.end,
                "n_eval_splits": n_eval_splits,
                "train_deep_learning": train_deep_learning,
            })

            # ── 1. Data ────────────────────────────────────────────────────
            t0 = time.perf_counter()
            self._ingest(force_refresh)
            mlflow.log_metric("ingestion_seconds", time.perf_counter() - t0)

            # ── 2. Features ────────────────────────────────────────────────
            t0 = time.perf_counter()
            self._build_features()
            mlflow.log_metric("feature_seconds", time.perf_counter() - t0)
            mlflow.log_metric("n_features", self._features.shape[1])
            mlflow.log_metric("n_samples", len(self._features))

            # ── 3. Regime detection ────────────────────────────────────────
            t0 = time.perf_counter()
            self._train_regime_detector()
            mlflow.log_metric("regime_seconds", time.perf_counter() - t0)

            regime_summary = self._regime_detector.regime_summary(self._regimes)
            for regime_name, row in regime_summary.iterrows():
                mlflow.log_metric(f"regime_pct_{regime_name}", float(row.get("pct_time", 0)))

            # ── 4. Individual models ───────────────────────────────────────
            t0 = time.perf_counter()
            self._train_models(train_deep_learning)
            mlflow.log_metric("model_training_seconds", time.perf_counter() - t0)

            # ── 5. Ensemble ────────────────────────────────────────────────
            self._build_ensemble()

            # ── 6. Evaluation ──────────────────────────────────────────────
            metrics = self._evaluate(n_eval_splits)
            mlflow.log_metrics(metrics)

            # ── 7. Backtesting ─────────────────────────────────────────────
            bt_metrics = self._backtest()
            mlflow.log_metrics({f"bt_{k}": v for k, v in bt_metrics.items()})

            # ── 8. Save ────────────────────────────────────────────────────
            self._save_artefacts()
            mlflow.log_artifacts(str(self.model_dir))

            logger.info(f"Training complete | Sharpe={bt_metrics.get('sharpe', 0):.3f}")

        return {"metrics": metrics, "backtest": bt_metrics, "run_id": run_id}

    # ── Pipeline stages ────────────────────────────────────────────────────────

    def _ingest(self, force_refresh: bool) -> None:
        logger.info(f"[Pipeline] Ingesting {self.ticker}")
        pipeline = IngestionPipeline()
        self._bundle = pipeline.run(
            tickers=[self.ticker] + settings.data.equity_tickers[:5],
            start=self.start,
            end=self.end,
            force_refresh=force_refresh,
        )

    def _build_features(self) -> None:
        logger.info("[Pipeline] Engineering features")
        feat_pipeline = FeaturePipeline()
        feature_map = feat_pipeline.run(self._bundle)
        if self.ticker not in feature_map:
            raise ValueError(f"Features not generated for {self.ticker}")
        self._features = feature_map[self.ticker]
        ohlcv = self._bundle.ohlcv[self.ticker]
        close = ohlcv["close"].reindex(self._features.index)
        self._returns = np.log(close / close.shift(1)).rename("returns").dropna()

    def _train_regime_detector(self) -> None:
        logger.info("[Pipeline] Training regime detector")
        feat_aligned = self._features.reindex(self._returns.index)
        self._regime_detector = RegimeDetector(method="ensemble")
        self._regime_detector.fit(feat_aligned, self._returns)
        self._regimes = self._regime_detector.predict(feat_aligned, self._returns)
        self._features = self._features.join(
            self._regimes[["regime", "regime_confidence"]], how="left"
        )

    def _train_models(self, train_deep_learning: bool) -> None:
        logger.info("[Pipeline] Training individual models")
        feat_cols = [c for c in self._features.columns
                     if c not in ("regime", "regime_confidence")]
        X = self._features[feat_cols].reindex(self._returns.index).ffill().bfill().dropna()
        y = self._returns.reindex(X.index).shift(-1).dropna()
        X = X.reindex(y.index)

        train_end = int(len(X) * 0.8)
        X_tr, y_tr = X.iloc[:train_end], y.iloc[:train_end]
        X_val, y_val = X.iloc[train_end:], y.iloc[train_end:]

        # ── Econometric ────────────────────────────────────────────────────
        arima = ARIMAForecaster(horizon=1)
        arima.fit(pd.DataFrame(index=y_tr.index), y_tr)
        self._models["arima"] = arima

        garch = GARCHForecaster(horizon=1)
        garch.fit(pd.DataFrame(index=y_tr.index), y_tr)
        self._models["garch"] = garch

        # ── Machine learning ───────────────────────────────────────────────
        xgb = XGBoostForecaster(horizon=1)
        xgb.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        self._models["xgboost"] = xgb

        lgbm = LightGBMForecaster(horizon=1)
        lgbm.fit(X_tr, y_tr, eval_set=(X_val, y_val))
        self._models["lightgbm"] = lgbm

        rf = RandomForestForecaster(horizon=1)
        rf.fit(X_tr, y_tr)
        self._models["random_forest"] = rf

        # ── Deep learning ──────────────────────────────────────────────────
        if train_deep_learning:
            lstm = LSTMForecaster(horizon=1, cell_type="LSTM", max_epochs=50)
            lstm.fit(X_tr, y_tr)
            self._models["lstm"] = lstm

            transformer = TransformerForecaster(horizon=1, max_epochs=50)
            transformer.fit(X_tr, y_tr)
            self._models["transformer"] = transformer

            # Layer 2+3: QCPAC — joint HMM + HRT + physics-informed training
            if _QCPAC_AVAILABLE:
                try:
                    macro_cols = [c for c in X_tr.columns if any(
                        k in c for k in ("vix", "yield", "spread", "cpi", "unemployment")
                    )]
                    macro_tr = X_tr[macro_cols] if macro_cols else None
                    qcpac = QCPACForecaster(
                        n_regimes=settings.regimes.n_regimes,
                        n_qubits=32,
                        seq_len=20,
                        max_epochs=50,
                    )
                    qcpac.fit(X_tr, y_tr, macro_df=macro_tr)
                    self._models["qcpac"] = qcpac
                    logger.info("[Pipeline] QCPAC trained successfully")
                except Exception as exc:
                    logger.warning(f"[Pipeline] QCPAC training failed (non-fatal): {exc}")

        logger.info(f"[Pipeline] Trained {len(self._models)} models")

    def _build_ensemble(self) -> None:
        logger.info("[Pipeline] Building regime-adaptive ensemble")
        feat_cols = [c for c in self._features.columns
                     if c not in ("regime", "regime_confidence")]
        X = self._features[feat_cols].reindex(self._returns.index).ffill().bfill()
        y = self._returns.shift(-1).dropna()
        X = X.reindex(y.index)

        self._ensemble = RegimeAdaptiveEnsemble(self._regime_detector)
        for name, model in self._models.items():
            regimes = {
                "arima": ["sideways"],
                "garch": ["high_volatility", "bear"],
                "xgboost": ["bull", "sideways"],
                "lightgbm": ["bull", "bear"],
                "random_forest": None,
                "lstm": ["bull", "bear"],
                "transformer": ["high_volatility"],
                "qcpac": None,  # active in all regimes — learns per-regime via HRT
            }
            self._ensemble.register(name, model, regimes=regimes.get(name))

        self._ensemble.fit_weights(X, y, self._regimes)

    def _evaluate(self, n_splits: int) -> dict[str, float]:
        logger.info("[Pipeline] Running walk-forward evaluation")
        feat_cols = [c for c in self._features.columns
                     if c not in ("regime", "regime_confidence")]
        X = self._features[feat_cols].reindex(self._returns.index).ffill().bfill()
        y = self._returns.shift(-1).dropna()
        X = X.reindex(y.index)

        evaluator = ModelEvaluator(n_splits=n_splits)
        summary = evaluator.evaluate(
            self._models, X, y, regimes=self._regimes
        )
        if summary.empty:
            return {}

        best_model = summary["rmse_mean"].idxmin()
        metrics = {
            f"{col}_{best_model}": float(summary.loc[best_model, col])
            for col in summary.columns if "_mean" in col
        }
        return metrics

    def _backtest(self) -> dict[str, float]:
        logger.info("[Pipeline] Running backtest")
        ohlcv = self._bundle.ohlcv.get(self.ticker, pd.DataFrame())
        if ohlcv.empty:
            return {}

        feat_cols = [c for c in self._features.columns
                     if c not in ("regime", "regime_confidence")]
        X = self._features[feat_cols].reindex(self._returns.index).ffill().bfill()

        forecasts = self._ensemble.predict(X, self._regimes)
        if forecasts.empty:
            return {}

        engine = BacktestEngine()
        signal = forecasts["forecast"].dropna()
        prices = ohlcv["close"].reindex(signal.index)

        # Layer 1: apply NoTradeGate multipliers if available
        gate_series = None
        if "size_multiplier" in self._regimes.columns:
            gate_series = self._regimes["size_multiplier"].reindex(signal.index)
        elif hasattr(self._regime_detector, "_no_trade_gate"):
            try:
                online_df = self._regime_detector.predict_online(
                    self._features.reindex(self._returns.index)
                )
                if "size_multiplier" in online_df.columns:
                    gate_series = online_df["size_multiplier"].reindex(signal.index)
            except Exception:
                pass

        result = engine.run(
            signal, prices,
            strategy="long_short",
            regimes=self._regimes,
            no_trade_gate_series=gate_series,
        )
        return result.metrics

    def _save_artefacts(self) -> None:
        logger.info(f"[Pipeline] Saving artefacts to {self.model_dir}")
        self._regime_detector.save(self.model_dir / "regime_detector.pkl")
        self._ensemble.save(self.model_dir / "ensemble.pkl")
        for name, model in self._models.items():
            try:
                model.save(self.model_dir / f"{name}.pkl")
            except Exception as exc:
                logger.warning(f"Could not save {name}: {exc}")


def main() -> None:
    """CLI entrypoint: python -m aurora.pipelines.training."""
    import typer

    app = typer.Typer()

    @app.command()
    def train(
        ticker: str = typer.Option("SPY", help="Ticker to train on"),
        start: str = typer.Option("2010-01-01"),
        end: str = typer.Option("2023-12-31"),
        deep_learning: bool = typer.Option(True, help="Train deep learning models"),
        force_refresh: bool = typer.Option(False, help="Re-ingest data"),
    ) -> None:
        pipeline = TrainingPipeline(ticker=ticker, start=start, end=end)
        result = pipeline.run(
            force_refresh=force_refresh,
            train_deep_learning=deep_learning,
        )
        typer.echo(f"Training complete: {result['run_id']}")

    app()


if __name__ == "__main__":
    main()
