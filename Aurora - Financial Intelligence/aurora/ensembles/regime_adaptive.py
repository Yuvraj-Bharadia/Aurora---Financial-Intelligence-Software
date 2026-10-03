"""
Regime-Adaptive Ensemble — the core Aurora research contribution.

This module implements dynamic model switching conditioned on the detected
market regime. The ensemble:

1. Maintains a registry of specialist forecasters, each tuned for a specific
   market regime (bull, bear, high-volatility, sideways).

2. At inference, queries the RegimeDetector for the current regime and
   confidence score.

3. Weights models by:
   - Regime alignment (primary weight: did you win for this regime?)
   - Recent out-of-sample performance (rolling Sharpe / RMSE)
   - Confidence decay for low-certainty regime calls

4. Combines weighted predictions into a final ensemble forecast.

The regime → model mapping is learned from walk-forward validation rather
than hard-coded, ensuring empirical rather than assumed relationships.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from scipy.special import softmax

from aurora.configs.config import settings
from aurora.models.base import BaseForecaster
from aurora.regimes.detector import RegimeDetector
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class RegimeAdaptiveEnsemble:
    """
    Dynamic, regime-conditioned ensemble of forecasting models.

    Usage::

        ensemble = RegimeAdaptiveEnsemble(regime_detector)
        ensemble.register("arima", arima_model, regimes=["sideways"])
        ensemble.register("lstm", lstm_model, regimes=["bull", "bear"])
        ensemble.register("transformer", tfm_model)  # active in all regimes
        ensemble.fit_weights(features, returns, regimes_df)
        forecast = ensemble.predict(features, regimes_df)
    """

    def __init__(
        self,
        regime_detector: RegimeDetector,
        weight_lookback: int | None = None,
        min_weight: float | None = None,
        confidence_scaling: bool = True,
    ) -> None:
        self.regime_detector = regime_detector
        self.weight_lookback = weight_lookback or settings.ensemble.weight_lookback
        self.min_weight = min_weight or settings.ensemble.min_model_weight
        self.confidence_scaling = confidence_scaling

        self._models: dict[str, BaseForecaster] = {}
        self._model_regimes: dict[str, list[str]] = {}  # model → active regimes
        self._regime_weights: dict[str, dict[str, float]] = {}  # regime → model → weight
        self._performance_history: dict[str, list[float]] = {}  # model → [errors]

    # ── Model Registration ─────────────────────────────────────────────────────

    def register(
        self,
        name: str,
        model: BaseForecaster,
        regimes: list[str] | None = None,
    ) -> "RegimeAdaptiveEnsemble":
        """
        Register a forecaster as an ensemble component.

        Args:
            name: Unique identifier for this model.
            model: Fitted BaseForecaster instance.
            regimes: Regime names this model is preferred for.
                     None → active in all regimes.
        """
        self._models[name] = model
        self._model_regimes[name] = regimes or list(settings.regimes.regime_labels)
        self._performance_history[name] = []
        logger.info(f"[Ensemble] Registered '{name}' for regimes: {self._model_regimes[name]}")
        return self

    # ── Weight Learning ────────────────────────────────────────────────────────

    def fit_weights(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        regimes: pd.DataFrame,
    ) -> "RegimeAdaptiveEnsemble":
        """
        Learn per-regime model weights from walk-forward errors.

        For each regime, computes each model's historical RMSE within that
        regime period and converts to inverse-error weights via softmax.
        """
        aligned = features.join(regimes[["regime"]], how="inner")
        aligned["target"] = returns.shift(-1).reindex(aligned.index)
        aligned = aligned.dropna(subset=["target"])

        for regime_name in settings.regimes.regime_labels:
            regime_mask = aligned["regime"] == regime_name
            if regime_mask.sum() < 20:
                logger.warning(f"Too few '{regime_name}' samples for weight learning")
                continue

            regime_data = aligned[regime_mask]
            feat_cols = [c for c in regime_data.columns if c not in ["regime", "target"]]
            X_reg = regime_data[feat_cols]
            y_reg = regime_data["target"]

            errors: dict[str, float] = {}
            for model_name, model in self._models.items():
                if regime_name not in self._model_regimes[model_name]:
                    errors[model_name] = 1e9  # penalize irrelevant models
                    continue
                try:
                    preds = model.predict(X_reg)
                    n = min(len(preds), len(y_reg))
                    rmse = np.sqrt(np.mean((y_reg.values[:n] - preds[:n]) ** 2))
                    errors[model_name] = rmse
                except Exception as exc:
                    logger.warning(f"[Ensemble] {model_name} error in {regime_name}: {exc}")
                    errors[model_name] = 1e9

            # Convert RMSE → weights via softmax of negative errors
            model_names = list(errors.keys())
            err_arr = np.array([errors[m] for m in model_names])
            # Replace very large errors with max of finite values
            finite_max = err_arr[np.isfinite(err_arr)].max() if np.isfinite(err_arr).any() else 1.0
            err_arr = np.clip(err_arr, 0, finite_max * 10)
            weights = softmax(-err_arr / (err_arr.std() + 1e-9))
            weights = np.maximum(weights, self.min_weight)
            weights /= weights.sum()

            self._regime_weights[regime_name] = dict(zip(model_names, weights))
            logger.info(f"[Ensemble] {regime_name} weights: "
                        + ", ".join(f"{m}={w:.3f}" for m, w in
                                    sorted(self._regime_weights[regime_name].items(),
                                           key=lambda x: x[1], reverse=True)))

        return self

    def fit_weights_oof(
        self,
        features: pd.DataFrame,
        returns: pd.Series,
        regimes: pd.DataFrame,
        n_splits: int = 5,
        test_size: int = 63,
    ) -> "RegimeAdaptiveEnsemble":
        """
        Layer 4: learn ensemble weights using OOFStore + OOFRouter.

        Trains each model on walk-forward in-sample folds and stores
        out-of-fold predictions, then trains a GatingMLP that combines
        regime posterior, confidence, and transition risk to route
        allocation across models with bounded turnover.

        Falls back to fit_weights() if OOF infrastructure unavailable.
        """
        try:
            from aurora.ensembles.oof_router import OOFRouter, OOFStore
            from aurora.evaluation.evaluator import WalkForwardEvaluator
        except ImportError:
            logger.warning("[Ensemble] OOFRouter unavailable — falling back to fit_weights()")
            return self.fit_weights(features, returns, regimes)

        feat_cols = [c for c in features.columns if c not in ("regime", "regime_confidence")]
        X = features[feat_cols].ffill().bfill()
        y = returns.shift(-1).dropna()
        X = X.reindex(y.index)

        cv = WalkForwardEvaluator(n_splits=n_splits, test_size=test_size, expanding=True)
        splits = cv.split(len(X))

        store = OOFStore(model_names=list(self._models.keys()))
        model_names = list(self._models.keys())

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr, y_tr = X.iloc[train_idx], y.iloc[train_idx]
            X_te = X.iloc[test_idx]
            actuals = y.iloc[test_idx].values

            preds_dict: dict[str, np.ndarray] = {}
            for name, model in self._models.items():
                try:
                    model.fit(X_tr, y_tr)
                    p = model.predict(X_te)
                    preds_dict[name] = np.asarray(p)[:len(actuals)]
                except Exception as exc:
                    logger.warning(f"[Ensemble-OOF] {name} fold {fold_idx}: {exc}")
                    preds_dict[name] = np.zeros(len(actuals))

            # Extract regime state vectors for this fold if available
            svs = None
            reg_slice = regimes.reindex(X_te.index)
            if "transition_risk" in reg_slice.columns:
                svs = reg_slice

            store.add_fold(fold_idx, preds_dict, actuals, svs)

        self._oof_router = OOFRouter(
            store=store,
            model_names=model_names,
        )
        self._oof_router.fit()
        logger.info("[Ensemble] OOFRouter fitted on walk-forward out-of-fold predictions")
        return self

    # ── Prediction ─────────────────────────────────────────────────────────────

    def predict(
        self,
        features: pd.DataFrame,
        regimes: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Produce regime-conditioned ensemble forecasts.

        Args:
            features: Feature matrix for the forecast period.
            regimes: Output of RegimeDetector.predict() aligned to features.

        Returns:
            DataFrame with columns:
            - forecast: Weighted ensemble point prediction
            - forecast_lower, forecast_upper: Uncertainty bounds
            - dominant_model: Model with highest weight in current regime
            - regime: Current regime label
        """
        results: list[dict[str, Any]] = []

        for idx in features.index:
            if idx not in regimes.index:
                continue

            regime = str(regimes.loc[idx, "regime"])
            confidence = float(regimes.loc[idx].get("regime_confidence", 1.0))
            X_row = features.loc[[idx]]

            weighted_pred, lo, hi, dominant = self._weighted_prediction(
                X_row, regime, confidence
            )

            results.append({
                "timestamp": idx,
                "forecast": weighted_pred,
                "forecast_lower": lo,
                "forecast_upper": hi,
                "dominant_model": dominant,
                "regime": regime,
                "regime_confidence": confidence,
            })

        df = pd.DataFrame(results).set_index("timestamp")
        return df

    def _weighted_prediction(
        self,
        X: pd.DataFrame,
        regime: str,
        confidence: float,
    ) -> tuple[float, float, float, str]:
        """Compute the weighted ensemble prediction for one row."""
        weights = self._regime_weights.get(regime, {})

        # Fall back to equal weights if regime not seen during training
        if not weights:
            weights = {m: 1.0 / len(self._models) for m in self._models}

        if self.confidence_scaling:
            # Scale down weights toward uniform when confidence is low
            uniform = {m: 1.0 / len(self._models) for m in self._models}
            weights = {
                m: confidence * weights.get(m, 1.0 / len(self._models))
                   + (1 - confidence) * uniform[m]
                for m in self._models
            }

        predictions: dict[str, float] = {}
        lowers: dict[str, float] = {}
        uppers: dict[str, float] = {}

        for model_name, model in self._models.items():
            try:
                pred = float(model.predict(X)[0])
                predictions[model_name] = pred
                if model.supports_intervals:
                    lo_arr, hi_arr = model.predict_interval(X)
                    lowers[model_name] = float(lo_arr[0])
                    uppers[model_name] = float(hi_arr[0])
                else:
                    lowers[model_name] = pred
                    uppers[model_name] = pred
            except Exception as exc:
                logger.debug(f"[Ensemble] {model_name} predict failed: {exc}")

        if not predictions:
            return 0.0, 0.0, 0.0, "none"

        # Normalize weights to available models
        avail_weights = {m: weights.get(m, 0.0) for m in predictions}
        total_w = sum(avail_weights.values())
        if total_w < 1e-9:
            avail_weights = {m: 1.0 / len(predictions) for m in predictions}
            total_w = 1.0

        ensemble_pred = sum(
            (avail_weights[m] / total_w) * predictions[m] for m in predictions
        )
        ensemble_lo = sum(
            (avail_weights[m] / total_w) * lowers.get(m, predictions[m]) for m in predictions
        )
        ensemble_hi = sum(
            (avail_weights[m] / total_w) * uppers.get(m, predictions[m]) for m in predictions
        )
        dominant = max(avail_weights, key=avail_weights.get)
        return ensemble_pred, ensemble_lo, ensemble_hi, dominant

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"[Ensemble] Saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "RegimeAdaptiveEnsemble":
        return joblib.load(path)

    # ── Diagnostics ────────────────────────────────────────────────────────────

    def weight_summary(self) -> pd.DataFrame:
        """Display the learned per-regime model weights as a pivot table."""
        rows = []
        for regime, weights in self._regime_weights.items():
            for model, w in weights.items():
                rows.append({"regime": regime, "model": model, "weight": w})
        if not rows:
            return pd.DataFrame()
        return (
            pd.DataFrame(rows)
            .pivot(index="regime", columns="model", values="weight")
            .fillna(0.0)
            .round(3)
        )
