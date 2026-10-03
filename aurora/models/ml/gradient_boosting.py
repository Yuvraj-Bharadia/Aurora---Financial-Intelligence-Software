"""
Gradient boosting forecasters: XGBoost, LightGBM, CatBoost.

All three models share a common interface inheriting BaseForecaster.
They support SHAP-based feature importance for interpretability.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import Any, Literal

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from aurora.configs.config import settings
from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class XGBoostForecaster(BaseForecaster):
    """XGBoost regression forecaster with early stopping."""

    name = "xgboost"
    supports_intervals = True  # via quantile regression

    def __init__(
        self,
        horizon: int = 1,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        subsample: float | None = None,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int | None = None,
        scale_features: bool = False,
        objective: str = "reg:squarederror",
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        cfg = settings.models
        self.n_estimators = n_estimators or cfg.gbm_n_estimators
        self.learning_rate = learning_rate or cfg.gbm_learning_rate
        self.max_depth = max_depth or cfg.gbm_max_depth
        self.subsample = subsample or cfg.gbm_subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds or cfg.gbm_early_stopping_rounds
        self.scale_features = scale_features
        self.objective = objective
        self._model: Any = None
        self._scaler: StandardScaler | None = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        **kwargs: Any,
    ) -> "XGBoostForecaster":
        try:
            import xgboost as xgb
        except ImportError as e:
            raise ImportError("xgboost not installed") from e

        self._feature_names = list(X.columns)
        X_arr, y_arr = self._prepare(X, y, fit_scaler=True)

        eval_list = None
        if eval_set is not None:
            X_val, y_val = eval_set
            X_val_arr = self._prepare(X_val, y_val, fit_scaler=False)[0]
            eval_list = [(X_arr, y_arr), (X_val_arr, y_val.values)]

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            objective=self.objective,
            early_stopping_rounds=self.early_stopping_rounds if eval_list else None,
            random_state=42,
            n_jobs=-1,
            verbosity=0,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X_arr, y_arr,
                eval_set=eval_list,
                verbose=False,
            )

        best = getattr(self._model, "best_iteration", self.n_estimators)
        logger.info(f"[XGBoost] Fitted: best_iteration={best}")
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        X_arr = self._prepare(X, fit_scaler=False)[0]
        return self._model.predict(X_arr)

    def feature_importance(self, importance_type: str = "gain") -> pd.Series:
        """SHAP-compatible feature importance."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        scores = self._model.get_booster().get_score(importance_type=importance_type)
        return (
            pd.Series(scores)
            .reindex(self._feature_names)
            .fillna(0)
            .sort_values(ascending=False)
        )

    def shap_values(self, X: pd.DataFrame) -> np.ndarray:
        """Compute SHAP values for model interpretability."""
        try:
            import shap
            explainer = shap.TreeExplainer(self._model)
            X_arr = self._prepare(X, fit_scaler=False)[0]
            return explainer.shap_values(X_arr)
        except ImportError:
            raise ImportError("shap not installed: pip install shap")

    def _prepare(
        self,
        X: pd.DataFrame,
        y: pd.Series | None = None,
        fit_scaler: bool = False,
    ) -> tuple[np.ndarray, np.ndarray]:
        X_vals = X[self._feature_names].values if self._feature_names else X.values
        if self.scale_features:
            if fit_scaler:
                self._scaler = StandardScaler()
                X_vals = self._scaler.fit_transform(X_vals)
            elif self._scaler is not None:
                X_vals = self._scaler.transform(X_vals)
        y_vals = y.values if y is not None else np.array([])
        return X_vals, y_vals

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "XGBoostForecaster":
        return joblib.load(path)


class LightGBMForecaster(BaseForecaster):
    """LightGBM forecaster — faster than XGBoost on large feature sets."""

    name = "lightgbm"
    supports_intervals = True

    def __init__(
        self,
        horizon: int = 1,
        n_estimators: int | None = None,
        learning_rate: float | None = None,
        max_depth: int | None = None,
        num_leaves: int = 63,
        subsample: float | None = None,
        colsample_bytree: float = 0.8,
        early_stopping_rounds: int | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        cfg = settings.models
        self.n_estimators = n_estimators or cfg.gbm_n_estimators
        self.learning_rate = learning_rate or cfg.gbm_learning_rate
        self.max_depth = max_depth or cfg.gbm_max_depth
        self.num_leaves = num_leaves
        self.subsample = subsample or cfg.gbm_subsample
        self.colsample_bytree = colsample_bytree
        self.early_stopping_rounds = early_stopping_rounds or cfg.gbm_early_stopping_rounds
        self._model: Any = None
        self._feature_names: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        eval_set: tuple[pd.DataFrame, pd.Series] | None = None,
        **kwargs: Any,
    ) -> "LightGBMForecaster":
        try:
            import lightgbm as lgb
        except ImportError as e:
            raise ImportError("lightgbm not installed") from e

        self._feature_names = list(X.columns)
        callbacks = [lgb.early_stopping(self.early_stopping_rounds, verbose=False),
                     lgb.log_evaluation(-1)]

        eval_data = None
        if eval_set is not None:
            X_val, y_val = eval_set
            eval_data = [(X_val[self._feature_names], y_val)]

        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            max_depth=self.max_depth,
            num_leaves=self.num_leaves,
            subsample=self.subsample,
            colsample_bytree=self.colsample_bytree,
            random_state=42,
            n_jobs=-1,
            verbose=-1,
        )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self._model.fit(
                X[self._feature_names], y,
                eval_set=eval_data,
                callbacks=callbacks if eval_data else [lgb.log_evaluation(-1)],
            )

        logger.info(f"[LightGBM] Fitted: best_iteration={self._model.best_iteration_}")
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self._model.predict(X[self._feature_names])

    def feature_importance(self) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return (
            pd.Series(self._model.feature_importances_, index=self._feature_names)
            .sort_values(ascending=False)
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "LightGBMForecaster":
        return joblib.load(path)


class RandomForestForecaster(BaseForecaster):
    """Scikit-learn Random Forest with conformal prediction intervals."""

    name = "random_forest"
    supports_intervals = True

    def __init__(
        self,
        horizon: int = 1,
        n_estimators: int | None = None,
        max_depth: int | None = None,
        n_jobs: int = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(horizon=horizon, **kwargs)
        cfg = settings.models
        self.n_estimators = n_estimators or cfg.rf_n_estimators
        self.max_depth = max_depth
        self.n_jobs = n_jobs
        self._model: Any = None
        self._feature_names: list[str] = []

    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: Any) -> "RandomForestForecaster":
        from sklearn.ensemble import RandomForestRegressor

        self._feature_names = list(X.columns)
        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            n_jobs=self.n_jobs,
            random_state=42,
        )
        self._model.fit(X[self._feature_names], y)
        logger.info(f"[RF] Fitted: {self.n_estimators} trees")
        self._fitted = True
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return self._model.predict(X[self._feature_names])

    def predict_interval(
        self, X: pd.DataFrame, alpha: float = 0.05
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quantile-based intervals using individual tree predictions."""
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        tree_preds = np.stack([
            t.predict(X[self._feature_names].values) for t in self._model.estimators_
        ])
        lo = np.percentile(tree_preds, alpha / 2 * 100, axis=0)
        hi = np.percentile(tree_preds, (1 - alpha / 2) * 100, axis=0)
        return lo, hi

    def feature_importance(self) -> pd.Series:
        if not self._fitted:
            raise RuntimeError("Not fitted.")
        return (
            pd.Series(self._model.feature_importances_, index=self._feature_names)
            .sort_values(ascending=False)
        )

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @classmethod
    def load(cls, path: str | Path) -> "RandomForestForecaster":
        return joblib.load(path)
