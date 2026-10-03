"""
Hyperparameter optimization using Optuna.

Supports:
- Tree-structured Parzen Estimator (TPE) — Bayesian optimization
- CMA-ES for continuous parameter spaces
- Grid / random search

Objective functions optimize:
- Forecast RMSE (walk-forward)
- Out-of-sample Sharpe ratio
- Calmar ratio (CAGR / max drawdown)

Results are logged to MLflow and stored in an Optuna SQLite study.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable, Literal

import numpy as np
import optuna
import pandas as pd

from aurora.configs.config import settings
from aurora.evaluation.metrics import forecast_metrics, sharpe_ratio, calmar_ratio
from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


class HyperparameterOptimizer:
    """
    Optuna-based hyperparameter search for Aurora forecasters.

    Usage::

        optimizer = HyperparameterOptimizer(objective="sharpe")
        best_params = optimizer.optimize(
            model_class=LightGBMForecaster,
            X_train=X, y_train=y,
            n_trials=100,
        )
    """

    def __init__(
        self,
        objective: Literal["rmse", "sharpe", "calmar"] = "sharpe",
        sampler: Literal["tpe", "cmaes", "random"] = "tpe",
        n_cv_splits: int = 3,
        test_size: int = 63,
        study_name: str | None = None,
        storage: str | None = None,
        direction: str = "minimize",
    ) -> None:
        self.objective_metric = objective
        self.sampler = sampler
        self.n_cv_splits = n_cv_splits
        self.test_size = test_size
        self.study_name = study_name or f"aurora_{objective}_{int(time.time())}"
        self.storage = storage or f"sqlite:///{settings.models.model_dir}/optuna.db"
        self.direction = "minimize" if objective == "rmse" else "maximize"
        self._study: optuna.Study | None = None

    def optimize(
        self,
        model_class: type[BaseForecaster],
        X: pd.DataFrame,
        y: pd.Series,
        param_space: dict[str, Any] | None = None,
        n_trials: int = 50,
        timeout: int | None = None,
        n_jobs: int = 1,
    ) -> dict[str, Any]:
        """
        Run hyperparameter optimization.

        Args:
            model_class: Forecaster class to optimize.
            X: Feature matrix.
            y: Target series.
            param_space: Dict defining the search space via Optuna suggest_ calls.
                         Keys map to model constructor kwargs.
                         If None, uses the default space for model_class.
            n_trials: Number of Optuna trials.
            timeout: Maximum wall-clock time in seconds.
            n_jobs: Parallel jobs (-1 = all cores).

        Returns:
            Best hyperparameter dict.
        """
        param_space = param_space or self._default_param_space(model_class)

        sampler_obj = {
            "tpe": optuna.samplers.TPESampler(seed=42),
            "cmaes": optuna.samplers.CmaEsSampler(seed=42),
            "random": optuna.samplers.RandomSampler(seed=42),
        }[self.sampler]

        try:
            Path(self.storage.replace("sqlite:///", "")).parent.mkdir(parents=True, exist_ok=True)
            self._study = optuna.create_study(
                study_name=self.study_name,
                storage=self.storage,
                sampler=sampler_obj,
                direction=self.direction,
                load_if_exists=True,
            )
        except Exception:
            self._study = optuna.create_study(
                study_name=self.study_name,
                sampler=sampler_obj,
                direction=self.direction,
            )

        objective_fn = self._make_objective(model_class, X, y, param_space)
        self._study.optimize(
            objective_fn,
            n_trials=n_trials,
            timeout=timeout,
            n_jobs=n_jobs,
            catch=(Exception,),
        )

        best = self._study.best_params
        best_value = self._study.best_value
        logger.info(
            f"[Optuna] Optimization complete: {self.objective_metric}={best_value:.6f} | "
            f"best_params={best}"
        )
        return best

    def _make_objective(
        self,
        model_class: type[BaseForecaster],
        X: pd.DataFrame,
        y: pd.Series,
        param_space: dict[str, Any],
    ) -> Callable[[optuna.Trial], float]:
        """Create a walk-forward cross-validation objective function."""
        splits = self._get_splits(len(X))
        metric = self.objective_metric

        def objective(trial: optuna.Trial) -> float:
            params = self._suggest_params(trial, param_space)
            scores: list[float] = []

            for train_idx, test_idx in splits:
                try:
                    model = model_class(**params)
                    model.fit(X.iloc[train_idx], y.iloc[train_idx])
                    preds = model.predict(X.iloc[test_idx])
                    actual = y.iloc[test_idx].values
                    n = min(len(preds), len(actual))

                    if metric == "rmse":
                        score = float(np.sqrt(np.mean((actual[:n] - preds[:n]) ** 2)))
                    elif metric == "sharpe":
                        # Use forecast as signal: long when positive
                        signal = np.sign(preds[:n])
                        strategy_rets = signal * actual[:n]
                        score = sharpe_ratio(strategy_rets)
                    else:  # calmar
                        signal = np.sign(preds[:n])
                        strategy_rets = pd.Series(signal * actual[:n])
                        score = calmar_ratio(strategy_rets)

                    scores.append(score)
                except Exception as exc:
                    logger.debug(f"[Optuna] Trial failed: {exc}")
                    return float("inf") if metric == "rmse" else float("-inf")

            return float(np.mean(scores)) if scores else (
                float("inf") if metric == "rmse" else float("-inf")
            )

        return objective

    @staticmethod
    def _suggest_params(trial: optuna.Trial, space: dict[str, Any]) -> dict[str, Any]:
        """Translate space definition into Optuna suggest_ calls."""
        params: dict[str, Any] = {}
        for key, spec in space.items():
            kind = spec[0]
            if kind == "int":
                params[key] = trial.suggest_int(key, spec[1], spec[2])
            elif kind == "float":
                params[key] = trial.suggest_float(key, spec[1], spec[2], log=spec[3] if len(spec) > 3 else False)
            elif kind == "categorical":
                params[key] = trial.suggest_categorical(key, spec[1])
            elif kind == "loguniform":
                params[key] = trial.suggest_float(key, spec[1], spec[2], log=True)
        return params

    @staticmethod
    def _default_param_space(model_class: type[BaseForecaster]) -> dict[str, Any]:
        """Return a sensible default search space for common model classes."""
        name = getattr(model_class, "name", "")
        if name in ("xgboost", "lightgbm"):
            return {
                "n_estimators":  ("int",         200, 2000),
                "learning_rate": ("loguniform",   1e-3, 0.3),
                "max_depth":     ("int",          3, 10),
                "subsample":     ("float",        0.5, 1.0),
                "colsample_bytree": ("float",     0.5, 1.0),
            }
        elif name == "random_forest":
            return {
                "n_estimators": ("int",   100, 1000),
                "max_depth":    ("int",   3,   30),
            }
        elif name in ("lstm", "gru"):
            return {
                "hidden_dim":   ("int",         32, 256),
                "num_layers":   ("int",          1,   4),
                "dropout":      ("float",      0.0,  0.5),
                "learning_rate": ("loguniform", 1e-4, 1e-2),
                "seq_len":      ("int",         20, 120),
            }
        else:
            return {}

    def _get_splits(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Simple expanding walk-forward splits."""
        splits = []
        test_start = int(n * 0.7)
        chunk = (n - test_start) // self.n_cv_splits

        for i in range(self.n_cv_splits):
            t_start = test_start + i * chunk
            t_end = min(t_start + chunk, n)
            train_idx = np.arange(0, t_start)
            test_idx = np.arange(t_start, t_end)
            if len(train_idx) > 50 and len(test_idx) > 5:
                splits.append((train_idx, test_idx))

        return splits

    def importance_plot(self) -> Any:
        """Return Optuna parameter importance figure."""
        if self._study is None:
            raise RuntimeError("No study found.")
        try:
            return optuna.visualization.plot_param_importances(self._study)
        except Exception:
            return None

    def history_plot(self) -> Any:
        """Return Optuna optimization history figure."""
        if self._study is None:
            raise RuntimeError("No study found.")
        return optuna.visualization.plot_optimization_history(self._study)
