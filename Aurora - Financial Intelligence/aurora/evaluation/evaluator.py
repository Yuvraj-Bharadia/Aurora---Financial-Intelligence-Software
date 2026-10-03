"""
End-to-end model evaluation orchestrator.

Runs walk-forward out-of-sample evaluation across all registered models,
computes all metrics, and produces a structured comparison report.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from aurora.configs.config import settings
from aurora.evaluation.metrics import (
    deflated_sharpe_ratio,
    diebold_mariano_test,
    directional_metrics,
    forecast_metrics,
    mincer_zarnowitz,
    trading_metrics,
)
from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


class WalkForwardEvaluator:
    """
    Walk-forward (rolling / expanding window) cross-validation.

    For each fold:
    1. Train on the in-sample window.
    2. Predict on the out-of-sample window.
    3. Record errors and compute metrics.

    Args:
        n_splits: Number of test folds.
        test_size: Number of observations per test window.
        gap: Gap between train end and test start (to avoid leakage).
        expanding: Use expanding (vs rolling) training window.
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 63,  # one quarter
        gap: int = 1,
        expanding: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.gap = gap
        self.expanding = expanding

    def split(
        self, n: int
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Generate (train_indices, test_indices) tuples."""
        splits = []
        total_test = self.n_splits * self.test_size
        min_train = max(int(n * 0.5), 252)

        if n < min_train + total_test:
            raise ValueError(
                f"Not enough data: {n} rows for {self.n_splits} splits"
            )

        test_start = n - total_test
        for i in range(self.n_splits):
            t_start = test_start + i * self.test_size
            t_end = t_start + self.test_size
            train_end = t_start - self.gap

            if self.expanding:
                train_indices = np.arange(0, train_end)
            else:
                window = max(train_end - 252 * 2, 0)  # 2-year rolling
                train_indices = np.arange(window, train_end)

            test_indices = np.arange(t_start, min(t_end, n))
            if len(train_indices) > 0 and len(test_indices) > 0:
                splits.append((train_indices, test_indices))

        return splits


class ModelEvaluator:
    """
    Comprehensive model benchmarking with walk-forward validation.

    Produces:
    - Per-fold forecast accuracy metrics
    - Directional accuracy metrics
    - Regime-stratified performance
    - Diebold-Mariano pairwise tests
    - Summary ranking table
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 63,
        expanding: bool = True,
        risk_free_rate: float | None = None,
        use_purged_cv: bool = True,
        embargo_days: int = 5,
        max_lookback: int = 252,
    ) -> None:
        self.rfr = risk_free_rate or settings.backtest.risk_free_rate
        self._results: dict[str, pd.DataFrame] = {}
        if use_purged_cv:
            try:
                from aurora.evaluation.proof_package import PurgedWalkForward
                self.cv = PurgedWalkForward(
                    n_splits=n_splits,
                    test_size=test_size,
                    expanding=expanding,
                    embargo_days=embargo_days,
                    max_lookback=max_lookback,
                )
            except Exception:
                self.cv = WalkForwardEvaluator(
                    n_splits=n_splits, test_size=test_size, expanding=expanding
                )
        else:
            self.cv = WalkForwardEvaluator(
                n_splits=n_splits, test_size=test_size, expanding=expanding
            )

    def evaluate(
        self,
        models: dict[str, BaseForecaster],
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """
        Run walk-forward evaluation for all models.

        Args:
            models: Dict of model name → fitted BaseForecaster.
            X: Feature matrix.
            y: Target series (returns).
            regimes: Optional regime assignments for regime-stratified analysis.

        Returns:
            Summary DataFrame with all metrics, models as rows.
        """
        splits = self.cv.split(len(X))
        all_results: dict[str, list[dict[str, Any]]] = {name: [] for name in models}
        all_preds: dict[str, list[np.ndarray]] = {name: [] for name in models}
        all_actuals: list[np.ndarray] = []

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_train = X.iloc[train_idx]
            y_train = y.iloc[train_idx]
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx].values

            all_actuals.append(y_test)
            logger.info(
                f"Fold {fold_idx + 1}/{len(splits)}: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

            for model_name, model in models.items():
                try:
                    # Re-fit on this fold's training data
                    model.fit(X_train, y_train)
                    preds = model.predict(X_test)
                    n = min(len(preds), len(y_test))
                    preds, actual = preds[:n], y_test[:n]

                    fold_metrics = forecast_metrics(actual, preds)
                    dir_metrics = directional_metrics(actual, preds)
                    fold_metrics.update(dir_metrics)
                    fold_returns = pd.Series(preds * np.sign(actual))
                    fold_metrics["deflated_sharpe"] = deflated_sharpe_ratio(
                        fold_returns, n_trials=len(models)
                    )
                    fold_metrics["fold"] = fold_idx + 1
                    fold_metrics["model"] = model_name
                    all_results[model_name].append(fold_metrics)
                    all_preds[model_name].append(preds)

                except Exception as exc:
                    logger.error(f"Fold {fold_idx + 1} | {model_name}: {exc}")

        summary = self._build_summary(all_results)
        self._add_dm_tests(summary, all_preds, all_actuals)

        if regimes is not None:
            regime_summary = self._regime_stratified(
                models, X, y, regimes, splits
            )
            self._results["regime_performance"] = regime_summary

        self._results["summary"] = summary
        return summary

    def _build_summary(
        self, all_results: dict[str, list[dict[str, Any]]]
    ) -> pd.DataFrame:
        """Aggregate fold results into mean ± std summary."""
        rows = []
        for model_name, fold_list in all_results.items():
            if not fold_list:
                continue
            df_folds = pd.DataFrame(fold_list)
            numeric = df_folds.select_dtypes(include=np.number)
            row = {"model": model_name}
            for col in numeric.columns:
                row[f"{col}_mean"] = float(numeric[col].mean())
                row[f"{col}_std"] = float(numeric[col].std())
            rows.append(row)

        if not rows:
            return pd.DataFrame()
        return pd.DataFrame(rows).set_index("model").sort_values("rmse_mean")

    def _add_dm_tests(
        self,
        summary: pd.DataFrame,
        all_preds: dict[str, list[np.ndarray]],
        all_actuals: list[np.ndarray],
    ) -> None:
        """Append pairwise DM test p-values to the summary."""
        actuals = np.concatenate(all_actuals)
        model_names = list(all_preds.keys())
        for m1 in model_names:
            if not all_preds[m1]:
                continue
            p1 = np.concatenate(all_preds[m1])
            n = min(len(actuals), len(p1))
            e1 = actuals[:n] - p1[:n]

            for m2 in model_names:
                if m1 == m2 or not all_preds[m2]:
                    continue
                p2 = np.concatenate(all_preds[m2])
                n = min(len(actuals), len(p2), len(p1))
                e2 = actuals[:n] - p2[:n]
                dm = diebold_mariano_test(e1[:n], e2[:n])
                col = f"dm_vs_{m2}_pval"
                if m1 in summary.index:
                    summary.loc[m1, col] = dm["p_value"]

    def _regime_stratified(
        self,
        models: dict[str, BaseForecaster],
        X: pd.DataFrame,
        y: pd.Series,
        regimes: pd.DataFrame,
        splits: list[tuple[np.ndarray, np.ndarray]],
    ) -> pd.DataFrame:
        """Compute per-regime RMSE for each model."""
        rows = []
        regime_col = regimes["regime"].reindex(X.index)

        for _, test_idx in splits:
            X_test = X.iloc[test_idx]
            y_test = y.iloc[test_idx]
            r_test = regime_col.iloc[test_idx]

            for regime_name in r_test.unique():
                mask = r_test == regime_name
                if mask.sum() < 5:
                    continue
                for model_name, model in models.items():
                    try:
                        preds = model.predict(X_test[mask])
                        actual = y_test[mask].values
                        n = min(len(preds), len(actual))
                        err = float(np.sqrt(np.mean((actual[:n] - preds[:n]) ** 2)))
                        rows.append({
                            "model": model_name, "regime": regime_name, "rmse": err
                        })
                    except Exception:
                        pass

        if not rows:
            return pd.DataFrame()
        df = pd.DataFrame(rows)
        return df.groupby(["model", "regime"])["rmse"].mean().unstack().round(6)

    def get_results(self) -> dict[str, pd.DataFrame]:
        """Return all stored evaluation results."""
        return self._results
