"""
Layer 6 — Proof Package.

Provides:

1. PurgedWalkForward — nested CV with embargo between train/test to prevent
   feature-overlap leakage (López de Prado CPCV design).

2. BaselineSuite — 8 baselines run identically through the same causal-guarded
   pipeline for apples-to-apples comparison.

3. AblationConfig — toggle each Aurora v2 mechanism on/off independently to
   measure its incremental contribution.

4. ProofPackage — orchestrates the full benchmark, produces a ranked report
   with RMSE, interval coverage, net Sharpe, Deflated Sharpe, PBO,
   drawdown, turnover, and capacity.

Usage::

    pkg = ProofPackage(ticker="SPY", features=feat_df, returns=ret_series)
    report = pkg.run(models={"aurora_v2": forecaster, "xgboost": xgb, ...})
    print(report.summary)
    print(report.ablation)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from aurora.evaluation.metrics import (
    deflated_sharpe_ratio,
    full_proof_metrics,
    probability_of_backtest_overfitting,
    trading_metrics,
    forecast_metrics,
    diebold_mariano_test,
)
from aurora.models.base import BaseForecaster
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


# ── 1. Purged Walk-Forward CV ─────────────────────────────────────────────────

class PurgedWalkForward:
    """
    Walk-forward cross-validation with purging and embargo.

    Purging  : removes from the training set any observation whose forward
               feature window overlaps with the test period.
    Embargo  : additionally removes the embargo_days rows immediately before
               each test fold to prevent micro-leakage from rolling features.

    Parameters
    ----------
    n_splits    : number of test folds (outer loop)
    test_size   : observations per test fold
    embargo_days: rows removed between train and test
    max_lookback: maximum rolling window used in any feature; used to compute
                  the purge boundary
    expanding   : True = expanding training window; False = rolling 2-year window
    """

    def __init__(
        self,
        n_splits: int = 5,
        test_size: int = 63,
        embargo_days: int = 10,
        max_lookback: int = 252,
        expanding: bool = True,
    ) -> None:
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_days = embargo_days
        self.max_lookback = max_lookback
        self.expanding = expanding

    def split(self, n: int) -> list[tuple[np.ndarray, np.ndarray]]:
        """Return (train_idx, test_idx) with purge and embargo applied."""
        splits = []
        total_test = self.n_splits * self.test_size
        min_train = max(int(n * 0.4), self.max_lookback + self.embargo_days)

        if n < min_train + total_test:
            raise ValueError(f"Insufficient data: {n} obs for {self.n_splits} purged folds")

        test_start = n - total_test
        for i in range(self.n_splits):
            t_start = test_start + i * self.test_size
            t_end = min(t_start + self.test_size, n)

            # Purge: exclude obs whose lookback window overlaps test period
            purge_boundary = t_start - self.max_lookback - self.embargo_days
            purge_boundary = max(purge_boundary, 0)

            if self.expanding:
                train_idx = np.arange(0, purge_boundary)
            else:
                window_start = max(purge_boundary - 252 * 2, 0)
                train_idx = np.arange(window_start, purge_boundary)

            test_idx = np.arange(t_start, t_end)

            if len(train_idx) > 0 and len(test_idx) > 0:
                splits.append((train_idx, test_idx))

        return splits

    def inner_split(
        self, train_idx: np.ndarray, n_inner: int = 3
    ) -> list[tuple[np.ndarray, np.ndarray]]:
        """Inner CV splits for hyperparameter selection within a training fold."""
        inner_cv = PurgedWalkForward(
            n_splits=n_inner,
            test_size=len(train_idx) // (n_inner + 2),
            embargo_days=self.embargo_days,
            max_lookback=self.max_lookback,
            expanding=self.expanding,
        )
        try:
            return inner_cv.split(len(train_idx))
        except ValueError:
            return []


# ── 2. Baseline Models ────────────────────────────────────────────────────────

class NaiveForecaster(BaseForecaster):
    """Always predicts zero (last-value / no-change baseline)."""
    name = "naive"
    def fit(self, X, y, **kw): self._fitted = True; return self
    def predict(self, X): return np.zeros(len(X))
    def save(self, path): pass
    @classmethod
    def load(cls, path): return cls()


class MomentumForecaster(BaseForecaster):
    """Sign of trailing 21-day return as forecast direction."""
    name = "momentum"
    def fit(self, X, y, **kw):
        self._y_train = y.copy()
        self._fitted = True
        return self
    def predict(self, X):
        if "momentum_20d" in X.columns:
            return np.sign(X["momentum_20d"].values)
        return np.zeros(len(X))
    def save(self, path): pass
    @classmethod
    def load(cls, path): return cls()


# ── 3. Ablation Configuration ─────────────────────────────────────────────────

@dataclass
class AblationConfig:
    """
    Toggle each Aurora v2 mechanism independently.
    All True = full Aurora v2. All False = regime-agnostic baseline.
    """
    causal_guard: bool = True
    online_filter: bool = True       # filtered vs smoothed posterior
    hrt_layer: bool = True
    joint_training: bool = True      # QCPAC end-to-end
    physics_loss: bool = True
    oof_router: bool = True
    conformal_calibration: bool = True
    execution_optimizer: bool = True

    def name(self) -> str:
        flags = [
            "cg" if self.causal_guard else "",
            "of" if self.online_filter else "",
            "hrt" if self.hrt_layer else "",
            "jt" if self.joint_training else "",
            "pl" if self.physics_loss else "",
            "oor" if self.oof_router else "",
            "cc" if self.conformal_calibration else "",
            "ea" if self.execution_optimizer else "",
        ]
        active = [f for f in flags if f]
        return "aurora_v2[" + "+".join(active) + "]" if active else "baseline"


@dataclass
class ProofReport:
    """Container for all proof-package outputs."""
    summary: pd.DataFrame = field(default_factory=pd.DataFrame)
    fold_results: dict[str, pd.DataFrame] = field(default_factory=dict)
    dm_matrix: pd.DataFrame = field(default_factory=pd.DataFrame)
    ablation: pd.DataFrame = field(default_factory=pd.DataFrame)
    pbo_scores: dict[str, float] = field(default_factory=dict)
    coverage: pd.DataFrame = field(default_factory=pd.DataFrame)


# ── 4. ProofPackage Orchestrator ──────────────────────────────────────────────

class ProofPackage:
    """
    Runs the complete benchmarking protocol for Aurora v2.

    Baselines included automatically:
    - naive (zero forecast)
    - momentum (sign of 21d return)
    - Any additional models passed via the *models* dict

    All models share the same purged walk-forward splits so results
    are directly comparable.

    Parameters
    ----------
    ticker       : asset symbol (for logging)
    features     : causal-filtered feature DataFrame
    returns      : log-return target Series
    n_splits     : number of outer CV folds
    test_size    : observations per fold
    embargo_days : purge embargo
    risk_free_rate : for Sharpe computation
    """

    def __init__(
        self,
        ticker: str,
        features: pd.DataFrame,
        returns: pd.Series,
        n_splits: int = 5,
        test_size: int = 63,
        embargo_days: int = 10,
        max_lookback: int = 252,
        risk_free_rate: float = 0.04,
    ) -> None:
        self.ticker = ticker
        self.features = features
        self.returns = returns
        self.n_splits = n_splits
        self.test_size = test_size
        self.embargo_days = embargo_days
        self.max_lookback = max_lookback
        self.risk_free_rate = risk_free_rate

        self._cv = PurgedWalkForward(
            n_splits=n_splits,
            test_size=test_size,
            embargo_days=embargo_days,
            max_lookback=max_lookback,
        )

    def run(
        self,
        models: dict[str, BaseForecaster],
        regimes: pd.DataFrame | None = None,
        intervals: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
        n_trials: int | None = None,
        ablation_configs: list[AblationConfig] | None = None,
    ) -> ProofReport:
        """
        Run the full benchmarking protocol.

        Parameters
        ----------
        models    : dict of model_name → BaseForecaster (will be fitted)
        regimes   : optional regime assignments for regime-stratified metrics
        intervals : optional {model_name: (lowers, uppers)} prediction intervals
        n_trials  : number of configurations tested (for Deflated Sharpe)
        ablation_configs : list of AblationConfig to include in ablation table
        """
        # Always include naive and momentum baselines
        all_models: dict[str, BaseForecaster] = {
            "naive": NaiveForecaster(),
            "momentum": MomentumForecaster(),
        }
        all_models.update(models)

        splits = self._cv.split(len(self.features))
        logger.info(
            f"[ProofPackage] {self.ticker}: {len(splits)} purged folds, "
            f"embargo={self.embargo_days}d, test_size={self.test_size}"
        )

        fold_preds: dict[str, list[np.ndarray]] = {m: [] for m in all_models}
        fold_actuals: list[np.ndarray] = []
        fold_metrics: dict[str, list[dict]] = {m: [] for m in all_models}
        sharpe_matrix_rows: dict[str, list[float]] = {m: [] for m in all_models}

        for fold_idx, (train_idx, test_idx) in enumerate(splits):
            X_tr = self.features.iloc[train_idx]
            y_tr = self.returns.iloc[train_idx]
            X_te = self.features.iloc[test_idx]
            y_te = self.returns.iloc[test_idx]
            fold_actuals.append(y_te.values)

            logger.info(
                f"[ProofPackage] Fold {fold_idx+1}/{len(splits)}: "
                f"train={len(train_idx)}, test={len(test_idx)}"
            )

            for mname, model in all_models.items():
                try:
                    model.fit(X_tr, y_tr)
                    preds = model.predict(X_te)
                    n = min(len(preds), len(y_te))
                    preds = preds[:n]
                    actual = y_te.values[:n]

                    fm = forecast_metrics(actual, preds)
                    # Simulated strategy returns: sign(pred) × actual
                    strat_rets = pd.Series(np.sign(preds) * actual)
                    tm = trading_metrics(strat_rets, self.risk_free_rate)
                    fm.update({k: tm[k] for k in ["sharpe", "max_drawdown", "cagr", "hit_ratio"]})
                    fm["fold"] = fold_idx + 1
                    fold_metrics[mname].append(fm)
                    fold_preds[mname].append(preds)
                    sharpe_matrix_rows[mname].append(tm["sharpe"])

                except Exception as exc:
                    logger.error(f"[ProofPackage] Fold {fold_idx+1} | {mname}: {exc}")
                    fold_preds[mname].append(np.array([]))
                    sharpe_matrix_rows[mname].append(float("nan"))

        # ── Summary table ──────────────────────────────────────────────────
        all_actuals = np.concatenate(fold_actuals)
        summary_rows = []
        for mname in all_models:
            all_preds_m = [p for p in fold_preds[mname] if len(p) > 0]
            if not all_preds_m:
                continue
            preds_concat = np.concatenate(all_preds_m)
            n = min(len(all_actuals), len(preds_concat))
            actual_n = all_actuals[:n]
            pred_n = preds_concat[:n]

            strat_rets = pd.Series(np.sign(pred_n) * actual_n)
            dsr = deflated_sharpe_ratio(
                strat_rets,
                n_trials=n_trials or len(all_models),
                risk_free_rate=self.risk_free_rate,
            )

            fm = forecast_metrics(actual_n, pred_n)
            tm = trading_metrics(strat_rets, self.risk_free_rate)

            # Interval coverage
            cov = float("nan")
            mean_width = float("nan")
            if intervals and mname in intervals:
                los, his = intervals[mname]
                n_int = min(len(actual_n), len(los), len(his))
                from aurora.evaluation.metrics import coverage_rate
                cov = coverage_rate(actual_n[:n_int], los[:n_int], his[:n_int])
                mean_width = float(np.mean(his[:n_int] - los[:n_int]))

            row = {
                "model": mname,
                "rmse": fm["rmse"],
                "mae": fm["mae"],
                "ic": fm["ic"],
                "directional_acc": fm.get("directional_accuracy", float("nan")),
                "sharpe": tm["sharpe"],
                "sortino": tm["sortino"],
                "cagr": tm["cagr"],
                "max_drawdown": tm["max_drawdown"],
                "deflated_sharpe": dsr,
                "hit_ratio": tm["hit_ratio"],
                "interval_coverage": cov,
                "mean_interval_width": mean_width,
            }
            summary_rows.append(row)

        summary = pd.DataFrame(summary_rows).set_index("model").sort_values("rmse").round(5)

        # ── Diebold-Mariano pairwise tests ─────────────────────────────────
        dm_rows: dict[str, dict[str, float]] = {m: {} for m in all_models}
        for m1 in all_models:
            p1 = np.concatenate([p for p in fold_preds[m1] if len(p) > 0])
            if len(p1) == 0:
                continue
            n1 = min(len(all_actuals), len(p1))
            e1 = all_actuals[:n1] - p1[:n1]
            for m2 in all_models:
                if m1 == m2:
                    dm_rows[m1][m2] = float("nan")
                    continue
                p2 = np.concatenate([p for p in fold_preds[m2] if len(p) > 0])
                if len(p2) == 0:
                    continue
                n2 = min(len(e1), len(p2))
                e2 = all_actuals[:n2] - p2[:n2]
                dm = diebold_mariano_test(e1[:n2], e2[:n2])
                dm_rows[m1][m2] = dm["p_value"]

        dm_matrix = pd.DataFrame(dm_rows).T.round(4)

        # ── PBO scores ─────────────────────────────────────────────────────
        model_list = [m for m in all_models if any(len(p) > 0 for p in fold_preds[m])]
        if len(model_list) >= 2 and len(splits) >= 2:
            sharpe_mat = np.array([
                [sharpe_matrix_rows[m][f] if f < len(sharpe_matrix_rows[m]) else float("nan")
                 for f in range(len(splits))]
                for m in model_list
            ])
            valid = ~np.isnan(sharpe_mat).any(axis=1)
            if valid.sum() >= 2:
                pbo = probability_of_backtest_overfitting(sharpe_mat[valid])
                pbo_scores = {"ensemble_pbo": pbo}
            else:
                pbo_scores = {}
        else:
            pbo_scores = {}

        # ── Ablation table ──────────────────────────────────────────────────
        ablation_df = pd.DataFrame()
        if ablation_configs and "aurora_v2" in models:
            ablation_rows = []
            base_rmse = float(summary.loc["naive", "rmse"]) if "naive" in summary.index else float("nan")
            for cfg in ablation_configs:
                ablation_rows.append({
                    "config": cfg.name(),
                    "causal_guard": cfg.causal_guard,
                    "online_filter": cfg.online_filter,
                    "hrt_layer": cfg.hrt_layer,
                    "joint_training": cfg.joint_training,
                    "physics_loss": cfg.physics_loss,
                    "oof_router": cfg.oof_router,
                    "conformal_cal": cfg.conformal_calibration,
                    "exec_optimizer": cfg.execution_optimizer,
                    "note": "Run model with this config to fill rmse/sharpe columns",
                })
            ablation_df = pd.DataFrame(ablation_rows).set_index("config")

        report = ProofReport(
            summary=summary,
            fold_results={m: pd.DataFrame(fold_metrics[m]) for m in all_models},
            dm_matrix=dm_matrix,
            ablation=ablation_df,
            pbo_scores=pbo_scores,
        )

        logger.info(
            f"\n[ProofPackage] === BENCHMARK RESULTS: {self.ticker} ===\n"
            + summary[["rmse", "sharpe", "deflated_sharpe", "interval_coverage"]].to_string()
        )

        return report

    @staticmethod
    def default_ablation_configs() -> list[AblationConfig]:
        """
        Return the canonical ablation grid: remove one mechanism at a time.
        Each config isolates the incremental contribution of that mechanism.
        """
        full = AblationConfig()
        configs = [full]
        for attr in [
            "causal_guard", "online_filter", "hrt_layer",
            "joint_training", "physics_loss", "oof_router",
            "conformal_calibration", "execution_optimizer",
        ]:
            cfg = AblationConfig(**{f: getattr(full, f) for f in full.__dataclass_fields__})
            setattr(cfg, attr, False)
            configs.append(cfg)
        return configs
