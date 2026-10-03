"""
Comprehensive evaluation metrics for Aurora forecasting and trading systems.

Covers:
  Forecast Accuracy  : MSE, RMSE, MAE, MAPE, SMAPE, R², IC
  Directional        : accuracy, precision, recall, F1
  Trading            : Sharpe, Sortino, Calmar, max drawdown, CAGR, hit ratio
  Statistical Tests  : Diebold-Mariano, Mincer-Zarnowitz regression
"""

from __future__ import annotations

import warnings
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


# ── Forecast accuracy ─────────────────────────────────────────────────────────

def mse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean((actual - predicted) ** 2))


def rmse(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.sqrt(mse(actual, predicted)))


def mae(actual: np.ndarray, predicted: np.ndarray) -> float:
    return float(np.mean(np.abs(actual - predicted)))


def mape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-9) -> float:
    """Mean Absolute Percentage Error (excludes near-zero actuals)."""
    mask = np.abs(actual) > eps
    return float(np.mean(np.abs((actual[mask] - predicted[mask]) / actual[mask])) * 100)


def smape(actual: np.ndarray, predicted: np.ndarray, eps: float = 1e-9) -> float:
    """Symmetric MAPE: bounded [0, 200%], scale-invariant."""
    denom = (np.abs(actual) + np.abs(predicted)) / 2 + eps
    return float(np.mean(np.abs(actual - predicted) / denom) * 100)


def r_squared(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Coefficient of determination."""
    ss_res = np.sum((actual - predicted) ** 2)
    ss_tot = np.sum((actual - actual.mean()) ** 2)
    return float(1 - ss_res / ss_tot) if ss_tot > 0 else 0.0


def information_coefficient(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Rank IC: Spearman correlation between predicted and actual ranks."""
    return float(stats.spearmanr(actual, predicted).statistic)


def forecast_metrics(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """Compute the full suite of forecast accuracy metrics."""
    n = min(len(actual), len(predicted))
    a, p = actual[:n], predicted[:n]
    return {
        "mse": mse(a, p),
        "rmse": rmse(a, p),
        "mae": mae(a, p),
        "mape": mape(a, p),
        "smape": smape(a, p),
        "r2": r_squared(a, p),
        "ic": information_coefficient(a, p),
    }


# ── Directional metrics ───────────────────────────────────────────────────────

def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Fraction of time the predicted direction matches actual direction."""
    correct = np.sign(actual) == np.sign(predicted)
    return float(correct.mean())


def directional_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    """Precision, recall, F1 for binary directional prediction."""
    from sklearn.metrics import classification_report
    y_true = (actual > 0).astype(int)
    y_pred = (predicted > 0).astype(int)
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    return {
        "directional_accuracy": directional_accuracy(actual, predicted),
        "precision": float(report.get("1", {}).get("precision", 0.0)),
        "recall": float(report.get("1", {}).get("recall", 0.0)),
        "f1": float(report.get("1", {}).get("f1-score", 0.0)),
    }


# ── Trading / portfolio metrics ───────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.04,
    annualization: int = 252,
) -> float:
    """Annualized Sharpe ratio."""
    r = pd.Series(returns).dropna()
    excess = r - risk_free_rate / annualization
    if excess.std() < 1e-9:
        return 0.0
    return float(excess.mean() / excess.std() * np.sqrt(annualization))


def sortino_ratio(
    returns: pd.Series | np.ndarray,
    risk_free_rate: float = 0.04,
    annualization: int = 252,
) -> float:
    """Annualized Sortino ratio (only downside deviation)."""
    r = pd.Series(returns).dropna()
    excess = r - risk_free_rate / annualization
    downside = excess[excess < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    return float(excess.mean() / downside_std * np.sqrt(annualization))


def calmar_ratio(returns: pd.Series | np.ndarray, annualization: int = 252) -> float:
    """CAGR / Max drawdown."""
    r = pd.Series(returns).dropna()
    cumulative = (1 + r).cumprod()
    cagr_val = cagr(r, annualization)
    mdd = max_drawdown(cumulative)
    if abs(mdd) < 1e-9:
        return 0.0
    return float(cagr_val / abs(mdd))


def max_drawdown(cum_returns: pd.Series | np.ndarray) -> float:
    """Maximum peak-to-trough drawdown."""
    cum = pd.Series(cum_returns)
    rolling_max = cum.cummax()
    drawdowns = (cum - rolling_max) / rolling_max
    return float(drawdowns.min())


def cagr(returns: pd.Series | np.ndarray, annualization: int = 252) -> float:
    """Compound Annual Growth Rate."""
    r = pd.Series(returns).dropna()
    total_return = (1 + r).prod()
    years = len(r) / annualization
    if years <= 0 or total_return <= 0:
        return 0.0
    return float(total_return ** (1 / years) - 1)


def hit_ratio(returns: pd.Series | np.ndarray) -> float:
    """Fraction of periods with positive return."""
    r = pd.Series(returns).dropna()
    return float((r > 0).mean())


def trading_metrics(
    returns: pd.Series,
    risk_free_rate: float = 0.04,
    annualization: int = 252,
) -> dict[str, float]:
    """Full suite of trading performance metrics."""
    cum = (1 + returns).cumprod()
    return {
        "sharpe": sharpe_ratio(returns, risk_free_rate, annualization),
        "sortino": sortino_ratio(returns, risk_free_rate, annualization),
        "calmar": calmar_ratio(returns, annualization),
        "cagr": cagr(returns, annualization),
        "max_drawdown": max_drawdown(cum),
        "hit_ratio": hit_ratio(returns),
        "ann_volatility": float(returns.std() * np.sqrt(annualization)),
        "skewness": float(returns.skew()),
        "kurtosis": float(returns.kurt()),
        "total_return": float(cum.iloc[-1] - 1) if len(cum) > 0 else 0.0,
    }


# ── Statistical tests ─────────────────────────────────────────────────────────

def diebold_mariano_test(
    e1: np.ndarray,
    e2: np.ndarray,
    loss_fn: str = "mse",
    h: int = 1,
) -> dict[str, float]:
    """
    Diebold-Mariano test for equal predictive accuracy.

    H0: Both models have equal predictive accuracy.
    Reject H0 (p < 0.05) → one model is statistically significantly better.

    Args:
        e1, e2: Forecast error arrays for model 1 and model 2.
        loss_fn: Loss differential ('mse', 'mae').
        h: Forecast horizon (for HAC correction).

    Returns:
        Dict with dm_stat and p_value.
    """
    if loss_fn == "mse":
        d = e1**2 - e2**2
    else:
        d = np.abs(e1) - np.abs(e2)

    n = len(d)
    d_bar = d.mean()
    # Newey-West variance estimator for h-step-ahead forecasts
    gamma0 = np.var(d, ddof=1)
    nw_var = gamma0
    for lag in range(1, h):
        gamma_l = np.cov(d[:-lag], d[lag:])[0, 1]
        nw_var += 2 * (1 - lag / (h + 1)) * gamma_l

    se = np.sqrt(nw_var / n)
    dm_stat = d_bar / (se + 1e-12)
    p_value = 2 * (1 - stats.t.cdf(abs(dm_stat), df=n - 1))
    return {"dm_stat": float(dm_stat), "p_value": float(p_value)}


def mincer_zarnowitz(
    actual: np.ndarray, predicted: np.ndarray
) -> dict[str, float]:
    """
    Mincer-Zarnowitz forecast rationality regression.

    Regresses actual on predicted: actual = a + b * predicted + error.
    Efficient forecast → a ≈ 0, b ≈ 1, R² high.
    """
    x = np.column_stack([np.ones(len(predicted)), predicted])
    result = np.linalg.lstsq(x, actual, rcond=None)
    intercept, slope = result[0]
    fitted = x @ result[0]
    r2 = r_squared(actual, fitted)
    return {"intercept": float(intercept), "slope": float(slope), "r2": float(r2)}


# ── Advanced proof-package metrics ────────────────────────────────────────────

def deflated_sharpe_ratio(
    returns: pd.Series | np.ndarray,
    n_trials: int = 1,
    risk_free_rate: float = 0.04,
    annualization: int = 252,
) -> float:
    """
    Deflated Sharpe Ratio (Bailey & López de Prado, 2016).

    Adjusts the observed Sharpe for:
    - Multiple testing (number of strategy trials)
    - Non-normality of returns (skewness and excess kurtosis)
    - Finite sample length

    A DSR < 0 indicates the strategy is likely overfit.

    Parameters
    ----------
    returns   : strategy daily return series
    n_trials  : total number of strategy configurations tested (inc. rejected)
    """
    from scipy.special import ndtri  # inverse normal CDF

    r = pd.Series(returns).dropna()
    n = len(r)
    if n < 4:
        return float("nan")

    sr = sharpe_ratio(r, risk_free_rate, annualization)
    skew = float(r.skew())
    kurt = float(r.kurt())  # excess kurtosis

    # Expected maximum Sharpe across n_trials under normality
    gamma_em = 0.5772156649  # Euler-Mascheroni constant
    if n_trials > 1:
        e_max_sr = (
            (1 - gamma_em) * ndtri(1 - 1.0 / n_trials)
            + gamma_em * ndtri(1 - 1.0 / (n_trials * np.e))
        )
    else:
        e_max_sr = 0.0

    # Non-normality adjustment factor
    adjustment = np.sqrt(
        (1 - skew * sr + (kurt / 4) * sr ** 2) / (n - 1)
    )
    if adjustment <= 0 or np.isnan(adjustment):
        return float("nan")

    from scipy.stats import norm
    dsr = norm.cdf((sr - e_max_sr) / adjustment)
    return float(dsr)


def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_train_combinations: int | None = None,
) -> float:
    """
    Probability of Backtest Overfitting (Bailey et al., 2017) via
    Combinatorial Purged Cross-Validation (CPCV).

    Parameters
    ----------
    returns_matrix : (n_configs, n_periods) array of OOS Sharpe ratios
                     one row per strategy configuration, one col per OOS fold
    n_train_combinations : number of training set combinations used
                           (defaults to n_periods // 2)

    Returns
    -------
    PBO ∈ [0, 1]; near 0 = low overfit risk, near 0.5 = random.
    """
    if returns_matrix.ndim != 2:
        raise ValueError("returns_matrix must be 2D: (n_configs, n_folds)")

    n_configs, n_folds = returns_matrix.shape
    if n_configs < 2 or n_folds < 2:
        return float("nan")

    # For each fold, rank configs by IS Sharpe (all folds except this one)
    # and record whether the IS winner also won OOS
    is_winner_oos_rank: list[float] = []
    for fold in range(n_folds):
        oos_col = returns_matrix[:, fold]
        is_cols = np.delete(returns_matrix, fold, axis=1)
        is_sharpe = is_cols.mean(axis=1)
        oos_sharpe = oos_col

        is_winner_idx = int(np.argmax(is_sharpe))
        oos_rank = int(np.sum(oos_sharpe < oos_sharpe[is_winner_idx])) / max(n_configs - 1, 1)
        is_winner_oos_rank.append(oos_rank)

    pbo = float(np.mean([r < 0.5 for r in is_winner_oos_rank]))
    return pbo


def coverage_rate(
    actuals: np.ndarray,
    lowers: np.ndarray,
    uppers: np.ndarray,
) -> float:
    """Fraction of actuals falling inside [lower, upper] prediction intervals."""
    covered = (actuals >= lowers) & (actuals <= uppers)
    return float(covered.mean())


def capacity_estimate(
    returns: pd.Series,
    base_sharpe: float,
    impact_coeff: float = 0.1,
    target_sharpe_fraction: float = 0.75,
    annualization: int = 252,
) -> float:
    """
    Estimate the AUM at which market impact degrades Sharpe by (1 - target_sharpe_fraction).

    Uses Almgren-Chriss: impact ∝ √AUM, so we find AUM where
    Sharpe(AUM) = base_sharpe * target_sharpe_fraction.

    Returns AUM in the same currency units as assumed by the returns.
    """
    r = pd.Series(returns).dropna()
    daily_vol = float(r.std())
    if daily_vol < 1e-9 or base_sharpe <= 0:
        return float("nan")

    # Sharpe degrades as: SR(AUM) = SR_0 - impact_coeff * sqrt(AUM) / (daily_vol * sqrt(252))
    target_sr = base_sharpe * target_sharpe_fraction
    sr_loss = base_sharpe - target_sr
    denom = impact_coeff / (daily_vol * np.sqrt(annualization))
    if denom <= 0:
        return float("nan")
    capacity = (sr_loss / denom) ** 2
    return float(capacity)


def full_proof_metrics(
    returns: pd.Series,
    actuals: np.ndarray,
    predicted: np.ndarray,
    lowers: np.ndarray | None = None,
    uppers: np.ndarray | None = None,
    n_trials: int = 1,
    risk_free_rate: float = 0.04,
) -> dict[str, float]:
    """Compute the complete proof-package metric suite in one call."""
    metrics: dict[str, float] = {}
    metrics.update(forecast_metrics(actuals, predicted))
    metrics.update(directional_metrics(actuals, predicted))
    metrics.update(trading_metrics(returns, risk_free_rate))
    metrics["deflated_sharpe"] = deflated_sharpe_ratio(returns, n_trials, risk_free_rate)
    metrics["capacity_usd"] = capacity_estimate(returns, metrics.get("sharpe", 0.0))

    if lowers is not None and uppers is not None:
        metrics["interval_coverage"] = coverage_rate(actuals, lowers, uppers)
        metrics["mean_interval_width"] = float(np.mean(uppers - lowers))

    dm = diebold_mariano_test(actuals - predicted, np.zeros_like(actuals))
    metrics["dm_vs_naive_pval"] = dm["p_value"]

    mz = mincer_zarnowitz(actuals, predicted)
    metrics["mz_intercept"] = mz["intercept"]
    metrics["mz_slope"] = mz["slope"]
    metrics["mz_r2"] = mz["r2"]

    return metrics
