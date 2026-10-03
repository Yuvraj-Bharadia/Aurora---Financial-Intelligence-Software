"""
Unit tests for evaluation metrics and the walk-forward evaluator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.evaluation.metrics import (
    mse, rmse, mae, r_squared, information_coefficient,
    sharpe_ratio, sortino_ratio, max_drawdown, cagr,
    directional_accuracy, forecast_metrics, trading_metrics,
    diebold_mariano_test, mincer_zarnowitz,
)
from aurora.evaluation.evaluator import WalkForwardEvaluator


# ── Forecast metrics ───────────────────────────────────────────────────────────

class TestForecastMetrics:
    def test_perfect_forecast(self):
        y = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert mse(y, y) == pytest.approx(0.0)
        assert rmse(y, y) == pytest.approx(0.0)
        assert mae(y, y) == pytest.approx(0.0)
        assert r_squared(y, y) == pytest.approx(1.0)

    def test_constant_forecast_r2_zero(self):
        y = np.random.default_rng(42).normal(0, 1, 100)
        const = np.zeros(100)
        r2 = r_squared(y, const)
        assert r2 <= 0.0

    def test_ic_perfect_rank(self):
        y = np.arange(100, dtype=float)
        ic = information_coefficient(y, y)
        assert ic == pytest.approx(1.0)

    def test_directional_accuracy_range(self):
        rng = np.random.default_rng(7)
        y = rng.normal(0, 0.01, 500)
        pred = rng.normal(0, 0.01, 500)
        da = directional_accuracy(y, pred)
        assert 0 <= da <= 1

    def test_forecast_metrics_keys(self):
        rng = np.random.default_rng(3)
        actual = rng.normal(0, 1, 100)
        predicted = actual + rng.normal(0, 0.1, 100)
        result = forecast_metrics(actual, predicted)
        for key in ["mse", "rmse", "mae", "mape", "r2", "ic"]:
            assert key in result


# ── Trading metrics ────────────────────────────────────────────────────────────

class TestTradingMetrics:
    @pytest.fixture
    def random_returns(self):
        rng = np.random.default_rng(99)
        dates = pd.bdate_range("2019-01-01", periods=252, tz="UTC")
        return pd.Series(rng.normal(0.0005, 0.01, 252), index=dates)

    def test_sharpe_positive_expectation(self, random_returns):
        """Positive mean returns should yield positive Sharpe (on average)."""
        positive = pd.Series(
            np.abs(random_returns.values) * 0.01 + 0.001, index=random_returns.index
        )
        sr = sharpe_ratio(positive, risk_free_rate=0.0)
        assert sr > 0

    def test_max_drawdown_non_positive(self, random_returns):
        cum = (1 + random_returns).cumprod()
        dd = max_drawdown(cum)
        assert dd <= 0

    def test_cagr_sign(self, random_returns):
        positive = pd.Series(
            np.full(252, 0.0003), index=random_returns.index
        )
        c = cagr(positive)
        assert c > 0

    def test_trading_metrics_keys(self, random_returns):
        m = trading_metrics(random_returns)
        for key in ["sharpe", "sortino", "calmar", "cagr", "max_drawdown", "hit_ratio"]:
            assert key in m


# ── Statistical tests ──────────────────────────────────────────────────────────

class TestStatisticalTests:
    def test_dm_test_equal_models(self):
        """Two identical forecasts should have DM stat ≈ 0."""
        rng = np.random.default_rng(5)
        e = rng.normal(0, 1, 200)
        result = diebold_mariano_test(e, e)
        assert abs(result["dm_stat"]) < 0.01

    def test_dm_test_better_model(self):
        """A clearly better model should reject H0."""
        rng = np.random.default_rng(5)
        e1 = rng.normal(0, 1, 500)      # larger errors
        e2 = rng.normal(0, 0.1, 500)    # much smaller errors
        result = diebold_mariano_test(e1, e2)
        # Model 2 is better → positive DM stat (e1² > e2²)
        assert result["p_value"] < 0.05

    def test_mz_efficient_forecast(self):
        """Perfect forecast should yield slope≈1, intercept≈0."""
        rng = np.random.default_rng(8)
        y = rng.normal(0, 1, 300)
        noise = rng.normal(0, 0.01, 300)
        result = mincer_zarnowitz(y, y + noise)
        assert abs(result["slope"] - 1.0) < 0.1
        assert abs(result["intercept"]) < 0.1


# ── Walk-forward evaluator ─────────────────────────────────────────────────────

class TestWalkForwardEvaluator:
    def test_split_count(self):
        ev = WalkForwardEvaluator(n_splits=5, test_size=50)
        splits = ev.split(800)
        assert len(splits) == 5

    def test_no_overlap(self):
        ev = WalkForwardEvaluator(n_splits=3, test_size=50, gap=1)
        splits = ev.split(500)
        for (tr, te), (tr2, te2) in zip(splits, splits[1:]):
            assert te.max() < te2.min()

    def test_train_before_test(self):
        ev = WalkForwardEvaluator(n_splits=4, test_size=40)
        splits = ev.split(600)
        for train, test in splits:
            assert train.max() < test.min()
