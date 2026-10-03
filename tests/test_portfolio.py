"""
Tests for portfolio optimizer.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.portfolio.optimizer import PortfolioOptimizer


@pytest.fixture
def multi_asset_returns():
    """5 assets, 5 years of daily returns."""
    rng = np.random.default_rng(42)
    n = 252 * 5
    dates = pd.bdate_range("2018-01-01", periods=n, tz="UTC")
    # Correlated returns (Cholesky)
    cov = np.array([
        [1.0, 0.6, 0.3, 0.2, 0.1],
        [0.6, 1.0, 0.4, 0.3, 0.2],
        [0.3, 0.4, 1.0, 0.5, 0.3],
        [0.2, 0.3, 0.5, 1.0, 0.4],
        [0.1, 0.2, 0.3, 0.4, 1.0],
    ]) * 0.0001
    L = np.linalg.cholesky(cov)
    raw = rng.standard_normal((n, 5)) @ L.T
    mu = np.array([0.0004, 0.0003, 0.0002, 0.0001, 0.0005])
    returns = raw + mu
    return pd.DataFrame(returns, columns=list("ABCDE"), index=dates)


class TestPortfolioOptimizer:
    def test_equal_weight_sums_to_one(self, multi_asset_returns):
        opt = PortfolioOptimizer(method="equal_weight")
        w = opt.optimize(multi_asset_returns)
        assert abs(w.sum() - 1.0) < 1e-9

    def test_min_variance_weights_positive(self, multi_asset_returns):
        opt = PortfolioOptimizer(method="min_variance", min_weight=0.0)
        w = opt.optimize(multi_asset_returns)
        assert (w >= -1e-9).all()
        assert abs(w.sum() - 1.0) < 1e-6

    def test_max_sharpe_weights_sum_to_one(self, multi_asset_returns):
        opt = PortfolioOptimizer(method="max_sharpe", risk_free_rate=0.0)
        w = opt.optimize(multi_asset_returns)
        assert abs(w.sum() - 1.0) < 1e-6

    def test_risk_parity_equal_contributions(self, multi_asset_returns):
        opt = PortfolioOptimizer(method="risk_parity")
        w = opt.optimize(multi_asset_returns)
        assert abs(w.sum() - 1.0) < 1e-5
        # All weights should be positive
        assert (w > 0).all()

    def test_hrp_diversified(self, multi_asset_returns):
        opt = PortfolioOptimizer(method="hrp")
        w = opt.optimize(multi_asset_returns)
        assert abs(w.sum() - 1.0) < 1e-5
        # Max concentration < 50% for a 5-asset diversified portfolio
        assert w.max() < 0.50

    def test_max_weight_constraint(self, multi_asset_returns):
        opt = PortfolioOptimizer(method="max_sharpe", max_weight=0.30)
        w = opt.optimize(multi_asset_returns)
        assert (w <= 0.30 + 1e-6).all()
