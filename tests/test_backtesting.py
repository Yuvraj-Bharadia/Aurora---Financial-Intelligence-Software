"""
Tests for the Aurora backtesting engine.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from aurora.backtesting.engine import BacktestEngine, BacktestResult


class TestBacktestEngine:
    @pytest.fixture
    def engine(self):
        return BacktestEngine(
            initial_capital=100_000,
            commission_bps=5,
            slippage_bps=2,
        )

    @pytest.fixture
    def signal_and_prices(self, synthetic_returns, synthetic_prices):
        signal = synthetic_returns.rolling(20).mean().dropna()
        prices = synthetic_prices.reindex(signal.index)
        return signal, prices

    def test_run_returns_result(self, engine, signal_and_prices):
        signal, prices = signal_and_prices
        result = engine.run(signal, prices, strategy="long_short")
        assert isinstance(result, BacktestResult)

    def test_returns_shape(self, engine, signal_and_prices):
        signal, prices = signal_and_prices
        result = engine.run(signal, prices, strategy="long_short")
        assert len(result.returns) > 0

    def test_returns_are_finite(self, engine, signal_and_prices):
        signal, prices = signal_and_prices
        result = engine.run(signal, prices, strategy="long_long")
        assert np.isfinite(result.returns.values).all()

    def test_long_only_no_short_positions(self, engine, signal_and_prices):
        signal, prices = signal_and_prices
        result = engine.run(signal, prices, strategy="long_only")
        assert (result.positions["weight"] >= 0).all()

    def test_metrics_present(self, engine, signal_and_prices):
        signal, prices = signal_and_prices
        result = engine.run(signal, prices, strategy="long_short")
        for key in ["sharpe", "sortino", "max_drawdown", "cagr", "hit_ratio"]:
            assert key in result.metrics

    def test_portfolio_value_starts_at_capital(self, engine, signal_and_prices):
        signal, prices = signal_and_prices
        result = engine.run(signal, prices, strategy="long_only")
        assert abs(result.portfolio_value.iloc[0] - 100_000) < 1_000

    def test_all_strategies_run(self, engine, signal_and_prices):
        signal, prices = signal_and_prices
        for strategy in ["long_only", "long_short", "market_neutral", "volatility_targeting"]:
            result = engine.run(signal, prices, strategy=strategy)
            assert result.metrics is not None
