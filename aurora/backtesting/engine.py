"""
Institutional-grade backtesting engine for Aurora.

Simulates strategy P&L with realistic market microstructure:
- Bid-ask spread / slippage modelling
- Per-trade commissions in basis points
- Position sizing constraints
- Turnover tracking and cost attribution

Supports:
  - Long-only
  - Long-short
  - Market neutral
  - Volatility targeting

All returns are computed on a daily basis and compounded correctly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from aurora.configs.config import settings
from aurora.evaluation.metrics import max_drawdown, sharpe_ratio, sortino_ratio, cagr, trading_metrics
from aurora.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Trade:
    """Single trade record."""
    date: pd.Timestamp
    ticker: str
    direction: Literal["buy", "sell"]
    shares: float
    price: float
    commission: float
    slippage: float

    @property
    def gross_value(self) -> float:
        return abs(self.shares * self.price)

    @property
    def net_cost(self) -> float:
        return self.gross_value + self.commission + self.slippage


@dataclass
class BacktestResult:
    """Container for all backtest outputs."""
    returns: pd.Series
    positions: pd.DataFrame
    trades: list[Trade]
    portfolio_value: pd.Series
    metrics: dict[str, float] = field(default_factory=dict)
    regime_returns: dict[str, pd.Series] = field(default_factory=dict)


class BacktestEngine:
    """
    Event-driven backtesting engine with realistic cost modelling.

    Usage::

        engine = BacktestEngine()
        result = engine.run(
            signals=forecast_df["forecast"],
            prices=ohlcv["close"],
            strategy="long_short",
        )
    """

    def __init__(
        self,
        initial_capital: float | None = None,
        commission_bps: float | None = None,
        slippage_bps: float | None = None,
        max_position_size: float | None = None,
        risk_free_rate: float | None = None,
        annualization: int | None = None,
    ) -> None:
        cfg = settings.backtest
        self.initial_capital = initial_capital or cfg.initial_capital
        self.commission_bps = commission_bps or cfg.commission_bps
        self.slippage_bps = slippage_bps or cfg.slippage_bps
        self.max_position_size = max_position_size or cfg.max_position_size
        self.risk_free_rate = risk_free_rate or cfg.risk_free_rate
        self.annualization = annualization or cfg.annualization_factor

    def run(
        self,
        signals: pd.Series,
        prices: pd.DataFrame | pd.Series,
        strategy: Literal["long_only", "long_short", "market_neutral", "volatility_targeting"] = "long_short",
        regimes: pd.DataFrame | None = None,
        vol_target: float = 0.15,  # for volatility targeting
        rebalance_freq: int = 1,    # days between rebalances
        no_trade_gate_series: pd.Series | None = None,  # Layer 1: size_multipliers from NoTradeGate
    ) -> BacktestResult:
        """
        Run the backtest.

        Args:
            signals: Numeric signal series (positive → long, negative → short).
            prices: Price series or panel (DatetimeIndex aligned with signals).
            strategy: Trading strategy variant.
            regimes: Optional regime assignments for attribution.
            vol_target: Annualized target volatility for vol-targeting strategy.
            rebalance_freq: Rebalance every N days.

        Returns:
            BacktestResult with all statistics and trade log.
        """
        if isinstance(prices, pd.DataFrame):
            prices = prices.iloc[:, 0]

        aligned = pd.concat([signals, prices], axis=1, join="inner").dropna()
        aligned.columns = ["signal", "price"]

        # Layer 1: apply NoTradeGate size multipliers before position sizing
        if no_trade_gate_series is not None:
            gate = no_trade_gate_series.reindex(aligned.index).ffill().fillna(1.0)
            aligned["signal"] = aligned["signal"] * gate

        positions = self._compute_positions(
            aligned["signal"], aligned["price"], strategy, vol_target, rebalance_freq
        )
        trades = self._generate_trades(positions, aligned["price"])
        returns = self._compute_returns(positions, aligned["price"], trades)
        portfolio_value = self.initial_capital * (1 + returns).cumprod()

        metrics = trading_metrics(returns, self.risk_free_rate, self.annualization)
        metrics["turnover"] = self._turnover(positions)
        metrics["total_costs_bps"] = self._total_cost_bps(trades, portfolio_value)

        result = BacktestResult(
            returns=returns,
            positions=positions,
            trades=trades,
            portfolio_value=portfolio_value,
            metrics=metrics,
        )

        if regimes is not None:
            result.regime_returns = self._regime_attribution(returns, regimes)

        logger.info(
            f"Backtest complete: Sharpe={metrics['sharpe']:.3f} | "
            f"CAGR={metrics['cagr']:.1%} | MaxDD={metrics['max_drawdown']:.1%}"
        )
        return result

    # ── Position sizing ────────────────────────────────────────────────────────

    def _compute_positions(
        self,
        signal: pd.Series,
        price: pd.Series,
        strategy: str,
        vol_target: float,
        rebalance_freq: int,
    ) -> pd.DataFrame:
        """Map signals → position weights."""
        positions = pd.DataFrame(index=signal.index, columns=["weight"], dtype=float)

        if strategy == "long_only":
            # Rank signal: above median = long, else 0
            rolling_median = signal.rolling(20).median()
            raw_weight = (signal > rolling_median).astype(float)
            positions["weight"] = raw_weight.clip(0, self.max_position_size)

        elif strategy == "long_short":
            # Normalized signal: z-score → weight
            z = (signal - signal.rolling(60).mean()) / (signal.rolling(60).std() + 1e-9)
            positions["weight"] = z.clip(-1, 1) * self.max_position_size

        elif strategy == "market_neutral":
            # Equal-weight long-short; net exposure ≈ 0
            z = (signal - signal.rolling(60).mean()) / (signal.rolling(60).std() + 1e-9)
            positions["weight"] = z.clip(-self.max_position_size, self.max_position_size)
            # Demean to enforce market neutrality
            positions["weight"] -= positions["weight"].mean()

        elif strategy == "volatility_targeting":
            # Scale position inversely with realized volatility to hit vol_target
            ret = price.pct_change()
            realized_vol = ret.rolling(21).std() * np.sqrt(self.annualization)
            raw_z = (signal - signal.rolling(60).mean()) / (signal.rolling(60).std() + 1e-9)
            vol_scale = (vol_target / realized_vol.replace(0, np.nan)).clip(0.1, 5.0)
            positions["weight"] = (raw_z * vol_scale).clip(-self.max_position_size, self.max_position_size)

        # Enforce rebalance frequency (only update on rebalance days)
        if rebalance_freq > 1:
            mask = np.arange(len(positions)) % rebalance_freq == 0
            positions.loc[~positions.index[~mask], "weight"] = np.nan
            positions["weight"] = positions["weight"].ffill()

        return positions.fillna(0)

    def _generate_trades(
        self, positions: pd.DataFrame, price: pd.Series
    ) -> list[Trade]:
        """Log trades whenever position weight changes."""
        trades: list[Trade] = []
        prev_weight = 0.0
        portfolio_val = self.initial_capital

        for date, row in positions.iterrows():
            weight = float(row["weight"])
            delta = weight - prev_weight
            if abs(delta) < 1e-6:
                prev_weight = weight
                continue

            px = float(price.loc[date])
            shares = delta * portfolio_val / px
            gross = abs(shares * px)
            commission = gross * self.commission_bps / 10_000
            slippage = gross * self.slippage_bps / 10_000

            trades.append(Trade(
                date=date,
                ticker="portfolio",
                direction="buy" if delta > 0 else "sell",
                shares=shares,
                price=px,
                commission=commission,
                slippage=slippage,
            ))
            prev_weight = weight

        return trades

    def _compute_returns(
        self, positions: pd.DataFrame, price: pd.Series, trades: list[Trade]
    ) -> pd.Series:
        """Compute net daily returns after transaction costs."""
        price_returns = price.pct_change()
        strategy_returns = positions["weight"].shift(1) * price_returns

        # Subtract transaction costs as fraction of portfolio
        cost_series = pd.Series(0.0, index=positions.index)
        for trade in trades:
            if trade.date in cost_series.index:
                cost_frac = (trade.commission + trade.slippage) / self.initial_capital
                cost_series[trade.date] += cost_frac

        return (strategy_returns - cost_series).dropna()

    def _turnover(self, positions: pd.DataFrame) -> float:
        """Annualized one-way turnover."""
        weight_changes = positions["weight"].diff().abs()
        return float(weight_changes.mean() * self.annualization)

    def _total_cost_bps(
        self, trades: list[Trade], portfolio_value: pd.Series
    ) -> float:
        """Total transaction costs as basis points of average AUM."""
        total_cost = sum(t.commission + t.slippage for t in trades)
        avg_aum = float(portfolio_value.mean()) if len(portfolio_value) > 0 else self.initial_capital
        return float(total_cost / avg_aum * 10_000)

    def _regime_attribution(
        self, returns: pd.Series, regimes: pd.DataFrame
    ) -> dict[str, pd.Series]:
        """Split returns by regime for attribution analysis."""
        regime_col = regimes["regime"].reindex(returns.index)
        result: dict[str, pd.Series] = {}
        for regime_name in regime_col.dropna().unique():
            mask = regime_col == regime_name
            result[regime_name] = returns[mask]
        return result
